from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent
from spatial_omics_toolkit.application.intake import read_project_samples_bytes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(run_dir: Path, event: StageEvent | dict[str, Any]) -> None:
    payload = event.model_dump(mode="json") if isinstance(event, StageEvent) else event
    event_path = run_dir / "events.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps({"timestamp": utc_now(), **payload}, default=_json_default, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

@contextmanager
def _status_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".status.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_status(status_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {}
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return status if isinstance(status, dict) else {}


def update_status(run_dir: Path, status: str, message: str | None = None, **details: Any) -> None:
    status_path = run_dir / "status.json"
    with _status_lock(run_dir):
        current = _read_status(status_path)
        current.update({"status": status, "updated_at": utc_now(), **details})
        if message is not None:
            current["message"] = message
        atomic_json(status_path, current)


def update_status_if(
    run_dir: Path,
    expected_statuses: set[str],
    status: str,
    message: str,
    expected_job_id: str,
    **details: Any,
) -> bool:
    status_path = run_dir / "status.json"
    job_path = run_dir / "job.json"
    with _status_lock(run_dir):
        current = _read_status(status_path)
        if current.get("status") not in expected_statuses:
            return False
        if current.get("slurm_job_id") != expected_job_id:
            return False
        job_metadata = _read_status(job_path)
        if job_metadata.get("slurm_job_id") != expected_job_id:
            return False
        current.update({"status": status, "updated_at": utc_now(), **details})
        current["message"] = message
        atomic_json(status_path, current)
        return True


def record_slurm_job(run_dir: Path, job_id: str, job_type: str, resources: dict[str, Any], attempt: int) -> None:
    job_path = run_dir / "job.json"
    status_path = run_dir / "status.json"
    with _status_lock(run_dir):
        submitted_at = utc_now()
        atomic_json(
            job_path,
            {
                "slurm_job_id": job_id,
                "submitted_at": submitted_at,
                "job_type": job_type,
            },
        )
        current = _read_status(status_path)
        current.update(
            {
                "slurm_job_id": job_id,
                "job_type": job_type,
                "resources": resources,
                "attempt": attempt,
                "updated_at": submitted_at,
            }
        )
        atomic_json(status_path, current)


def create_run_bundle(request: RunRequest, descriptor: dict[str, Any], resolved_parameters: dict[str, Any]) -> dict[str, Any]:
    if request.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {request.run_dir}")
    annotation_bytes = request.annotation_csv.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    expected_annotation_sha256 = resolved_parameters.get("annotation_csv_sha256")
    if expected_annotation_sha256 and expected_annotation_sha256 != annotation_sha256:
        raise ValueError("project_samples.csv changed after final preflight; rerun preflight before submission.")
    samples, blockers = read_project_samples_bytes(annotation_bytes, str(request.annotation_csv))
    if blockers:
        raise ValueError("Cannot submit invalid project_samples.csv: " + "; ".join(blockers))
    request.run_dir.mkdir(parents=True, exist_ok=False)
    (request.run_dir / "inputs").mkdir()
    (request.run_dir / "logs").mkdir()
    (request.run_dir / "results").mkdir()
    annotation_snapshot = request.run_dir / "inputs" / "project_samples.csv"
    snapshot_temporary = annotation_snapshot.with_suffix(".csv.tmp")
    snapshot_temporary.write_bytes(annotation_bytes)
    os.replace(snapshot_temporary, annotation_snapshot)
    settings = resolved_parameters.get("_settings_profile")
    if isinstance(settings, dict):
        for key, filename in (("source_toml", "settings_source.toml"), ("effective_toml", "settings_effective.toml")):
            text = settings.get(key)
            if not isinstance(text, str):
                raise ValueError(f"Missing frozen {key}")
            snapshot = request.run_dir / "inputs" / filename
            snapshot.write_text(text, encoding="utf-8")
            snapshot.chmod(0o440)
        config_files = settings.get("config_files", {})
        if not isinstance(config_files, dict):
            raise ValueError("Invalid frozen configuration file metadata")
        config_snapshots: dict[str, dict[str, str]] = {}
        for index, (name, info) in enumerate(sorted(config_files.items())):
            if not isinstance(info, dict) or not isinstance(info.get("path"), str) or not isinstance(info.get("sha256"), str):
                raise ValueError(f"Invalid frozen configuration metadata for {name}")
            snapshot = request.run_dir / "inputs" / f"settings_config_{index}.json"
            digest = hashlib.sha256()
            with Path(info["path"]).open("rb") as source, snapshot.open("xb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Configuration file for {name} changed after preflight")
            snapshot.chmod(0o440)
            config_snapshots[name] = {"path": f"inputs/{snapshot.name}", "sha256": digest.hexdigest()}
        atomic_json(request.run_dir / "inputs" / "settings_configs.json", config_snapshots)
    annotation_snapshot.chmod(0o440)
    atomic_json(request.run_dir / "inputs" / "manifest.json", {
        "modality_dir": str(request.modality_dir.absolute()),
        "modality_dir_resolved": str(request.modality_dir.resolve()),
        "project_root": str(request.project_root.absolute()),
        "project_root_resolved": str(request.project_root.resolve()),
        "annotation_csv": str(request.annotation_csv.absolute()),
        "annotation_csv_snapshot": str(annotation_snapshot.resolve()),
        "annotation_csv_sha256": annotation_sha256,
        "inventory_status": "pending_worker_inventory",
    })
    metadata = {
        "schema_version": "1.0",
        "run_id": str(request.run_id),
        "workflow_id": request.workflow_id,
        "modality": request.modality,
        "created_at": utc_now(),
        "project": {
            "root": str(request.project_root.absolute()),
            "root_resolved": str(request.project_root.resolve()),
            "annotation_csv": str(request.annotation_csv.absolute()),
            "annotation_csv_resolved": str(request.annotation_csv.resolve()),
            "annotation_csv_snapshot": str(annotation_snapshot.resolve()),
            "annotation_csv_sha256": annotation_sha256,
            "samples": samples,
        },
        "input": {
            "modality_dir": str(request.modality_dir.absolute()),
            "modality_dir_resolved": str(request.modality_dir.resolve()),
            "inventory": "inputs/manifest.json",
            "annotation_csv_snapshot": "inputs/project_samples.csv",
        },
        "output_root": str(request.output_root.resolve()),
        "run_dir": str(request.run_dir.resolve()),
        "workflow": descriptor,
        "parameters": resolved_parameters,
    }
    atomic_json(request.run_dir / "mapping_receipt.json", {
        "annotation_csv": str(request.annotation_csv.absolute()),
        "annotation_csv_resolved": str(request.annotation_csv.resolve()),
        "annotation_csv_snapshot": str(annotation_snapshot.resolve()),
        "annotation_csv_sha256": annotation_sha256,
        "specimen_mapping": resolved_parameters.get("specimen_mapping", {}),
        "core_to_specimen": resolved_parameters.get("core_to_specimen", {}),
        "specimen_mapping_confirmed": resolved_parameters.get("specimen_mapping_confirmed", False),
        "acquisition_groups": resolved_parameters.get("acquisition_groups", []),
    })
    atomic_json(request.run_dir / "run.json", metadata)
    atomic_json(request.run_dir / "resolved_parameters.json", resolved_parameters)
    atomic_json(request.run_dir / "status.json", {"status": "SUBMITTED", "created_at": utc_now(), "updated_at": utc_now(), "message": "Run bundle created; awaiting SLURM worker."})
    atomic_json(request.run_dir / "artifacts.json", {"artifacts": []})
    append_event(request.run_dir, {"stage_id": "submission", "status": "CREATED", "message": "Run bundle created", "tool_versions": {}, "parameters": resolved_parameters, "artifacts": []})
    return metadata
