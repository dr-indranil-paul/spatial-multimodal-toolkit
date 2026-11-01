from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import (
    FunctionReference,
    ParameterField,
    PreflightResult,
    RunRequest,
    StageDefinition,
    StageEvent,
    ToolReference,
    WorkflowDescriptor,
    WorkflowEmitter,
    WorkflowResult,
)


import hashlib
import platform
from datetime import datetime, timezone


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _stage_parameters(parameters: dict[str, Any], stage_id: str) -> dict[str, Any]:
    if stage_id == "intake":
        return {"intake": parameters.get("intake", {})}
    if stage_id == "conversion":
        return {"conversion": parameters.get("conversion", {})}
    if stage_id == "registration":
        return {"intake": parameters.get("intake", {}), "registration": {key: value for key, value in parameters.get("registration", {}).items() if key != "marker_panel_csv" and key != "marker_panel_confirmed"}}
    if stage_id == "export":
        panel = parameters.get("_resolved_marker_panel", {})
        return {"marker_panel_sha256": panel.get("sha256"), "marker_map": panel.get("marker_map"), "registration": parameters.get("registration", {}).get("export_config", {})}
    return {"segmentation": parameters.get("segmentation", {})}


def _input_fingerprint(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "inputs" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run input inventory is missing; checkpoint receipt cannot be created.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = sorted((str(entry.get("path")), str(entry.get("sha256"))) for entry in manifest.get("files", []) if entry.get("sha256"))
    if not hashes:
        raise ValueError("Run input inventory contains no hashed files.")
    return {"inventory_sha256": _canonical_hash(hashes), "annotation_csv_sha256": manifest.get("annotation_csv_sha256")}


def _source_code_identity(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in sorted(paths.items())}


def _tool_identity(stage_id: str) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    shared_sources = {
        "application/workflows/cycif.py": Path(__file__).resolve(),
        "application/worker.py": package_root / "application" / "worker.py",
        "application/intake.py": package_root / "application" / "intake.py",
        "application/runs.py": package_root / "application" / "runs.py",
        "cycif/__init__.py": package_root / "cycif" / "__init__.py",
        "cycif/config.py": package_root / "cycif" / "config.py",
    }
    stage_sources = {
        "intake": ("cycif/intake.py",),
        "conversion": ("cycif/convert.py",),
        "registration": ("cycif/register.py",),
        "export": ("cycif/register.py", "cycif/export_images.py"),
        "segmentation": ("cycif/export_images.py", "cycif/segment.py"),
    }
    if stage_id not in stage_sources:
        raise ValueError(f"Unsupported checkpoint stage for tool identity: {stage_id}")
    source_paths = {
        **shared_sources,
        **{
            name: package_root / name
            for name in stage_sources[stage_id]
        },
    }
    identity = {
        "spatial_omics_toolkit": _version("spatial-omics-toolkit"),
        "python": platform.python_version(),
        **{f"source:{name}": digest for name, digest in _source_code_identity(source_paths).items()},
    }
    for package in ("numpy", "opencv-python", "opencv-python-headless", "tifffile"):
        version = _version(package)
        if version != "not installed":
            identity[package] = version
    if stage_id == "segmentation":
        software_root = os.environ.get("PAULI_FAST_SOFTWARES")
        if software_root:
            mcmicro_home = Path(software_root) / "envs" / "spatial-proteomics" / "mcmicro"
            runtime_files = {
                "mcmicro_runtime_lock": Path(software_root) / "config" / "workflow-runtime.lock.sh",
                "mcmicro_main_nf": mcmicro_home / "main.nf",
            }
            identity.update({
                f"runtime:{name}": _sha256(path)
                for name, path in runtime_files.items()
                if path.is_file()
            })
    return identity


def _write_checkpoint_receipts(request: RunRequest, events: list[StageEvent]) -> None:
    fingerprint = _input_fingerprint(request.run_dir)
    receipts: dict[str, Any] = {}
    for stage_id in ("intake", "conversion", "registration", "export", "segmentation"):
        stage_events = [event for event in events if event.stage_id == stage_id]
        if not stage_events or any(event.status not in {"RUNNING", "SUCCEEDED", "WARNING"} for event in stage_events) or not any(event.status in {"SUCCEEDED", "WARNING"} for event in stage_events):
            continue
        if stage_id == "intake" and any(
            event.status == "WARNING"
            and any(event.parameters.get(key) for key in ("check_warnings", "coverage_missing", "unparsed"))
            for event in stage_events
        ):
            continue
        paths = sorted({Path(path).resolve() for event in stage_events for path in event.artifacts})
        if not paths or any(not path.is_file() for path in paths):
            continue
        artifacts = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
        settings = _stage_parameters(request.parameters, stage_id)
        tool_identity = _tool_identity(stage_id)
        for event in stage_events:
            if event.status in {"SUCCEEDED", "WARNING"}:
                tool_identity.update(event.tool_versions)
        receipts[stage_id] = {
            "workflow_id": request.workflow_id,
            "stage_id": stage_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint,
            "settings_sha256": _canonical_hash(settings),
            "settings": settings,
            "tool_identity": tool_identity,
            "stage_status": "WARNING" if any(event.status == "WARNING" for event in stage_events) else "SUCCEEDED",
            "artifacts": artifacts,
        }
    if receipts:
        from spatial_omics_toolkit.application.runs import atomic_json
        atomic_json(request.run_dir / "checkpoint_receipts.json", {"workflow_id": request.workflow_id, "checkpoints": receipts})



def validate_continuation(request: RunRequest) -> tuple[list[str], list[dict[str, str]]]:
    """Validate immutable source receipts against the new run's current inventory."""
    selection = request.parameters.get("continuation", {})
    if not isinstance(selection, dict) or selection.get("mode") != "continue":
        return [], []
    source = Path(str(selection.get("parent_run_dir", ""))).expanduser().resolve()
    canonical_runs = (request.output_root / "CycIF" / "runs").resolve()
    if not source.is_relative_to(canonical_runs):
        return ["Selected source run must resolve beneath this project's CyCIF/runs directory."], []
    start_stage = selection.get("start_stage")
    required = {
        "conversion": ("intake",),
        "registration": ("intake",),
        "export": ("intake", "registration"),
        "segmentation": ("intake", "registration", "export"),
    }
    if start_stage not in required:
        return ["Choose a supported continuation boundary after intake, conversion, registration, or export."], []
    reasons: list[str] = []
    if not source.is_dir():
        return [f"Selected source run does not exist: {source}"], []
    try:
        metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
        parent_inventory = json.loads((source / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        manifest_inventory = json.loads((request.run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        child_metadata = json.loads((request.run_dir / "run.json").read_text(encoding="utf-8"))
        source_annotation = Path(child_metadata["project"]["annotation_csv"]).expanduser()
        from spatial_omics_toolkit.application.intake import sha256_file
        current_fingerprint = _input_fingerprint(request.run_dir)
        live_annotation_hash = sha256_file(source_annotation)
        if live_annotation_hash != manifest_inventory.get("annotation_csv_sha256"):
            return ["Current project_samples.csv changed after child-run submission; re-inventory and submit a fresh continuation."], []
        if live_annotation_hash != current_fingerprint.get("annotation_csv_sha256"):
            return ["project_samples.csv snapshot differs from the current project metadata."], []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"Source run is legacy or missing required checkpoint receipts/inventory: {exc}"], []
    if metadata.get("workflow_id") != "cycif-raw-czi" or receipt.get("workflow_id") != "cycif-raw-czi":
        reasons.append("Source workflow ID does not match cycif-raw-czi.")
    if source.parent.name != "runs" or source.parent.parent.name != "CycIF":
        reasons.append("Selected source is not a CyCIF run bundle under a canonical `CycIF/runs/<run_id>` path.")
    if str(metadata.get("run_id", "")) != source.name or metadata.get("modality") != "CycIF":
        reasons.append("Source run identity or modality does not match its run-bundle path.")
    if status.get("status") not in {"SUCCEEDED", "WARNING", "FAILED", "BLOCKED"}:
        reasons.append(f"Source run has no terminal workflow status: {status.get('status', 'missing')}.")
    if receipt.get("checkpoints") is not None and receipt.get("checkpoints", {}).get("intake", {}).get("input_fingerprint") != current_fingerprint:
        reasons.append("Current raw-CZI inventory or project_samples.csv fingerprint differs from the source run.")
    parent_hashes = sorted((str(entry.get("path")), str(entry.get("sha256"))) for entry in parent_inventory.get("files", []) if entry.get("sha256"))
    if _canonical_hash(parent_hashes) != current_fingerprint.get("inventory_sha256"):
        reasons.append("Current raw-CZI file hashes differ from the source run inventory.")
    checkpoints = receipt.get("checkpoints", {})
    try:
        source_events = [json.loads(line) for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        source_events = []
    latest_status: dict[str, str] = {}
    for source_event in source_events:
        stage_name = source_event.get("stage_id")
        if isinstance(stage_name, str):
            latest_status[stage_name] = str(source_event.get("status", ""))

    source_stage_status: dict[str, str] = {}
    warning_stages: list[str] = []
    for stage_id in required[start_stage]:
        checkpoint = checkpoints.get(stage_id)
        latest = latest_status.get(stage_id, "missing")
        if latest == "REUSED" and isinstance(checkpoint, dict):
            latest = str(checkpoint.get("stage_status", latest))
        source_stage_status[stage_id] = latest
        if latest == "WARNING":
            warning_stages.append(stage_id)
    if warning_stages and selection.get("warning_review_confirmed") is not True:
        reasons.append(f"Explicitly review the source QC report and confirm reuse of warning checkpoints: {warning_stages}.")
    reused: list[dict[str, str]] = []
    for stage_id in required[start_stage]:
        checkpoint = checkpoints.get(stage_id)
        if not isinstance(checkpoint, dict):
            reasons.append(f"Source run has no reusable {stage_id} checkpoint receipt.")
            continue
        stage_status = source_stage_status[stage_id]
        if stage_status not in {"SUCCEEDED", "WARNING", "REUSED"}:
            reasons.append(f"{stage_id} upstream stage is not complete and reusable; latest status is {stage_status}.")
        if checkpoint.get("workflow_id") != "cycif-raw-czi" or checkpoint.get("input_fingerprint") != current_fingerprint:
            reasons.append(f"{stage_id} checkpoint is bound to a different workflow or input fingerprint.")
            continue
        current_settings = _stage_parameters(request.parameters, stage_id)
        if checkpoint.get("settings_sha256") != _canonical_hash(current_settings):
            reasons.append(f"{stage_id} checkpoint settings differ from the current configuration.")
        expected_tools = _tool_identity(stage_id)
        if checkpoint.get("tool_identity") != expected_tools:
            reasons.append(f"{stage_id} checkpoint tool/runtime identity differs from the current worker.")
        artifacts = checkpoint.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            reasons.append(f"{stage_id} checkpoint has no artifact hashes.")
            continue
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256(path) != artifact.get("sha256"):
                reasons.append(f"{stage_id} checkpoint artifact is missing or modified: {path}")
            else:
                reused.append({"stage_id": stage_id, "path": str(path), "sha256": str(artifact["sha256"]), "source_status": stage_status})
    return reasons, reused


def _warning_reuse_messages(reused: list[dict[str, str]]) -> list[str]:
    warning_stages = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
    return [
        f"Reused {stage_id} from a WARNING checkpoint; the source QC limitations remain applicable."
        for stage_id in warning_stages
    ]


def _execute_export_resume(request: RunRequest, emit: WorkflowEmitter, reused: list[dict[str, str]]) -> WorkflowResult:
    from spatial_omics_toolkit.cycif import step04_export_registered

    selection = request.parameters["continuation"]
    source = Path(selection["parent_run_dir"]).expanduser().resolve()
    source_receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    checkpoints = source_receipt["checkpoints"]
    handoffs = sorted(
        Path(item["path"])
        for item in checkpoints["registration"]["artifacts"]
        if Path(item["path"]).name.endswith("_registration_handoff.json")
    )
    if not handoffs:
        message = "Validated registration checkpoint has no durable per-scene registration handoff."
        event = StageEvent(stage_id="export", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    marker_panel = request.parameters.get("_resolved_marker_panel")
    marker_map = marker_panel.get("marker_map") if isinstance(marker_panel, dict) else None
    if not isinstance(marker_map, dict) or not marker_map:
        message = "Export continuation requires the same explicitly confirmed marker panel used by the source registration run."
        event = StageEvent(stage_id="export", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    preflight_result = preflight(request)
    if not preflight_result.ready:
        message = "Export continuation preflight blocked: " + "; ".join(preflight_result.blockers)
        event = StageEvent(stage_id="export", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=preflight_result.blockers)
    resume_warnings = list(preflight_result.warnings) + _warning_reuse_messages(reused)
    reused_events: list[StageEvent] = []
    for stage_id in ("intake", "registration"):
        source_status = next((item.get("source_status", "SUCCEEDED") for item in reused if item["stage_id"] == stage_id), "SUCCEEDED")
        event = StageEvent(
            stage_id=stage_id,
            status="REUSED",
            message=f"Reused verified {stage_id} checkpoint from parent run {source.name} (source status {source_status}).",
            parameters={"parent_run_dir": str(source), "start_stage": "export", "source_status": source_status},
            artifacts=[item["path"] for item in reused if item["stage_id"] == stage_id],
        )
        emit(event)
        reused_events.append(event)
    events: list[StageEvent] = []
    output_artifacts: list[str] = []
    for handoff_path in handoffs:
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        scene_label = str(handoff["scene_label"])
        reg_result = _load_registration_handoff(handoff_path)
        if set(marker_map) != set(reg_result["round_order"]):
            message = f"Marker-panel rounds {sorted(marker_map)} do not match registered rounds for {scene_label}: {reg_result['round_order']}."
            event = StageEvent(stage_id="export", status="BLOCKED", message=message, parameters={"parent_run_dir": str(source), "scene_label": scene_label})
            emit(event)
            events.append(event)
            return WorkflowResult(status="BLOCKED", stages=[*reused_events, *events], artifacts=output_artifacts, warnings=[message])
        export_dir = request.run_dir / "cycif" / "exported" / scene_label
        export_result = step04_export_registered(
            reg_result,
            scene_label,
            str(export_dir),
            marker_map=marker_map,
            write_pipex_channels=False,
            verbose=False,
            **request.parameters.get("registration", {}).get("export_config", {}),
        )
        scene_artifacts = [str(path) for path in export_dir.rglob("*") if path.is_file()]
        if not scene_artifacts:
            raise RuntimeError(f"step04_export_registered produced no outputs for {scene_label}: {export_result}")
        output_artifacts.extend(scene_artifacts)
        event = StageEvent(
            stage_id="export",
            status="SUCCEEDED",
            message=f"Exported {scene_label} from verified registered TIFFs and transform metadata.",
            tool_versions={"spatial_omics_toolkit": _version("spatial-omics-toolkit")},
            parameters={"parent_run_dir": str(source), "registration_handoff": str(handoff_path), "marker_panel": marker_panel},
            artifacts=scene_artifacts,
        )
        emit(event)
        events.append(event)
        segmentation = request.parameters.get("segmentation", {})
        if isinstance(segmentation, dict) and segmentation.get("enabled") is True:
            if segmentation.get("backend") != "MCMICRO":
                message = "Export continuation with segmentation requires the supported MCMICRO backend."
                blocked = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
                emit(blocked)
                events.append(blocked)
                return WorkflowResult(status="BLOCKED", stages=[*reused_events, *events], artifacts=output_artifacts, warnings=[message])
            if not os.environ.get("SLURM_JOB_ID"):
                message = "MCMICRO execution is blocked outside an allocated SLURM worker."
                blocked = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
                emit(blocked)
                events.append(blocked)
                return WorkflowResult(status="BLOCKED", stages=[*reused_events, *events], artifacts=output_artifacts, warnings=[message])
            from spatial_omics_toolkit.cycif import MCMICROConfig, step05_segment_cells

            runtime = preflight_result.resolved_parameters["mcmicro_runtime"]
            version_process = subprocess.run([runtime["wrapper"], "-version"], check=True, capture_output=True, text=True, timeout=60)
            mcmicro_version = (version_process.stdout or version_process.stderr).strip() or "version command returned no output"
            composite_path = export_result.get("composite_path")
            if not composite_path or not Path(str(composite_path)).is_file():
                raise RuntimeError(f"step04_export_registered returned no usable composite for segmentation of {scene_label}.")
            config_values = segmentation.get("parameters", {})
            config = MCMICROConfig(mcmicro_bin=runtime["wrapper"], mcmicro_nf=str(Path(runtime["home"]) / "main.nf"), **config_values)
            project_dir = request.run_dir / "cycif" / "mcmicro" / scene_label
            emit(StageEvent(stage_id="segmentation", status="RUNNING", message=f"Running MCMICRO for {scene_label} after resumed export.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": _version("spatial-omics-toolkit")}, parameters={"project_dir": str(project_dir), "composite_path": str(composite_path)}))
            segmentation_result = step05_segment_cells("mcmicro", scene_label=scene_label, composite_ometiff_path=str(composite_path), marker_map=marker_map, project_dir=str(project_dir), mcmicro_config=config, run=True, load_results=True, verbose=False)
            segmentation_artifacts = [str(path) for path in project_dir.rglob("*") if path.is_file()]
            if not segmentation_artifacts:
                raise RuntimeError(f"MCMICRO returned no artifacts for {scene_label}: {segmentation_result}")
            output_artifacts.extend(segmentation_artifacts)
            completed = StageEvent(stage_id="segmentation", status="SUCCEEDED", message=f"MCMICRO {mcmicro_version} completed for {scene_label} after resumed export.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": _version("spatial-omics-toolkit")}, parameters={"backend": "MCMICRO", "segmenter": config.segmenter, "result": str(segmentation_result)}, artifacts=segmentation_artifacts)
            emit(completed)
            events.append(completed)
    parent_metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
    from spatial_omics_toolkit.application.runs import atomic_json
    warning_stage_ids = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
    atomic_json(request.run_dir / "resume_receipt.json", {
        "parent_run_id": parent_metadata.get("run_id"),
        "parent_run_path": str(source),
        "start_stage": "export",
        "raw_input_fingerprint": _input_fingerprint(request.run_dir),
        "reused_checkpoints": reused,
        "warning_review_confirmed": selection.get("warning_review_confirmed") is True,
        "warning_stage_ids": warning_stage_ids,
        "warning_review_recorded_at_utc": datetime.now(timezone.utc).isoformat() if warning_stage_ids else None,
    })
    _write_checkpoint_receipts(request, [*reused_events, *events])
    child_receipts = json.loads((request.run_dir / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    child_receipts["checkpoints"] = {
        "intake": checkpoints["intake"],
        "registration": checkpoints["registration"],
        **child_receipts["checkpoints"],
    }
    atomic_json(request.run_dir / "checkpoint_receipts.json", child_receipts)
    return WorkflowResult(status="WARNING" if resume_warnings else "SUCCEEDED", stages=[*reused_events, *events], artifacts=[item["path"] for item in reused] + output_artifacts, warnings=resume_warnings)


def _execute_segmentation_resume(request: RunRequest, emit: WorkflowEmitter, reused: list[dict[str, str]]) -> WorkflowResult:
    selection = request.parameters["continuation"]
    source = Path(selection["parent_run_dir"]).expanduser().resolve()
    receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    checkpoints = receipt["checkpoints"]
    export_checkpoint = checkpoints["export"]
    marker_map = export_checkpoint["settings"].get("marker_map")
    if not isinstance(marker_map, dict) or not marker_map:
        message = "Validated export checkpoint lacks the marker map required by segmentation."
        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    composites = sorted(
        Path(item["path"])
        for item in export_checkpoint["artifacts"]
        if Path(item["path"]).name.endswith("_composite.ome.tif")
    )
    if not composites:
        message = "Validated export checkpoint has no composite OME-TIFF for segmentation."
        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    check = preflight(request)
    if not check.ready:
        message = "Segmentation continuation preflight blocked: " + "; ".join(check.blockers)
        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=check.blockers)
    resume_warnings = list(check.warnings) + _warning_reuse_messages(reused)
    if not os.environ.get("SLURM_JOB_ID"):
        message = "MCMICRO execution is blocked outside an allocated SLURM worker."
        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    from spatial_omics_toolkit.cycif import MCMICROConfig, step05_segment_cells

    segmentation = request.parameters.get("segmentation", {})
    if segmentation.get("enabled") is not True or segmentation.get("backend") != "MCMICRO":
        message = "Continue after export requires segmentation.enabled=true and backend='MCMICRO'."
        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
    runtime = check.resolved_parameters["mcmicro_runtime"]
    version_process = subprocess.run([runtime["wrapper"], "-version"], check=True, capture_output=True, text=True, timeout=60)
    mcmicro_version = (version_process.stdout or version_process.stderr).strip() or "version command returned no output"
    config_values = segmentation.get("parameters", {})
    if not isinstance(config_values, dict):
        raise ValueError("segmentation.parameters must be an object of MCMICROConfig settings.")
    config = MCMICROConfig(mcmicro_bin=runtime["wrapper"], mcmicro_nf=str(Path(runtime["home"]) / "main.nf"), **config_values)
    reused_events: list[StageEvent] = []
    for stage_id in ("intake", "registration", "export"):
        stage_artifacts = [item["path"] for item in reused if item["stage_id"] == stage_id]
        source_status = next((item.get("source_status", "SUCCEEDED") for item in reused if item["stage_id"] == stage_id), "SUCCEEDED")
        event = StageEvent(
            stage_id=stage_id,
            status="REUSED",
            message=f"Reused verified {stage_id} checkpoint from parent run {source.name} (source status {source_status}).",
            parameters={"parent_run_dir": str(source), "start_stage": "segmentation", "source_status": source_status},
            artifacts=stage_artifacts,
        )
        emit(event)
        reused_events.append(event)
    results: list[StageEvent] = []
    for composite in composites:
        scene_label = composite.name.removesuffix("_composite.ome.tif")
        project_dir = request.run_dir / "cycif" / "mcmicro" / scene_label
        emit(StageEvent(
            stage_id="segmentation",
            status="RUNNING",
            message=f"Running MCMICRO for {scene_label} from the verified parent export.",
            tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]},
            parameters={"parent_composite": str(composite), "project_dir": str(project_dir), "marker_map": marker_map},
        ))
        result = step05_segment_cells(
            "mcmicro",
            scene_label=scene_label,
            composite_ometiff_path=str(composite),
            marker_map=marker_map,
            project_dir=str(project_dir),
            mcmicro_config=config,
            run=True,
            load_results=True,
            verbose=False,
        )
        artifacts = [str(path) for path in project_dir.rglob("*") if path.is_file()]
        if not artifacts:
            raise RuntimeError(f"MCMICRO returned without run-scoped artifacts for {scene_label}: {result}")
        event = StageEvent(
            stage_id="segmentation",
            status="SUCCEEDED",
            message=f"MCMICRO {mcmicro_version} completed for {scene_label} from a verified export checkpoint.",
            tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]},
            parameters={"backend": "MCMICRO", "segmenter": config.segmenter, "parent_composite": str(composite), "result": str(result)},
            artifacts=artifacts,
        )
        emit(event)
        results.append(event)
    parent_metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
    warning_stage_ids = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
    atomic_payload = {
        "parent_run_id": parent_metadata.get("run_id"),
        "parent_run_path": str(source),
        "start_stage": "segmentation",
        "raw_input_fingerprint": _input_fingerprint(request.run_dir),
        "reused_checkpoints": reused,
        "warning_review_confirmed": selection.get("warning_review_confirmed") is True,
        "warning_stage_ids": warning_stage_ids,
        "warning_review_recorded_at_utc": datetime.now(timezone.utc).isoformat() if warning_stage_ids else None,
    }
    from spatial_omics_toolkit.application.runs import atomic_json
    atomic_json(request.run_dir / "resume_receipt.json", atomic_payload)
    child_checkpoints = {stage: checkpoints[stage] for stage in ("intake", "registration", "export")}
    for event in results:
        if event.artifacts:
            child_checkpoints["segmentation"] = {
                "workflow_id": "cycif-raw-czi",
                "stage_id": "segmentation",
                "input_fingerprint": _input_fingerprint(request.run_dir),
                "settings": _stage_parameters(request.parameters, "segmentation"),
                "settings_sha256": _canonical_hash(_stage_parameters(request.parameters, "segmentation")),
                "tool_identity": {**_tool_identity("segmentation"), "MCMICRO": mcmicro_version},
                "artifacts": [{"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))} for path in event.artifacts],
            }
    atomic_json(request.run_dir / "checkpoint_receipts.json", {"workflow_id": "cycif-raw-czi", "checkpoints": child_checkpoints})
    return WorkflowResult(status="WARNING" if resume_warnings else "SUCCEEDED", stages=[*reused_events, *results], artifacts=[item["path"] for item in reused] + [path for event in results for path in event.artifacts], warnings=resume_warnings)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _intake_options(parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate user-provided acquisition assumptions to the canonical intake config."""
    from spatial_omics_toolkit.cycif import RawImageIntakeConfig

    supplied = parameters.get("intake", {})
    if not isinstance(supplied, dict):
        raise ValueError("parameters.intake must be an object of RawImageIntakeConfig fields.")
    if "filename_pattern" not in supplied:
        raise ValueError("A user-reviewed intake.filename_pattern is required; acquisition naming is not inferred from PREVAIL defaults.")
    missing_assumptions = {"expected_rounds", "expected_channels"} - set(supplied)
    if missing_assumptions:
        raise ValueError(f"Explicit acquisition assumptions are required: {sorted(missing_assumptions)}; PREVAIL round/channel defaults are not applied.")
    try:
        if not supplied.get("expected_rounds") or int(supplied["expected_channels"]) < 1:
            raise ValueError("Explicit expected_rounds (non-empty) and expected_channels (positive) are required.")
    except (TypeError, ValueError) as exc:
        raise ValueError("Explicit expected_rounds (non-empty) and positive expected_channels are required.") from exc
    allowed = {"filename_pattern", "expected_rounds", "expected_channels", "raw_extension", "ome_xml_suffix", "expected_files_per_slide", "pixel_size_bounds_um"}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown intake configuration field(s): {', '.join(unknown)}")
    options = dict(supplied)
    if "expected_rounds" in options:
        options["expected_rounds"] = tuple(options["expected_rounds"])
    if "pixel_size_bounds_um" in options and options["pixel_size_bounds_um"] is not None:
        options["pixel_size_bounds_um"] = tuple(options["pixel_size_bounds_um"])
    try:
        compiled = re.compile(str(options["filename_pattern"]))
    except re.error as exc:
        raise ValueError(f"Invalid intake.filename_pattern: {exc}") from exc
    required_groups = {"round", "markers", "slide", "date", "acq_id", "scene"}
    missing_groups = sorted(required_groups - set(compiled.groupindex))
    if missing_groups:
        raise ValueError(
            "filename_pattern must define named groups 'round', 'markers', 'slide', 'date', 'acq_id', and 'scene'; "
            f"include explicit empty captures for fields absent from source filenames (missing: {missing_groups})."
        )
    return {"config": RawImageIntakeConfig(**options)}


def _resolve_marker_panel(parameters: dict[str, Any]) -> dict[str, Any] | None:
    registration = parameters.get("registration")
    if not isinstance(registration, dict):
        return None
    panel_value = registration.get("marker_panel_csv")
    if not panel_value:
        return None
    if registration.get("marker_panel_confirmed") is not True:
        raise ValueError("Marker panel preview must be explicitly confirmed before export or segmentation.")
    panel_path = Path(str(panel_value)).expanduser().resolve()
    if not panel_path.is_file():
        raise ValueError(f"Selected marker_panel_metadata.csv does not exist: {panel_path}")
    import csv
    import hashlib

    with panel_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"round", "channel_index", "marker_name", "target_cell_compartment", "fluorophore", "notes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Marker panel CSV is missing required schema columns: {missing}")
        rows = list(reader)
    grouped: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        try:
            index = int(row["channel_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Marker panel has invalid zero-based channel_index {row.get('channel_index')!r}.") from exc
        marker_name = (row.get("marker_name") or "").strip()
        round_name = (row.get("round") or "").strip()
        if not round_name or not marker_name or index < 0:
            raise ValueError("Marker panel rows require non-empty round/marker_name and non-negative channel_index.")
        grouped.setdefault(round_name, []).append((index, marker_name))
    if not grouped:
        raise ValueError("Marker panel CSV contains no channel rows.")
    resolved: dict[str, list[str]] = {}
    for round_name, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        indices = [index for index, _ in entries]
        if indices != list(range(len(indices))):
            raise ValueError(f"Marker panel channel_index values for round {round_name!r} must be unique and contiguous from zero; received {indices}.")
        resolved[round_name] = [name for _, name in entries]
    panel = {"source_path": str(panel_path), "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(), "marker_map": resolved}
    parameters["_resolved_marker_panel"] = panel
    return panel


def describe() -> WorkflowDescriptor:
    toolkit_version = _version("spatial-omics-toolkit")
    return WorkflowDescriptor(
        workflow_id="cycif-raw-czi",
        modality="CycIF",
        title="CyCIF raw CZI intake and staged image analysis",
        description="Run canonical notebook-derived CyCIF steps in one SLURM-allocated guarded Python worker. Intake scans raw files and OME metadata; downstream stages require explicit, complete configuration and preserve tool failures.",
        stages=[
            StageDefinition(stage_id="intake", title="Intake", purpose="Inventory CZI files, parse configured acquisition names, validate cycle/scene coverage and OME metadata, and write the canonical QC report.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step01_scan_intake", version=toolkit_version, role="Canonical raw-CZI intake and OME metadata QC", documentation_url=None)], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/intake.py", function_name="step01_scan_intake", role="Raw-CZI intake and metadata QC")]),
            StageDefinition(stage_id="conversion", title="Conversion", purpose="Convert raw CZI to OME-TIFF through the toolkit's SLURM-dispatch interface when explicitly configured.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step02_convert_raw_images", version=toolkit_version, role="Canonical CZI conversion", documentation_url=None)], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/convert.py", function_name="step02_convert_raw_images", role="Optional CZI conversion")]),
            StageDefinition(stage_id="registration", title="Registration", purpose="Register configured acquisition rounds against a user-selected reference round; failures are retained as stage failures.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step03_register_rounds", version=toolkit_version, role="Canonical round registration", documentation_url=None)], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/register.py", function_name="step03_register_rounds", role="Optional round registration")]),
            StageDefinition(stage_id="export", title="Marker export", purpose="Export registered scenes only after explicit marker/channel mapping confirmation; missing or mismatched mappings block export.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step04_export_registered", version=toolkit_version, role="Canonical registered-image export", documentation_url=None)], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/export_images.py", function_name="step04_export_registered", role="Confirmed marker export")]),
            StageDefinition(stage_id="segmentation", title="Segmentation", purpose="Run the configured MCMICRO segmentation workflow on exported scenes inside the allocated worker.", tools=[ToolReference(name="MCMICRO", version=toolkit_version, role="Cell segmentation", documentation_url=None)], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/segment.py", function_name="step05_segment_cells", role="Toolkit MCMICRO invocation")]),
        ],
        parameter_fields=[
            ParameterField(name="intake.filename_pattern", label="Acquisition filename regex", kind="text", required=True, section="Intake", stage_id="intake", help_text="User-reviewed regex with named groups round, markers, slide, date, acq_id, and scene. Use explicit empty named captures for fields absent from source filenames; PREVAIL defaults are not applied."),
            ParameterField(name="intake.expected_rounds", label="Expected rounds", kind="multiselect", required=True, default=[], dynamic_choices_key="available_rounds", section="Intake", stage_id="intake", help_text="Select expected round labels for coverage QC."),
            ParameterField(name="intake.expected_channels", label="Expected channels per image", kind="int", required=True, min_value=1, step=1, section="Intake", stage_id="intake", help_text="Enter the channel count reviewed from the raw acquisition metadata; it is not inferred from filenames or an example panel."),
            ParameterField(name="intake.raw_extension", label="Raw image extension", kind="text", default=".czi", section="Intake", stage_id="intake"),
            ParameterField(name="conversion.enabled", label="Retain converted OME-TIFF outputs", kind="bool", default=False, section="Conversion", stage_id="conversion", help_text="Optional; registration reads original CZI and does not consume conversion outputs."),
            ParameterField(name="conversion.config.generate_mip", label="Generate maximum-intensity projections", kind="bool", default=True, advanced=True, section="Conversion", stage_id="conversion"),
            ParameterField(name="conversion.config.generate_composite", label="Generate pseudocolor composites", kind="bool", default=True, advanced=True, section="Conversion", stage_id="conversion"),
            ParameterField(name="registration.enabled", label="Register and export scenes", kind="bool", default=False, section="Registration/export", stage_id="registration", help_text="Requires confirmation of parsed slide/scene/round grouping."),
            ParameterField(name="registration.confirmed_grouping", label="Confirm parsed acquisition grouping", kind="bool", default=False, section="Registration/export", stage_id="registration", help_text="Review acquisition_groups and confirm before registration."),
            ParameterField(name="registration.config.ref_round", label="Reference round", kind="select", required=True, dynamic_choices_key="available_rounds", section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.config.dapi_channel", label="Zero-based DAPI channel index", kind="int", default=0, min_value=0, section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.config.scale_factor", label="Registration scale factor", kind="float", default=1.0, min_value=0.01, max_value=1.0, step=0.01, section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.marker_panel_csv", label="Marker panel metadata CSV", kind="config_file", section="Registration/export", stage_id="export", help_text="Preview and confirm its schema/map before setting marker_panel_confirmed."),
            ParameterField(name="registration.marker_panel_confirmed", label="Confirm marker/channel mapping", kind="bool", default=False, section="Registration/export", stage_id="export"),
            ParameterField(name="segmentation.backend", label="Segmentation engine", kind="select", choices=["MCMICRO"], default="MCMICRO", section="Segmentation", stage_id="segmentation", help_text="The toolkit runs MCMICRO for cell segmentation."),
            ParameterField(name="segmentation.parameters.segmenter", label="MCMICRO segmenter", kind="select", choices=["mccellpose", "cellpose", "unmicst"], default="mccellpose", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.dapi_channel", label="MCMICRO DAPI channel (1-based)", kind="int", default=1, min_value=1, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_model", label="Cellpose model", kind="text", default="cyto3", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_diameter", label="Cellpose diameter", kind="int", default=30, min_value=1, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_expansion", label="Cell expansion", kind="int", default=10, min_value=0, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.use_gpu", label="Use GPU", kind="bool", default=False, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.enabled", label="Run segmentation", kind="bool", default=False, section="Segmentation", stage_id="segmentation"),
        ],
        input_description="One complete raw-CZI modality folder, project root, and project_samples.csv. Naming pattern, expected cycles/channels, and round/channel mapping are acquisition-specific parameters; no PREVAIL biological labels are assumed.",
        output_description="Run-scoped intake inventory, checks, report, and any explicitly configured stage outputs under run_dir.",
        limitations=["Conversion, registration, export, and segmentation are not implied by successful intake; they require explicit configuration and available tools/resources.", "No universal biological labels or cell-type configuration is assumed.", "Heavy stages run only when execute is called in the allocated Python worker."],
    )


def _acquisition_groups(root: Path, filename_pattern: str) -> list[dict[str, Any]]:
    pattern = re.compile(filename_pattern)
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".czi":
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        fields = match.groupdict()
        if not {"slide", "scene", "round"}.issubset(fields):
            continue
        slide = str(fields["slide"])
        scene = str(fields["scene"])
        group_key = f"slide={slide};scene={scene}"
        group = groups.setdefault(group_key, {"group_key": group_key, "label": f"{slide}_{scene}", "input_paths": []})
        group["input_paths"].append(str(path))
    return [groups[key] for key in sorted(groups)]


def _available_rounds(groups: list[dict[str, Any]], filename_pattern: str) -> list[str]:
    pattern = re.compile(filename_pattern)
    return sorted({str(match.group("round")) for group in groups for path in group["input_paths"] if (match := pattern.match(Path(path).name)) is not None})


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    warnings: list[str] = []
    conversion = request.parameters.get("conversion")
    params: dict[str, Any] = {"toolkit_version": _version("spatial-omics-toolkit"), "conversion_enabled": bool(isinstance(conversion, dict) and conversion.get("enabled")), "conversion_policy": "optional retained deliverable; registration reads original CZI"}
    intake_values = request.parameters.get("intake")
    if isinstance(intake_values, dict):
        params["acquisition_assumptions"] = {key: {"value": intake_values[key], "source": "explicit user configuration"} for key in ("filename_pattern", "expected_rounds", "expected_channels") if key in intake_values}
    if request.modality != "CycIF":
        blockers.append(f"Expected modality 'CycIF', received {request.modality!r}.")
    for label, path in (("project root", request.project_root), ("modality folder", request.modality_dir)):
        if not path.is_dir():
            blockers.append(f"Selected {label} is not a directory: {path}")
    if not request.annotation_csv.is_file():
        blockers.append(f"project_samples.csv is missing: {request.annotation_csv}")
    elif request.annotation_csv.name != "project_samples.csv":
        warnings.append(f"Annotation file is named {request.annotation_csv.name!r}; expected project_samples.csv.")
    if request.parameters.get("_preflight_mode") == "quick":
        try:
            _intake_options(request.parameters)
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
        else:
            if request.modality_dir.is_dir():
                pattern = str(request.parameters["intake"]["filename_pattern"])
                groups = _acquisition_groups(request.modality_dir, pattern)
                params["acquisition_groups"] = groups
                params["available_rounds"] = _available_rounds(groups, pattern)
        return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)
    try:
        _intake_options(request.parameters)
    except (TypeError, ValueError) as exc:
        blockers.append(str(exc))
    if request.modality_dir.is_dir():
        czi_files = sorted(p for p in request.modality_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".czi")
        ome_files = sorted(p for p in request.modality_dir.rglob("*") if p.is_file() and p.name.lower().endswith(".ome.xml"))
        pattern_value = request.parameters.get("intake", {}).get("filename_pattern") if isinstance(request.parameters.get("intake"), dict) else None
        groups = _acquisition_groups(request.modality_dir, str(pattern_value)) if pattern_value else []
        if pattern_value:
            params.update({"acquisition_groups": groups, "available_rounds": _available_rounds(groups, str(pattern_value))})
        params.update({"czi_file_count": len(czi_files), "ome_xml_file_count": len(ome_files), "czi_files": [str(p) for p in czi_files], "ome_xml_files": [str(p) for p in ome_files]})
        if not czi_files:
            blockers.append("The selected modality folder contains no raw .czi files.")
        if not ome_files:
            warnings.append("No .ome.xml sidecars were found; step01_scan_intake will report metadata availability from the image files it can inspect.")
    segmentation = request.parameters.get("segmentation", {})
    if isinstance(segmentation, dict) and segmentation.get("enabled") is True:
        if segmentation.get("backend", "MCMICRO") != "MCMICRO":
            blockers.append("The CyCIF app supports MCMICRO segmentation only.")
        else:
            software_root = os.environ.get("PAULI_FAST_SOFTWARES")
            if not software_root:
                blockers.append("MCMICRO runtime unavailable: PAULI_FAST_SOFTWARES is not set in the worker environment.")
            else:
                mcmicro_home = Path(software_root) / "envs" / "spatial-proteomics" / "mcmicro"
                mcmicro_bin = Path(software_root) / "bin" / "mcmicro"
                lock_file = Path(software_root) / "config" / "workflow-runtime.lock.sh"
                if not mcmicro_home.is_dir() or not (mcmicro_home / "main.nf").is_file():
                    blockers.append(f"Pinned MCMICRO workflow is unavailable at {mcmicro_home}.")
                if not mcmicro_bin.is_file() or not os.access(mcmicro_bin, os.X_OK):
                    blockers.append(f"Pinned MCMICRO wrapper is unavailable or not executable: {mcmicro_bin}.")
                if not lock_file.is_file():
                    blockers.append(f"Pinned MCMICRO runtime lock is missing: {lock_file}.")
                params["mcmicro_runtime"] = {"home": str(mcmicro_home), "wrapper": str(mcmicro_bin), "runtime_lock": str(lock_file)}
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)


def _write_registration_handoff(result: dict[str, Any], scene_label: str, registration_dir: Path, dapi_channel: int) -> Path:
    handoff: dict[str, Any] = {
        "scene_label": scene_label,
        "reference_shape": list(result["reference_shape"]),
        "round_order": list(result["round_order"]),
        "pixel_size_um": result.get("pixel_size_um"),
        "params": result.get("params", {}),
        "rounds": {},
    }
    for round_name in result["round_order"]:
        item = result["rounds"][round_name]
        channels = item.get("channels")
        channel_path = registration_dir / f"{round_name}_registered.tif"
        if channels is not None and not channel_path.is_file():
            raise RuntimeError(f"Registration handoff is missing its durable TIFF stack: {channel_path}")
        handoff["rounds"][round_name] = {
            "registered": bool(item["registered"]),
            "status": str(item["status"]),
            "transform": item["transform"].tolist() if item.get("transform") is not None else None,
            "inliers": int(item.get("inliers", 0)),
            "matches": int(item.get("matches", 0)),
            "czi_path": str(item.get("czi_path", "")),
            "czi_size_bytes": int(item.get("czi_size_bytes", 0)),
            "mov_shape": list(item["mov_shape"]) if item.get("mov_shape") is not None else None,
            "channels_path": str(channel_path.resolve()) if channels is not None else None,
            "dapi_channel": int(dapi_channel),
        }
    handoff_path = registration_dir / f"{scene_label}_registration_handoff.json"
    handoff_path.write_text(json.dumps(handoff, sort_keys=True, indent=2), encoding="utf-8")
    return handoff_path


def _load_registration_handoff(path: Path) -> dict[str, Any]:
    import numpy as np
    import tifffile

    handoff = json.loads(path.read_text(encoding="utf-8"))
    rounds: dict[str, Any] = {}
    for round_name in handoff["round_order"]:
        item = handoff["rounds"][round_name]
        channels = tifffile.imread(item["channels_path"]) if item.get("channels_path") else None
        rounds[round_name] = {
            "channels": channels,
            "dapi": channels[int(item["dapi_channel"])] if channels is not None else None,
            "transform": np.asarray(item["transform"], dtype=np.float64) if item.get("transform") is not None else None,
            "inliers": int(item["inliers"]),
            "matches": int(item["matches"]),
            "registered": bool(item["registered"]),
            "status": str(item["status"]),
            "mov_shape": tuple(item["mov_shape"]) if item.get("mov_shape") is not None else None,
            "czi_path": item.get("czi_path"),
            "czi_size_bytes": int(item.get("czi_size_bytes", 0)),
        }
    return {
        "rounds": rounds,
        "round_order": list(handoff["round_order"]),
        "reference_shape": tuple(handoff["reference_shape"]),
        "pixel_size_um": handoff.get("pixel_size_um"),
        "params": handoff.get("params", {}),
    }


def _execute_raw(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        event = StageEvent(stage_id="intake", status="BLOCKED", message="CyCIF preflight blocked: " + "; ".join(check.blockers), parameters=check.resolved_parameters)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=check.blockers + check.warnings)
    from spatial_omics_toolkit.cycif import step01_scan_intake

    stages: list[StageEvent] = []
    warnings = list(check.warnings)
    conversion = request.parameters.get("conversion")
    intake_dir = request.run_dir / "cycif" / "intake"
    artifacts: list[str] = []
    current_stage = "intake"
    try:
        options = _intake_options(request.parameters)
        selection = request.parameters.get("continuation", {})
        resumed_start = selection.get("start_stage") if isinstance(selection, dict) and selection.get("mode") == "continue" else None
        inherited_intake = [item for item in request.parameters.get("_reused_checkpoints", []) if item.get("stage_id") == "intake"]
        if resumed_start in {"conversion", "registration"}:
            source = Path(selection["parent_run_dir"]).expanduser().resolve()
            source_status = next((item.get("source_status", "SUCCEEDED") for item in inherited_intake), "SUCCEEDED")
            event = StageEvent(
                stage_id="intake", status="REUSED",
                message=f"Reused verified intake checkpoint from parent run {source.name} (source status {source_status}).",
                parameters={"parent_run_dir": str(source), "start_stage": resumed_start, "source_status": source_status},
                artifacts=[item["path"] for item in inherited_intake],
            )
            emit(event)
            stages.append(event)
            artifacts = [item["path"] for item in inherited_intake]
            warnings.extend(_warning_reuse_messages(inherited_intake))
            has_intake_issues = False
            intake_issues: dict[str, Any] = {}
        else:
            emit(StageEvent(stage_id="intake", status="RUNNING", message="Scanning raw CZI inventory and validating naming, cycle/scene coverage, and OME metadata.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"modality_dir": str(request.modality_dir), **request.parameters.get("intake", {})}))
            result = step01_scan_intake(raw_dir=str(request.modality_dir), reports_dir=str(intake_dir), generate_html_report=True, generate_dapi_preview=False, strict=False, verbose=False, **options)
            artifacts = [str(p) for p in sorted(intake_dir.rglob("*")) if p.is_file()]
            checks = getattr(result, "checks", [])
            check_rows = checks.to_dict(orient="records") if hasattr(checks, "to_dict") else []
            failed_checks = [row for row in check_rows if str(row.get("Status", "")).strip() == "⚠"]
            unassessed_checks = [row for row in check_rows if str(row.get("Status", "")).strip() == "—" and str(row.get("Check", "")).startswith(("Channel count", "Pixel size"))]
            intake_issues = {
                "check_warnings": failed_checks,
                "checks_unassessed": unassessed_checks,
                "coverage_missing": [str(x) for x in getattr(result, "coverage_missing", []) or []],
                "unparsed": [str(x) for x in getattr(result, "unparsed", []) or []],
                "preflight_warnings": check.warnings,
            }
            has_intake_issues = bool(failed_checks or intake_issues["coverage_missing"] or intake_issues["unparsed"])
            has_intake_warnings = bool(unassessed_checks or check.warnings)
            status = "WARNING" if has_intake_issues or has_intake_warnings else "SUCCEEDED"
            if has_intake_warnings:
                warnings.append(json.dumps({"checks_unassessed": unassessed_checks, "preflight_warnings": check.warnings}, default=str))
            message = "Intake found failed QC checks; review the report. Downstream imaging stages are blocked." if has_intake_issues else ("Intake completed with unassessed QC checks; review the report and warning details before continuing." if has_intake_warnings else "Canonical step01_scan_intake completed.")
            event = StageEvent(stage_id="intake", status=status, message=message, tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters=intake_issues, artifacts=artifacts)
            emit(event)
            stages.append(event)
            if has_intake_issues:
                warnings.append(json.dumps(intake_issues, default=str))
                gate = StageEvent(stage_id="conversion", status="BLOCKED", message="Downstream imaging stages were not started because intake reported QC warnings.", parameters=intake_issues)
                emit(gate)
                stages.append(gate)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings)
        if resumed_start == "registration":
            conversion = None
        elif isinstance(conversion, dict) and conversion.get("enabled") is True:
            pass
        else:
            conversion = None
        current_stage = "conversion"
        if conversion:
            if not __import__("os").environ.get("SLURM_JOB_ID"):
                message = "CZI conversion is blocked outside an allocated SLURM worker."
                event = StageEvent(stage_id="conversion", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            from spatial_omics_toolkit.cycif import ImageConversionConfig, step02_convert_raw_images

            conversion_options = conversion.get("config", {})
            if not isinstance(conversion_options, dict):
                raise ValueError("parameters.conversion.config must be an object of ImageConversionConfig fields.")
            conversion_config = ImageConversionConfig(**conversion_options)
            czi_files = [Path(value) for value in check.resolved_parameters.get("czi_files", [])]
            input_folders = sorted({path.parent for path in czi_files})
            if not czi_files or not input_folders:
                raise RuntimeError("No raw CZI files were inventoried for conversion.")
            emit(StageEvent(stage_id="conversion", status="RUNNING", message=f"Converting {len(czi_files)} inventoried CZI file(s) across {len(input_folders)} physical folder(s) in the allocated SLURM worker.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"dispatch": "inprocess within allocated worker", "input_files": [str(path) for path in czi_files], **conversion_options}))
            conversion_outputs: list[str] = []
            conversion_results: dict[str, Any] = {}
            for input_folder in input_folders:
                relative = input_folder.relative_to(request.modality_dir)
                folder_key = relative.as_posix().replace("/", "__") or "root"
                conversion_dir = request.run_dir / "cycif" / "converted" / folder_key
                conversion_dir.mkdir(parents=True, exist_ok=True)
                expected_inputs = [str(path) for path in czi_files if path.parent == input_folder]
                conversion_results[str(input_folder)] = step02_convert_raw_images(str(input_folder), str(conversion_dir), dispatch="inprocess", conversion_config=conversion_config, strict=True, verbose=False)
                folder_outputs = [str(p) for p in sorted(conversion_dir.rglob("*")) if p.is_file()]
                if not folder_outputs:
                    raise RuntimeError(f"Canonical conversion produced no outputs for inventoried inputs: {expected_inputs}")
                conversion_outputs.extend(folder_outputs)
            if not conversion_outputs:
                raise RuntimeError("Canonical conversion returned without producing output files.")
            event = StageEvent(stage_id="conversion", status="SUCCEEDED", message=f"Canonical CZI conversion completed for {len(czi_files)} inventoried input file(s) in {len(input_folders)} folder(s).", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"results_by_input_folder": conversion_results}, artifacts=conversion_outputs)
            emit(event)
            stages.append(event)
        segmentation_completed = 0
        current_stage = "registration"
        registration = request.parameters.get("registration")
        registration_enabled = isinstance(registration, dict) and registration.get("enabled") is True
        if not registration_enabled:
            registration = None
        if registration:
            if not isinstance(registration, dict) or not registration.get("confirmed_grouping"):
                message = "Registration requires user-confirmed filename grouping in parameters.registration.confirmed_grouping=true."
                event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            import re
            from spatial_omics_toolkit.cycif import step03_register_rounds, step04_export_registered

            pattern = re.compile(str(request.parameters["intake"]["filename_pattern"]))
            grouped: dict[tuple[str, str], dict[str, str]] = {}
            for path in sorted(request.modality_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() != ".czi":
                    continue
                match = pattern.match(path.name)
                if match is None or not {"round", "slide", "scene"}.issubset(match.groupdict()):
                    continue
                key = (str(match.group("slide")), str(match.group("scene")))
                round_label = str(match.group("round"))
                if round_label in grouped.setdefault(key, {}):
                    message = f"Ambiguous duplicate CZI for slide={key[0]}, scene={key[1]}, round={round_label}."
                    event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                    emit(event)
                    stages.append(event)
                    return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
                grouped[key][round_label] = str(path)
            if not grouped:
                message = "No CZI files matched the user-configured filename pattern with named slide, scene, and round groups."
                event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            try:
                marker_panel = _resolve_marker_panel(request.parameters)
            except ValueError as exc:
                marker_panel = None
                warnings.append(str(exc))
            marker_map = marker_panel["marker_map"] if marker_panel is not None else None
            if marker_map is None:
                message = "Export/segmentation requires a user-selected and explicitly confirmed marker_panel_metadata.csv with the required schema."
                warnings.append(message)
            for (slide, scene), round_files in grouped.items():
                label = f"{slide}_{scene}"
                reg_dir = request.run_dir / "cycif" / "registered" / label
                reg_dir.mkdir(parents=True, exist_ok=True)
                emit(StageEvent(stage_id="registration", status="RUNNING", message=f"Registering {len(round_files)} configured rounds for {label}.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"slide": slide, "scene": scene, "rounds": sorted(round_files)}))
                reg_result = step03_register_rounds(round_files, output_dir=str(reg_dir), verbose=False, **registration.get("config", {}))
                round_results = reg_result.get("rounds", {})
                bad_rounds = {name: info.get("status", "unregistered") for name, info in round_results.items() if not info.get("registered", False)}
                if not bad_rounds:
                    _write_registration_handoff(reg_result, label, reg_dir, int(registration.get("config", {}).get("dapi_channel", 0)))
                reg_artifacts = [str(p) for p in reg_dir.rglob("*") if p.is_file()]
                artifacts.extend(reg_artifacts)
                reg_status = "FAILED" if bad_rounds else "SUCCEEDED"
                reg_message = f"Registration transform failures preserved: {bad_rounds}" if bad_rounds else f"Registered scene {label}."
                event = StageEvent(stage_id="registration", status=reg_status, message=reg_message, tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"slide": slide, "scene": scene, "round_status": {name: info.get("status") for name, info in round_results.items()}}, artifacts=reg_artifacts)
                emit(event)
                stages.append(event)
                if bad_rounds:
                    return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings + [reg_message])
                if marker_map is None:
                    message = "Registration succeeded, but export is blocked until a marker-panel CSV is selected and explicitly confirmed."
                    event = StageEvent(stage_id="export", status="BLOCKED", message=message, parameters={"scene_label": label})
                    emit(event)
                    stages.append(event)
                    continue
                if set(marker_map) != set(round_files):
                    message = f"Confirmed marker panel rounds {sorted(marker_map)} do not match scene {label} rounds {sorted(round_files)}; registration QC is retained but export is blocked."
                    event = StageEvent(stage_id="export", status="BLOCKED", message=message, parameters={"scene_label": label, "marker_panel": marker_panel})
                    emit(event)
                    stages.append(event)
                    warnings.append(message)
                    continue
                current_stage = "export"
                export_dir = request.run_dir / "cycif" / "exported" / label
                export_dir.mkdir(parents=True, exist_ok=True)
                emit(StageEvent(stage_id="export", status="RUNNING", message=f"Exporting registered scene {label} with explicitly confirmed channel/marker mapping.", parameters={"slide": slide, "scene": scene, "marker_panel": marker_panel}))
                export_result = step04_export_registered(
                    reg_result,
                    label,
                    str(export_dir),
                    marker_map=marker_map,
                    write_pipex_channels=False,
                    verbose=False,
                    **registration.get("export_config", {}),
                )
                export_artifacts = [str(path) for path in export_dir.rglob("*") if path.is_file()]
                if not export_artifacts:
                    raise RuntimeError(f"step04_export_registered produced no artifacts for {label}: {export_result}")
                artifacts.extend(export_artifacts)
                export_event = StageEvent(stage_id="export", status="SUCCEEDED", message=f"Exported scene {label} with the confirmed marker/channel map.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"slide": slide, "scene": scene, "marker_map": marker_map}, artifacts=export_artifacts)
                emit(export_event)
                stages.append(export_event)
                segmentation = request.parameters.get("segmentation")
                if isinstance(segmentation, dict) and segmentation.get("enabled") is True and segmentation.get("backend") == "MCMICRO":
                    current_stage = "segmentation"
                    if not os.environ.get("SLURM_JOB_ID"):
                        message = "MCMICRO execution is blocked outside an allocated SLURM worker."
                        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
                        emit(event)
                        stages.append(event)
                        return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
                    from spatial_omics_toolkit.cycif import MCMICROConfig, step05_segment_cells

                    runtime = check.resolved_parameters["mcmicro_runtime"]
                    version_process = subprocess.run([runtime["wrapper"], "-version"], check=True, capture_output=True, text=True, timeout=60)
                    mcmicro_version = (version_process.stdout or version_process.stderr).strip() or "version command returned no output"
                    config_values = segmentation.get("parameters", {})
                    if not isinstance(config_values, dict):
                        raise ValueError("segmentation.parameters must be an object of MCMICROConfig settings.")
                    mcmicro_config = MCMICROConfig(mcmicro_bin=runtime["wrapper"], mcmicro_nf=str(Path(runtime["home"]) / "main.nf"), **config_values)
                    project_dir = request.run_dir / "cycif" / "mcmicro" / label
                    composite_path = export_result.get("composite_path")
                    if not composite_path or not Path(str(composite_path)).is_file():
                        raise RuntimeError(f"step04_export_registered did not return a usable composite OME-TIFF for MCMICRO scene {label}.")
                    emit(StageEvent(stage_id="segmentation", status="RUNNING", message=f"Running the pinned MCMICRO backend for {label} inside the allocated SLURM worker.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"backend": "MCMICRO", "segmenter": mcmicro_config.segmenter, "marker_panel": marker_panel, "config": config_values, "project_dir": str(project_dir)}))
                    segmentation_result = step05_segment_cells("mcmicro", scene_label=label, composite_ometiff_path=str(composite_path), marker_map=marker_map, project_dir=str(project_dir), mcmicro_config=mcmicro_config, run=True, load_results=True, verbose=False)
                    segmentation_artifacts = [str(p) for p in project_dir.rglob("*") if p.is_file()]
                    if not segmentation_artifacts:
                        raise RuntimeError(f"MCMICRO returned without run-scoped artifacts for {label}: {segmentation_result}")
                    artifacts.extend(segmentation_artifacts)
                    event = StageEvent(stage_id="segmentation", status="SUCCEEDED", message=f"MCMICRO {mcmicro_version} completed for {label}.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"backend": "MCMICRO", "segmenter": mcmicro_config.segmenter, "marker_panel": marker_panel, "result": str(segmentation_result)}, artifacts=segmentation_artifacts)
                    emit(event)
                    segmentation_completed += 1
                    stages.append(event)
        segmentation = request.parameters.get("segmentation")
        if isinstance(segmentation, dict) and segmentation.get("enabled") is True and segmentation.get("backend") == "MCMICRO" and segmentation_completed == 0:
            message = "MCMICRO was selected but no scene reached segmentation; registration, export, and a confirmed marker map are required."
            event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
            emit(event)
            stages.append(event)
            return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
        final_status = "BLOCKED" if any(stage.status == "BLOCKED" for stage in stages) else ("WARNING" if warnings or any(stage.status == "WARNING" for stage in stages) else "SUCCEEDED")
        return WorkflowResult(status=final_status, stages=stages, artifacts=artifacts, warnings=warnings)
    except Exception as exc:
        event = StageEvent(stage_id=current_stage, status="FAILED", message=f"{current_stage} stage failed: {type(exc).__name__}: {exc}", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]})
        emit(event)
        stages.append(event)
        return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings + [event.message])


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    """Run raw or verified-checkpoint continuation and persist receipts."""
    selection = request.parameters.get("continuation", {})
    if isinstance(selection, dict) and selection.get("mode") == "continue":
        start_stage = selection.get("start_stage")
        try:
            if start_stage in {"conversion", "registration"}:
                if start_stage == "conversion" and not (isinstance(request.parameters.get("conversion"), dict) and request.parameters["conversion"].get("enabled") is True):
                    raise ValueError("Conversion continuation requires conversion.enabled=true.")
                if start_stage == "registration" and not (isinstance(request.parameters.get("registration"), dict) and request.parameters["registration"].get("enabled") is True):
                    raise ValueError("Registration continuation requires registration.enabled=true.")
            elif start_stage not in {"export", "segmentation"}:
                raise ValueError(f"Unsupported continuation boundary: {start_stage!r}.")
            if start_stage in {"export", "segmentation"}:
                _resolve_marker_panel(request.parameters)
        except ValueError as exc:
            event = StageEvent(stage_id=str(start_stage or "intake"), status="BLOCKED", message=f"Cannot execute selected continuation: {exc}")
            emit(event)
            return WorkflowResult(status="BLOCKED", stages=[event], warnings=[event.message])
        reasons, reused = validate_continuation(request)
        if reasons:
            message = "CyCIF continuation blocked: " + "; ".join(reasons)
            event = StageEvent(stage_id=str(start_stage), status="BLOCKED", message=message)
            emit(event)
            return WorkflowResult(status="BLOCKED", stages=[event], warnings=[message])
        if start_stage in {"export", "segmentation"}:
            if start_stage == "export":
                return _execute_export_resume(request, emit, reused)
            return _execute_segmentation_resume(request, emit, reused)
        request.parameters["_reused_checkpoints"] = reused
        observed: list[StageEvent] = []
        def record(event: StageEvent) -> None:
            observed.append(event)
            emit(event)
        result = _execute_raw(request, record)
        source = Path(selection["parent_run_dir"]).expanduser().resolve()
        source_receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
        source_checkpoint = source_receipt["checkpoints"]["intake"]
        parent_metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
        from spatial_omics_toolkit.application.runs import atomic_json
        warning_stage_ids = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
        atomic_json(request.run_dir / "resume_receipt.json", {
            "parent_run_id": parent_metadata.get("run_id"), "parent_run_path": str(source),
            "start_stage": start_stage, "raw_input_fingerprint": _input_fingerprint(request.run_dir),
            "reused_checkpoints": reused, "warning_review_confirmed": selection.get("warning_review_confirmed") is True,
            "warning_stage_ids": warning_stage_ids,
            "warning_review_recorded_at_utc": datetime.now(timezone.utc).isoformat() if warning_stage_ids else None,
        })
        _write_checkpoint_receipts(request, observed)
        receipt_path = request.run_dir / "checkpoint_receipts.json"
        child_receipts = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"workflow_id": request.workflow_id, "checkpoints": {}}
        child_receipts.setdefault("checkpoints", {})["intake"] = source_checkpoint
        atomic_json(receipt_path, child_receipts)
        return result
    observed: list[StageEvent] = []
    def record(event: StageEvent) -> None:
        observed.append(event)
        emit(event)
    result = _execute_raw(request, record)
    try:
        _write_checkpoint_receipts(request, observed)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return result
