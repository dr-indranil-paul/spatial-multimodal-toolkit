import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import RunRequest
from spatial_omics_toolkit.application.runs import (
    atomic_json,
    create_run_bundle,
    record_slurm_job,
    update_status,
    update_status_if,
)


def _request(tmp_path: Path) -> RunRequest:
    project_root = tmp_path / "project"
    modality_dir = project_root / "visium"
    output_root = tmp_path / "runs"
    modality_dir.mkdir(parents=True)
    annotation_csv = project_root / "project_samples.csv"
    annotation_csv.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "unit-1,specimen-1,control,tissue\n",
        encoding="utf-8",
    )
    run_id = uuid4()
    run_dir = output_root / "Visium" / "runs" / str(run_id)
    return RunRequest(
        workflow_id="visium-native-qc",
        modality="Visium",
        project_root=project_root,
        modality_dir=modality_dir,
        annotation_csv=annotation_csv,
        output_root=output_root,
        run_id=run_id,
        run_dir=run_dir,
    )


def test_run_bundle_freezes_reviewed_project_samples(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source_bytes = request.annotation_csv.read_bytes()
    expected_sha256 = hashlib.sha256(source_bytes).hexdigest()

    metadata = create_run_bundle(
        request,
        descriptor={"workflow_id": request.workflow_id},
        resolved_parameters={"annotation_csv_sha256": expected_sha256},
    )

    snapshot_path = Path(metadata["project"]["annotation_csv_snapshot"])
    assert snapshot_path.read_bytes() == source_bytes
    assert metadata["project"]["annotation_csv_sha256"] == expected_sha256

    request.annotation_csv.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "unit-2,specimen-1,case,tissue\n",
        encoding="utf-8",
    )
    assert snapshot_path.read_bytes() == source_bytes
    run_metadata = json.loads((request.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["project"]["annotation_csv_snapshot"] == str(snapshot_path)
    assert run_metadata["input"]["annotation_csv_snapshot"] == "inputs/project_samples.csv"
    assert run_metadata["project"]["samples"][0]["condition"] == "control"


def test_run_bundle_rejects_changed_project_samples(tmp_path: Path) -> None:
    request = _request(tmp_path)
    expected_sha256 = hashlib.sha256(request.annotation_csv.read_bytes()).hexdigest()
    request.annotation_csv.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "unit-2,specimen-1,case,tissue\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed after final preflight"):
        create_run_bundle(
            request,
            descriptor={"workflow_id": request.workflow_id},
            resolved_parameters={"annotation_csv_sha256": expected_sha256},
        )

    assert not request.run_dir.exists()
def test_slurm_reconciliation_preserves_terminal_worker_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_json(run_dir / "status.json", {"status": "RUNNING"})
    record_slurm_job(run_dir, "job-1", "guarded_python_worker", {}, 0)
    update_status(run_dir, "SUCCEEDED", "Worker recorded success")

    changed = update_status_if(
        run_dir,
        {"SUBMITTED", "QUEUED", "RUNNING"},
        "FAILED",
        "SLURM reconciliation fallback",
        expected_job_id="job-1",
    )

    current = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert changed is False
    assert current["status"] == "SUCCEEDED"


def test_slurm_reconciliation_ignores_stale_job_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_json(run_dir / "status.json", {"status": "QUEUED"})
    record_slurm_job(run_dir, "job-old", "guarded_python_worker", {}, 0)
    record_slurm_job(run_dir, "job-new", "guarded_python_worker", {}, 1)

    changed = update_status_if(
        run_dir,
        {"SUBMITTED", "QUEUED", "RUNNING"},
        "FAILED",
        "Stale job completion",
        expected_job_id="job-old",
    )

    current = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    assert changed is False
    assert current["status"] == "QUEUED"
    assert current["slurm_job_id"] == "job-new"
    assert job["slurm_job_id"] == "job-new"
def test_run_bundle_preserves_workspace_symlink_path_and_target(tmp_path: Path) -> None:
    request = _request(tmp_path)
    link_path = request.modality_dir
    link_path.rmdir()
    target_path = tmp_path / "shared_data" / "visium"
    target_path.mkdir(parents=True)
    link_path.symlink_to(target_path, target_is_directory=True)
    expected_sha256 = hashlib.sha256(request.annotation_csv.read_bytes()).hexdigest()

    metadata = create_run_bundle(
        request,
        descriptor={"workflow_id": request.workflow_id},
        resolved_parameters={"annotation_csv_sha256": expected_sha256},
    )

    manifest = json.loads(
        (request.run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    assert metadata["input"]["modality_dir"] == str(link_path.absolute())
    assert metadata["input"]["modality_dir_resolved"] == str(target_path.resolve())
    assert manifest["modality_dir"] == str(link_path.absolute())
    assert manifest["modality_dir_resolved"] == str(target_path.resolve())
