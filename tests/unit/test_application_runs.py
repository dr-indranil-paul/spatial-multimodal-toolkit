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
from spatial_omics_toolkit.application import service
from spatial_omics_toolkit.application.worker import _validate_acquisition_snapshots, run_worker


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


def test_acquisition_snapshots_preserve_exact_bytes_and_worker_checks_them(tmp_path: Path) -> None:
    request = _request(tmp_path)
    sources = {
        "acquisition_metadata.csv": b"acquisition_id,field\r\na,scale\r\n",
        "acquisition_relationships.csv": b"acquisition_id,relationship\r\na,candidate\r\n",
    }
    hashes = {}
    for name, content in sources.items():
        (request.project_root / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    metadata = create_run_bundle(request, {}, {"_acquisition_csv_sha256": hashes})
    for name, content in sources.items():
        snapshot = request.run_dir / "inputs" / name
        assert snapshot.read_bytes() == content
        assert metadata["project"]["acquisition_snapshots"][name]["sha256"] == hashes[name]
        (request.project_root / name).write_bytes(b"changed source")
    _validate_acquisition_snapshots(request.run_dir, metadata["project"])
    (request.run_dir / "inputs" / "acquisition_metadata.csv").chmod(0o600)
    (request.run_dir / "inputs" / "acquisition_metadata.csv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot changed"):
        _validate_acquisition_snapshots(request.run_dir, metadata["project"])
    assert run_worker(request.run_dir) == 1
    status = json.loads((request.run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert "snapshot changed" in status["message"]


def test_acquisition_changed_after_preflight_rejects_bundle(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = request.project_root / "acquisition_metadata.csv"
    source.write_bytes(b"reviewed")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after final preflight"):
        create_run_bundle(request, {}, {"_acquisition_csv_sha256": {source.name: digest}})
    assert not request.run_dir.exists()


def test_preflight_absent_declarations_are_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(service, "ALLOWED_ROOT", tmp_path)
    assert service._review_acquisition_declarations(request.project_root) == ({}, [])


def test_preflight_reviews_and_excludes_invalid_optional_declarations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(service, "ALLOWED_ROOT", tmp_path)
    metadata = request.project_root / "acquisition_metadata.csv"
    metadata.write_text(
        "acquisition_id,modality,source_path,specimen_id,field,value,unit,provenance,source_detail,reviewed\n"
        "a,Visium,visium,specimen-1,pixel_size_x,1,um,user,operator log,true\n",
        encoding="utf-8",
    )
    relationships = request.project_root / "acquisition_relationships.csv"
    relationships.write_text("wrong,columns\nx,y\n", encoding="utf-8")
    hashes, warnings = service._review_acquisition_declarations(request.project_root)
    assert hashes == {metadata.name: hashlib.sha256(metadata.read_bytes()).hexdigest()}
    assert any(relationships.name in warning and "excluded" in warning for warning in warnings)


def test_preflight_excludes_declarations_changed_during_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(service, "ALLOWED_ROOT", tmp_path)
    metadata = request.project_root / "acquisition_metadata.csv"
    metadata.write_text(
        "acquisition_id,modality,source_path,specimen_id,field,value,unit,provenance,source_detail,reviewed\n"
        "a,Visium,visium,specimen-1,pixel_size_x,1,um,user,operator log,true\n",
        encoding="utf-8",
    )
    validate = service.validate_acquisition_metadata

    def changed_during_validation(*args: object, **kwargs: object) -> list[str]:
        metadata.write_bytes(b"after")
        return validate(*args, **kwargs)

    monkeypatch.setattr(service, "validate_acquisition_metadata", changed_during_validation)
    hashes, warnings = service._review_acquisition_declarations(request.project_root, request.annotation_csv)
    assert hashes == {}
    assert any("changed" in warning for warning in warnings)


def test_preflight_validates_captured_metadata_and_samples_despite_aba(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(service, "ALLOWED_ROOT", tmp_path)
    metadata = request.project_root / "acquisition_metadata.csv"
    original = (
        b"acquisition_id,modality,source_path,specimen_id,field,value,unit,provenance,source_detail,reviewed\n"
        b"a,Visium,visium,specimen-2,pixel_size_x,1,um,user,operator log,true\n"
    )
    metadata.write_bytes(original)
    sample_bytes = request.annotation_csv.read_bytes()
    validate = service.validate_acquisition_metadata

    def swapped_during_validation(*args: object, **kwargs: object) -> list[str]:
        metadata.write_bytes(original.replace(b"specimen-2", b"specimen-1"))
        request.annotation_csv.write_bytes(sample_bytes.replace(b"specimen-1", b"specimen-2"))
        try:
            return validate(*args, **kwargs)
        finally:
            metadata.write_bytes(original)
            request.annotation_csv.write_bytes(sample_bytes)

    monkeypatch.setattr(service, "validate_acquisition_metadata", swapped_during_validation)
    hashes, warnings = service._review_acquisition_declarations(
        request.project_root, request.annotation_csv, annotation_bytes=sample_bytes
    )
    assert hashes == {}
    assert any("Unknown project specimen_id: specimen-2" in warning for warning in warnings)


def test_preflight_relationships_use_captured_metadata_despite_aba(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(service, "ALLOWED_ROOT", tmp_path)
    metadata = request.project_root / "acquisition_metadata.csv"
    original = (
        b"acquisition_id,modality,source_path,specimen_id,field,value,unit,provenance,source_detail,reviewed\n"
        b"a,Visium,visium,specimen-1,pixel_size_x,1,um,user,operator log,true\n"
    )
    metadata.write_bytes(original)
    relationships = request.project_root / "acquisition_relationships.csv"
    relationships.write_text(
        "acquisition_id,related_acquisition_id,relationship,evidence,reviewed\n"
        "a,b,serial,operator,true\n", encoding="utf-8"
    )
    validate = service.validate_acquisition_relationships

    def swapped_during_validation(*args: object, **kwargs: object) -> list[str]:
        metadata.write_bytes(original + original.splitlines(keepends=True)[1].replace(b"a,Visium", b"b,Visium"))
        try:
            return validate(*args, **kwargs)
        finally:
            metadata.write_bytes(original)

    monkeypatch.setattr(service, "validate_acquisition_relationships", swapped_during_validation)
    hashes, warnings = service._review_acquisition_declarations(request.project_root, request.annotation_csv)
    assert hashes == {metadata.name: hashlib.sha256(original).hexdigest()}
    assert any("unknown acquisition ID" in warning for warning in warnings)


def test_worker_rejects_inputs_directory_symlink_outside_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    source = request.project_root / "acquisition_metadata.csv"
    source.write_bytes(b"declaration")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = create_run_bundle(request, {}, {"_acquisition_csv_sha256": {source.name: digest}})
    inputs = request.run_dir / "inputs"
    outside = tmp_path / "external-inputs"
    inputs.rename(outside)
    inputs.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inputs resolve outside"):
        _validate_acquisition_snapshots(request.run_dir, metadata["project"])
    assert run_worker(request.run_dir) == 1
    status = json.loads((request.run_dir / "status.json").read_text(encoding="utf-8"))
    assert "inputs resolve outside" in status["message"]


def test_acquisition_snapshot_rejects_outside_symlink(tmp_path: Path) -> None:
    request = _request(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"outside")
    source = request.project_root / "acquisition_metadata.csv"
    source.symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside the project root"):
        create_run_bundle(request, {}, {"_acquisition_csv_sha256": {source.name: digest}})
    assert not request.run_dir.exists()


def test_missing_optional_acquisition_snapshots_are_compatible(tmp_path: Path) -> None:
    request = _request(tmp_path)
    metadata = create_run_bundle(request, {}, {})
    assert metadata["project"]["acquisition_snapshots"] == {}
    _validate_acquisition_snapshots(request.run_dir, metadata["project"])
    _validate_acquisition_snapshots(request.run_dir, {})

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
