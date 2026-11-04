from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent
from spatial_omics_toolkit.application.workflows.cycif import (
    _canonical_hash,
    _input_fingerprint,
    _source_code_identity,
    _stage_parameters,
    _tool_identity,
    _intake_options,
    _write_checkpoint_receipts,
    validate_continuation,
    execute,
)

from spatial_omics_toolkit.cycif.intake import parse_intake_filename


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intake_parameters() -> dict[str, Any]:
    return {
        "filename_pattern": r"^(?P<slide>sample)_(?P<scene>sceneA)_(?P<round>r1)(?P<markers>)(?P<date>)(?P<acq_id>)\.czi$",
        "expected_rounds": ["r1"],
        "expected_channels": 1,
    }


def _request(tmp_path: Path, source: Path) -> RunRequest:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    czi = raw / "sample_sceneA_r1.czi"
    czi.write_bytes(b"same raw image")
    annotation = tmp_path / "project_samples.csv"
    annotation.write_text("biological_unit_id,specimen_id,condition,tissue_source\nu1,s1,c,t\n", encoding="utf-8")
    manifest = {
        "files": [{"path": str(czi), "sha256": _sha(czi)}],
        "annotation_csv_sha256": _sha(annotation),
    }
    run_dir = tmp_path / "child"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "inputs" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({"workflow_id": "cycif-raw-czi", "project": {"annotation_csv": str(annotation)}}), encoding="utf-8")
    return RunRequest(
        workflow_id="cycif-raw-czi",
        modality="CycIF",
        project_root=tmp_path,
        modality_dir=raw,
        annotation_csv=annotation,
        output_root=tmp_path,
        run_id=uuid4(),
        run_dir=run_dir,
        parameters={
            "continuation": {"mode": "continue", "parent_run_dir": str(source), "start_stage": "conversion", "warning_review_confirmed": True},
            "intake": _intake_parameters(),
        },
    )


def _source(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = tmp_path / "CycIF" / "runs" / str(uuid4())
    (source / "inputs").mkdir(parents=True)
    artifact = source / "intake-report.json"
    artifact.write_text("complete", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    czi = raw / "sample_sceneA_r1.czi"
    czi.write_bytes(b"same raw image")
    annotation = tmp_path / "project_samples.csv"
    annotation.write_text("biological_unit_id,specimen_id,condition,tissue_source\nu1,s1,c,t\n", encoding="utf-8")
    manifest = {"files": [{"path": str(czi), "sha256": _sha(czi)}], "annotation_csv_sha256": _sha(annotation)}
    (source / "inputs" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    parameters = {"intake": _intake_parameters()}
    fingerprint = _input_fingerprint(source)
    checkpoint = {
        "workflow_id": "cycif-raw-czi",
        "stage_id": "intake",
        "stage_status": "SUCCEEDED",
        "input_fingerprint": fingerprint,
        "settings": _stage_parameters(parameters, "intake"),
        "settings_sha256": _canonical_hash(_stage_parameters(parameters, "intake")),
        "tool_identity": _tool_identity("intake"),
        "artifacts": [{"path": str(artifact.resolve()), "sha256": _sha(artifact)}],
    }
    (source / "run.json").write_text(json.dumps({"workflow_id": "cycif-raw-czi", "run_id": source.name, "modality": "CycIF"}), encoding="utf-8")
    (source / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}), encoding="utf-8")
    (source / "events.jsonl").write_text(json.dumps({"stage_id": "intake", "status": "SUCCEEDED"}) + "\n", encoding="utf-8")
    (source / "checkpoint_receipts.json").write_text(json.dumps({"workflow_id": "cycif-raw-czi", "checkpoints": {"intake": checkpoint}}), encoding="utf-8")
    return source, artifact, checkpoint


def test_valid_receipt_allows_checkpoint_reuse(tmp_path: Path) -> None:
    source, artifact, _ = _source(tmp_path)
    reasons, reused = validate_continuation(_request(tmp_path, source))
    assert reasons == []
    assert reused == [{"stage_id": "intake", "path": str(artifact.resolve()), "sha256": _sha(artifact), "source_status": "SUCCEEDED"}]


def test_raw_input_mismatch_blocks_reuse(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    raw_path = request.modality_dir / "sample_sceneA_r1.czi"
    raw_path.write_bytes(b"changed raw image")
    manifest_path = request.run_dir / "inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [{"path": str(raw_path), "sha256": _sha(raw_path)}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reasons, _ = validate_continuation(request)
    assert any("hashes differ" in reason for reason in reasons)


def test_raw_example_pattern_parses_all_intake_group_names() -> None:
    pattern = r"^(?P<slide>cycif_tma_core)_(?P<scene>a1)_(?P<round>r[1-5]q?)(?P<markers>)(?P<date>)(?P<acq_id>)\.czi$"
    options = _intake_options(
        {
            "intake": {
                "filename_pattern": pattern,
                "expected_rounds": ["r1", "r2", "r3", "r4", "r5", "r5q"],
                "expected_channels": 5,
            }
        }
    )
    parsed = parse_intake_filename("cycif_tma_core_a1_r1.czi", options["config"].filename_pattern)

    assert parsed == {
        "slide": "cycif_tma_core",
        "scene": "a1",
        "round": "r1",
        "markers": "",
        "date": "",
        "acq_id": "",
    }


def test_missing_or_modified_checkpoint_artifact_blocks_reuse(tmp_path: Path) -> None:
    source, artifact, _ = _source(tmp_path)
    artifact.write_text("modified", encoding="utf-8")
    reasons, _ = validate_continuation(_request(tmp_path, source))
    assert any("missing or modified" in reason for reason in reasons)


def test_source_code_identity_changes_with_module_bytes(tmp_path: Path) -> None:
    module = tmp_path / "workflow.py"
    module.write_text("STAGE = 1\n", encoding="utf-8")
    first = _source_code_identity({"workflow": module})
    module.write_text("STAGE = 2\n", encoding="utf-8")
    second = _source_code_identity({"workflow": module})
    assert first["workflow"] != second["workflow"]


def test_changed_source_identity_blocks_checkpoint_reuse(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    receipt_path = source / "checkpoint_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoints"]["intake"]["tool_identity"]["source:application/workflows/cycif.py"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    reasons, _ = validate_continuation(_request(tmp_path, source))
    assert any("tool/runtime identity differs" in reason for reason in reasons)


def test_parent_run_must_be_inside_selected_output_root(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    request.output_root = tmp_path / "different-output"

    reasons, _ = validate_continuation(request)

    assert any("beneath this project's CyCIF/runs directory" in reason for reason in reasons)


def test_warning_checkpoint_requires_and_records_explicit_review(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    (source / "status.json").write_text(json.dumps({"status": "WARNING"}), encoding="utf-8")
    (source / "events.jsonl").write_text(json.dumps({"stage_id": "intake", "status": "WARNING"}) + "\n", encoding="utf-8")
    receipt_path = source / "checkpoint_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoints"]["intake"]["stage_status"] = "WARNING"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    request = _request(tmp_path, source)
    request.parameters["continuation"]["warning_review_confirmed"] = False

    reasons, _ = validate_continuation(request)

    assert any("confirm reuse of warning checkpoints" in reason for reason in reasons)
    request.parameters["continuation"]["warning_review_confirmed"] = True
    reasons, reused = validate_continuation(request)
    assert reasons == []
    assert reused[0]["source_status"] == "WARNING"


def test_warning_status_survives_a_child_run_for_next_continuation(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    receipt_path = source / "checkpoint_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoints"]["intake"]["stage_status"] = "WARNING"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    child_source = source.parent / "resumed-child"
    shutil.copytree(source, child_source)
    child_metadata = json.loads((child_source / "run.json").read_text(encoding="utf-8"))
    child_metadata["run_id"] = child_source.name
    (child_source / "run.json").write_text(json.dumps(child_metadata), encoding="utf-8")
    (child_source / "status.json").write_text(json.dumps({"status": "WARNING"}), encoding="utf-8")
    (child_source / "events.jsonl").write_text(json.dumps({"stage_id": "intake", "status": "REUSED"}) + "\n", encoding="utf-8")
    request = _request(tmp_path, child_source)
    request.parameters["continuation"]["warning_review_confirmed"] = False

    reasons, _ = validate_continuation(request)

    assert any("confirm reuse of warning checkpoints" in reason for reason in reasons)


def test_intake_warning_checkpoints_only_when_qc_completed(tmp_path: Path) -> None:
    request = _request(tmp_path, _source(tmp_path)[0])
    artifact = request.run_dir / "intake-report.html"
    artifact.write_text("intake", encoding="utf-8")
    fatal_warning = StageEvent(
        stage_id="intake",
        status="WARNING",
        message="An intake check failed.",
        parameters={"check_warnings": [{"Check": "Filename parsing"}], "coverage_missing": [], "unparsed": []},
        artifacts=[str(artifact)],
    )
    _write_checkpoint_receipts(request, [fatal_warning])
    assert not (request.run_dir / "checkpoint_receipts.json").exists()

    reviewable_warning = StageEvent(
        stage_id="intake",
        status="WARNING",
        message="Metadata checks were not available.",
        parameters={"check_warnings": [], "coverage_missing": [], "unparsed": [], "checks_unassessed": [{"Check": "Channel count"}]},
        artifacts=[str(artifact)],
    )
    _write_checkpoint_receipts(request, [reviewable_warning])
    receipts = json.loads((request.run_dir / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    assert receipts["checkpoints"]["intake"]["artifacts"]


def test_export_checkpoint_is_reused_by_new_segmentation_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import spatial_omics_toolkit.application.workflows.cycif as adapter
    import spatial_omics_toolkit.cycif as cycif_api
    from spatial_omics_toolkit.application.contracts import PreflightResult

    source, _, _ = _source(tmp_path)
    composite = source / "sceneA_composite.ome.tif"
    composite.write_bytes(b"ome-tiff")
    request = _request(tmp_path, source)
    request.parameters["continuation"]["start_stage"] = "segmentation"
    request.parameters["segmentation"] = {"enabled": True, "backend": "MCMICRO", "parameters": {}}
    export_settings = {
        "marker_panel_sha256": "panel-hash",
        "marker_map": {"R1": ["DAPI"]},
        "registration": {},
    }
    export_checkpoint = {
        "workflow_id": "cycif-raw-czi",
        "stage_id": "export",
        "input_fingerprint": _input_fingerprint(source),
        "settings": export_settings,
        "settings_sha256": _canonical_hash(export_settings),
        "tool_identity": json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))["checkpoints"]["intake"]["tool_identity"],
        "artifacts": [{"path": str(composite.resolve()), "sha256": _sha(composite)}],
    }
    receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    receipt["checkpoints"]["registration"] = {**receipt["checkpoints"]["intake"], "stage_id": "registration"}
    receipt["checkpoints"]["export"] = export_checkpoint
    (source / "checkpoint_receipts.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "preflight",
        lambda _: PreflightResult(
            ready=True,
            resolved_parameters={
                "toolkit_version": "test",
                "mcmicro_runtime": {"wrapper": "/fake/mcmicro", "home": "/fake/runtime"},
            },
        ),
    )
    monkeypatch.setattr(adapter.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(stdout="mcmicro-test", stderr=""))
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.segmenter = kwargs.get("segmenter", "mccellpose")

    def fake_segmenter(_backend: str, *, project_dir: str | Path, **kwargs: Any) -> dict[str, str]:
        project = Path(project_dir)
        project.mkdir(parents=True)
        output = project / "segmentation.txt"
        output.write_text("completed", encoding="utf-8")
        return {"result": str(output)}

    monkeypatch.setattr(cycif_api, "MCMICROConfig", FakeConfig)
    monkeypatch.setattr(cycif_api, "step05_segment_cells", fake_segmenter)
    reused = [
        {"stage_id": stage_id, "path": str(composite), "sha256": _sha(composite), "source_status": "SUCCEEDED"}
        for stage_id in ("intake", "registration", "export")
    ]
    monkeypatch.setattr(adapter, "_resolve_marker_panel", lambda _: None)
    monkeypatch.setattr(adapter, "validate_continuation", lambda _: ([], reused))
    emitted = []
    result = adapter.execute(request, emitted.append)
    assert result.status == "SUCCEEDED"
    assert [event.status for event in emitted if event.status == "REUSED"] == ["REUSED", "REUSED", "REUSED"]
    assert (request.run_dir / "resume_receipt.json").is_file()
    assert json.loads((request.run_dir / "resume_receipt.json").read_text(encoding="utf-8"))["parent_run_path"] == str(source.resolve())
    assert (request.run_dir / "checkpoint_receipts.json").is_file()


def test_invalid_workflow_or_source_status_blocks_reuse(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    (source / "run.json").write_text(json.dumps({"workflow_id": "cycif.tabular_analysis"}), encoding="utf-8")
    (source / "status.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
    (source / "events.jsonl").write_text(json.dumps({"stage_id": "intake", "status": "BLOCKED"}) + "\n", encoding="utf-8")
    reasons, _ = validate_continuation(_request(tmp_path, source))
    assert any("workflow ID" in reason for reason in reasons)
    assert any("upstream stage is not complete" in reason for reason in reasons)



def test_blocked_parent_can_reuse_successful_upstream_checkpoint(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    (source / "status.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
    reasons, reused = validate_continuation(_request(tmp_path, source))
    assert reasons == []
    assert reused and reused[0]["stage_id"] == "intake"


@pytest.mark.parametrize(
    ("start_stage", "configuration"),
    [("conversion", {"conversion": {"enabled": True}}), ("registration", {"registration": {"enabled": True}})],
)
def test_intake_receipt_is_valid_at_conversion_and_registration_boundaries(
    tmp_path: Path, start_stage: str, configuration: dict[str, Any]
) -> None:
    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    request.parameters["continuation"]["start_stage"] = start_stage
    request.parameters.update(configuration)

    reasons, reused = validate_continuation(request)

    assert reasons == []
    assert [item["stage_id"] for item in reused] == ["intake"]


@pytest.mark.parametrize(
    ("start_stage", "configuration", "message"),
    [
        ("conversion", {}, "conversion.enabled=true"),
        ("registration", {}, "registration.enabled=true"),
    ],
)
def test_early_continuation_blocks_when_selected_stage_is_disabled(
    tmp_path: Path, start_stage: str, configuration: dict[str, Any], message: str
) -> None:
    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    request.parameters["continuation"]["start_stage"] = start_stage
    request.parameters.update(configuration)
    emitted: list[StageEvent] = []

    result = execute(request, emitted.append)

    assert result.status == "BLOCKED"
    assert message in result.warnings[0]
    assert emitted[-1].status == "BLOCKED"


def test_missing_intake_receipt_blocks_early_continuation(tmp_path: Path) -> None:
    source, _, _ = _source(tmp_path)
    receipt_path = source / "checkpoint_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoints"].clear()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    request = _request(tmp_path, source)
    request.parameters["conversion"] = {"enabled": True}

    reasons, reused = validate_continuation(request)

    assert any("no reusable intake checkpoint receipt" in reason for reason in reasons)
    assert reused == []


def test_conversion_continuation_reuses_intake_and_dispatches_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types
    import spatial_omics_toolkit.cycif as cycif
    import spatial_omics_toolkit.application.workflows.cycif as adapter
    from spatial_omics_toolkit.application.contracts import PreflightResult

    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    request.parameters["conversion"] = {"enabled": True, "config": {}}
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    conversion_calls: list[tuple[str, str]] = []

    def forbidden_intake(**kwargs: Any) -> None:
        raise AssertionError("raw step01 intake must not run during continuation")

    def convert(input_folder: str, output_folder: str, **kwargs: Any) -> dict[str, str]:
        conversion_calls.append((input_folder, output_folder))
        output = Path(output_folder) / "converted.ome.tif"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"converted")
        return {"output": str(output)}

    class Config:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(cycif, "step01_scan_intake", forbidden_intake, raising=False)
    monkeypatch.setattr(cycif, "ImageConversionConfig", Config, raising=False)
    monkeypatch.setattr(cycif, "step02_convert_raw_images", convert, raising=False)
    monkeypatch.setattr(
        adapter,
        "preflight",
        lambda req: PreflightResult(
            ready=True, blockers=[], warnings=[],
            resolved_parameters={"toolkit_version": "test", "czi_files": [str(req.modality_dir / "sample_sceneA_r1.czi")]},
        ),
    )
    emitted: list[StageEvent] = []

    result = execute(request, emitted.append)

    assert result.status == "SUCCEEDED"
    assert conversion_calls
    assert any(event.stage_id == "intake" and event.status == "REUSED" for event in emitted)
    assert any(event.stage_id == "conversion" and event.status == "SUCCEEDED" for event in emitted)
    resume = json.loads((request.run_dir / "resume_receipt.json").read_text(encoding="utf-8"))
    assert resume["start_stage"] == "conversion"
    assert resume["parent_run_path"] == str(source.resolve())





def test_registration_continuation_reuses_intake_and_skips_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import spatial_omics_toolkit.application.workflows.cycif as adapter
    import spatial_omics_toolkit.cycif as cycif
    from spatial_omics_toolkit.application.contracts import PreflightResult

    source, _, _ = _source(tmp_path)
    request = _request(tmp_path, source)
    raw_path = request.modality_dir / "sample_sceneA_r1.czi"
    intake_parameters = _intake_parameters()
    for run_dir in (source, request.run_dir):
        manifest_path = run_dir / "inputs" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [{"path": str(raw_path), "sha256": _sha(raw_path)}]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_receipt_path = source / "checkpoint_receipts.json"
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    intake_settings = _stage_parameters({"intake": intake_parameters}, "intake")
    intake_checkpoint = source_receipt["checkpoints"]["intake"]
    intake_checkpoint["input_fingerprint"] = _input_fingerprint(source)
    intake_checkpoint["settings"] = intake_settings
    intake_checkpoint["settings_sha256"] = _canonical_hash(intake_settings)
    source_receipt_path.write_text(json.dumps(source_receipt), encoding="utf-8")
    request.parameters["continuation"]["start_stage"] = "registration"
    request.parameters["intake"] = intake_parameters
    request.parameters["conversion"] = {"enabled": True, "config": {}}
    request.parameters["registration"] = {
        "enabled": True,
        "confirmed_grouping": True,
        "config": {"ref_round": "r1"},
    }
    monkeypatch.setattr(
        adapter,
        "preflight",
        lambda _: PreflightResult(
            ready=True,
            resolved_parameters={"toolkit_version": "test", "czi_files": [str(raw_path)]},
        ),
    )

    def forbidden_step(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("intake and conversion must not rerun when starting at registration")

    registration_calls: list[dict[str, str]] = []

    def register(
        czi_files: dict[str, str],
        *,
        output_dir: str,
        verbose: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        registration_calls.append(dict(czi_files))
        czi_path = next(iter(czi_files.values()))
        return {
            "round_order": ["r1"],
            "reference_shape": (1, 1),
            "pixel_size_um": None,
            "params": {"ref_round": "r1", "dapi_channel": 0},
            "rounds": {
                "r1": {
                    "channels": None,
                    "registered": True,
                    "status": "reference",
                    "transform": None,
                    "inliers": 0,
                    "matches": 0,
                    "czi_path": czi_path,
                    "czi_size_bytes": Path(czi_path).stat().st_size,
                    "mov_shape": None,
                }
            },
        }

    monkeypatch.setattr(cycif, "step01_scan_intake", forbidden_step, raising=False)
    monkeypatch.setattr(cycif, "step02_convert_raw_images", forbidden_step, raising=False)
    monkeypatch.setattr(cycif, "step03_register_rounds", register, raising=False)
    monkeypatch.setattr(cycif, "step04_export_registered", forbidden_step, raising=False)
    emitted: list[StageEvent] = []

    result = execute(request, emitted.append)

    assert result.status == "BLOCKED"  # Export remains gated until a reviewed marker panel is supplied.
    assert len(registration_calls) == 1
    assert registration_calls[0] == {"r1": str(raw_path)}
    assert any(event.stage_id == "intake" and event.status == "REUSED" for event in emitted)
    assert any(event.stage_id == "registration" and event.status == "SUCCEEDED" for event in emitted)
    child_receipt = json.loads((request.run_dir / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    assert child_receipt["checkpoints"]["registration"]["stage_status"] == "SUCCEEDED"
    assert (request.run_dir / "resume_receipt.json").is_file()
def test_registration_handoff_roundtrips_registered_tiffs_and_metadata(tmp_path: Path) -> None:
    import numpy as np
    import tifffile

    from spatial_omics_toolkit.application.workflows.cycif import (
        _load_registration_handoff,
        _write_registration_handoff,
    )

    registration_dir = tmp_path / "registered"
    registration_dir.mkdir()
    first = np.arange(64, dtype=np.uint16).reshape(2, 4, 8)
    second = first + 3
    tifffile.imwrite(registration_dir / "R1_registered.tif", first)
    tifffile.imwrite(registration_dir / "R2_registered.tif", second)
    result = {
        "round_order": ["R1", "R2"],
        "reference_shape": (4, 8),
        "pixel_size_um": 0.5,
        "params": {"ref_round": "R1", "dapi_channel": 0},
        "rounds": {
            "R1": {"channels": first, "dapi": first[0], "transform": np.eye(2, 3), "inliers": 0, "matches": 0, "registered": True, "status": "reference", "mov_shape": (4, 8), "czi_path": "/raw/R1.czi", "czi_size_bytes": 100},
            "R2": {"channels": second, "dapi": second[0], "transform": np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]), "inliers": 10, "matches": 12, "registered": True, "status": "ok", "mov_shape": (4, 8), "czi_path": "/raw/R2.czi", "czi_size_bytes": 120},
        },
    }
    handoff_path = _write_registration_handoff(result, "sceneA", registration_dir, 0)
    restored = _load_registration_handoff(handoff_path)
    assert restored["round_order"] == result["round_order"]
    assert restored["reference_shape"] == result["reference_shape"]
    assert restored["params"] == result["params"]
    assert np.array_equal(restored["rounds"]["R1"]["channels"], first)
    assert np.array_equal(restored["rounds"]["R2"]["channels"], second)
    assert np.array_equal(restored["rounds"]["R2"]["transform"], result["rounds"]["R2"]["transform"])


def test_registration_checkpoint_resumes_at_export_without_reregistering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import spatial_omics_toolkit.application.workflows.cycif as adapter
    import spatial_omics_toolkit.cycif as cycif_api
    from spatial_omics_toolkit.application.contracts import PreflightResult

    source, _, _ = _source(tmp_path)
    handoff = source / "sceneA_registration_handoff.json"
    handoff.write_text(json.dumps({"scene_label": "sceneA", "round_order": ["R1"]}), encoding="utf-8")
    request = _request(tmp_path, source)
    request.parameters["continuation"]["start_stage"] = "export"
    request.parameters["registration"] = {"export_config": {}}
    request.parameters["_resolved_marker_panel"] = {"sha256": "panel-hash", "marker_map": {"R1": ["DAPI"]}}
    receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
    receipt["checkpoints"]["registration"] = {
        **receipt["checkpoints"]["intake"],
        "stage_id": "registration",
        "artifacts": [{"path": str(handoff.resolve()), "sha256": _sha(handoff)}],
    }
    (source / "checkpoint_receipts.json").write_text(json.dumps(receipt), encoding="utf-8")
    reused = [
        {"stage_id": "intake", "path": str(source / "intake-report.json"), "sha256": _sha(source / "intake-report.json"), "source_status": "SUCCEEDED"},
        {"stage_id": "registration", "path": str(handoff), "sha256": _sha(handoff), "source_status": "SUCCEEDED"},
    ]
    monkeypatch.setattr(adapter, "_resolve_marker_panel", lambda parameters: parameters["_resolved_marker_panel"])
    monkeypatch.setattr(adapter, "validate_continuation", lambda _: ([], reused))
    monkeypatch.setattr(adapter, "_load_registration_handoff", lambda _: {"round_order": ["R1"]})
    monkeypatch.setattr(adapter, "preflight", lambda _: PreflightResult(ready=True, resolved_parameters={}))

    def fake_export(_result: dict[str, Any], scene_label: str, output_dir: str | Path, **kwargs: Any) -> dict[str, str]:
        composite = Path(output_dir) / f"{scene_label}_composite.ome.tif"
        composite.parent.mkdir(parents=True)
        composite.write_bytes(b"new child composite")
        return {"composite_path": str(composite)}

    monkeypatch.setattr(cycif_api, "step04_export_registered", fake_export)
    monkeypatch.setattr(adapter, "_execute_raw", lambda *_: (_ for _ in ()).throw(AssertionError("step03 raw path executed")))
    emitted = []
    result = adapter.execute(request, emitted.append)
    assert result.status == "SUCCEEDED"
    assert any(event.stage_id == "registration" and event.status == "REUSED" for event in emitted)
    assert any(event.stage_id == "export" and event.status == "SUCCEEDED" for event in emitted)
    resume_receipt = json.loads((request.run_dir / "resume_receipt.json").read_text(encoding="utf-8"))
    assert resume_receipt["start_stage"] == "export"


def test_streamlit_boundary_selector_rejects_non_raw_parent_without_exception(tmp_path: Path) -> None:
    import hashlib

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]

    output_root = tmp_path / "outputs"
    parent = output_root / "CycIF" / "runs" / "legacy-tabular-run"
    parent.mkdir(parents=True)
    (parent / "run.json").write_text(json.dumps({"workflow_id": "cycif.tabular_analysis"}), encoding="utf-8")
    (parent / "checkpoint_receipts.json").write_text(
        json.dumps({"workflow_id": "cycif.tabular_analysis", "checkpoints": {}}),
        encoding="utf-8",
    )
    (parent / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}), encoding="utf-8")
    raw_workflow = next(item for item in list_workflows() if item.workflow_id == "cycif-raw-czi")
    app = AppTest.from_file(str(Path(__file__).parents[2] / "apps" / "streamlit" / "app.py"))
    app.session_state["spatial_page"] = "CycIF"
    app.session_state["workspace_section_CycIF"] = "Setup"
    app.session_state["data_mode_CycIF"] = "own"
    app.session_state[f"selected_workflow_{key('CycIF')}"] = raw_workflow.title
    app.session_state["output_root"] = str(output_root)
    app.session_state[f"cycif_mode_{key('cycif-raw-czi')}"] = "continue"
    app.session_state[f"cycif_parent_{key('cycif-raw-czi')}"] = str(parent)

    app.run()

    assert not app.exception
    continuation = app.session_state["workflow_parameters_cycif-raw-czi"]["continuation"]
    assert continuation["start_stage"] == ""
    assert not any(widget.label == "Continue at stage" for widget in app.selectbox)


def test_streamlit_offers_conversion_and_registration_from_intake_checkpoint(tmp_path: Path) -> None:
    import hashlib

    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]

    source, _, _ = _source(tmp_path)
    raw_workflow = next(item for item in list_workflows() if item.workflow_id == "cycif-raw-czi")
    app = AppTest.from_file(str(Path(__file__).parents[2] / "apps" / "streamlit" / "app.py"))
    app.session_state["spatial_page"] = "CycIF"
    app.session_state["workspace_section_CycIF"] = "Setup"
    app.session_state["data_mode_CycIF"] = "own"
    app.session_state[f"selected_workflow_{key('CycIF')}"] = raw_workflow.title
    app.session_state["output_root"] = str(tmp_path)
    app.session_state[f"cycif_mode_{key('cycif-raw-czi')}"] = "continue"
    app.session_state[f"cycif_parent_{key('cycif-raw-czi')}"] = str(source)

    app.run()

    assert not app.exception
    boundary = next(widget for widget in app.selectbox if widget.label == "Continue at stage")
    assert boundary.options == ["conversion", "registration"]
