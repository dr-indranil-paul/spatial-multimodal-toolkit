from pathlib import Path
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, WorkflowResult
from spatial_omics_toolkit.application.workflows import flim


def _request(root: Path) -> RunRequest:
    return RunRequest(
        workflow_id=flim.WORKFLOW_ID, modality="FLIM", project_root=root,
        modality_dir=root, annotation_csv=root / "samples.csv", output_root=root,
        run_id=uuid4(), run_dir=root / "run", parameters={},
    )


def test_flim_dag_preserves_independent_reference_branches() -> None:
    stages = {stage.stage_id: stage for stage in flim.describe().stages}
    assert stages["flim_reference_raw_qc"].depends_on == ["flim_reference_inventory"]
    assert stages["flim_reference_phasor"].depends_on == ["flim_reference_inventory"]
    assert set(stages["flim_reference_calibration"].depends_on) == {
        "flim_count_mask", "flim_filter", "flim_reference_phasor"
    }
    assert all(stage.condition and stage.inputs and stage.outputs for stage in stages.values())


@pytest.mark.parametrize("status", ["QC_PENDING", "BLOCKED", "FAILED"])
def test_inventory_gate_prevents_downstream_work(tmp_path: Path, status: str) -> None:
    events = []
    review = {"status": "QC_PENDING"} if status == "QC_PENDING" else None
    result = flim.execute_stage(
        _request(tmp_path), "flim_phasor",
        {"flim_inventory": WorkflowResult(status=status, review_request=review)}, events.append,
    )
    assert result.status == status
    assert result.review_request == review
    assert len(events) == 1
    assert not (tmp_path / "run").exists()


def test_missing_dependency_blocks_stage(tmp_path: Path) -> None:
    result = flim.execute_stage(_request(tmp_path), "flim_raw_qc", {}, lambda event: None)
    assert result.status == "BLOCKED"
    assert not (tmp_path / "run").exists()


def test_relative_reference_and_pairing_paths_use_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "target.sdt").write_bytes(b"target")
    (tmp_path / "reference.sdt").write_bytes(b"reference")
    (tmp_path / "pairs.csv").write_text("pair\n", encoding="utf-8")
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("clinical_roi_flim.py", "clinical_roi_flim_plots.py"):
        (sources / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(flim, "_source_dir", lambda: sources)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    monkeypatch.chdir(task_dir)
    request = _request(tmp_path).model_copy(update={
        "modality_dir": data,
        "parameters": {
            "analysis_scope": "raw_qc",
            "hs_flim_pairing_manifest": "pairs.csv",
            "reference_calibration": {
                "path": "reference.sdt", "block_indices": [0],
                "known_lifetime_ns": 1.0, "provenance": "measured standard",
                "instrument_settings_match_confirmed": True,
            },
        },
    })
    result = flim.preflight(request)
    assert result.ready, result.blockers
    assert result.resolved_parameters["reference_calibration"]["path"] == str(tmp_path / "reference.sdt")
    assert result.resolved_parameters["hs_flim_pairing_manifest"] == str(tmp_path / "pairs.csv")
