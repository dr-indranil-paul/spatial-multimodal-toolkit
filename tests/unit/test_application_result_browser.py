from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
from spatial_omics_toolkit.application.ui import results


def test_artifact_inventory_deduplicates_paths_and_retains_latest_stage_state(tmp_path: Path) -> None:
    plot = tmp_path / "figure.png"
    plot.write_bytes(b"placeholder image content is not decoded in this inventory test")
    events: list[dict[str, Any]] = [
        {"stage_id": "intake", "status": "RUNNING", "artifacts": ["figure.png"]},
        {"stage_id": "intake", "status": "SUCCEEDED", "artifacts": [str(plot)]},
    ]
    manifest = {"artifacts": [
        {"path": str(plot), "stage_id": "intake", "media_type": "image/png", "role": "preview"},
        {"path": "report.pdf", "stage_id": "intake", "media_type": "application/pdf"},
        {"path": "report-page.png", "stage_id": "intake", "source_pdf": "report.pdf", "page": 1},
    ]}

    records = results._artifact_records(tmp_path, events, manifest)

    by_name = {record["path"].name: record for record in records}
    assert [record["path"].name for record in records].count("figure.png") == 1
    assert by_name["figure.png"]["status"] == "SUCCEEDED"
    assert by_name["figure.png"]["metadata"]["role"] == "preview"
    assert by_name["report-page.png"]["metadata"]["source_pdf"] == "report.pdf"
    assert set(by_name) == {"figure.png", "report.pdf", "report-page.png"}


def test_run_bundle_selection_requires_modality_workflow_and_output_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPATIAL_TOOLKIT_WORKSPACE_ROOTS", str(tmp_path))
    output = tmp_path / "outputs"
    run_dir = output / "CycIF" / "runs" / "saved"
    run_dir.mkdir(parents=True)
    metadata = {"modality": "CycIF", "workflow_id": "cycif-raw-czi", "run_id": "saved"}
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert results._bundle_metadata(run_dir, "CycIF", {"cycif-raw-czi"}, output) == metadata
    assert results._bundle_metadata(run_dir, "FLIM", {"flim.raw_sdt"}, output) is None
    assert results._bundle_metadata(run_dir, "CycIF", {"cycif.tabular_analysis"}, output) is None
    unrelated_output = tmp_path / "other"
    unrelated_output.mkdir()
    assert results._bundle_metadata(run_dir, "CycIF", {"cycif-raw-czi"}, unrelated_output) is None


def test_example_and_own_run_histories_stay_separate_in_shared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPATIAL_TOOLKIT_WORKSPACE_ROOTS", str(tmp_path))
    for scope in ("own", "example"):
        run_dir = tmp_path / "H&E" / "runs" / scope
        run_dir.mkdir(parents=True)
        metadata = {
            "modality": "H&E", "workflow_id": "he-image-intake",
            "parameters": {"example_authorization": {"scope": "technical_qc_only"}} if scope == "example" else {},
        }
        (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        assert results._bundle_metadata(
            run_dir, "H&E", {"he-image-intake"}, tmp_path, state_scope=scope,
        ) == metadata
        opposite = "example" if scope == "own" else "own"
        assert results._bundle_metadata(
            run_dir, "H&E", {"he-image-intake"}, tmp_path, state_scope=opposite,
        ) is None
