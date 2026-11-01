from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import RunRequest
from spatial_omics_toolkit.application.examples import get_example, prepare_example
from spatial_omics_toolkit.application.worker import _example_input_blockers
from spatial_omics_toolkit.application.workflows import flim


def _request(root: Path, parameters: dict[str, Any], modality: str = "FLIM") -> RunRequest:
    return RunRequest(
        workflow_id="flim.raw_sdt" if modality == "FLIM" else "he-image-intake",
        modality=modality, project_root=root, modality_dir=root / "data",
        annotation_csv=root / "project_samples.csv", output_root=root / "outputs",
        run_id=uuid4(), run_dir=root / "outputs" / "run", parameters=parameters,
    )


def test_example_preparation_requires_explicit_run_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        prepare_example("CycIF", tmp_path)


def test_example_source_receipt_blocks_changed_missing_or_extra_raw_files(tmp_path: Path) -> None:
    source = tmp_path / "source.sdt"
    source.write_bytes(b"source bytes for inventory verification, not decoded")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {"path": str(source), "sha256": digest}
    request = _request(tmp_path, {"example_authorization": {
        "scope": "technical_qc_only", "source_files": [record],
    }})
    assert _example_input_blockers(request, {"files": [record]}) == []
    changed = {"path": str(source), "sha256": hashlib.sha256(b"changed").hexdigest()}
    assert _example_input_blockers(request, {"files": [changed]})
    assert _example_input_blockers(request, {"files": []})
    assert _example_input_blockers(request, {"files": [record, {"path": str(tmp_path / "extra.sdt"), "sha256": digest}]})
    own_request = _request(tmp_path, {})
    assert _example_input_blockers(own_request, {"files": [changed]}) == []


def test_corrupt_example_provenance_fails_before_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPATIAL_TOOLKIT_EXAMPLES", str(tmp_path))
    project = tmp_path / "quickstart" / "he"
    project.mkdir(parents=True)
    original = b'{"source_files":[]}'
    (project / "source_provenance.json").write_bytes(b'{"source_files":[],"changed":true}')
    entry = {
        "example_id": "he.example", "modality": "H&E", "workflow_id": "he-image-intake",
        "title": "Image example", "summary": "Technical input inspection", "layout": "Single sample",
        "folder": "he", "input_help": "One image file", "expected_outputs": [], "limitations": [],
        "source_paths": [], "parameters": {"example_authorization": {"provenance_sha256": hashlib.sha256(original).hexdigest()}},
    }
    (project.parent / "catalog.json").write_text(json.dumps({"examples": {"H&E": entry}}), encoding="utf-8")
    assert get_example("H&E").project_root == project
    with pytest.raises(RuntimeError, match="provenance changed"):
        prepare_example("H&E", tmp_path / "outputs", authorize_technical_qc=True)


def test_all_block_flim_policy_is_raw_qc_only_and_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "data"
    raw.mkdir()
    (raw / "input.sdt").write_bytes(b"preflight inventories only; not measurement data")
    sources = tmp_path / "canonical"
    sources.mkdir()
    for name in ("clinical_roi_flim.py", "clinical_roi_flim_plots.py"):
        (sources / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(flim, "_source_dir", lambda: sources)
    base = {"analysis_scope": "raw_qc", "raw_qc_all_blocks": True, "technical_qc_authorized": True}
    assert flim.preflight(_request(tmp_path, base)).ready
    assert not flim.preflight(_request(tmp_path, {**base, "technical_qc_authorized": False})).ready
    assert not flim.preflight(_request(tmp_path, {**base, "analysis_scope": "full"})).ready
    assert not flim.preflight(_request(tmp_path, {**base, "block_indices": [0]})).ready
