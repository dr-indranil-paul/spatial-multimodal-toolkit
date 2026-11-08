from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.application import plotting
from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent, WorkflowResult
from spatial_omics_toolkit.application.workflows import hyperspectral_native as native


def _request(tmp_path: Path) -> RunRequest:
    data = tmp_path / "data"
    data.mkdir()
    (data / "source.czi").write_bytes(b"native source")
    return RunRequest(
        workflow_id=native.WORKFLOW_ID, modality="HS", project_root=tmp_path,
        modality_dir=data, annotation_csv=tmp_path / "samples.csv",
        output_root=tmp_path / "outputs", run_id=uuid4(), run_dir=tmp_path / "run",
    )


def test_native_reader_and_plot_are_distinct_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("clinical_roi_hs_flim.py", "clinical_roi_hyperspectral.py"):
        (sources / name).write_text("# source identity\n")
    monkeypatch.setenv("SPATIAL_TOOLKIT_HS_SOURCE_DIR", str(sources))
    calls: list[Path] = []

    def read_moments(path: Path, *, max_working_bytes: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
        calls.append(path)
        assert max_working_bytes == 4 * 1024**3
        moments = np.ones((5, 2, 2), dtype=float)
        return moments, pd.DataFrame({"band": [1, 2, 3, 4, 5]}), {"native_shape_yx": [2, 2], "channel_count": 5}

    monkeypatch.setattr(native, "_reader", lambda: SimpleNamespace(read_native_hs_moments=read_moments))
    monkeypatch.setattr(plotting, "ensure_arial", lambda: None)
    events: list[StageEvent] = []
    stages = native.describe().stages
    assert [stage.stage_id for stage in stages] == ["hs_native_moments", "hs_native_qc_plots"]
    assert stages[0].depends_on == []
    assert stages[1].depends_on == ["hs_native_moments"]
    result = native.execute_stage(request, "hs_native_moments", {}, events.append)
    assert result.status == "SUCCEEDED"
    assert len(calls) == 1
    assert not (request.run_dir / "diagnostics").exists()
    metadata = json.loads((request.run_dir / "native_source_metadata.json").read_text())
    assert metadata["native_metadata"]["channel_count"] == 5
    assert metadata["source_sha256"]
    assert (request.run_dir / "tables/native_spectral_band_audit.csv").is_file()
    plotted = native.execute_stage(request, "hs_native_qc_plots", {"hs_native_moments": result}, events.append)
    assert plotted.status == "SUCCEEDED"
    assert len(calls) == 1
    assert (request.run_dir / "diagnostics/native_intensity_qc.pdf").is_file()
    assert (request.run_dir / "diagnostics/native_spectral_phasor_density_qc.pdf").is_file()
    receipt = json.loads((request.run_dir / "adapter_receipt.json").read_text())
    assert set(result.artifacts).issubset(receipt["artifacts"])


def test_native_plot_blocks_without_successful_moments(tmp_path: Path) -> None:
    request = _request(tmp_path)
    events: list[StageEvent] = []
    failed = WorkflowResult(status="QC_PENDING")
    result = native.execute_stage(request, "hs_native_qc_plots", {"hs_native_moments": failed}, events.append)
    assert result.status == "BLOCKED"
    assert not request.run_dir.exists()
    assert events[-1].stage_id == "hs_native_qc_plots"


def test_native_reader_rejects_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setenv("SPATIAL_TOOLKIT_HS_SOURCE_DIR", str(tmp_path / "missing"))
    result = native.execute_stage(request, "hs_native_moments", {}, lambda event: None)
    assert result.status == "BLOCKED"
    assert not request.run_dir.exists()
