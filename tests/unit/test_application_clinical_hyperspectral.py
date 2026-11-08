from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from spatial_omics_toolkit.application.contracts import PreflightResult, RunRequest, WorkflowResult
from spatial_omics_toolkit.application.workflows import hyperspectral


def _request(tmp_path: Path, **parameters: Any) -> RunRequest:
    return RunRequest(workflow_id="hs.clinical-roi-hyperspectral", modality="HS",
                      project_root=tmp_path, modality_dir=tmp_path,
                      annotation_csv=tmp_path / "samples.csv", output_root=tmp_path,
                      run_id=uuid4(), run_dir=tmp_path / "run", parameters=parameters)


class _Cache:
    def __init__(self, path: Path, **kwargs: Any) -> None:
        self.events: list[dict[str, Any]] = []
        self.manifest_sha256 = "manifest"
        self.queue_sha256 = "queue"


def _source(calls: list[tuple[str, str]]) -> SimpleNamespace:
    def audit(final: pd.DataFrame, *, cache: _Cache) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(("audit", ""))
        return pd.DataFrame({"path": ["first", "second"], "band": [1.0, 2.0]}), final

    def phasors(final: pd.DataFrame, channels: pd.DataFrame, *, assigned_tma: str,
                cache: _Cache, table_dir: Path, plot_dir: Path, **kwargs: Any) -> tuple[pd.DataFrame, Path, Path]:
        calls.append(("phasors", assigned_tma))
        summary = pd.DataFrame({"path": ["first"]})
        summary.attrs["analysis_receipt_path"] = str(table_dir / f"{assigned_tma}_receipt.json")
        summary.attrs["source_bound"] = True
        return summary, plot_dir / f"{assigned_tma}_spectrum.pdf", plot_dir / f"{assigned_tma}_phasor.pdf"

    def diagnostic(summary: pd.DataFrame, *, assigned_tma: str, **kwargs: Any) -> tuple[None, list[Path]]:
        assert summary.attrs["source_bound"]
        calls.append(("diagnostic", assigned_tma))
        return None, [Path(kwargs["table_dir"]) / f"{assigned_tma}_diagnostic.csv"]

    return SimpleNamespace(pd=pd, HSCache=_Cache, _full_file_hash=lambda path: "source",
                           _hs_channel_map_sha256=lambda channels: "bands",
                           load_final_analysis_manifest=lambda *args, **kwargs: pd.DataFrame({"path": ["first", "second"]}),
                           audit_spectral_channel_maps=audit, analyze_tma_spectral_phasors=phasors,
                           analyze_tma_phasor_filter_sensitivity=diagnostic,
                           analyze_tma_phasor_gmm_sensitivity=diagnostic)


def _patch_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    monkeypatch.setattr(hyperspectral, "preflight", lambda request: PreflightResult(ready=True))
    monkeypatch.setattr(hyperspectral, "_paths", lambda request: (tmp_path / "source.py", {
        key: tmp_path / key for key in ("final_manifest", "ledger", "assignment_manifest", "queue", "settings_manifest")
    }))
    monkeypatch.setattr(hyperspectral, "_load_source", lambda script: _source(calls))


def test_clinical_dag_has_independent_tma_branches() -> None:
    stages = {stage.stage_id: stage for stage in hyperspectral.describe().stages}
    assert set(stages) == {"metadata_audit", "tma1_analysis", "tma2_analysis"}
    assert stages["metadata_audit"].depends_on == []
    assert stages["tma1_analysis"].depends_on == ["metadata_audit"]
    assert stages["tma2_analysis"].depends_on == ["metadata_audit"]


def test_mapping_gate_retains_metadata_without_phasors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    _patch_source(monkeypatch, tmp_path, calls)
    request = _request(tmp_path, specimen_mapping_confirmed=True,
                       specimen_mapping={"acquisition:first": "S1", "acquisition:second": "UNMAPPED"})
    metadata = hyperspectral.execute_stage(request, "metadata_audit", {}, lambda event: None)
    assert metadata.status == "WARNING"
    assert metadata.state["mapping_complete"] is False
    assert Path(metadata.state["channel_map"]).is_file()
    for stage in ("tma1_analysis", "tma2_analysis"):
        result = hyperspectral.execute_stage(request, stage, {"metadata_audit": metadata}, lambda event: None)
        assert result.status == "SKIPPED"
    assert calls == [("audit", "")]


@pytest.mark.parametrize("diagnostic", ["none", "filter", "gmm"])
def test_tma_stages_only_run_their_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: str) -> None:
    calls: list[tuple[str, str]] = []
    _patch_source(monkeypatch, tmp_path, calls)
    request = _request(tmp_path, specimen_mapping_confirmed=True, exploratory_diagnostic=diagnostic,
                       specimen_mapping={"acquisition:first": "S1", "acquisition:second": "S2"})
    metadata = hyperspectral.execute_stage(request, "metadata_audit", {}, lambda event: None)
    assert metadata.status == "SUCCEEDED"
    calls.clear()
    result = hyperspectral.execute_stage(request, "tma2_analysis", {"metadata_audit": metadata}, lambda event: None)
    assert result.status == "SUCCEEDED"
    assert calls == [("phasors", "TMA2")] + ([] if diagnostic == "none" else [("diagnostic", "TMA2")])
    assert any(path.endswith("hs_tma2_cache_events.json") for path in result.artifacts)


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED", "QC_PENDING"])
def test_upstream_gate_prevents_analysis(tmp_path: Path, status: str) -> None:
    result = hyperspectral.execute_stage(_request(tmp_path), "tma1_analysis",
                                         {"metadata_audit": WorkflowResult(status=status)}, lambda event: None)
    assert result.status == status
