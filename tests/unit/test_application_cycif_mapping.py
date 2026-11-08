import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent, WorkflowResult
from spatial_omics_toolkit.application.worker import _mapping_blockers
from spatial_omics_toolkit.application.workflows import cycif_analysis
from spatial_omics_toolkit.application.workflows.cycif_analysis import preflight
from tests.stage_runner import run_stages


def _request(tmp_path: Path, use_specimen_id_column: bool) -> RunRequest:
    project_root = tmp_path / "project"
    modality_dir = project_root / "cycif"
    modality_dir.mkdir(parents=True)
    annotation_csv = project_root / "project_samples.csv"
    annotation_csv.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "unit-1,specimen-1,control,tissue\n",
        encoding="utf-8",
    )
    feature_path = modality_dir / "features.csv"
    feature_path.write_text(
        "acquisition,core,x,y,marker,specimen_id\n"
        "acq-1,core-1,1,2,3.5,specimen-1\n",
        encoding="utf-8",
    )
    parameters = {
        "feature_table_path": str(feature_path),
        "columns": {
            "acquisition_group": "acquisition",
            "core_id": "core",
            "x": "x",
            "y": "y",
            "intensity_columns": ["marker"],
            "specimen_id": "specimen_id" if use_specimen_id_column else None,
        },
        "specimen_mapping_confirmed": True,
        "use_specimen_id_column": use_specimen_id_column,
        "core_to_specimen": {} if use_specimen_id_column else {"core-1": "specimen-1"},
    }
    return RunRequest(
        workflow_id="cycif.tabular_analysis",
        modality="CycIF",
        project_root=project_root,
        modality_dir=modality_dir,
        annotation_csv=annotation_csv,
        output_root=tmp_path / "outputs",
        run_id=uuid4(),
        run_dir=tmp_path / "outputs" / "run",
        parameters=parameters,
    )


def test_manual_cycif_mapping_keeps_worker_contract_string_values(tmp_path: Path) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)

    result = preflight(request)
    request.parameters = {**request.parameters, **result.resolved_parameters}

    assert result.ready, result.blockers
    assert request.parameters["core_to_specimen"] == {"core-1": "specimen-1"}
    assert request.parameters["resolved_core_to_specimen"]["core-1"]["biological_unit_id"] == "unit-1"
    assert _mapping_blockers(request, result.resolved_parameters["acquisition_groups"]) == []


def test_existing_specimen_column_does_not_conflict_with_resolved_mapping(tmp_path: Path) -> None:
    request = _request(tmp_path, use_specimen_id_column=True)

    result = preflight(request)
    request.parameters = {**request.parameters, **result.resolved_parameters}

    assert result.ready, result.blockers
    assert request.parameters["core_to_specimen"] == {}
    assert request.parameters["resolved_core_to_specimen"]["core-1"]["specimen_id"] == "specimen-1"
    assert _mapping_blockers(request, result.resolved_parameters["acquisition_groups"]) == []


def test_optional_stages_pass_mapped_table_without_analysis(tmp_path: Path) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    events: list[StageEvent] = []

    result = run_stages(cycif_analysis, request, events.append)

    assert result.status in {"SUCCEEDED", "WARNING"}
    assert {event.stage_id for event in events if event.status == "SKIPPED"} == {
        "step06", "step07", "step08", "step09", "step10", "step11", "step12"
    }
    mapped = pd.read_csv(request.run_dir / "cycif_analysis" / "mapped_cell_features.csv")
    assert mapped.loc[0, "biological_unit_id"] == "unit-1"
    assert mapped.loc[0, "condition"] == "control"


def test_qc_runs_once_and_passes_receipt_through_skipped_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    request.parameters["qc"] = {}
    calls: list[pd.DataFrame] = []

    def qc(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        calls.append(frame.copy())
        result = frame + 1
        result.attrs["marker_snr"] = {"marker": 1.0}
        return result

    monkeypatch.setattr(cycif_analysis, "step06_qc_filter_intensities", qc)
    events: list[StageEvent] = []
    result = run_stages(cycif_analysis, request, events.append)

    assert result.status in {"SUCCEEDED", "WARNING"}
    assert len(calls) == 1
    corrected = pd.read_csv(request.run_dir / "cycif_analysis" / "step06_qc" / "cell_features_qc.csv")
    assert corrected.loc[0, "marker"] == 4.5
    assert any(event.stage_id == "step06" and event.status == "SUCCEEDED" for event in events)
    assert any(event.stage_id == "step07" and event.status == "SKIPPED" for event in events)


@pytest.mark.parametrize("absolute", [False, True])
def test_config_files_resolve_from_project_not_stage_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    config_path = request.project_root / "qc.json"
    mapping_path = request.project_root / "mapping.json"
    config_path.write_text('{"background_percentile": 12}', encoding="utf-8")
    mapping_path.write_text('{"core-1": "specimen-1"}', encoding="utf-8")
    request.parameters["qc"] = str(config_path) if absolute else "qc.json"
    request.parameters["core_to_specimen"] = str(mapping_path) if absolute else "mapping.json"
    stage_cwd = tmp_path / "nextflow-work"
    stage_cwd.mkdir()
    monkeypatch.chdir(stage_cwd)
    calls: list[object] = []

    def qc(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs["background_percentile"])
        return frame

    monkeypatch.setattr(cycif_analysis, "step06_qc_filter_intensities", qc)
    check = preflight(request)
    assert check.ready, check.blockers
    assert check.resolved_parameters["stage_configs"]["qc"]["background_percentile"] == 12
    events: list[StageEvent] = []
    result = cycif_analysis.execute_stage(request, "step06", {}, events.append)
    assert result.status == "SUCCEEDED"
    assert calls == [12]


def test_stage_receipts_and_failure_gate(tmp_path: Path) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    events: list[StageEvent] = []
    first = cycif_analysis.execute_stage(request, "step06", {}, events.append)
    second = cycif_analysis.execute_stage(request, "step07", {"step06": first}, events.append)
    assert first.status == second.status == "SKIPPED"
    assert first.state["cell_table"] == second.state["cell_table"]
    json.dumps(second.state)
    blocked = cycif_analysis.execute_stage(request, "step08", {}, events.append)
    assert blocked.status == "BLOCKED"
    for status in ("FAILED", "BLOCKED", "QC_PENDING"):
        gated = cycif_analysis.execute_stage(
            request, "step08", {"step07": WorkflowResult(status=status)}, events.append
        )
        assert gated.status == "BLOCKED"


def test_tabular_stage_graph_retains_parallel_spatial_and_aggregate_branches() -> None:
    stages = {stage.stage_id: stage for stage in cycif_analysis.describe().stages}
    assert stages["step07"].depends_on == ["step06"]
    assert stages["step08"].depends_on == ["step07"]
    assert stages["step09"].depends_on == ["step08"]
    assert stages["step10"].depends_on == ["step08"]
    assert stages["step11"].depends_on == ["step08"]
    assert "matching spatial_features pair and k configuration" in stages["step11"].condition
    assert set(stages["step12"].depends_on) == {"step08", "step09"}
    assert all(stage.condition and stage.inputs and stage.outputs and stage.tools for stage in stages.values())


def test_aggregation_weights_cells_directly_across_cores(tmp_path: Path) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    Path(request.parameters["feature_table_path"]).write_text(
        "acquisition,core,x,y,marker,cell_type\n"
        "acq-1,core-1,1,2,0,A\n"
        "acq-1,core-2,2,3,6,A\n"
        "acq-1,core-2,3,4,6,A\n",
        encoding="utf-8",
    )
    request.parameters["columns"]["specimen_id"] = None
    request.parameters["core_to_specimen"] = {"core-1": "specimen-1", "core-2": "specimen-1"}
    request.parameters["aggregate"] = {"level": "biological_unit_id"}
    events: list[StageEvent] = []

    result = run_stages(cycif_analysis, request, events.append)

    assert result.status in {"SUCCEEDED", "WARNING"}
    aggregate = pd.read_csv(request.run_dir / "cycif_analysis" / "step09_aggregation" / "biological_unit_features.csv")
    assert len(aggregate) == 1
    assert aggregate.loc[0, "total_cells"] == 3
    assert aggregate.loc[0, "marker"] == 4.0


def test_typed_handoffs_preserve_strings_nulls_categories_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, use_specimen_id_column=False)
    request.annotation_csv.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "unit-1,specimen-1,control,tissue\n"
        "unit-2,specimen-2,control,tissue\n"
        "unit-3,specimen-3,treated,tissue\n"
        "unit-4,specimen-4,treated,tissue\n",
        encoding="utf-8",
    )
    source = pd.DataFrame(
        {
            "acquisition": ["acq-1"] * 4,
            "core": ["core-1", "core-2", "core-3", "core-4"],
            "x": [1, 2, 3, 4],
            "y": [2, 3, 4, 5],
            "marker": [1.0, 2.0, 3.0, 4.0],
            "cell_type": pd.Categorical(["NA", "NULL", "", None], categories=["NA", "NULL", ""]),
        },
        index=pd.Index([7, 9, 11, 13], name="cell_index"),
    )
    source_path = request.modality_dir / "typed_features.parquet"
    source.to_parquet(source_path, index=True)
    request.parameters.update(
        feature_table_path=str(source_path),
        core_to_specimen={f"core-{n}": f"specimen-{n}" for n in range(1, 5)},
        qc={},
        classification={"marker_config": {
            "epithelial_markers": ["marker"], "mesenchymal_markers": ["marker"],
            "immune_marker": "marker", "myeloid_markers": ["marker"],
            "antigen_presentation_marker": "marker", "stromal_marker": "marker",
            "endothelial_marker": "marker", "proliferation_marker": "marker",
            "z_score_markers": ["marker"],
        }},
        programs={"definitions": [{"name": "test", "marker_columns": ["marker"]}]},
        aggregate={"level": "biological_unit_id"},
        compare_groups={"metrics": ["marker"]},
    )
    request.parameters["columns"]["specimen_id"] = None
    seen_cells: list[pd.DataFrame] = []
    seen_aggregates: list[pd.DataFrame] = []

    def qc(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return frame

    def classify(frame: pd.DataFrame, config: object, **kwargs: object) -> pd.DataFrame:
        return frame

    def programs(frame: pd.DataFrame, definitions: object, **kwargs: object) -> pd.DataFrame:
        return frame

    def aggregate(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        seen_cells.append(frame.copy())
        return pd.DataFrame({
            "biological_unit_id": frame["biological_unit_id"].to_numpy(),
            "marker": frame["marker"].to_numpy(),
            "test_label": frame["cell_type"].reset_index(drop=True),
        })

    def compare(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        seen_aggregates.append(frame.copy())
        return pd.DataFrame({"metric": ["marker"]})

    monkeypatch.setattr(cycif_analysis, "step06_qc_filter_intensities", qc)
    monkeypatch.setattr(cycif_analysis, "step07_classify_cells", classify)
    monkeypatch.setattr(cycif_analysis, "step08_score_programs", programs)
    monkeypatch.setattr(cycif_analysis, "step09_aggregate_per_core", aggregate)
    monkeypatch.setattr(cycif_analysis, "step12_compare_groups", compare)
    events: list[StageEvent] = []
    step06 = cycif_analysis.execute_stage(request, "step06", {}, events.append)
    step07 = cycif_analysis.execute_stage(request, "step07", {"step06": step06}, events.append)
    step08 = cycif_analysis.execute_stage(request, "step08", {"step07": step07}, events.append)
    step09 = cycif_analysis.execute_stage(request, "step09", {"step08": step08}, events.append)
    step12 = cycif_analysis.execute_stage(request, "step12", {"step08": step08, "step09": step09}, events.append)

    assert [step06.status, step07.status, step08.status, step09.status, step12.status] == [
        "SUCCEEDED", "SUCCEEDED", "SUCCEEDED", "SUCCEEDED", "SUCCEEDED"
    ]
    for result in (step06, step07, step08):
        assert result.state["cell_table"].endswith(".parquet")
        assert result.state["cell_table"] in result.artifacts
    assert step09.state["aggregate_table"].endswith(".parquet")
    assert step09.state["aggregate_table"] in step09.artifacts
    pd.testing.assert_series_equal(seen_cells[0]["cell_type"], source["cell_type"])
    pd.testing.assert_series_equal(
        seen_aggregates[0]["test_label"],
        source["cell_type"].reset_index(drop=True).rename("test_label"),
    )
