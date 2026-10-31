from pathlib import Path
from uuid import uuid4

from spatial_omics_toolkit.application.contracts import RunRequest
from spatial_omics_toolkit.application.worker import _mapping_blockers
from spatial_omics_toolkit.application.workflows.cycif_analysis import preflight


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
