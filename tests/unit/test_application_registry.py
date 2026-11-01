from spatial_omics_toolkit.application.workflows.registry import (
    SUPPORTED_MODALITIES,
    list_workflows,
    registry_warnings,
)


def test_registry_exposes_each_requested_modality_without_pla() -> None:
    workflows = list_workflows()

    assert registry_warnings() == []
    assert SUPPORTED_MODALITIES == frozenset({"CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"})
    assert {workflow.modality for workflow in workflows} == set(SUPPORTED_MODALITIES)
    assert "pla" not in {workflow.workflow_id.lower() for workflow in workflows}
    assert "cycif.tabular_analysis" in {workflow.workflow_id for workflow in workflows}


def test_raw_cycif_exposes_required_channel_count() -> None:
    workflow = next(workflow for workflow in list_workflows() if workflow.workflow_id == "cycif-raw-czi")
    channel_field = next(field for field in workflow.parameter_fields if field.name == "intake.expected_channels")

    assert channel_field.kind == "int"
    assert channel_field.required is True
    assert channel_field.min_value == 1
