from spatial_omics_toolkit.application.workflows.registry import (
    SUPPORTED_MODALITIES,
    list_workflows,
    registry_warnings,
)


def test_registry_exposes_each_requested_modality_without_pla() -> None:
    workflows = list_workflows()

    assert registry_warnings() == []
    assert SUPPORTED_MODALITIES == frozenset({"CycIF", "HS", "FLIM", "H&E", "Visium"})
    assert {workflow.modality for workflow in workflows} == set(SUPPORTED_MODALITIES)
    assert "pla" not in {workflow.workflow_id.lower() for workflow in workflows}
    assert "cycif.tabular_analysis" in {workflow.workflow_id for workflow in workflows}
