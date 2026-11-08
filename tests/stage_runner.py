from __future__ import annotations

from types import ModuleType

from spatial_omics_toolkit.application.contracts import RunRequest, WorkflowDescriptor, WorkflowEmitter, WorkflowResult
from spatial_omics_toolkit.application.nextflow import _combine_results, validate_dag


def run_stages(adapter: ModuleType, request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    """Exercise adapter stages without a Nextflow runtime, only in adapter tests."""
    descriptor = WorkflowDescriptor.model_validate(adapter.describe())
    stages = {stage.stage_id: stage for stage in descriptor.stages}
    results: dict[str, WorkflowResult] = {}
    gated: set[str] = set()
    for stage_id in validate_dag(descriptor):
        stage = stages[stage_id]
        if any(parent in gated or results[parent].status.upper() == "QC_PENDING" for parent in stage.depends_on):
            gated.add(stage_id)
            continue
        upstream = {parent: results[parent] for parent in stage.depends_on}
        result = WorkflowResult.model_validate(adapter.execute_stage(request, stage_id, upstream, emit))
        results[stage_id] = result
        if result.status.upper() in {"FAILED", "BLOCKED"}:
            break
    return _combine_results(list(results.values()))
