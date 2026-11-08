from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, StageDefinition, WorkflowDescriptor, WorkflowEmitter, WorkflowResult
from spatial_omics_toolkit.application.nextflow import _allocated_resources, run_nextflow, run_stage, validate_dag
from spatial_omics_toolkit.application.runs import atomic_json


def _stage(stage_id: str, depends_on: list[str] | None = None) -> StageDefinition:
    return StageDefinition(stage_id=stage_id, title=stage_id, purpose=stage_id, depends_on=depends_on or [])


def _descriptor(stages: list[StageDefinition]) -> WorkflowDescriptor:
    return WorkflowDescriptor(workflow_id="test", modality="HS", title="Test", description="Test", stages=stages, input_description="input", output_description="output")


def _request(path: Path) -> RunRequest:
    return RunRequest(workflow_id="test", modality="HS", project_root=path, modality_dir=path, annotation_csv=path / "samples.csv", output_root=path, run_id=uuid4(), run_dir=path, parameters={"resources": {"cpus": 2, "memory_gb": 4}})


def test_graph_is_topological_and_preserves_branch_edges() -> None:
    descriptor = _descriptor([_stage("join", ["left", "right"]), _stage("left", ["root"]), _stage("root"), _stage("right", ["root"])])
    assert validate_dag(descriptor) == ["root", "left", "right", "join"]


@pytest.mark.parametrize("stages", [
    [_stage("a"), _stage("a")],
    [_stage("a", ["missing"])],
    [_stage("a", ["b"]), _stage("b", ["a"])],
    [_stage("bad stage")],
    [_stage("a"), _stage("b", ["a", "a"])],
    [],
])
def test_invalid_graphs_are_rejected(stages: list[StageDefinition]) -> None:
    with pytest.raises(ValueError):
        validate_dag(_descriptor(stages))


def test_stage_blocks_changed_scientific_config_before_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor([_stage("root")])
    module = ModuleType("test_adapter")
    calls: list[str] = []
    def execute(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
        calls.append(stage_id)
        return WorkflowResult(status="SUCCEEDED")
    module.execute_stage = execute  # type: ignore[attr-defined]
    monkeypatch.setattr("spatial_omics_toolkit.application.workflows.registry.get_workflow", lambda workflow_id: (module, descriptor))
    monkeypatch.setattr("spatial_omics_toolkit.application.nextflow.validate_frozen_config_files", lambda request: ["Scientific config changed"])
    request_path = tmp_path / "request.json"
    atomic_json(request_path, _request(tmp_path).model_dump(mode="json"))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    assert run_stage(request_path, receipts, "root", tmp_path / "output.json", {}) == 1
    assert calls == []
    result = json.loads((receipts / "root.json").read_text())
    assert result["status"] == "BLOCKED"
    assert "Scientific config changed" in result["warnings"]


def test_failed_stage_is_nonzero_and_does_not_swallow_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor([_stage("root")])
    module = ModuleType("test_adapter")
    def fail(*args: object) -> WorkflowResult:
        raise RuntimeError("scientific tool failed")
    module.execute_stage = fail  # type: ignore[attr-defined]
    monkeypatch.setattr("spatial_omics_toolkit.application.workflows.registry.get_workflow", lambda workflow_id: (module, descriptor))
    request_path = tmp_path / "request.json"
    atomic_json(request_path, _request(tmp_path).model_dump(mode="json"))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    assert run_stage(request_path, receipts, "root", tmp_path / "output.json", {}) == 1
    assert "scientific tool failed" in (receipts / "root.error.log").read_text()
    assert json.loads((receipts / "root.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("status", ["QC_PENDING", "BLOCKED", "FAILED"])
def test_stage_cli_never_executes_behind_terminal_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    descriptor = _descriptor([_stage("root"), _stage("child", ["root"])])
    module = ModuleType("test_adapter")
    calls: list[str] = []
    def execute(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
        calls.append(stage_id)
        return WorkflowResult(status="SUCCEEDED")
    module.execute_stage = execute  # type: ignore[attr-defined]
    monkeypatch.setattr("spatial_omics_toolkit.application.workflows.registry.get_workflow", lambda workflow_id: (module, descriptor))
    request_path, upstream_path = tmp_path / "request.json", tmp_path / "root.json"
    atomic_json(request_path, _request(tmp_path).model_dump(mode="json"))
    atomic_json(upstream_path, WorkflowResult(status=status).model_dump(mode="json"))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    assert run_stage(request_path, receipts, "child", tmp_path / "output.json", {"root": upstream_path}) == 1
    assert calls == []


def test_runner_preserves_qc_gate_without_downstream_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor([_stage("review"), _stage("downstream", ["review"])])
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")
    monkeypatch.setattr("spatial_omics_toolkit.application.nextflow.shutil.which", lambda name: "/bin/nextflow")
    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert "-resume" not in command
        invocation = Path(kwargs["cwd"])
        atomic_json(invocation / "receipts" / "review.json", WorkflowResult(status="QC_PENDING", review_request={"reason": "review"}).model_dump(mode="json"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("spatial_omics_toolkit.application.nextflow.subprocess.run", run)
    result = run_nextflow(_request(tmp_path), descriptor, lambda event: None)
    assert result.status == "QC_PENDING"
    assert result.review_request == {"reason": "review"}
    assert not (tmp_path / "nextflow" / "attempt-0" / "receipts" / "downstream.json").exists()


def test_resources_require_and_cannot_exceed_slurm_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM"):
        _allocated_resources(request)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")
    assert _allocated_resources(request) == (2, 4)
    request.parameters["resources"]["cpus"] = 3
    with pytest.raises(RuntimeError, match="exceed"):
        _allocated_resources(request)
