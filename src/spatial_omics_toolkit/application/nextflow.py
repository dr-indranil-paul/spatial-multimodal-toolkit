from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import cast

from spatial_omics_toolkit.application.contracts import (
    RunRequest, StageEvent, StageStatus, WorkflowDescriptor, WorkflowEmitter, WorkflowResult,
)
from spatial_omics_toolkit.application.runs import append_event, atomic_json, update_status
from spatial_omics_toolkit.application.settings import validate_frozen_config_files

_LAUNCHER = "/home/groups/EmiliLab/pauli/00_environments/bin/spatial-python"
_STAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TERMINAL = {"FAILED", "BLOCKED", "QC_PENDING"}
_STATUSES = {"SUCCEEDED", "WARNING", "REUSED", "SKIPPED", *_TERMINAL}


def validate_dag(descriptor: WorkflowDescriptor) -> list[str]:
    """Validate stage identifiers and return a stable topological ordering."""
    stages = descriptor.stages
    ids = [stage.stage_id for stage in stages]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Workflow stages must be nonempty and have unique identifiers")
    if any(not _STAGE_ID.fullmatch(stage_id) for stage_id in ids):
        raise ValueError("Stage identifiers must contain only ASCII letters, digits, underscores or hyphens")
    known = set(ids)
    dependencies: dict[str, set[str]] = {}
    for stage in stages:
        if len(stage.depends_on) != len(set(stage.depends_on)):
            raise ValueError(f"Duplicate dependency for stage {stage.stage_id}")
        missing = set(stage.depends_on) - known
        if missing:
            raise ValueError(f"Unknown dependencies for stage {stage.stage_id}: {sorted(missing)}")
        dependencies[stage.stage_id] = set(stage.depends_on)
    order: list[str] = []
    while dependencies:
        ready = [stage_id for stage_id in ids if stage_id in dependencies and not dependencies[stage_id]]
        if not ready:
            raise ValueError("Workflow stage dependencies contain a cycle")
        for stage_id in ready:
            order.append(stage_id)
            del dependencies[stage_id]
        for remaining in dependencies.values():
            remaining.difference_update(ready)
    return order


def _encoded(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path.resolve()).encode()).decode()


def _decoded(value: str) -> Path:
    return Path(base64.urlsafe_b64decode(value.encode()).decode())


def generate_pipeline(descriptor: WorkflowDescriptor, request_path: Path, receipt_dir: Path) -> str:
    """Generate one actual DSL2 process for each descriptor stage and its real edges."""
    order = validate_dag(descriptor)
    by_id = {stage.stage_id: stage for stage in descriptor.stages}
    names = {stage_id: f"STAGE_{index:03d}" for index, stage_id in enumerate(order)}
    lines = ["import groovy.json.JsonSlurper", "nextflow.enable.dsl=2", ""]
    for stage_id in order:
        stage = by_id[stage_id]
        name = names[stage_id]
        lines.extend([f"process {name} {{", "    input:"])
        if stage.depends_on:
            lines.extend(f"    path dep_{index}" for index in range(len(stage.depends_on)))
        else:
            lines.append("    val trigger")
        lines.extend(["    output:", f"    path '{name}.json'", "    script:", '    """'])
        command = (f"{_LAUNCHER} -m spatial_omics_toolkit.application.nextflow stage "
                   f"--request-b64 {_encoded(request_path)} --receipts-b64 {_encoded(receipt_dir)} "
                   f"--stage {stage_id} --output {name}.json")
        for index, dependency in enumerate(stage.depends_on):
            command += f' --upstream {dependency} "${{dep_{index}}}"'
        lines.extend([f"    {command}", '    """', "}", ""])
    lines.extend(["workflow {", "    def passGate = { p -> new JsonSlurper().parse(p.toFile()).status != 'QC_PENDING' }"])
    for stage_id in order:
        dependencies = by_id[stage_id].depends_on
        args = ", ".join(f"{names[dependency]}.out.filter(passGate)" for dependency in dependencies)
        lines.append(f"    {names[stage_id]}({args if dependencies else 'channel.value(true)'})")
    lines.extend(["}", ""])
    return "\n".join(lines)


def _allocated_resources(request: RunRequest) -> tuple[int, int]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Nextflow execution requires an existing SLURM allocation")
    resources = request.parameters.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Workflow resources must be an object")
    cpus, memory_gb = resources.get("cpus", 1), resources.get("memory_gb", 8)
    if (not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1 or
            not isinstance(memory_gb, int) or isinstance(memory_gb, bool) or memory_gb < 1):
        raise ValueError("Invalid workflow CPU or memory request")
    try:
        allocated_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        if "SLURM_MEM_PER_NODE" in os.environ:
            allocated_mb = int(os.environ["SLURM_MEM_PER_NODE"])
        else:
            allocated_mb = int(os.environ["SLURM_MEM_PER_CPU"]) * allocated_cpus
    except (KeyError, ValueError) as exc:
        raise RuntimeError("SLURM allocation CPU and memory limits are required") from exc
    if cpus > allocated_cpus or memory_gb * 1024 > allocated_mb:
        raise RuntimeError("Workflow resources exceed the SLURM allocation")
    return cpus, memory_gb


def _combine_results(results: list[WorkflowResult]) -> WorkflowResult:
    statuses = {result.status.upper() for result in results}
    status = next((s for s in ("FAILED", "BLOCKED", "QC_PENDING", "WARNING") if s in statuses), "SUCCEEDED")
    reviews = [result.review_request for result in results if result.status.upper() == "QC_PENDING"]
    if len(reviews) > 1:
        raise ValueError("Multiple simultaneous QC gates require separate review handling")
    return WorkflowResult(
        status=status,
        stages=[stage for result in results for stage in result.stages],
        artifacts=list(dict.fromkeys(artifact for result in results for artifact in result.artifacts)),
        warnings=list(dict.fromkeys(warning for result in results for warning in result.warnings)),
        review_request=reviews[0] if reviews else None,
    )


def run_nextflow(request: RunRequest, descriptor: WorkflowDescriptor, emit: WorkflowEmitter) -> WorkflowResult:
    """Run Nextflow locally inside the guarded worker; never fall back to direct execution."""
    cpus, memory_gb = _allocated_resources(request)
    executable = shutil.which("nextflow")
    if not executable:
        raise RuntimeError("Nextflow is not available in the guarded environment")
    base = request.run_dir / "nextflow"
    base.mkdir(parents=True, exist_ok=True)
    attempt = int(request.parameters.get("review_attempt", 0))
    invocation = base / f"attempt-{attempt}"
    invocation.mkdir(parents=True, exist_ok=False)
    receipts = invocation / "receipts"
    receipts.mkdir()
    request_path = invocation / "request.json"
    atomic_json(request_path, request.model_dump(mode="json"))
    source = invocation / "workflow.nf"
    source.write_text(generate_pipeline(descriptor, request_path, receipts), encoding="utf-8")
    config = invocation / "nextflow.config"
    task_memory_mb = memory_gb * 1024 - 512
    config.write_text(
        "process.executor = 'local'\n"
        f"process.cpus = {cpus}\nprocess.memory = '{task_memory_mb} MB'\n"
        f"executor.cpus = {cpus}\nexecutor.memory = '{task_memory_mb} MB'\n"
        "executor.queueSize = 1\nprocess.errorStrategy = 'terminate'\n",
        encoding="utf-8",
    )
    # A fresh session and fresh work directory are intentional: Nextflow's cache
    # cannot validate external adapter outputs or the full scientific environment.
    command = [executable, "-c", str(config), "run", str(source), "-w", str(invocation / "work"),
               "-with-trace", str(invocation / "trace.tsv"), "-with-report", str(invocation / "report.html"),
               "-with-dag", str(invocation / "dag.dot")]
    environment = os.environ.copy()
    # The controller and any nested Nextflow controller must share the allocation
    # with stage tools, rather than each inheriting the environment's 4 GiB heap.
    environment["NXF_OPTS"] = "-Xms64m -Xmx384m"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(cpus)
    emit(StageEvent(stage_id="nextflow", status="RUNNING", message="Running descriptor DAG inside the SLURM allocation", parameters={"cpus": cpus, "memory_gb": memory_gb, "task_memory_mb": task_memory_mb, "controller_reserved_mb": 512, "attempt": attempt}, artifacts=[str(source), str(config)]))
    with (invocation / "nextflow.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=invocation, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    results: list[WorkflowResult] = []
    stage_results: dict[str, WorkflowResult] = {}
    for stage_id in validate_dag(descriptor):
        receipt = receipts / f"{stage_id}.json"
        if receipt.is_file():
            stage_results[stage_id] = WorkflowResult.model_validate_json(receipt.read_text(encoding="utf-8"))
            results.append(stage_results[stage_id])
    result = _combine_results(results)
    if completed.returncode != 0 and result.status not in {"FAILED", "BLOCKED"}:
        message = f"Nextflow exited with status {completed.returncode}; see {invocation / 'nextflow.log'}"
        emit(StageEvent(stage_id="nextflow", status="FAILED", message=message))
        result.status = "FAILED"
        result.warnings.append(message)
    elif completed.returncode == 0 and not results:
        raise RuntimeError("Nextflow completed without stage receipts")
    elif completed.returncode == 0:
        gated: set[str] = set()
        for stage_id in validate_dag(descriptor):
            stage = next(item for item in descriptor.stages if item.stage_id == stage_id)
            if any(parent in gated or (parent in stage_results and stage_results[parent].status == "QC_PENDING") for parent in stage.depends_on):
                gated.add(stage_id)
            elif stage_id not in stage_results:
                raise RuntimeError(f"Nextflow completed without a receipt for stage {stage_id}")
    return result


def run_stage(request_path: Path, receipts: Path, stage_id: str, output: Path, upstream_paths: dict[str, Path]) -> int:
    """Execute exactly one registered adapter stage, persisting its receipt and events."""
    from spatial_omics_toolkit.application.workflows.registry import get_workflow

    request = RunRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    module, descriptor = get_workflow(request.workflow_id)
    validate_dag(descriptor)
    stage = next((item for item in descriptor.stages if item.stage_id == stage_id), None)
    if stage is None or set(upstream_paths) != set(stage.depends_on):
        raise ValueError("Stage or upstream receipt set does not match the registered DAG")
    upstream = {key: WorkflowResult.model_validate_json(path.read_text(encoding="utf-8")) for key, path in upstream_paths.items()}

    def emit(event: StageEvent) -> None:
        append_event(request.run_dir, event)
        if event.status in _TERMINAL:
            update_status(request.run_dir, event.status, event.message, current_stage=event.stage_id)
        else:
            update_status(request.run_dir, "RUNNING", event.message, current_stage=event.stage_id)

    try:
        emit(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Nextflow started stage {stage_id}"))
        if any(result.status.upper() in _TERMINAL for result in upstream.values()):
            raise ValueError("Refusing to execute a stage behind an upstream terminal gate")
        config_blockers = validate_frozen_config_files(request)
        if config_blockers:
            message = "; ".join(config_blockers)
            event = StageEvent(stage_id=stage_id, status="BLOCKED", message=message)
            result = WorkflowResult(status="BLOCKED", stages=[event], warnings=config_blockers)
        else:
            result = WorkflowResult.model_validate(module.execute_stage(request, stage_id, upstream, emit))
        result.status = result.status.upper()
        if result.status not in _STATUSES:
            raise ValueError(f"Invalid stage result status: {result.status}")
        if result.status == "QC_PENDING" and result.review_request is None:
            raise ValueError("QC_PENDING requires a review request")
        completion = StageEvent(
            stage_id=stage_id, status=cast(StageStatus, result.status),
            message=result.stages[-1].message if result.stages else str(result.state.get("reason", f"Stage {stage_id} finished with {result.status}")),
            parameters={"execution_engine": "Nextflow", "workflow_id": request.workflow_id, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "task_work_dir": str(Path.cwd()), "upstream_stages": list(upstream), "receipt": str(receipts / f"{stage_id}.json")}, artifacts=result.artifacts,
        )
        emit(completion)
        result.stages.append(completion)
    except Exception as exc:
        message = f"Stage {stage_id} failed: {exc}"
        event = StageEvent(stage_id=stage_id, status="FAILED", message=message)
        emit(event)
        (receipts / f"{stage_id}.error.log").write_text(traceback.format_exc(), encoding="utf-8")
        result = WorkflowResult(status="FAILED", stages=[event], warnings=[message])
    payload = result.model_dump(mode="json")
    atomic_json(receipts / f"{stage_id}.json", payload)
    atomic_json(output, payload)
    return 1 if result.status in {"FAILED", "BLOCKED"} else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stage"])
    parser.add_argument("--request-b64", required=True)
    parser.add_argument("--receipts-b64", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", action="append", nargs=2, default=[])
    arguments = parser.parse_args()
    pairs = arguments.upstream
    upstream = {key: Path(value) for key, value in pairs}
    if len(upstream) != len(pairs):
        parser.error("Duplicate upstream stage")
    raise SystemExit(run_stage(_decoded(arguments.request_b64), _decoded(arguments.receipts_b64), arguments.stage, arguments.output, upstream))


if __name__ == "__main__":
    main()
