from __future__ import annotations

import getpass
import json
import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from spatial_omics_toolkit.application.contracts import PreflightResult, RunRequest
from spatial_omics_toolkit.application.intake import ALLOWED_ROOT, lexical_path_is_under, path_is_under, read_project_samples_bytes, sha256_file
from spatial_omics_toolkit.application.runs import append_event, atomic_json, create_run_bundle, update_status, utc_now
from spatial_omics_toolkit.application.workflows.registry import get_workflow
from spatial_omics_toolkit.application.scheduler import submit_controller
from spatial_omics_toolkit.application.settings import PROFILE_KEY, config_file_hashes, default_settings, freeze_profile, profile_metadata


def prepare_request(workflow_id: str, project_root: Path, annotation_csv: Path, modality_dir: Path,
                    output_root: Path, parameters: dict[str, Any] | None = None) -> tuple[RunRequest, PreflightResult]:
    workflow, descriptor = get_workflow(workflow_id)
    run_id = uuid4()
    normalized_modality = descriptor.modality
    run_dir = output_root / normalized_modality / "runs" / str(run_id)
    request = RunRequest(workflow_id=workflow_id, modality=normalized_modality,
                         project_root=project_root, modality_dir=modality_dir,
                         annotation_csv=annotation_csv, output_root=output_root,
                         run_id=run_id, run_dir=run_dir, parameters=parameters or {})
    blockers: list[str] = []
    warnings: list[str] = []
    for label, path in (("Project root", project_root), ("Modality folder", modality_dir), ("project_samples.csv", annotation_csv)):
        if not path.exists():
            blockers.append(f"{label} does not exist: {path}")
        elif not lexical_path_is_under(path, ALLOWED_ROOT):
            blockers.append(f"{label} must be selected from under {ALLOWED_ROOT}")
    if project_root.exists() and not project_root.is_dir():
        blockers.append("Project root must be a directory")
    if modality_dir.exists() and not modality_dir.is_dir():
        blockers.append("Modality folder must be a directory")
    if annotation_csv.exists() and not annotation_csv.is_file():
        blockers.append("project_samples.csv must be a file")
    if project_root.exists() and annotation_csv.exists() and not lexical_path_is_under(annotation_csv, project_root):
        blockers.append("project_samples.csv must be located under the selected project root")
    if project_root.exists() and modality_dir.exists() and not lexical_path_is_under(modality_dir, project_root):
        warnings.append("Selected modality folder is outside the project root; paths are recorded as supplied.")
    if output_root.exists():
        if not output_root.is_dir():
            blockers.append("Output root must be a directory")
        elif not path_is_under(output_root, ALLOWED_ROOT):
            blockers.append(f"Output root must resolve under {ALLOWED_ROOT}")
    else:
        if not lexical_path_is_under(output_root, ALLOWED_ROOT) or not path_is_under(output_root, ALLOWED_ROOT):
            blockers.append(f"Output root must resolve under {ALLOWED_ROOT}")
    if modality_dir.exists() and path_is_under(output_root, modality_dir):
        blockers.append("Output root cannot be inside the raw modality folder")
    if run_dir.exists():
        blockers.append(f"Run output already exists; refusing overwrite: {run_dir}")
    try:
        annotation_bytes = annotation_csv.read_bytes()
        annotation_csv_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
        _, sample_blockers = read_project_samples_bytes(annotation_bytes, str(annotation_csv))
        blockers.extend(sample_blockers)
    except OSError as exc:
        annotation_csv_sha256 = None
        blockers.append(f"Unable to read project_samples.csv for this run: {exc}")
    try:
        initial_config_hashes = config_file_hashes(descriptor, request.parameters, project_root)
    except (ValueError, OSError) as exc:
        initial_config_hashes = None
        blockers.append(f"Settings configuration: {exc}")
    quick_request = request.model_copy(deep=True)
    quick_request.parameters = {**request.parameters, "_preflight_mode": "quick"}
    try:
        modality_preflight = workflow.preflight(quick_request)
        blockers.extend(modality_preflight.blockers)
        warnings.extend(modality_preflight.warnings)
    except Exception as exc:
        blockers.append(f"Workflow preflight failed: {exc}")
        modality_preflight = None
    if annotation_csv_sha256 is not None:
        try:
            if sha256_file(annotation_csv) != annotation_csv_sha256:
                blockers.append("project_samples.csv changed during preflight; rerun preflight before submission.")
        except OSError as exc:
            blockers.append(f"Unable to recheck project_samples.csv after preflight: {exc}")
    resolved: dict[str, Any] = dict(parameters or {})
    if modality_preflight is not None:
        resolved = {**modality_preflight.resolved_parameters, **resolved}
    if annotation_csv_sha256 is not None:
        resolved["annotation_csv_sha256"] = annotation_csv_sha256
    resources = resolved.get("resources", {})
    if not isinstance(resources, dict):
        blockers.append("SLURM task resource settings must be an object")
    else:
        cpus = resources.get("cpus")
        memory_gb = resources.get("memory_gb")
        time_limit = resources.get("time_limit")
        if not isinstance(cpus, int) or isinstance(cpus, bool) or not 1 <= cpus <= 32:
            blockers.append("SLURM task CPUs must be an integer from 1 through 32")
        if not isinstance(memory_gb, int) or isinstance(memory_gb, bool) or not 1 <= memory_gb <= 512:
            blockers.append("SLURM task memory_gb must be an integer from 1 through 512")
        if time_limit not in {"8h", "24h", "48h", "72h", "168h"}:
            blockers.append("SLURM task time_limit must be one of 8h, 24h, 48h, 72h, or 168h")
    try:
        profile = resolved.get(PROFILE_KEY)
        if profile is None:
            resolved[PROFILE_KEY] = profile_metadata(default_settings(descriptor))
        elif isinstance(profile, dict) and profile.get("project_root") is not None and Path(str(profile["project_root"])).resolve() != project_root.resolve():
            raise ValueError("The selected settings file belongs to a different project")
        frozen = freeze_profile(descriptor, resolved, project_root)
        if frozen is not None and initial_config_hashes is not None and frozen["config_files"] != initial_config_hashes:
            raise ValueError("A settings configuration file changed during preflight; run checks again")
        if frozen is not None:
            resolved[PROFILE_KEY] = frozen
    except (ValueError, OSError, UnicodeError) as exc:
        blockers.append(f"Settings profile: {exc}")
    request.parameters = resolved
    return request, PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=resolved)


def submit_request(request: RunRequest, preflight: PreflightResult) -> str:
    if not preflight.ready:
        raise ValueError("Cannot submit a blocked workflow request")
    _, descriptor = get_workflow(request.workflow_id)
    frozen = freeze_profile(descriptor, request.parameters, request.project_root)
    if frozen is None or frozen != preflight.resolved_parameters.get(PROFILE_KEY) or frozen != request.parameters.get(PROFILE_KEY):
        raise ValueError("Settings changed after preflight; run checks again before submission")
    create_run_bundle(request, descriptor.model_dump(mode="json"), {**preflight.resolved_parameters, **request.parameters})
    try:
        job_id = submit_controller(request)
    except Exception as exc:
        from spatial_omics_toolkit.application.runs import append_event, update_status
        append_event(request.run_dir, {"stage_id": "submission", "status": "FAILED", "message": f"SLURM submission failed: {exc}", "tool_versions": {}, "parameters": {}, "artifacts": []})
        update_status(request.run_dir, "FAILED", f"SLURM submission failed: {exc}")
        raise
    return job_id


def submit_review_decision(run_dir: Path, block_indices: list[int], blocks_review_confirmed: bool) -> str:
    run_dir = run_dir.resolve()
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "QC_PENDING":
        raise ValueError("This run is not waiting for QC review")
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    review_request_path = Path(status.get("review_request_path", "review_request.json"))
    if not review_request_path.is_absolute():
        review_request_path = run_dir / review_request_path
    review_request = json.loads(review_request_path.read_text(encoding="utf-8"))
    run_id = str(metadata["run_id"])
    if str(review_request.get("run_id", run_id)) != run_id:
        raise ValueError("Review request belongs to a different run")
    manifest_path = run_dir / "inputs" / "manifest.json"
    inventory_receipt_sha256 = sha256_file(manifest_path)
    if review_request.get("inventory_receipt_sha256") != inventory_receipt_sha256:
        raise ValueError("Input inventory receipt changed after QC review; review must be repeated")
    flim_receipt_sha256 = review_request.get("flim_inventory_receipt_sha256")
    if flim_receipt_sha256 is not None:
        flim_receipt_path = Path(str(review_request.get("inventory_receipt_path", ""))).expanduser()
        if not flim_receipt_path.is_absolute():
            flim_receipt_path = run_dir / flim_receipt_path
        flim_receipt_path = flim_receipt_path.resolve()
        if not path_is_under(flim_receipt_path, run_dir) or not flim_receipt_path.is_file():
            raise ValueError("Canonical modality inventory receipt is missing or outside this run.")
        if sha256_file(flim_receipt_path) != flim_receipt_sha256:
            raise ValueError("Canonical modality inventory receipt changed after review; review must be repeated.")
    choices = review_request.get("block_indices")
    if not isinstance(choices, list):
        choices = review_request.get("choices", {}).get("block_indices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("QC review request has no allowed block choices")
    if blocks_review_confirmed is not True or not block_indices:
        raise ValueError("Select and explicitly confirm one or more SDT blocks")
    if len(block_indices) != len(set(block_indices)) or any(index not in choices for index in block_indices):
        raise ValueError("Selected block indices must be unique members of the inspected choices")
    resolved_path = run_dir / "resolved_parameters.json"
    resolved_parameters = json.loads(resolved_path.read_text(encoding="utf-8"))
    attempt = int(resolved_parameters.get("review_attempt", 0)) + 1
    decision_parameters = {
        "block_indices": block_indices,
        "blocks_review_confirmed": True,
        "inventory_receipt_sha256": inventory_receipt_sha256,
        "review_attempt": attempt,
    }
    if flim_receipt_sha256 is not None:
        decision_parameters["flim_inventory_receipt_sha256"] = flim_receipt_sha256
    receipt = {
        "run_id": run_id,
        **decision_parameters,
        "reviewer": getpass.getuser(),
        "reviewed_at": utc_now(),
        "parameters": decision_parameters,
    }
    atomic_json(run_dir / "review_decision.json", receipt)
    with (run_dir / "review_decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
        stream.flush()
    resolved_parameters.update(decision_parameters)
    atomic_json(resolved_path, resolved_parameters)
    append_event(run_dir, {"stage_id": "qc_review", "status": "CREATED", "message": "QC review decision recorded; resume submitted", "tool_versions": {}, "parameters": receipt, "artifacts": [str(run_dir / "review_decision.json")]})
    request = RunRequest(
        workflow_id=metadata["workflow_id"],
        modality=metadata["modality"],
        project_root=Path(metadata["project"]["root"]),
        modality_dir=Path(metadata["input"]["modality_dir"]),
        annotation_csv=Path(metadata["project"].get("annotation_csv_snapshot", metadata["project"]["annotation_csv"])),
        output_root=Path(metadata["output_root"]),
        run_id=metadata["run_id"],
        run_dir=run_dir,
        parameters=resolved_parameters,
    )
    atomic_json(run_dir / "job.json", {})
    update_status(run_dir, "QUEUED", "QC review confirmed; submitting resume worker", review_attempt=attempt, slurm_job_id=None)
    try:
        return submit_controller(request)
    except Exception as exc:
        update_status(run_dir, "FAILED", f"Resume submission failed: {exc}", review_attempt=attempt)
        raise
