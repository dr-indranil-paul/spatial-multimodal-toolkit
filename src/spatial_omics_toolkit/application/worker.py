from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent
from spatial_omics_toolkit.application.intake import build_input_inventory, read_project_samples, sha256_file
from spatial_omics_toolkit.application.runs import append_event, atomic_json, update_status
from spatial_omics_toolkit.application.workflows.registry import get_workflow



def _mapping_blockers(request: RunRequest, groups: Any) -> list[str]:
    if not isinstance(groups, list):
        return ["Workflow acquisition_groups must be a list."]
    if not groups:
        return []
    keys: list[str] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_key"), str) or not group["group_key"].strip():
            return ["Workflow returned an acquisition group without a nonempty group_key."]
        keys.append(group["group_key"])
    if len(keys) != len(set(keys)):
        return ["Workflow returned duplicate acquisition group keys; mapping is ambiguous."]
    parameters = request.parameters
    if parameters.get("specimen_mapping_confirmed") is not True:
        return ["Explicit specimen mapping confirmation is required before workflow execution."]
    if request.workflow_id == "cycif.tabular_analysis" and parameters.get("use_specimen_id_column") is True:
        if parameters.get("has_specimen_id") is not True and parameters.get("existing_specimen_id_column") is not True:
            return ["The selected specimen_id column was not confirmed by tabular preflight."]
        return []
    field = "core_to_specimen" if request.workflow_id == "cycif.tabular_analysis" else "specimen_mapping"
    mapping = parameters.get(field)
    if not isinstance(mapping, dict):
        return [f"Explicit `{field}` mapping is missing."]
    if set(mapping) != set(keys):
        missing = sorted(set(keys) - set(mapping))
        extra = sorted(set(mapping) - set(keys))
        return [f"Mapping keys must exactly match acquisition groups; missing={missing}, extra={extra}."]
    try:
        samples, sample_errors = read_project_samples(request.annotation_csv)
    except Exception as exc:
        return [f"Unable to validate specimen mapping against project_samples.csv: {exc}"]
    if sample_errors:
        return sample_errors
    specimen_metadata: dict[str, tuple[str, str, str]] = {}
    ambiguous_specimens: set[str] = set()
    for row in samples:
        specimen_id = row["specimen_id"]
        identity = (row["biological_unit_id"], row["condition"], row["tissue_source"])
        if specimen_id in specimen_metadata and specimen_metadata[specimen_id] != identity:
            ambiguous_specimens.add(specimen_id)
        specimen_metadata[specimen_id] = identity
    specimen_ids = set(specimen_metadata)
    invalid = [key for key, value in mapping.items() if not isinstance(value, str) or (value != "UNMAPPED" and value not in specimen_ids)]
    if invalid:
        return [f"Mapping values for {invalid} must be exact project specimen_id values or UNMAPPED."]
    ambiguous = sorted({value for value in mapping.values() if value in ambiguous_specimens})
    if ambiguous:
        return [f"Mapped specimen_id values have conflicting biological_unit_id/condition/tissue_source rows in project_samples.csv: {ambiguous}"]
    return []


def _example_input_blockers(request: RunRequest, inventory: dict[str, Any]) -> list[str]:
    """Bind a fixed example recipe to the raw assets prepared for that recipe."""
    authorization = request.parameters.get("example_authorization")
    if authorization is None:
        return []
    if not isinstance(authorization, dict) or authorization.get("scope") != "technical_qc_only":
        return ["The example authorization has no valid technical-QC scope."]
    expected_files = authorization.get("source_files")
    if not isinstance(expected_files, list) or not expected_files:
        return ["The example recipe is missing its prepared raw-source checksums."]
    expected: dict[str, str] = {}
    for item in expected_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            return ["The example raw-source checksum record is malformed."]
        expected[str(Path(item["path"]).resolve())] = item["sha256"]
    observed: dict[str, str] = {}
    for item in inventory.get("files", []):
        if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
            observed[str(Path(item["path"]).resolve())] = item["sha256"]
    if observed != expected:
        return ["Example inputs differ from the prepared source receipt. Restore the original example files or prepare and review a new example; no analysis ran."]
    return []


def _render_pdf_previews(pdf_path: Path, preview_dir: Path) -> tuple[list[Path], list[str], str | None]:
    warnings: list[str] = []
    try:
        if pdf_path.stat().st_size > 128 * 1024 * 1024:
            return [], [f"PDF preview unavailable for {pdf_path}: source exceeds 128 MiB safety limit"], None
    except OSError as exc:
        return [], [f"PDF preview unavailable for {pdf_path}: {exc}"], None
    try:
        import fitz
    except ImportError:
        return [], ["PDF previews unavailable: install the optional PyMuPDF preview dependency on worker nodes"], None
    rendered: list[Path] = []
    try:
        document = fitz.open(str(pdf_path))
        try:
            if document.page_count == 0:
                return [], [f"PDF preview unavailable for {pdf_path}: document has no pages"], str(getattr(fitz, "VersionBind", "unknown"))
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                rect = page.rect
                if rect.width <= 0 or rect.height <= 0:
                    warnings.append(f"Skipped invalid PDF page {page_index + 1} in {pdf_path}")
                    continue
                scale = min(1200 / rect.width, 1200 / rect.height, 1.0)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                if pixmap.width > 1200 or pixmap.height > 1200 or pixmap.width * pixmap.height > 1_440_000:
                    warnings.append(f"Skipped PDF page {page_index + 1} in {pdf_path}: preview exceeded pixel limits")
                    continue
                data = pixmap.tobytes("png")
                if len(data) > 5 * 1024 * 1024:
                    warnings.append(f"Skipped PDF page {page_index + 1} in {pdf_path}: preview exceeded 5 MiB")
                    continue
                page_path = preview_dir / f"page-{page_index + 1:04d}.png"
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_bytes(data)
                rendered.append(page_path)
        finally:
            document.close()
    except Exception as exc:
        warnings.append(f"PDF preview failed for {pdf_path}: {exc}")
    version = str(getattr(fitz, "VersionBind", "unknown"))
    return rendered, warnings, version


def _safe_stage_name(stage_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stage_id) or "workflow"


def run_worker(run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    run_file = run_dir / "run.json"
    try:
        metadata: dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))
        parameters: dict[str, Any] = dict(metadata["parameters"])
        resolved_path = run_dir / "resolved_parameters.json"
        if resolved_path.is_file():
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            if isinstance(resolved, dict):
                parameters.update(resolved)
        manifest_path = run_dir / "inputs" / "manifest.json"
        decision_path = run_dir / "review_decision.json"
        decision: dict[str, Any] | None = None
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if str(decision.get("run_id")) != str(metadata["run_id"]):
                raise ValueError("Review decision run_id does not match this run")
            if decision.get("inventory_receipt_sha256") != sha256_file(manifest_path):
                raise ValueError("Review decision inventory receipt does not match this run")
            if decision.get("blocks_review_confirmed") is not True:
                raise ValueError("Review decision lacks explicit block confirmation")
            parameters.update(decision.get("parameters", {}))
        annotation_snapshot = Path(
            metadata["project"].get(
                "annotation_csv_snapshot",
                metadata["project"]["annotation_csv"],
            )
        ).resolve()
        expected_annotation_sha256 = metadata["project"].get("annotation_csv_sha256")
        if not isinstance(expected_annotation_sha256, str) or sha256_file(annotation_snapshot) != expected_annotation_sha256:
            raise ValueError("Frozen project_samples.csv snapshot is missing or changed after run submission.")
        _, annotation_blockers = read_project_samples(annotation_snapshot)
        if annotation_blockers:
            raise ValueError("Frozen project_samples.csv snapshot is invalid: " + "; ".join(annotation_blockers))
        request = RunRequest(
            workflow_id=metadata["workflow_id"], modality=metadata["modality"],
            project_root=Path(metadata["project"]["root"]), modality_dir=Path(metadata["input"]["modality_dir"]),
            annotation_csv=annotation_snapshot, output_root=Path(metadata["output_root"]),
            run_id=metadata["run_id"], run_dir=run_dir, parameters=parameters,
        )
        module, descriptor = get_workflow(request.workflow_id)
        update_status(run_dir, "RUNNING", "Python workflow worker started")

        stage_for_artifact: dict[Path, str] = {}

        def emit(event: StageEvent) -> None:
            append_event(run_dir, event)
            for artifact in event.artifacts:
                path = Path(artifact)
                resolved = path if path.is_absolute() else run_dir / path
                stage_for_artifact[resolved.resolve()] = event.stage_id
            if event.status in {"FAILED", "BLOCKED", "QC_PENDING"}:
                update_status(run_dir, event.status, event.message, current_stage=event.stage_id)
            else:
                update_status(run_dir, "RUNNING", event.message, current_stage=event.stage_id)

        preflight = module.preflight(request)
        manifest_path = run_dir / "inputs" / "manifest.json"
        input_inventory: dict[str, Any] = {}
        if manifest_path.is_file():
            input_inventory = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory_input_paths = preflight.resolved_parameters.get("inventory_input_paths")
        selected_input_paths = (
            inventory_input_paths
            if isinstance(inventory_input_paths, list)
            and all(isinstance(path, str) for path in inventory_input_paths)
            else None
        )
        if not input_inventory.get("files"):
            input_inventory = build_input_inventory(
                request.modality_dir,
                request.project_root,
                request.annotation_csv,
                selected_input_paths=selected_input_paths,
            )
            atomic_json(manifest_path, input_inventory)
        if selected_input_paths is not None:
            expected_selected_paths = sorted(set(selected_input_paths))
            if input_inventory.get("selected_input_paths") != expected_selected_paths:
                message = "Workflow input selection changed after final preflight; submit a fresh run."
                emit(StageEvent(stage_id="input_integrity", status="BLOCKED", message=message))
                update_status(run_dir, "BLOCKED", message, current_stage="input_integrity")
                return 2
        if decision is not None:
            recorded_selected_paths = input_inventory.get("selected_input_paths")
            selected_for_revalidation = (
                recorded_selected_paths
                if isinstance(recorded_selected_paths, list)
                and all(isinstance(path, str) for path in recorded_selected_paths)
                else None
            )
            current_inventory = build_input_inventory(
                request.modality_dir,
                request.project_root,
                request.annotation_csv,
                selected_input_paths=selected_for_revalidation,
            )
            old_hashes = {item.get("path"): item.get("sha256") for item in input_inventory.get("files", []) if item.get("sha256")}
            new_hashes = {item.get("path"): item.get("sha256") for item in current_inventory.get("files", []) if item.get("sha256")}
            if old_hashes != new_hashes or input_inventory.get("annotation_csv_sha256") != current_inventory.get("annotation_csv_sha256"):
                raise ValueError("Input files changed after QC review; source hashes no longer match the inventory receipt")
            atomic_json(run_dir / "inputs" / f"revalidation-attempt-{request.parameters.get('review_attempt', 1)}.json", current_inventory)
        example_blockers = _example_input_blockers(request, input_inventory)
        if example_blockers:
            message = "; ".join(example_blockers)
            emit(StageEvent(stage_id="input_integrity", status="BLOCKED", message=message))
            update_status(run_dir, "BLOCKED", message, current_stage="input_integrity")
            return 2
        if not preflight.ready:
            for blocker in preflight.blockers:
                emit(StageEvent(stage_id="preflight", status="BLOCKED", message=blocker))
            update_status(run_dir, "BLOCKED", "; ".join(preflight.blockers))
            return 2
        groups = preflight.resolved_parameters.get("acquisition_groups", request.parameters.get("acquisition_groups", []))
        request.parameters = {**request.parameters, **preflight.resolved_parameters}
        mapping_blockers = _mapping_blockers(request, groups)
        if mapping_blockers:
            for blocker in mapping_blockers:
                emit(StageEvent(stage_id="mapping", status="BLOCKED", message=blocker, parameters=request.parameters))
            update_status(run_dir, "BLOCKED", "; ".join(mapping_blockers), current_stage="mapping")
            return 2
        atomic_json(run_dir / "resolved_parameters.json", request.parameters)
        atomic_json(run_dir / "mapping_receipt.json", {
            "annotation_csv": str(Path(metadata["project"]["annotation_csv"]).absolute()),
            "annotation_csv_resolved": str(Path(metadata["project"]["annotation_csv"]).resolve()),
            "annotation_csv_snapshot": str(request.annotation_csv.resolve()),
            "annotation_csv_sha256": expected_annotation_sha256,
            "specimen_mapping": request.parameters.get("specimen_mapping", {}),
            "core_to_specimen": request.parameters.get("core_to_specimen", {}),
            "use_specimen_id_column": request.parameters.get("use_specimen_id_column", False),
            "feature_table_path": request.parameters.get("feature_table_path"),
            "has_specimen_id": request.parameters.get("has_specimen_id", False),
            "specimen_mapping_confirmed": request.parameters.get("specimen_mapping_confirmed", False),
            "acquisition_groups": groups,
        })
        for warning in preflight.warnings:
            emit(StageEvent(stage_id="preflight", status="WARNING", message=warning))
        result = module.execute(request, emit)
        artifact_entries: list[dict[str, Any]] = []
        warnings = list(result.warnings)
        sources: set[Path] = set()
        for artifact in result.artifacts:
            path = Path(artifact)
            resolved = path if path.is_absolute() else run_dir / path
            sources.add(resolved.resolve())
        for stage in result.stages:
            for artifact in stage.artifacts:
                path = Path(artifact)
                sources.add((path if path.is_absolute() else run_dir / path).resolve())
        for path in run_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf" and "previews" not in path.parts:
                sources.add(path.resolve())
        for source in sorted(sources):
            if not source.is_file():
                artifact_entries.append({"path": str(source), "exists": False, "role": "original", "media_type": "application/octet-stream"})
                continue
            stage_id = stage_for_artifact.get(source, "workflow_output")
            media_type = "application/pdf" if source.suffix.lower() == ".pdf" else "image/" + source.suffix.lower().lstrip(".") if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else "application/octet-stream"
            artifact_entries.append({"path": str(source), "exists": True, "role": "original", "media_type": media_type, "stage_id": stage_id, "size_bytes": source.stat().st_size})
            if source.suffix.lower() != ".pdf":
                continue
            relative = source.relative_to(run_dir) if source.is_relative_to(run_dir) else Path(source.name)
            preview_dir = run_dir / "previews" / _safe_stage_name(stage_id) / relative.parent / relative.stem
            preview_paths, preview_messages, renderer_version = _render_pdf_previews(source, preview_dir)
            warnings.extend(preview_messages)
            for preview_path in preview_paths:
                artifact_entries.append({"path": str(preview_path), "exists": True, "role": "preview", "media_type": "image/png", "stage_id": stage_id, "source_pdf": str(source), "page": int(preview_path.stem.split("-")[-1]), "size_bytes": preview_path.stat().st_size, "renderer": "PyMuPDF", "renderer_version": renderer_version})
            if preview_paths:
                emit(StageEvent(stage_id=f"preview_{_safe_stage_name(stage_id)}", status="SUCCEEDED", message=f"Bounded PDF page previews from {source}", tool_versions={"PyMuPDF": renderer_version or "unknown"}, parameters={"source_pdf": str(source), "max_dimension_px": 1200, "max_pixels": 1_440_000}, artifacts=[str(path) for path in preview_paths]))
            for preview_message in preview_messages:
                emit(StageEvent(stage_id=f"preview_{_safe_stage_name(stage_id)}", status="WARNING", message=preview_message, parameters={"source_pdf": str(source)}))
        artifact_manifest_path = run_dir / "artifacts.json"
        previous_artifacts: list[dict[str, Any]] = []
        if artifact_manifest_path.is_file():
            previous_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(previous_manifest, dict) or not isinstance(previous_manifest.get("artifacts"), list):
                raise ValueError("Existing artifacts.json must contain an artifacts list.")
            previous_artifacts = previous_manifest["artifacts"]
        seen_artifacts = {(item.get("path"), item.get("role"), item.get("page")) for item in previous_artifacts if isinstance(item, dict)}
        artifact_entries = [*previous_artifacts, *[item for item in artifact_entries if (item.get("path"), item.get("role"), item.get("page")) not in seen_artifacts]]
        atomic_json(run_dir / "artifacts.json", {"artifacts": artifact_entries, "warnings": warnings})
        final_status = result.status.upper()
        if final_status not in {"SUCCEEDED", "WARNING", "FAILED", "BLOCKED", "QC_PENDING"}:
            final_status = "SUCCEEDED" if not warnings else "WARNING"
        if final_status == "QC_PENDING":
            if result.review_request is None:
                raise ValueError("Workflow returned QC_PENDING without a review_request")
            review_attempt = int(request.parameters.get("review_attempt", 0))
            review_path = run_dir / "review_request.json" if not (run_dir / "review_request.json").exists() else run_dir / f"review_request-attempt-{review_attempt}.json"
            review_payload = {
                **result.review_request,
                "run_id": str(request.run_id),
                "inventory_receipt_sha256": sha256_file(manifest_path),
                "review_attempt": review_attempt,
            }
            atomic_json(review_path, review_payload)
            review_path_resolved = review_path.resolve()
            artifact_entries = [
                item for item in artifact_entries
                if item.get("path") != str(review_path_resolved)
            ]
            artifact_entries.append({
                "path": str(review_path_resolved),
                "exists": True,
                "role": "review_request",
                "media_type": "application/json",
                "stage_id": "qc_review",
                "size_bytes": review_path.stat().st_size,
            })
            atomic_json(run_dir / "artifacts.json", {"artifacts": artifact_entries, "warnings": warnings})
            emit(StageEvent(stage_id="qc_review", status="QC_PENDING", message="QC review is required before analysis can resume", parameters={"review_request_path": str(review_path)}, artifacts=[str(review_path)]))
            update_status(run_dir, "QC_PENDING", "Review required before analysis can resume", review_request_path=str(review_path), review_attempt=review_attempt, warnings=warnings)
            return 0
        update_status(run_dir, final_status, "Workflow execution finished", warnings=warnings)
        return 0 if final_status in {"SUCCEEDED", "WARNING"} else 1
    except BaseException as exc:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        with (run_dir / "logs" / "worker-error.log").open("a", encoding="utf-8") as stream:
            stream.write(detail)
        if run_dir.exists():
            append_event(run_dir, {"stage_id": "worker", "status": "FAILED", "message": str(exc), "tool_versions": {}, "parameters": {}, "artifacts": []})
            update_status(run_dir, "FAILED", str(exc))
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run_worker(arguments.run_dir))


if __name__ == "__main__":
    main()
