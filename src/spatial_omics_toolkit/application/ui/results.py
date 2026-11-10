from __future__ import annotations

import json
import re

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import streamlit as st

from spatial_omics_toolkit.application.config import containing_workspace_root
from spatial_omics_toolkit.application.intake import path_is_under, path_is_under_workspace, sha256_file
from spatial_omics_toolkit.application.scheduler import read_run_status
from spatial_omics_toolkit.application.service import submit_review_decision
from spatial_omics_toolkit.application.workflows.registry import list_workflows


RunSection = Literal["run", "qc", "results", "files"]
_DOWNLOAD_LIMIT = 8 * 1024 * 1024
_IMAGE_PIXEL_LIMIT = 2048
_TEXT_PREVIEW_LIMIT = 50_000
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_WORKFLOW_SOURCE_RECEIPTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "flim.raw_sdt": ("adapter_receipt.json", "source_sha256", (
        "clinical_roi_flim.py", "clinical_roi_flim_plots.py",
    )),
    "hs.native_spectral_qc": ("native_source_metadata.json", "canonical_source_sha256", (
        "clinical_roi_hs_flim.py", "clinical_roi_hyperspectral.py",
    )),
}


def _receipt_source_versions(run_dir: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Collect only the known workflow-level script hashes from a saved receipt."""
    workflow_id = metadata.get("workflow_id")
    schema = _WORKFLOW_SOURCE_RECEIPTS.get(workflow_id) if isinstance(workflow_id, str) else None
    if schema is None:
        return {}
    try:
        resolved_run_dir = run_dir.resolve(strict=True)
        if not resolved_run_dir.is_dir() or not path_is_under_workspace(resolved_run_dir):
            return {}
        receipt_name, hash_field, scripts = schema
        receipt_path = (resolved_run_dir / receipt_name).resolve(strict=True)
        if not receipt_path.is_file() or not path_is_under(receipt_path, resolved_run_dir):
            return {}
        if receipt_path.stat().st_size > _DOWNLOAD_LIMIT:
            return {}
        with receipt_path.open("rb") as stream:
            content = stream.read(_DOWNLOAD_LIMIT + 1)
        if len(content) > _DOWNLOAD_LIMIT:
            return {}
        receipt = json.loads(content.decode("utf-8"))
    except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(receipt, dict) or receipt.get("workflow_id") != workflow_id:
        return {}
    if "run_id" in receipt and receipt.get("run_id") != metadata.get("run_id"):
        return {}
    hashes = receipt.get(hash_field)
    if not isinstance(hashes, dict):
        return {}
    return {
        f"Workflow source · {script} · {receipt_name} ({hash_field})": hashes[script]
        for script in scripts
        if isinstance(hashes.get(script), str) and _SHA256_PATTERN.fullmatch(hashes[script])
    }




def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with (run_dir / "events.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    except (OSError, UnicodeError):
        pass
    return events


def _safe_run_dir(run_dir: Path) -> bool:
    try:
        resolved = run_dir.resolve(strict=True)
        return resolved.is_dir() and path_is_under_workspace(resolved)
    except (OSError, RuntimeError):
        return False


def _bundle_metadata(
    run_dir: Path, modality: str, workflow_ids: set[str], output_root: Path,
    *, state_scope: Literal["own", "example"] = "own",
) -> dict[str, Any] | None:
    try:
        resolved = run_dir.resolve(strict=True)
        output_resolved = output_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_dir() or not path_is_under(resolved, output_resolved) or not path_is_under_workspace(resolved):
        return None
    metadata = _read_json(resolved / "run.json")
    workflow_id = metadata.get("workflow_id")
    if metadata.get("modality") != modality or not isinstance(workflow_id, str) or workflow_id not in workflow_ids:
        return None
    parameters = metadata.get("parameters", {})
    is_example = isinstance(parameters, dict) and isinstance(parameters.get("example_authorization"), dict)
    if is_example != (state_scope == "example"):
        return None
    return metadata


def _workflow_title(workflow_id: str) -> str:
    for workflow in list_workflows():
        if workflow.workflow_id == workflow_id:
            return workflow.title
    return "Workflow"


def _friendly_run_label(run_dir: Path, metadata: dict[str, Any]) -> str:
    created = str(metadata.get("created_at") or metadata.get("created") or "")
    if created:
        try:
            stamp = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().strftime("%b %d, %Y · %H:%M")
        except ValueError:
            stamp = created[:16]
    else:
        stamp = "Date unavailable"
    status = str(_read_json(run_dir / "status.json").get("status", "Status unavailable")).replace("_", " ").title()
    workflow = _workflow_title(str(metadata.get("workflow_id", "")))
    return f"{stamp}  ·  {workflow}  ·  {status}"

def _run_sort_timestamp(run_dir: Path, metadata: dict[str, Any]) -> float:
    created = metadata.get("created_at") or metadata.get("created")
    if isinstance(created, str):
        try:
            return datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0



def select_run(
    modality: str, workflow_ids: set[str], output_root: Path,
    *, state_scope: Literal["own", "example"] = "own",
) -> Path | None:
    if state_scope not in {"own", "example"}:
        raise ValueError(f"Unsupported run-selection scope: {state_scope}")
    selected_key = "selected_example_run_dirs" if state_scope == "example" else "selected_run_dirs"
    active_key = "active_example_run_dirs" if state_scope == "example" else "active_run_dirs"
    widget_scope = f"example_{modality}" if state_scope == "example" else modality
    pending_key = f"pending_example_run_selection_{modality}" if state_scope == "example" else f"pending_run_selection_{modality}"
    root = output_root / modality / "runs"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    try:
        directories = [item for item in root.iterdir() if item.is_dir()]
    except OSError:
        directories = []
    for directory in directories:
        metadata = _bundle_metadata(directory, modality, workflow_ids, output_root, state_scope=state_scope)
        if metadata is not None:
            candidates.append((directory.resolve(), metadata))

    selected_dirs = st.session_state.setdefault(selected_key, {})
    active_dirs = st.session_state.setdefault(active_key, {})
    if not isinstance(selected_dirs, dict):
        selected_dirs = {}
        st.session_state[selected_key] = selected_dirs
    if not isinstance(active_dirs, dict):
        active_dirs = {}
        st.session_state[active_key] = active_dirs

    pending_path = st.session_state.pop(pending_key, None)
    for saved_text in (pending_path, selected_dirs.get(modality), active_dirs.get(modality)):
        if not isinstance(saved_text, str):
            continue
        saved_candidate = Path(saved_text).expanduser()
        try:
            resolved_candidate = saved_candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if any(path == resolved_candidate for path, _ in candidates):
            continue
        metadata = _bundle_metadata(resolved_candidate, modality, workflow_ids, output_root, state_scope=state_scope)
        if metadata is not None:
            candidates.append((resolved_candidate, metadata))

    candidates.sort(key=lambda item: _run_sort_timestamp(item[0], item[1]), reverse=True)

    paths = [path for path, _ in candidates]
    option_labels = {
        str(path): f"{index + 1} · {_friendly_run_label(path, metadata)}"
        for index, (path, metadata) in enumerate(candidates)
    }
    if isinstance(pending_path, str) and pending_path in option_labels:
        st.session_state[f"run_history_{widget_scope}"] = pending_path
        selected_dirs[modality] = pending_path
    saved_path: str | None = None
    for saved_text in (selected_dirs.get(modality), active_dirs.get(modality)):
        if not isinstance(saved_text, str):
            continue
        try:
            candidate_path = str(Path(saved_text).expanduser().resolve(strict=True))
        except (OSError, RuntimeError):
            continue
        if candidate_path in option_labels:
            saved_path = candidate_path
            break
    default_path = saved_path if saved_path in option_labels else (str(paths[0]) if paths else None)
    if option_labels:
        options = list(option_labels)
        chosen_text = st.selectbox(
            "Run history",
            options,
            index=options.index(default_path) if default_path is not None else 0,
            format_func=lambda value: option_labels[value],
            key=f"run_history_{widget_scope}",
            help="Only runs for this modality and input mode are shown. Included examples and your own analyses have separate histories.",
        )
        chosen_path = Path(chosen_text)
        selected_dirs[modality] = chosen_text
    else:
        chosen_path = None
        st.info("No runs for this workflow yet. Submit a run from the Run section to see its progress here.")

    with st.expander("Advanced · open a run bundle", expanded=False):
        manual_text = st.text_input("Run bundle folder", key=f"manual_run_path_{widget_scope}", placeholder="Paste a run folder path under the workspace")
        if st.button("Open this run", key=f"manual_run_open_{widget_scope}"):
            manual = Path(manual_text).expanduser()
            if not manual.is_absolute():
                manual = output_root / manual
            manual_metadata = _bundle_metadata(manual, modality, workflow_ids, output_root, state_scope=state_scope)
            if manual_metadata is None:
                st.error("That folder is not a valid run bundle for this modality and input mode under the selected output workspace.")
            else:
                chosen_path = manual.resolve()
                chosen_text = str(chosen_path)
                selected_dirs[modality] = chosen_text
                st.session_state[pending_key] = chosen_text
                st.rerun()

    if chosen_path is None:
        return None
    if _bundle_metadata(chosen_path, modality, workflow_ids, output_root, state_scope=state_scope) is None:
        selected_dirs.pop(modality, None)
        st.warning("The selected run is no longer available for this modality.")
        return None
    return chosen_path


def _stage_titles(workflow_id: str) -> dict[str, str]:
    for workflow in list_workflows():
        if workflow.workflow_id == workflow_id:
            return {stage.stage_id: stage.title for stage in workflow.stages}
    return {}


def _stage_rows(events: list[dict[str, Any]], titles: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event in events:
        stage_id = str(event.get("stage_id", "Stage"))
        rows.append({
            "Stage": titles.get(stage_id, stage_id.replace("_", " ").replace(".", " · ").title()),
            "Status": str(event.get("status", "UNKNOWN")).replace("_", " ").title(),
            "Update": str(event.get("message", "")),
            "Updated": str(event.get("timestamp", "")),
        })
    return rows


def _render_qc_review(run_dir: Path, status: dict[str, Any], metadata: dict[str, Any]) -> None:
    if status.get("status") != "QC_PENDING":
        return
    st.warning("This run is paused for an explicit quality-control decision. No analysis resumes until the review is confirmed.")
    request_path = Path(str(status.get("review_request_path", "review_request.json"))).expanduser()
    if not request_path.is_absolute():
        request_path = run_dir / request_path
    try:
        if not path_is_under(request_path.resolve(strict=True), run_dir.resolve(strict=True)):
            st.error("The QC review request points outside this run. Resume is disabled.")
            return
    except (OSError, RuntimeError):
        st.error("The QC review request is unavailable. Resume is disabled.")
        return
    request = _read_json(request_path)
    manifest_path = run_dir / "inputs" / "manifest.json"
    try:
        receipt = sha256_file(manifest_path)
    except OSError as exc:
        st.error(f"Cannot verify the input inventory receipt: {exc}")
        return
    if str(request.get("run_id", "")) != str(metadata.get("run_id", "")) or request.get("inventory_receipt_sha256") != receipt:
        st.error("This review request is stale or belongs to another run. Resume is disabled until a valid request is recorded.")
        return
    choices = request.get("block_indices")
    if not isinstance(choices, list):
        nested = request.get("choices", {})
        choices = nested.get("block_indices", []) if isinstance(nested, dict) else []
    if not isinstance(choices, list) or not choices:
        st.error("The review request contains no allowed block choices. Resume is disabled.")
        return
    flim_receipt = request.get("flim_inventory_receipt_sha256")
    if flim_receipt is not None:
        receipt_path = Path(str(request.get("inventory_receipt_path", ""))).expanduser()
        if not receipt_path.is_absolute():
            receipt_path = run_dir / receipt_path
        try:
            receipt_path = receipt_path.resolve(strict=True)
            if not path_is_under(receipt_path, run_dir.resolve(strict=True)) or not receipt_path.is_file() or sha256_file(receipt_path) != flim_receipt:
                st.error("The modality-specific QC receipt is missing, changed, or outside this run. Resume is disabled.")
                return
        except (OSError, RuntimeError):
            st.error("The modality-specific QC receipt is unavailable. Resume is disabled.")
            return
    details = request.get("block_details")
    detail_rows: list[dict[str, Any]] = []
    if isinstance(details, list):
        for block in details:
            if not isinstance(block, dict) or not isinstance(block.get("records"), list):
                continue
            for record in block["records"]:
                if isinstance(record, dict):
                    detail_rows.append({"Block": block.get("value"), "Label": block.get("label"), **record})
    with st.expander("Inspected acquisition details", expanded=True):
        if detail_rows:
            st.dataframe(detail_rows[:200], use_container_width=True, hide_index=True)
            if len(detail_rows) > 200:
                st.caption(f"Showing 200 of {len(detail_rows)} inspected records.")
        else:
            st.info("No detailed acquisition records were supplied; review the saved inventory files before choosing blocks.")
    indices = st.multiselect("Choose the inspected blocks to include (none selected automatically)", choices, default=[], key=f"qc_blocks_{run_dir.resolve()}")
    confirmed = st.checkbox("I reviewed the acquisition details and confirm this selection.", key=f"qc_confirm_{run_dir.resolve()}")
    if indices and confirmed and st.button("Confirm review and resume this run", type="primary", key=f"qc_resume_{run_dir.resolve()}"):
        try:
            job_id = submit_review_decision(run_dir, indices, confirmed)
            st.success(f"Resume submitted as SLURM job {job_id}. This is the same run; its review receipt is retained.")
        except Exception as exc:
            st.error(f"Unable to resume this run: {exc}")


def _artifact_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, str):
        if value.strip():
            paths.append(value)
    elif isinstance(value, list):
        for item in value:
            paths.extend(_artifact_paths(item))
    elif isinstance(value, dict):
        path_value = value.get("path")
        if isinstance(path_value, str):
            paths.append(path_value)
        else:
            for item in value.values():
                paths.extend(_artifact_paths(item))
    return paths


def _artifact_records(run_dir: Path, events: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def add(path_text: str, stage: str, status: str, source: str, extra: dict[str, Any] | None = None) -> None:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = run_dir / path
        try:
            identity = str(path.resolve(strict=False))
        except (OSError, RuntimeError):
            identity = str(path.absolute())
        record = records.setdefault(identity, {"path": path, "stage": stage, "status": status, "metadata": {}, "sources": set()})
        if source == "event":
            record["stage"] = stage
            record["status"] = status
        elif source == "manifest" and stage not in {"Workflow", "Run artifacts"}:
            record["stage"] = stage
        record["sources"].add(source)
        if extra:
            record["metadata"].update(extra)

    for event in events:
        stage = str(event.get("stage_id", "Workflow"))
        status = str(event.get("status", "UNKNOWN"))
        for path_text in _artifact_paths(event.get("artifacts", [])):
            add(path_text, stage, status, "event")
    artifacts = manifest.get("artifacts", [])
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                continue
            stage = str(artifact.get("stage_id", "Run artifacts"))
            add(artifact["path"], stage, str(artifact.get("status", "")), "manifest", artifact)
    return list(records.values())


def _kind_label(path: Path, metadata: dict[str, Any]) -> str:
    media_type = str(metadata.get("media_type", "")).lower()
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"} or media_type.startswith("image/"):
        return "Image"
    if suffix == ".pdf" or media_type == "application/pdf":
        return "PDF report"
    if suffix in {".html", ".htm"} or media_type == "text/html":
        return "HTML report"
    if suffix in {".svg"} or media_type == "image/svg+xml":
        return "Vector figure"
    if suffix in {".json", ".jsonl"} or "json" in media_type:
        return "Data / metadata"
    if suffix in {".csv", ".tsv", ".parquet", ".feather"}:
        return "Table / data"
    if suffix in {".txt", ".log", ".md"}:
        return "Text / log"
    return "File"


def show_artifact(artifact_text: str, run_dir: Path, status: str | None = None, *, key_prefix: str = "") -> None:
    """Offer bounded download and a bounded, format-aware preview for one selected artifact."""
    artifact_path = Path(artifact_text).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = run_dir / artifact_path
    st.caption(f"File: {artifact_path}")
    if status:
        st.caption(f"Stage status: {status.replace('_', ' ').title()}")
    try:
        resolved = artifact_path.resolve(strict=True)
        if not resolved.is_file():
            st.info("This artifact is recorded, but is not currently an available file.")
            return
        size = resolved.stat().st_size
    except (OSError, RuntimeError) as exc:
        st.info(f"This artifact is unavailable: {exc}")
        return
    if not path_is_under_workspace(resolved):
        st.caption("File contents are not opened because this path resolves outside the approved workspace.")
        return
    if size <= _DOWNLOAD_LIMIT:
        try:
            st.download_button("Download file", data=resolved.read_bytes(), file_name=resolved.name, key=f"artifact_download_{key_prefix}_{resolved}")
        except OSError as exc:
            st.caption(f"Download unavailable: {exc}")
    else:
        st.caption(f"Original is {size:,} bytes. Browser download is limited to {_DOWNLOAD_LIMIT // (1024 * 1024)} MiB; the file remains available at the path above.")
    suffix = resolved.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if size > _DOWNLOAD_LIMIT:
            st.caption("Image preview omitted because the source exceeds the 8 MiB preview limit.")
            return
        try:
            from PIL import Image

            with Image.open(resolved) as image:
                if image.width > _IMAGE_PIXEL_LIMIT or image.height > _IMAGE_PIXEL_LIMIT or image.width * image.height > 4_194_304:
                    st.caption("Image preview omitted because its dimensions exceed 2048 × 2048 pixels.")
                else:
                    st.image(image, caption=resolved.name, use_container_width=True)
        except (OSError, ImportError, ValueError) as exc:
            st.caption(f"Image preview unavailable: {exc}")
    elif suffix == ".pdf":
        st.caption("PDF pages are previewed only through bounded PNG previews generated by the worker; open the PDF using the download above.")
    elif suffix in {".svg", ".html", ".htm", ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".log"}:
        if size > _DOWNLOAD_LIMIT:
            st.caption("Text preview omitted because the file exceeds the 8 MiB preview limit.")
            return
        try:
            text = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            st.caption(f"Text preview unavailable: {exc}")
            return
        if suffix in {".json", ".jsonl"} and suffix == ".json":
            try:
                st.json(json.loads(text))
                return
            except json.JSONDecodeError:
                pass
        clipped = text[:_TEXT_PREVIEW_LIMIT]
        language = "html" if suffix in {".html", ".htm", ".svg"} else None
        st.code(clipped, language=language, line_numbers=False)
        if len(text) > len(clipped):
            st.caption(f"Preview limited to the first {_TEXT_PREVIEW_LIMIT:,} characters; download for the complete file.")
    elif suffix in {".tif", ".tiff", ".bmp"}:
        st.caption("This image format is retained as a downloadable file but is not decoded in the browser.")
    else:
        st.caption("No in-app preview is available for this file type; use the bounded download or open the path on the workspace.")


def _render_saved_sources(
    run_dir: Path, metadata: dict[str, Any], events: list[dict[str, Any]],
    *, stage_id: str | None = None, section: str,
) -> None:
    from spatial_omics_toolkit.application.ui.implementation import render_step_sources

    workflow_id = str(metadata.get("workflow_id", ""))
    titles = _stage_titles(workflow_id)
    versions: dict[str, str] = {}
    for event in events:
        event_stage = str(event.get("stage_id", ""))
        if stage_id is not None and event_stage != stage_id:
            continue
        recorded = event.get("tool_versions")
        if isinstance(recorded, dict):
            for tool_name, version in recorded.items():
                label = str(tool_name) if stage_id is not None else f"{titles.get(event_stage, event_stage)} · {tool_name}"
                versions[label] = str(version)
    versions.update(_receipt_source_versions(run_dir, metadata))

    render_step_sources(
        workflow_id, stage_id=stage_id, recorded_versions=versions,
        key_prefix=f"{section}_{run_dir.resolve()}",
    )


def _render_run(run_dir: Path) -> None:
    metadata = _read_json(run_dir / "run.json")
    status = read_run_status(run_dir)
    status_name = str(status.get("status", "UNKNOWN"))
    events = _read_events(run_dir)
    titles = _stage_titles(str(metadata.get("workflow_id", "")))
    st.subheader("Run progress")
    st.metric("Current status", status_name.replace("_", " ").title())
    message = status.get("message")
    if message:
        st.write(str(message))
    if metadata.get("created_at"):
        st.caption(f"Started {metadata['created_at']}")
    job = _read_json(run_dir / "job.json")
    job_id = status.get("slurm_job_id") or job.get("slurm_job_id")
    if job_id:
        st.caption(f"SLURM job {job_id}")
    if st.button("Refresh run status", key=f"refresh_run_{run_dir.resolve()}"):
        st.rerun()
    rows = _stage_rows(events, titles)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("The worker has not recorded a stage update yet. Refresh to check for progress.")
    warnings = [event for event in events if event.get("status") == "WARNING"]
    if status_name == "SUCCEEDED":
        st.success("The workflow completed successfully.")
    elif status_name == "WARNING":
        st.warning("The workflow completed with warnings. Review the stage notes before using its outputs.")
    elif status_name in {"FAILED", "BLOCKED"}:
        st.error("The workflow did not complete successfully. See the stage updates for the recorded reason.")
    elif status_name == "QC_PENDING":
        st.warning("The workflow is waiting for a quality-control decision.")
    if warnings and status_name not in {"WARNING", "FAILED", "BLOCKED"}:
        st.warning(f"{len(warnings)} stage warning(s) were recorded; review them in Quality control.")
    _render_saved_sources(
        run_dir, metadata, events,
        stage_id=str(status["current_stage"]) if status.get("current_stage") else None,
        section="run",
    )


def _render_qc(run_dir: Path) -> None:
    status = read_run_status(run_dir)
    metadata = _read_json(run_dir / "run.json")
    events = _read_events(run_dir)
    _render_qc_review(run_dir, status, metadata)
    rows = _stage_rows(events, _stage_titles(str(metadata.get("workflow_id", ""))))
    qc_rows = [row for row in rows if row["Status"].upper() in {"WARNING", "FAILED", "BLOCKED", "QC PENDING"}]
    success_rows = [row for row in rows if row["Status"].upper() in {"SUCCEEDED", "REUSED"}]
    if qc_rows:
        st.subheader("Items to review")
        st.dataframe(qc_rows, use_container_width=True, hide_index=True)
    if success_rows:
        with st.expander(f"Successful stages ({len(success_rows)})", expanded=False):
            st.dataframe(success_rows, use_container_width=True, hide_index=True)
    if not events:
        st.info("No quality-control or stage events have been recorded for this run yet.")
    for index, event in enumerate(events):
        stage_id = str(event.get("stage_id", "Stage"))
        label = _stage_titles(str(metadata.get("workflow_id", ""))).get(stage_id, stage_id.replace("_", " ").title())
        with st.expander(f"{label} · {str(event.get('status', 'UNKNOWN')).replace('_', ' ').title()}", expanded=event.get("status") in {"WARNING", "FAILED", "BLOCKED", "QC_PENDING"}):
            if event.get("message"):
                st.write(str(event["message"]))
            if event.get("parameters"):
                st.write("Stage settings")
                st.json(event["parameters"], expanded=False)
            paths = _artifact_paths(event.get("artifacts", []))
            if paths:
                st.caption(f"{len(paths)} artifact(s) recorded for this stage; open the Files section for the complete inventory.")
    _render_saved_sources(run_dir, metadata, events, section="qc")


def _render_results(run_dir: Path) -> None:
    events = _read_events(run_dir)
    manifest = _read_json(run_dir / "artifacts.json")
    records = _artifact_records(run_dir, events, manifest)
    if not records:
        st.info("No result artifacts have been recorded for this run yet. Files appear here as workflow stages emit them.")
        return
    run_metadata = _read_json(run_dir / "run.json")
    stage_titles = _stage_titles(str(run_metadata.get("workflow_id", "")))
    for record in records:
        record["kind"] = _kind_label(record["path"], record["metadata"])
        record["stage_label"] = stage_titles.get(record["stage"], record["stage"].replace("_", " ").replace(".", " · ").title())
    result_priority = {
        "PDF report": 0, "HTML report": 1, "Image": 2, "Vector figure": 2,
        "Table / data": 3, "Data / metadata": 4, "Text / log": 5, "File": 6,
    }
    records.sort(key=lambda record: (
        result_priority[record["kind"]], record["stage_label"], record["path"].name.lower(),
    ))
    stages = sorted({record["stage_label"] for record in records})
    kinds = sorted({record["kind"] for record in records})
    col_browser, col_preview = st.columns([1, 1.35], gap="medium")
    with col_browser:
        with st.container(border=True):
            st.markdown('<div class="wb-pane-header"><p class="wb-pane-title">Result browser & filters</p></div>', unsafe_allow_html=True)
            col_stage, col_kind = st.columns(2)
            with col_stage:
                stage_filter = st.selectbox("Stage", ["All stages", *stages], key=f"results_stage_{run_dir.resolve()}")
            with col_kind:
                kind_filter = st.selectbox("File type", ["All types", *kinds], key=f"results_kind_{run_dir.resolve()}")
            query = st.text_input("Search results", key=f"results_search_{run_dir.resolve()}").strip().lower()
            filtered = [record for record in records if (stage_filter == "All stages" or record["stage_label"] == stage_filter) and (kind_filter == "All types" or record["kind"] == kind_filter) and (not query or query in record["path"].name.lower() or query in record["stage_label"].lower())]
            st.caption(f"{len(filtered)} of {len(records)} recorded files · grouped by their recorded stage and file type")
            if not filtered:
                st.info("No files match these filters.")
                return
            labels = [
                f"{record['stage_label']} · {record['path'].name} · {record['kind']} · file {index + 1}"
                for index, record in enumerate(filtered)
            ]
            choice = st.selectbox("Choose one file to preview", labels, key=f"results_artifact_{run_dir.resolve()}")
            selected = filtered[labels.index(choice)]
            _render_saved_sources(
                run_dir, run_metadata, events, stage_id=str(selected["stage"]), section="results",
            )
    with col_preview:
        with st.container(border=True):
            st.markdown(f"**{selected['path'].name}**")
            st.caption(f"{selected['stage_label']} · {selected['kind']} · recorded in {' and '.join(sorted(selected['sources']))}")
            show_artifact(str(selected["path"]), run_dir, selected["status"], key_prefix=f"results_{run_dir.resolve()}")
            from spatial_omics_toolkit.application.ui.plotting import render_plot_controls

            try:
                plot_root = containing_workspace_root(selected["path"])
            except ValueError:
                plot_root = None
            if plot_root is not None:
                render_plot_controls(selected["path"], run_dir, plot_root)

            def source_matches(record: dict[str, Any]) -> bool:
                source_text = str(record["metadata"].get("source_pdf", ""))
                if not source_text:
                    return False
                source_path = Path(source_text).expanduser()
                if not source_path.is_absolute():
                    source_path = run_dir / source_path
                try:
                    return source_path.resolve(strict=False) == selected["path"].resolve(strict=False)
                except (OSError, RuntimeError):
                    return source_text == selected["path"].name

            linked_previews = [record for record in records if source_matches(record)]
            if selected["path"].suffix.lower() == ".pdf" and linked_previews:
                st.caption("Worker-generated bounded page previews")
                page_labels = [f"Page {item['metadata'].get('page', '?')} · {item['path'].name}" for item in linked_previews]
                page_label = st.selectbox("Preview a PDF page", page_labels, key=f"pdf_page_{run_dir.resolve()}_{selected['path']}")
                preview = linked_previews[page_labels.index(page_label)]
                show_artifact(str(preview["path"]), run_dir, "Worker-generated preview", key_prefix=f"pdf_{run_dir.resolve()}_{selected['path']}")


def _file_inventory(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    try:
        for path in run_dir.rglob("*"):
            if path.is_file():
                files.append(path)
    except OSError:
        pass
    return sorted(files, key=lambda path: str(path.relative_to(run_dir)).lower())


def _render_files(run_dir: Path) -> None:
    metadata = _read_json(run_dir / "run.json")
    manifest = _read_json(run_dir / "artifacts.json")
    files = _file_inventory(run_dir)
    st.subheader("Run files")
    st.caption(f"{len(files)} files recorded in this run bundle. Large and unsupported files remain listed with their full paths.")
    inventory_paths = {str(path.resolve(strict=False)): path for path in files}
    for record in _artifact_records(run_dir, _read_events(run_dir), manifest):
        artifact_path = record["path"]
        identity = str(artifact_path.resolve(strict=False))
        inventory_paths.setdefault(identity, artifact_path)
    ordered_paths = sorted(inventory_paths.values(), key=lambda item: str(item).lower())
    rows: list[dict[str, str]] = []
    for path in ordered_paths:
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        try:
            display_path = str(path.relative_to(run_dir))
        except ValueError:
            display_path = str(path)
        rows.append({"File": display_path, "Size": f"{size:,} bytes", "Path": str(path)})
    st.caption(f"{len(ordered_paths)} files recorded or referenced by this run. Large and unsupported files remain listed with their full paths.")
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No files have been written to this run bundle yet.")
    with st.expander("Run configuration and provenance", expanded=False):
        st.json(metadata)
    with st.expander("Artifact manifest", expanded=False):
        if manifest:
            st.json(manifest)
        else:
            st.info("No artifact manifest has been recorded yet.")
    with st.expander("Stage event history", expanded=False):
        events = _read_events(run_dir)
        if events:
            st.json(events)
        else:
            st.info("No events have been recorded yet.")
    with st.expander("Logs", expanded=False):
        logs = [path for path in files if "log" in path.parts or path.suffix.lower() in {".log", ".out", ".err"}]
        if not logs:
            st.info("No log files have been recorded yet.")
        else:
            log_labels = [str(path.relative_to(run_dir)) for path in logs]
            selected_label = st.selectbox("Choose a log", log_labels, key=f"run_logs_{run_dir.resolve()}")
            show_artifact(str(logs[log_labels.index(selected_label)]), run_dir, key_prefix=f"logs_{run_dir.resolve()}")
    with st.expander("Tool and environment versions", expanded=False):
        versions = _read_json(run_dir / "software" / "versions.json")
        if versions:
            st.json(versions)
        else:
            found_versions = [event.get("tool_versions") for event in _read_events(run_dir) if event.get("tool_versions")]
            if found_versions:
                st.json(found_versions)
            else:
                st.info("No software version record is available for this run.")


def render_run(run_dir: Path, section: RunSection) -> None:
    """Render only the requested modality section for one validated run bundle."""
    if section not in {"run", "qc", "results", "files"}:
        raise ValueError(f"Unknown run section: {section}")
    if not _safe_run_dir(run_dir):
        st.error("This run folder is unavailable or outside the approved workspace.")
        return
    resolved = run_dir.resolve(strict=True)
    metadata = _read_json(resolved / "run.json")
    if not metadata.get("workflow_id") or not metadata.get("modality"):
        st.error("This folder does not contain a valid run bundle.")
        return
    if section == "run":
        _render_run(resolved)
    elif section == "qc":
        _render_qc(resolved)
    elif section == "results":
        _render_results(resolved)
    else:
        _render_files(resolved)
