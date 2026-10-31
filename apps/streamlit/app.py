from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any

import streamlit as st

from spatial_omics_toolkit.application.contracts import ParameterField, PreflightResult, RunRequest, WorkflowDescriptor
from spatial_omics_toolkit.application.intake import (
    ALLOWED_ROOT,
    PROJECT_SAMPLE_COLUMNS,
    lexical_path_is_under,
    read_project_samples,
    sha256_file,
)
from spatial_omics_toolkit.application.scheduler import read_run_status
from spatial_omics_toolkit.application.service import prepare_request, submit_request, submit_review_decision
from spatial_omics_toolkit.application.workflows.registry import list_workflows, registry_warnings


st.set_page_config(page_title="Spatial Toolkit", layout="wide", initial_sidebar_state="expanded")


def child_directories(path_text: str) -> list[str]:
    path = Path(path_text).expanduser()
    if not path.is_dir():
        return []
    try:
        return [str(item) for item in sorted(path.iterdir()) if item.is_dir()]
    except OSError:
        return []


def project_samples_template() -> str:
    """Return a header-only project metadata template; never invent sample rows."""
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow((*PROJECT_SAMPLE_COLUMNS, "batch_or_block", "section_id"))
    return output.getvalue()


def render_modality_folder_browser(project_root_text: str) -> str:
    project_root = Path(project_root_text).expanduser().absolute()
    if not project_root.is_dir() or not lexical_path_is_under(project_root, ALLOWED_ROOT):
        st.info("Choose a project root under the allowed institute workspace to browse modality folders.")
        return ""
    browser_key = "modality_browser_path"
    current = Path(st.session_state.get(browser_key, str(project_root))).expanduser().absolute()
    if not lexical_path_is_under(current, project_root) or not current.is_dir():
        current = project_root
    st.session_state[browser_key] = str(current)
    relative = current.relative_to(project_root)
    st.caption(f"Folder browser: /{relative.as_posix() if str(relative) != '.' else ''}")
    directories = child_directories(str(current))
    selected_child = st.selectbox(
        "Open a subfolder",
        ["(choose a folder)", *directories],
        key=f"browse_child_{confirmation_key(str(current))}",
    )
    open_column, parent_column, use_column = st.columns(3)
    with open_column:
        if st.button("Open selected folder", disabled=selected_child == "(choose a folder)"):
            st.session_state[browser_key] = selected_child
            st.experimental_rerun()
    with parent_column:
        can_go_parent = current != project_root
        if st.button("Parent folder", disabled=not can_go_parent):
            st.session_state[browser_key] = str(current.parent)
            st.experimental_rerun()
    with use_column:
        if st.button("Use this folder"):
            st.session_state["selected_modality_folder"] = str(current)
    return str(current)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def confirmation_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def preview_panel_csv(path_text: str) -> tuple[list[dict[str, str]], list[str]]:
    expected = {"round", "channel_index", "marker_name", "target_cell_compartment", "fluorophore", "notes"}
    path = Path(path_text).expanduser()
    if not path_text:
        return [], []
    if not path.is_file():
        return [], [f"Marker panel CSV is not a file: {path}"]
    try:
        if path.stat().st_size > 1024 * 1024:
            return [], ["Marker panel CSV exceeds the 1 MiB preview limit."]
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or [])
            missing = sorted(expected - headers)
            if missing:
                return [], [f"Marker panel CSV is missing columns: {', '.join(missing)}"]
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [f"Unable to preview marker panel CSV: {exc}"]
    if not rows:
        return [], ["Marker panel CSV contains no rows."]
    return rows, []


def _get_nested(value: dict[str, Any], dotted_name: str) -> Any:
    current: Any = value
    for part in dotted_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(value: dict[str, Any], dotted_name: str, item: Any) -> None:
    current = value
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = item


def _render_parameter_field(
    field: ParameterField,
    workflow_id: str,
    discovered: dict[str, Any],
    initial: dict[str, Any],
    key_prefix: str,
) -> tuple[Any, str | None]:
    label = f"{field.label} ({field.unit})" if field.unit else field.label
    label = f"{label} *" if field.required else label
    widget_key = f"{key_prefix}parameter_{confirmation_key(workflow_id, field.name)}"
    choices = field.choices
    dynamic_missing = False
    if field.dynamic_choices_key:
        dynamic = _get_nested(discovered, field.dynamic_choices_key)
        if isinstance(dynamic, list):
            choices = dynamic
        else:
            dynamic_missing = True
    previous_value = _get_nested(initial, field.name)
    default = previous_value if previous_value is not None else field.default
    if field.kind in {"select", "multiselect"} and not choices and dynamic_missing:
        st.info(f"{field.label}: run quick preflight to discover allowed options.")
        return None, None
    if field.kind == "bool":
        return st.checkbox(label, value=bool(default), help=field.help_text, key=widget_key), None
    if field.kind in {"int", "float"}:
        number_type = int if field.kind == "int" else float
        if default is None:
            raw = st.text_input(label, value="", help=field.help_text, key=widget_key)
            if not raw.strip():
                return None, f"{field.label} is required" if field.required else None
            try:
                parsed: Any = number_type(raw)
            except ValueError:
                return None, f"{field.label} must be a {field.kind}"
        else:
            parsed = st.number_input(
                label,
                min_value=number_type(field.min_value) if field.min_value is not None else None,
                max_value=number_type(field.max_value) if field.max_value is not None else None,
                value=number_type(default),
                step=number_type(field.step if field.step is not None else (1 if field.kind == "int" else 0.1)),
                help=field.help_text,
                key=widget_key,
            )
        if field.min_value is not None and parsed < field.min_value:
            return parsed, f"{field.label} must be at least {field.min_value}"
        if field.max_value is not None and parsed > field.max_value:
            return parsed, f"{field.label} must be at most {field.max_value}"
        return parsed, None
    if field.kind == "select":
        if not choices:
            if not field.required:
                st.caption(f"No choices are available for optional field {field.label}; it will remain unset.")
                return None, None
            return None, f"No choices are available for required field {field.label}; review workflow inputs."
        options: list[Any] = ["(Select a value)", *choices]
        selection = default if default in choices else "(Select a value)"
        selected = st.selectbox(label, options, index=options.index(selection), help=field.help_text, key=widget_key)
        if selected == "(Select a value)":
            return None, f"{field.label} requires an explicit selection" if field.required else None
        return selected, None
    if field.kind == "multiselect":
        selected_values = st.multiselect(label, choices, default=default or [], help=field.help_text, key=widget_key)
        if field.required and not selected_values:
            return selected_values, f"{field.label} requires at least one explicit selection"
        return selected_values, None
    if field.kind in {"path", "config_file", "text"}:
        initial_text = str(default) if default is not None else ""
        if field.kind == "config_file":
            st.caption("Server-side configuration file path; browser uploads are disabled.")
        entered = st.text_input(label, value=initial_text, help=field.help_text, key=widget_key)
        if field.required and not entered.strip():
            return None, f"{field.label} is required"
        return entered.strip() or None, None
    return None, f"Unsupported parameter control kind: {field.kind}"


def render_parameter_fields(
    descriptor: WorkflowDescriptor,
    discovered: dict[str, Any],
    fields: list[ParameterField] | None = None,
    initial: dict[str, Any] | None = None,
    key_prefix: str = "",
) -> tuple[dict[str, Any], list[str]]:
    parameters: dict[str, Any] = {}
    errors: list[str] = []
    grouped: dict[tuple[str, bool], list[ParameterField]] = {}
    core_fields = [
        ParameterField(name="resources.cpus", label="SLURM job CPUs", kind="int", required=True, default=1, min_value=1, max_value=32, step=1, unit="cores", section="SLURM job resources", help_text="CPU allocation for the single Python workflow job.", advanced=True),
        ParameterField(name="resources.memory_gb", label="SLURM job memory", kind="int", required=True, default=8, min_value=1, max_value=512, unit="GiB", section="SLURM job resources", help_text="Memory allocation for the Python workflow job. Choose within your cluster allocation policy.", advanced=True),
        ParameterField(name="resources.time_limit", label="SLURM job wall time", kind="select", required=True, default="72h", choices=["8h", "24h", "48h", "72h", "168h"], unit="wall time", section="SLURM job resources", help_text="Maximum runtime for the analysis job.", advanced=True),
    ]
    effective_fields = list(fields if fields is not None else descriptor.parameter_fields)
    existing_names = {field.name for field in effective_fields}
    effective_fields.extend(field for field in core_fields if field.name not in existing_names)
    for field in effective_fields:
        grouped.setdefault((field.section, field.advanced), []).append(field)
    for (section, advanced), section_fields in grouped.items():
        context = st.expander(f"Advanced settings — {section}", expanded=False) if advanced else st.container()
        with context:
            if not advanced:
                st.markdown(f"**{section}**")
            for field in section_fields:
                value, error = _render_parameter_field(field, descriptor.workflow_id, discovered, initial or {}, key_prefix)
                if error:
                    errors.append(error)
                    st.error(error)
                elif value is not None:
                    _set_nested(parameters, field.name, value)
    return parameters, errors


def run_files(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metadata = read_json(run_dir / "run.json")
    artifacts = read_json(run_dir / "artifacts.json")
    events: list[dict[str, Any]] = []
    event_path = run_dir / "events.jsonl"
    try:
        with event_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return metadata, events, artifacts


def show_artifact(artifact_text: str, run_dir: Path, status: str | None = None) -> None:
    artifact_path = Path(artifact_text).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = run_dir / artifact_path
    if status:
        st.caption(f"Artifact status: {status}")
    if not artifact_path.is_file():
        st.caption("Full-size artifact path is listed above; unavailable for preview.")
        return
    try:
        artifact_size = artifact_path.stat().st_size
        if artifact_size <= 8 * 1024 * 1024:
            st.download_button(
                "Download bounded artifact",
                data=artifact_path.read_bytes(),
                file_name=artifact_path.name,
                key=f"download_{confirmation_key(str(artifact_path))}",
            )
        else:
            st.caption(f"Full-size original ({artifact_size} bytes) remains on disk; browser download is capped at 8 MiB.")
    except OSError as exc:
        st.caption(f"Artifact unavailable: {exc}")
        return
    if artifact_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return
    try:
        if artifact_path.stat().st_size > 8 * 1024 * 1024:
            st.caption("Preview omitted (image exceeds the 8 MiB preview limit); full-size artifact remains on disk.")
            return
        from PIL import Image
        with Image.open(artifact_path) as image:
            if image.width > 2048 or image.height > 2048:
                st.caption("Preview omitted (image exceeds 2048 × 2048 pixels); full-size artifact remains on disk.")
                return
            st.image(image, caption=artifact_path.name, use_column_width=True)
    except (OSError, ImportError, ValueError) as exc:
        st.caption(f"Image preview unavailable: {exc}")


def render_qc_review(run_dir: Path, status: dict[str, Any]) -> None:
    if status.get("status") != "QC_PENDING":
        return
    st.warning("Review required: scientific analysis is paused until this same run receives an explicit QC decision.")
    review_path = Path(str(status.get("review_request_path", "review_request.json")))
    if not review_path.is_absolute():
        review_path = run_dir / review_path
    review_request = read_json(review_path)
    manifest_path = run_dir / "inputs" / "manifest.json"
    try:
        receipt = sha256_file(manifest_path)
    except OSError as exc:
        st.error(f"Cannot verify input inventory receipt: {exc}")
        return
    if review_request.get("run_id") != read_json(run_dir / "run.json").get("run_id") or review_request.get("inventory_receipt_sha256") != receipt:
        st.error("Review options are stale or belong to another run. Resume is disabled until the adapter regenerates a valid review request.")
        return
    choices = review_request.get("block_indices")
    if not isinstance(choices, list):
        choices = review_request.get("choices", {}).get("block_indices", [])
    if not isinstance(choices, list) or not choices:
        st.error("No reviewed block choices were recorded; resume is disabled.")
        return
    st.caption(f"Run ID: {review_request['run_id']} · inventory receipt SHA-256: {receipt}")
    block_details = review_request.get("block_details")
    detail_rows: list[dict[str, Any]] = []
    if isinstance(block_details, list):
        for block in block_details:
            if not isinstance(block, dict) or not isinstance(block.get("records"), list):
                continue
            for record in block["records"]:
                if not isinstance(record, dict):
                    continue
                detail_rows.append(
                    {
                        "block_index": block.get("value"),
                        "block_label": block.get("label"),
                        "source_path": record.get("path"),
                        "source_sha256": record.get("source_sha256"),
                        "detector_index": record.get("detector_index"),
                        "selection_kind": record.get("selection_kind"),
                        "shape": record.get("shape"),
                        "time_step_seconds": record.get("time_step_seconds"),
                        "window_seconds": record.get("window_seconds"),
                        "acquisition_frequency_mhz": record.get("acquisition_frequency_mhz"),
                    }
                )
    with st.expander("Inspected SDT block and acquisition details", expanded=True):
        if detail_rows:
            st.dataframe(
                detail_rows[:200],
                use_container_width=True,
                hide_index=True,
            )
            if len(detail_rows) > 200:
                st.caption(f"Showing the first 200 of {len(detail_rows)} inspected records.")
        else:
            st.info("The adapter supplied block choices without detailed acquisition records; review the saved inventory artifact before selection.")
    selected = st.multiselect(
        "Select SDT block indices for analysis (no blocks are selected automatically)",
        choices,
        default=[],
        key=f"review_blocks_{confirmation_key(str(run_dir), receipt)}",
    )
    confirmed = st.checkbox(
        "I reviewed the inspected blocks and confirm these selected indices.",
        key=f"review_confirmed_{confirmation_key(str(run_dir), receipt, *(str(value) for value in selected))}",
    )
    if selected and confirmed and st.button("Resume this run with selected blocks", type="primary"):
        try:
            job_id = submit_review_decision(run_dir, selected, confirmed)
            st.success(f"Resume worker submitted as SLURM job {job_id}; run ID is unchanged.")
        except Exception as exc:
            st.error(f"Unable to submit QC review decision: {exc}")


def display_run(run_dir: Path) -> None:
    status = read_run_status(run_dir)
    metadata, events, artifact_data = run_files(run_dir)
    status_name = str(status.get("status", "UNKNOWN"))
    st.subheader("Run progress")
    status_col, run_col = st.columns([1, 3])
    with status_col:
        st.metric("Overall status", status_name)
    with run_col:
        st.caption(status.get("message", ""))
        st.caption(f"Run directory: {run_dir.resolve()}")
    job_id = status.get("slurm_job_id") or read_json(run_dir / "job.json").get("slurm_job_id")
    if job_id:
        try:
            query = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True, timeout=5, check=False)
            queue_state = query.stdout.strip()
            if queue_state:
                st.caption(f"SLURM job {job_id}: {queue_state}")
        except (OSError, subprocess.TimeoutExpired):
            st.caption(f"SLURM job ID: {job_id}; queue state unavailable")

    stage_rows = [
        {
            "Stage": str(event.get("stage_id", f"event-{index}")),
            "Status": str(event.get("status", "UNKNOWN")),
            "Message": str(event.get("message", "")),
            "Artifacts": len(event.get("artifacts", [])) if isinstance(event.get("artifacts", []), list) else 0,
        }
        for index, event in enumerate(events, start=1)
    ]
    st.markdown("#### Stage overview")
    if stage_rows:
        st.dataframe(stage_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No stage events have been recorded yet.")
    render_qc_review(run_dir, status)

    if metadata:
        with st.expander("Run inputs and configuration", expanded=False):
            st.json({"project": metadata.get("project"), "input": metadata.get("input"), "parameters": metadata.get("parameters")})
            inventory_path = run_dir / "inputs" / "manifest.json"
            if inventory_path.is_file():
                inventory_size = inventory_path.stat().st_size
                with st.expander("Input inventory and hash scope", expanded=False):
                    st.code(str(inventory_path.resolve()), language=None)
                    if inventory_size <= 8 * 1024 * 1024:
                        input_inventory = read_json(inventory_path)
                        input_files = input_inventory.get("files", [])
                        st.json(
                            {
                                "inventory_scope": input_inventory.get("inventory_scope", "complete_modality_folder"),
                                "recorded_entries": len(input_files) if isinstance(input_files, list) else None,
                                "annotation_csv_sha256": input_inventory.get("annotation_csv_sha256"),
                                "inventory_errors": input_inventory.get("inventory_errors", []),
                            }
                        )
                        st.download_button(
                            "Download complete input manifest",
                            data=inventory_path.read_bytes(),
                            file_name="input-manifest.json",
                            mime="application/json",
                            key=f"input_manifest_{confirmation_key(str(inventory_path))}",
                        )
                    else:
                        st.caption(
                            f"Manifest size is {inventory_size} bytes. It was not loaded into the app; "
                            "use the path above to inspect or download it on the compute node."
                        )
            resolved_parameters = read_json(run_dir / "resolved_parameters.json")
            if resolved_parameters:
                st.markdown("**Resolved execution parameters**")
                st.json(resolved_parameters)
            mapping_receipt = read_json(run_dir / "mapping_receipt.json")
            if mapping_receipt:
                st.markdown("**Confirmed input-to-specimen mapping receipt**")
                st.json(mapping_receipt)

    if events:
        with st.expander("Stage details, QC, and artifacts", expanded=False):
            for index, event in enumerate(events, start=1):
                stage_id = str(event.get("stage_id", f"event-{index}"))
                stage_status = str(event.get("status", "UNKNOWN"))
                with st.expander(f"{stage_id} — {stage_status}", expanded=stage_status in {"RUNNING", "WARNING", "FAILED", "BLOCKED", "QC_PENDING"}):
                    st.write(event.get("message", ""))
                    if event.get("tool_versions"):
                        st.markdown("**Tool versions**")
                        st.json(event["tool_versions"])
                    if event.get("parameters"):
                        st.markdown("**Resolved parameters**")
                        st.json(event["parameters"])
                    for artifact in event.get("artifacts", []):
                        show_artifact(str(artifact), run_dir, stage_status)
    with st.expander("Complete artifact inventory", expanded=False):
        st.caption("Full-size files remain on disk; previews and downloads stay bounded.")
        if isinstance(artifact_data, dict):
            artifact_rows = [
                {"Artifact": str(name), "Details": json.dumps(value, sort_keys=True, default=str)}
                for name, value in artifact_data.items()
            ]
        elif isinstance(artifact_data, list):
            artifact_rows = [
                {"Artifact": str(index), "Details": json.dumps(value, sort_keys=True, default=str)}
                for index, value in enumerate(artifact_data, start=1)
            ]
        else:
            artifact_rows = []
        if artifact_rows:
            st.dataframe(artifact_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No artifact inventory has been recorded.")
    log_files = sorted((run_dir / "logs").glob("*")) if (run_dir / "logs").is_dir() else []
    if log_files:
        with st.expander("Worker logs", expanded=False):
            for log_path in log_files:
                if log_path.is_file():
                    st.markdown(f"**{log_path.name}**")
                    try:
                        with log_path.open("rb") as stream:
                            stream.seek(max(0, log_path.stat().st_size - 12000))
                            log_text = stream.read().decode("utf-8", errors="replace")
                        st.code(log_text)
                    except OSError as exc:
                        st.warning(f"Unable to read log: {exc}")


def render_workflow(descriptor: WorkflowDescriptor) -> None:
    st.header(descriptor.title)
    st.write(descriptor.description)
    st.write(f"**Inputs:** {descriptor.input_description}")
    st.write(f"**Outputs:** {descriptor.output_description}")
    if descriptor.limitations:
        with st.expander(f"Workflow notes and limitations ({len(descriptor.limitations)})", expanded=False):
            for limitation in descriptor.limitations:
                st.info(limitation)
    for index, stage in enumerate(descriptor.stages, start=1):
        with st.expander(f"Stage {index}: {stage.title}", expanded=False):
            st.write(stage.purpose)
            for tool in stage.tools:
                st.markdown(f"**{tool.name}** — version: `{tool.version}`; role: {tool.role}" + (f"; [documentation]({tool.documentation_url})" if tool.documentation_url else ""))


def render_preflight(result: PreflightResult) -> None:
    if result.blockers:
        for blocker in result.blockers:
            st.error(blocker)
    if result.warnings:
        with st.expander(f"Preflight notes ({len(result.warnings)})", expanded=False):
            for warning in result.warnings:
                st.warning(warning)
    if result.ready:
        st.success("Preflight ready. Submission schedules a guarded Python worker directly through SLURM.")
    with st.expander("Resolved preflight parameters", expanded=False):
        st.json(result.resolved_parameters)


def render_page_hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_markup = "".join(
        f'<span class="hero-chip">{escape(badge)}</span>'
        for badge in badges or []
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">SPATIAL OMICS WORKFLOW SUITE</div>
          <h1>{escape(title)}</h1>
          <p>{escape(subtitle)}</p>
          <div class="hero-sub">{badge_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0f1b2d;
            --navy2: #1e3a5f;
            --accent: #1d4ed8;
            --g50: #f8fafc;
            --g100: #f1f5f9;
            --g200: #e2e8f0;
            --g500: #64748b;
            --g700: #334155;
            --white: #ffffff;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--g50) !important;
            color: var(--g700) !important;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem; }
        h1, h2, h3, h4 { color: var(--navy) !important; letter-spacing: normal !important; }
        h1 { font-size: 1.72rem !important; font-weight: 800 !important; line-height: 1.25 !important; }
        h2 { font-size: 1.3rem !important; font-weight: 750 !important; }
        h3 { font-size: 1.02rem !important; font-weight: 700 !important; }
        p, label, .stMarkdown, .stCaption { color: var(--g700); line-height: 1.55; }
        section[data-testid="stSidebar"] {
            background: var(--white) !important;
            border-right: 1px solid var(--g200) !important;
            min-width: 19rem !important;
        }
        [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display: none !important; }
        section[data-testid="stSidebar"] {
            transform: none !important; min-width: 21rem !important; visibility: visible !important;
        }
        section[data-testid="stSidebar"] div[role="radio"] {
            border-radius: 7px; padding: .3rem .45rem; margin: .08rem 0;
        }
        section[data-testid="stSidebar"] div[role="radio"][aria-checked="true"] {
            background: #eff6ff; color: var(--accent); font-weight: 700;
        }
        input[type="radio"] { accent-color: var(--accent); }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea {
            background: var(--white) !important; color: var(--g700) !important;
            border: 1px solid var(--g200) !important; border-radius: 6px !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--white) !important; border-color: var(--g200) !important;
            border-radius: 6px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 7px; padding: .34rem .5rem; font-weight: 550;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: var(--g100);
        }
        .hero {
            background: linear-gradient(135deg, var(--navy) 0%, #1e3a8a 100%);
            padding: 1.7rem 2rem; border-radius: 12px; margin: .25rem 0 1.25rem;
            box-shadow: 0 2px 6px rgba(15,27,45,.08);
        }
        .hero-kicker {
            color: #bfdbfe; font-size: .68rem; font-weight: 750;
            letter-spacing: .08em; text-transform: uppercase; margin-bottom: .35rem;
        }
        .hero h1 { color: var(--white) !important; margin: 0 0 .35rem; }
        .hero p { color: #dbeafe !important; font-size: .87rem; margin: 0; max-width: 68rem; }
        .hero-sub { display: flex; gap: .5rem; margin-top: .85rem; flex-wrap: wrap; }
        .hero-chip {
            background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
            border-radius: 20px; padding: .22rem .7rem; color: #eff6ff;
            font-size: .7rem; font-weight: 600;
        }
        .section-card, [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--white); border: 1px solid var(--g200);
            border-radius: 9px; box-shadow: 0 1px 3px rgba(0,0,0,.035);
        }
        [data-testid="stMetric"] {
            background: var(--white); border: 1px solid var(--g200);
            border-top: 3px solid var(--accent); border-radius: 8px; padding: .75rem 1rem;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--g200); border-radius: 7px; }
        div[data-testid="stExpander"] { border: 1px solid var(--g200); border-radius: 8px; background: var(--white); }
        .stButton > button, .stDownloadButton > button {
            border-radius: 6px; font-weight: 650; border: 1px solid var(--g200);
        }
        .stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 4px; background: var(--g100); border-radius: 8px; padding: 4px;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 6px; font-size: .83rem; padding: .42rem .9rem; font-weight: 550;
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            background: var(--white) !important; color: var(--navy) !important;
            font-weight: 750 !important; box-shadow: 0 1px 3px rgba(0,0,0,.07);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    warnings = registry_warnings()
    workflows = list_workflows()
    destinations = ["Home", "Project & Data", "CycIF", "HS", "FLIM", "H&E", "Visium", "Review & Submit", "Run & Reports"]
    with st.sidebar:
        st.markdown("## Spatial Toolkit")
        st.caption("Reviewed inputs · guarded analysis · traceable outputs")
        page = st.radio("Workspace", destinations, key="spatial_page")
        st.caption("Steps: configure project → select assay → review gates → inspect run.")
        if warnings:
            with st.expander(f"Adapter notices ({len(warnings)})", expanded=False):
                for warning in warnings:
                    st.warning(f"Workflow adapter unavailable: {warning}")

    if not str(st.session_state.get("project_root_path", "")).strip():
        st.session_state["project_root_path"] = "/home/groups/EmiliLab/pauli/"
    if not str(st.session_state.get("output_root", "")).strip():
        st.session_state["output_root"] = "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/runs"
    if not str(st.session_state.get("annotation_csv", "")).strip():
        st.session_state["annotation_csv"] = str(Path(st.session_state["project_root_path"]).expanduser() / "project_samples.csv")
    if "selected_modality_folder" not in st.session_state:
        st.session_state["selected_modality_folder"] = ""
    if not workflows:
        st.error("No implemented modality workflows are currently registered. Workflow adapters are required before submission.")
        return

    by_modality: dict[str, list[WorkflowDescriptor]] = {
        modality: [item for item in workflows if item.modality == modality]
        for modality in ("CycIF", "HS", "FLIM", "H&E", "Visium")
    }
    project_root_text = str(st.session_state["project_root_path"])
    annotation_text = str(st.session_state["annotation_csv"])
    output_root_text = str(st.session_state["output_root"])
    modality_text = str(st.session_state.get("selected_modality_folder", ""))
    selected_descriptors: dict[str, WorkflowDescriptor] = {}
    for modality, items in by_modality.items():
        if not items:
            continue
        choice_key = f"selected_workflow_{confirmation_key(modality)}"
        labels = [f"{item.title} ({item.workflow_id})" for item in items]
        selected = st.session_state.get(choice_key, labels[0])
        if selected not in labels:
            selected = labels[0]
        selected_descriptors[modality] = items[labels.index(selected)]
    if page in by_modality and by_modality[page]:
        items = by_modality[page]
        choice_key = f"selected_workflow_{confirmation_key(page)}"
        labels = [f"{item.title} ({item.workflow_id})" for item in items]
        selected_label = st.selectbox("Implemented workflow", labels, key=choice_key)
        descriptor = items[labels.index(selected_label)]
        selected_descriptors[page] = descriptor
        st.session_state["active_workflow_id"] = descriptor.workflow_id
        st.session_state["active_workflow_modality"] = page
    active_id = st.session_state.get("active_workflow_id")
    descriptor = next((item for item in workflows if item.workflow_id == active_id), None)
    if descriptor is not None:
        selected_descriptors[descriptor.modality] = descriptor

    if page == "Home":
        render_page_hero(
            "Spatial Toolkit",
            "A reviewed, modality-specific workspace for spatial imaging and omics analysis.",
            [f"{len(by_modality)} modalities", f"{len(workflows)} workflows", "Python + SLURM"],
        )
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Workflow")
            st.write("Choose a project, configure an implemented assay workflow, and complete the review gates before submission.")
            st.info("Use the persistent sidebar to move between project data, assay settings, review, and run reports.")
        with right:
            st.subheader("Available workflows")
            st.dataframe(
                [{"Modality": item.modality, "Workflow": item.title, "ID": item.workflow_id} for item in workflows],
                use_container_width=True, hide_index=True,
            )
            st.caption("Only workflows registered by an implemented adapter are selectable.")

    elif page == "Project & Data":
        render_page_hero(
            "Project & Data",
            "Choose the project metadata, modality folder, and output location used by downstream steps.",
            ["Server-side paths", "Reviewed sample metadata"],
        )
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Project inputs")
            st.caption("Select server-side project and modality folders. Browser uploads are disabled for large imaging and spatial data.")
            project_root_text = st.text_input(
                "Project root",
                value=str(st.session_state["project_root_path"]),
                key="project_root_path",
            )
            project_root = Path(project_root_text).expanduser()
            annotation_default = str(project_root / "project_samples.csv")
            annotation_text = st.text_input(
                "Project sample metadata (project_samples.csv)",
                value=st.session_state.get("annotation_csv", annotation_default), key="annotation_csv",
            )
            st.download_button(
                "Download header-only project_samples.csv template", data=project_samples_template(),
                file_name="project_samples.csv", mime="text/csv",
                help="Headers only; no example identities or invented sample metadata.",
            )
            st.caption(
                "Use one row per physical specimen/section. `biological_unit_id` may repeat "
                "for multiple specimens from the same independent unit. Use literal `unknown` "
                "when a required source value is unavailable; do not guess."
            )
        with right:
            st.subheader("Validated project data")
            annotation_path = Path(annotation_text).expanduser()
            if annotation_path.is_file():
                sample_rows, sample_errors = read_project_samples(annotation_path)
                for sample_error in sample_errors:
                    st.error(sample_error)
                if sample_rows and not sample_errors:
                    st.success(f"Validated {len(sample_rows)} project sample row(s); each specimen_id is unique.")
                    st.dataframe(sample_rows[:200], use_container_width=True, hide_index=True)
                    if len(sample_rows) > 200:
                        st.caption("Showing the first 200 rows; all validated rows are used in the run contract.")
            else:
                st.warning("Select a valid project_samples.csv or download the header-only template and fill it with reviewed metadata.")
            render_modality_folder_browser(project_root_text)
            selected_modality = st.session_state.get("selected_modality_folder", "")
            with st.expander("Advanced: enter a server-side modality path", expanded=False):
                manual_modality = st.text_input("Manual complete modality folder path", value="", key="modality_manual_path")
            modality_text = manual_modality.strip() or selected_modality
            st.session_state["selected_modality_folder"] = modality_text
            st.info(f"Selected modality folder: {modality_text or '(none selected)'}")
            output_root_text = st.text_input(
                "Output root (must be under /home/groups/EmiliLab/pauli)",
                key="output_root",
            )

    elif page in by_modality:
        items = by_modality[page]
        if not items:
            render_page_hero(page, f"No implemented {page} workflow is currently registered.")
            st.info(f"No implemented {page} workflow is currently registered.")
        else:
            descriptor = selected_descriptors[page]
            render_page_hero(
                descriptor.title,
                descriptor.description,
                [page, f"{len(descriptor.stages)} stages", descriptor.workflow_id],
            )
            left, right = st.columns([1, 2], gap="large")
            with left:
                st.subheader("Workflow settings")
                st.caption("Fields marked * are required. Advanced settings remain grouped under collapsed sections.")
                previous_preflight = st.session_state.get("preflight")
                discovered_parameters: dict[str, Any] = {}
                if isinstance(previous_preflight, tuple) and len(previous_preflight) == 3:
                    previous_result = previous_preflight[1]
                    if isinstance(previous_result, dict):
                        candidate = previous_result.get("resolved_parameters", {})
                        if isinstance(candidate, dict):
                            discovered_parameters = candidate
                initial = st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {})
                if not isinstance(initial, dict):
                    initial = {}
                parameters, parameter_errors = render_parameter_fields(
                    descriptor, discovered_parameters, initial=initial,
                    key_prefix=f"{confirmation_key(descriptor.workflow_id)}_",
                )
                st.session_state[f"workflow_parameters_{descriptor.workflow_id}"] = parameters
                st.session_state[f"workflow_parameters_valid_{descriptor.workflow_id}"] = not parameter_errors
            with right:
                st.subheader("Workflow overview")
                overview_tab, tools_tab, limits_tab = st.tabs(["Overview", "Stages & tools", "Limitations"])
                with overview_tab:
                    st.caption("Workflow inputs and outputs")
                    st.markdown(f"**Inputs:** {descriptor.input_description}")
                    st.markdown(f"**Outputs:** {descriptor.output_description}")
                with tools_tab:
                    for index, stage in enumerate(descriptor.stages, start=1):
                        with st.expander(f"Stage {index}: {stage.title}", expanded=False):
                            st.write(stage.purpose)
                            for tool in stage.tools:
                                st.markdown(f"**{tool.name}** — version: `{tool.version}`; role: {tool.role}" + (f"; [documentation]({tool.documentation_url})" if tool.documentation_url else ""))
                with limits_tab:
                    if descriptor.limitations:
                        for limitation in descriptor.limitations:
                            st.warning(limitation)
                    else:
                        st.info("No additional workflow limitations are registered.")
                st.caption("Project paths and output location are maintained under Project & Data.")

    elif page == "Review & Submit":
        render_page_hero(
            "Review & Submit",
            "Confirm fresh inputs, explicit identity or technical-QC scope, and final readiness before submission.",
            ["Preflight", "Identity review", "Final gate"],
        )
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Active workflow")
            descriptor = next((item for item in workflows if item.workflow_id == st.session_state.get("active_workflow_id")), None)
            if descriptor is None:
                st.error("Choose an implemented workflow on an assay page before review or submission.")
            parameters = st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {}) if descriptor else {}
            params_valid = bool(descriptor) and bool(st.session_state.get(f"workflow_parameters_valid_{descriptor.workflow_id}", False))
            if descriptor and not params_valid:
                st.error("Workflow configuration is missing or invalid. Review the assay settings before continuing.")
            if descriptor:
                st.write(f"**{descriptor.modality}:** {descriptor.title}")
                st.caption(f"Workflow ID: `{descriptor.workflow_id}`")
        with right:
            if descriptor and params_valid:
                st.subheader("Preflight and identity review")
                st.caption("Preflight checks current inputs and parameters. Submission remains blocked until all required review gates are complete.")
                project_root_text = str(st.session_state.get("project_root_path", ""))
                annotation_text = str(st.session_state.get("annotation_csv", ""))
                output_root_text = str(st.session_state.get("output_root", ""))
                modality_text = str(st.session_state.get("selected_modality_folder", ""))
                project_root = Path(project_root_text).expanduser()
                panel_path_text = ""
                panel_confirmed = False
                grouping_confirmed = False
                panel_valid = True
                tabular_cycif = descriptor.workflow_id == "cycif.tabular_analysis"
                if descriptor.modality == "CycIF" and "analysis" not in descriptor.workflow_id.lower():
                    with st.expander("CycIF marker panel and grouping confirmation", expanded=True):
                        st.caption(
                            "Optional panel CSV enables export/segmentation only after confirmation. "
                            "channel_index is zero-based. Without a confirmed panel mapping, "
                            "registration may run, but export/segmentation remain BLOCKED."
                        )
                        panel_path_text = st.text_input("Marker panel CSV path (optional)", key="cycif_panel_csv")
                        panel_rows, panel_errors = preview_panel_csv(panel_path_text)
                        for panel_error in panel_errors:
                            st.error(panel_error)
                        if panel_rows:
                            st.dataframe(panel_rows, use_container_width=True, hide_index=True)
                        panel_key_parts = [panel_path_text]
                        if panel_path_text:
                            try:
                                panel_stat = Path(panel_path_text).expanduser().stat()
                                panel_key_parts.extend((str(panel_stat.st_size), str(panel_stat.st_mtime_ns)))
                            except OSError:
                                pass
                        panel_confirmed = st.checkbox(
                            "I reviewed and confirm this panel's zero-based channel-to-marker mapping.",
                            key=f"cycif_panel_confirmed_{confirmation_key(*panel_key_parts)}",
                        )
                        grouping_confirmed = st.checkbox(
                            "I confirm the selected grouping does not infer biological identity or replicate structure.",
                            key=f"cycif_grouping_confirmed_{confirmation_key(project_root_text, annotation_text, modality_text)}",
                        )
                        panel_valid = not panel_errors and grouping_confirmed
                        if not grouping_confirmed:
                            st.warning("Confirm grouping explicitly before CyCIF preflight.")
                        if panel_path_text and not panel_confirmed:
                            st.warning("A selected panel without explicit confirmation will not be used for export/segmentation.")
                        if not panel_path_text:
                            st.info("No panel selected: registration can proceed; adapter stages requiring marker mapping will report BLOCKED.")
                    registration = parameters.setdefault("registration", {})
                    if not isinstance(registration, dict):
                        st.error("The JSON `registration` parameter must be an object.")
                        panel_valid = False
                    else:
                        registration["confirmed_grouping"] = grouping_confirmed
                        registration["marker_panel_confirmed"] = panel_confirmed
                        if panel_path_text:
                            registration["marker_panel_csv"] = str(Path(panel_path_text).expanduser())
                        else:
                            registration.pop("marker_panel_csv", None)
                form_signature = confirmation_key(
                    descriptor.workflow_id, project_root_text, annotation_text, modality_text,
                    output_root_text, json.dumps(parameters, sort_keys=True, default=str),
                )
                if st.button("Run preflight", type="secondary", key="run_preflight") and params_valid and panel_valid:
                    try:
                        request, result = prepare_request(
                            descriptor.workflow_id, project_root, Path(annotation_text).expanduser(),
                            Path(modality_text).expanduser(), Path(output_root_text).expanduser(), parameters,
                        )
                        st.session_state["preflight"] = (
                            request.model_dump(mode="json"), result.model_dump(mode="json"), form_signature
                        )
                    except Exception as exc:
                        st.error(f"Unable to preflight request: {exc}")
                preflight_state = st.session_state.get("preflight")
                if preflight_state:
                    request_data, result_data, stored_signature = preflight_state
                    if stored_signature != form_signature:
                        st.warning("Inputs or parameters changed after preflight. Run preflight again before submission.")
                    else:
                        preflight = PreflightResult.model_validate(result_data)
                        render_preflight(preflight)
                        resolved_parameters = result_data.get("resolved_parameters", {})
                        group_records = resolved_parameters.get("acquisition_groups", [])
                        mapping_ui_valid = isinstance(group_records, list)
                        mapping_confirmed = False
                        specimen_mapping: dict[str, str] = {}
                        core_to_specimen: dict[str, str] = {}
                        use_specimen_id_column = False
                        if not mapping_ui_valid:
                            st.error("Adapter acquisition_groups must be a list of structured records; specimen mapping is blocked.")
                            group_records = []
                        samples, sample_errors = read_project_samples(Path(request_data["annotation_csv"]))
                        for sample_error in sample_errors:
                            st.error(sample_error)
                            mapping_ui_valid = False
                        specimen_ids = list(dict.fromkeys(row.get("specimen_id") or "" for row in samples if (row.get("specimen_id") or "").strip()))
                        manual_mapping = True
                        if tabular_cycif and bool(resolved_parameters.get("has_specimen_id")):
                            mapping_choice = st.radio(
                                "Feature-table mapping source",
                                ["Use existing specimen_id column", "Map exact core values to project specimen_id"],
                                key=f"cycif_mapping_source_{confirmation_key(request_data['workflow_id'], request_data['modality_dir'])}",
                            )
                            manual_mapping = mapping_choice == "Map exact core values to project specimen_id"
                        if tabular_cycif and not manual_mapping and bool(resolved_parameters.get("has_specimen_id")):
                            st.info("The adapter found a specimen_id column. It will be used only after explicit confirmation and validation against project_samples.csv; no names are inferred.")
                            mapping_confirmed = st.checkbox(
                                "I confirm the existing specimen_id column is the authoritative mapping to project_samples.csv.",
                                key=f"cycif_existing_specimen_confirmed_{confirmation_key(request_data['annotation_csv'], *specimen_ids)}",
                            )
                            use_specimen_id_column = mapping_confirmed
                            if not mapping_confirmed:
                                st.warning("Confirm the specimen_id column before submitting tabular analysis.")
                        elif group_records:
                            st.markdown("#### Confirm group-to-specimen mapping")
                            st.caption("Each group must map explicitly. Exact source paths are available on demand.")
                            seen_group_keys: set[str] = set()
                            for group in group_records:
                                if not isinstance(group, dict) or not isinstance(group.get("group_key"), str) or not group["group_key"].strip():
                                    st.error("Adapter returned an acquisition group without a nonempty group_key; mapping is blocked.")
                                    mapping_ui_valid = False
                                    continue
                                group_key = group["group_key"]
                                if group_key in seen_group_keys:
                                    st.error(f"Adapter returned duplicate acquisition group key: {group_key}")
                                    mapping_ui_valid = False
                                    continue
                                seen_group_keys.add(group_key)
                                input_paths = group.get("input_paths")
                                if not isinstance(input_paths, list) or any(not isinstance(item, str) for item in input_paths):
                                    st.error(f"Adapter group {group_key} must include input_paths as a list of exact paths.")
                                    mapping_ui_valid = False
                                    input_paths = []
                                with st.container(border=True):
                                    title_col, mapping_col = st.columns([2, 1])
                                    with title_col:
                                        st.markdown(f"**{group.get('label', group_key)}**")
                                        st.caption(f"Exact group key: `{group_key}`")
                                    with mapping_col:
                                        choice = st.selectbox(
                                            "Project specimen_id", ["UNMAPPED", *specimen_ids],
                                            key=f"specimen_mapping_{confirmation_key(group_key)}",
                                        )
                                with st.expander(f"Exact input paths · {group.get('label', group_key)}", expanded=False):
                                    for input_path in input_paths:
                                        st.code(input_path, language=None)
                                if tabular_cycif:
                                    core_to_specimen[group_key] = choice
                                else:
                                    specimen_mapping[group_key] = choice
                            selected_mapping = core_to_specimen if tabular_cycif else specimen_mapping
                            unmapped = [key for key, specimen_id in selected_mapping.items() if specimen_id == "UNMAPPED"]
                            if unmapped:
                                st.warning(f"{len(unmapped)} group(s) remain UNMAPPED. They are eligible only for technical QC; specimen aggregation/comparison/integration must remain BLOCKED.")
                            confirmation_parts = [request_data["annotation_csv"], *[f"{key}={value}" for key, value in sorted(selected_mapping.items())]]
                            mapping_confirmed = st.checkbox(
                                "I confirm these explicit group-to-specimen mappings, including any UNMAPPED technical-QC groups.",
                                key=f"specimen_mapping_confirmed_{confirmation_key(*confirmation_parts)}",
                            )
                            if not mapping_confirmed:
                                st.warning("Explicitly confirm the full mapping table before submission.")
                        elif descriptor.workflow_id in {"he-image-intake", "cycif-raw-czi"}:
                            st.info("This adapter reports no acquisition groups. This run can only perform source-bound technical QC and bounded image intake; it cannot attach specimen identity, pair images, or produce specimen-level analysis.")
                            mapping_confirmed = st.checkbox(
                                "I confirm this no-group run is technical QC only; no sample identity or biological comparison will be inferred.",
                                key=f"qc_only_confirmed_{confirmation_key(request_data['workflow_id'], request_data['modality_dir'])}",
                            )
                            if not mapping_confirmed:
                                st.warning("Explicitly confirm technical-QC-only scope before submission.")
                        else:
                            st.error("This workflow returned no acquisition groups for explicit identity mapping. Submission is blocked; check the modality inputs or adapter preflight.")
                            mapping_ui_valid = False
                        request_data["parameters"]["specimen_mapping"] = specimen_mapping
                        request_data["parameters"]["core_to_specimen"] = core_to_specimen
                        request_data["parameters"]["use_specimen_id_column"] = use_specimen_id_column
                        request_data["parameters"]["specimen_mapping_confirmed"] = mapping_confirmed
                        result_data["resolved_parameters"]["specimen_mapping"] = specimen_mapping
                        result_data["resolved_parameters"]["core_to_specimen"] = core_to_specimen
                        result_data["resolved_parameters"]["use_specimen_id_column"] = use_specimen_id_column
                        result_data["resolved_parameters"]["specimen_mapping_confirmed"] = mapping_confirmed
                        st.session_state["preflight"] = (request_data, result_data, stored_signature)
                        if mapping_ui_valid and mapping_confirmed and st.button("Submit guarded run to SLURM", type="primary", key="submit_run"):
                            try:
                                request, final_preflight = prepare_request(
                                    request_data["workflow_id"], Path(request_data["project_root"]),
                                    Path(request_data["annotation_csv"]), Path(request_data["modality_dir"]),
                                    Path(request_data["output_root"]), request_data["parameters"],
                                )
                                render_preflight(final_preflight)
                                if not final_preflight.ready:
                                    st.error("Final preflight is blocked with the confirmed specimen mapping; no SLURM job was submitted.")
                                else:
                                    job_id = submit_request(request, final_preflight)
                                    st.session_state["active_run_dir"] = str(request.run_dir)
                                    st.session_state["view_run_directory"] = str(request.run_dir)
                                    st.session_state.pop("preflight", None)
                                    st.success(f"Submitted SLURM Python worker job {job_id}; run ID {request.run_id}")
                            except Exception as exc:
                                st.error(f"Submission failed: {exc}")

    elif page == "Run & Reports":
        render_page_hero(
            "Run & Reports",
            "Inspect run status, stage tables, provenance, bounded previews, and saved artifacts.",
            ["Stage status", "Provenance", "Bounded previews"],
        )
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Run status")
            run_dir_text = st.text_input(
                "Run directory", value=st.session_state.get("active_run_dir", ""),
                key="view_run_directory",
            )
            st.caption("The active run location is retained across workspace navigation.")
        with right:
            if run_dir_text:
                run_dir = Path(run_dir_text).expanduser()
                if run_dir.is_dir():
                    display_run(run_dir)
                else:
                    st.caption("Run directory not found yet")
            else:
                st.info("Submit a guarded run or enter an existing run directory to view status and provenance.")


main()
