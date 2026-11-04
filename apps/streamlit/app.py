from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
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
)
from spatial_omics_toolkit.application.service import prepare_request, submit_request
from spatial_omics_toolkit.application.workflows.registry import list_workflows
from spatial_omics_toolkit.application.examples import ExampleSpec


st.set_page_config(page_title="Spatial Toolkit", layout="wide", initial_sidebar_state="collapsed")


def child_directories(path_text: str) -> list[str]:
    path = Path(path_text).expanduser()
    if not path.is_dir():
        return []
    try:
        return [str(item) for item in sorted(path.iterdir()) if item.is_dir()]
    except OSError:
        return []


def _restore_widget_state(widget_key: str, state_key: str, default: Any = "") -> None:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(state_key, default)


def _persist_widget_state(widget_key: str, state_key: str) -> None:
    st.session_state[state_key] = st.session_state.get(widget_key, "")

def _set_modality_section(modality: str, section: str) -> None:
    st.session_state[f"workspace_section_state_{modality}"] = section
    st.session_state[f"workspace_section_{modality}"] = section


def _submit_example_from_ui(modality: str) -> None:
    from spatial_omics_toolkit.application.examples import submit_example

    try:
        request, job_id = submit_example(
            modality,
            Path(str(st.session_state[f"example_output_{modality}"])).expanduser(),
            authorize_technical_qc=True,
        )
        active = dict(st.session_state.get("active_example_run_dirs", {}))
        selected = dict(st.session_state.get("selected_example_run_dirs", {}))
        active[modality] = str(request.run_dir)
        selected[modality] = str(request.run_dir)
        st.session_state["active_example_run_dirs"] = active
        st.session_state["selected_example_run_dirs"] = selected
        st.session_state[f"pending_example_run_selection_{modality}"] = str(request.run_dir)
        st.session_state[f"active_example_workflow_{modality}"] = request.workflow_id
        st.session_state[f"example_submission_notice_{modality}"] = f"Example submitted ({job_id}). Follow progress and results in this workspace."
        st.session_state.pop(f"example_submission_error_{modality}", None)
        _set_modality_section(modality, "Run")
    except Exception as exc:
        st.session_state[f"example_submission_error_{modality}"] = f"Example submission failed: {exc}"
        st.session_state.pop(f"example_submission_notice_{modality}", None)



def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def open_server_path_picker(kind: str, target_key: str, initial_path: str) -> None:
    """Open the picker without changing its destination until the user chooses."""
    allowed_root = _normalized_path(Path(ALLOWED_ROOT))
    current = _normalized_path(Path(initial_path))
    if kind in {"file", "czi"} and current.is_file():
        current = current.parent
    if not current.is_dir() or not lexical_path_is_under(current, allowed_root):
        current = allowed_root
    st.session_state["_server_picker_kind"] = kind
    st.session_state["_server_picker_target"] = target_key
    st.session_state[f"_server_picker_directory_{target_key}"] = str(current)
    st.session_state[f"_server_picker_path_{target_key}"] = str(current)
    st.session_state[f"_server_picker_pending_{target_key}"] = str(st.session_state.get(target_key, ""))
    st.session_state["_server_picker_open"] = True
    st.rerun()


def _close_server_path_picker() -> None:
    st.session_state["_server_picker_open"] = False


@st.dialog("Choose a folder or file", on_dismiss=_close_server_path_picker)
def server_path_picker_dialog() -> None:
    """Browse server paths under the institute workspace without uploading data."""
    allowed_root = _normalized_path(Path(ALLOWED_ROOT))
    kind = str(st.session_state.get("_server_picker_kind", "directory"))
    target_key = str(st.session_state.get("_server_picker_target", ""))
    browse_key = f"_server_picker_directory_{target_key}"
    current = _normalized_path(Path(str(st.session_state.get(browse_key, allowed_root))))
    if not current.is_dir() or not lexical_path_is_under(current, allowed_root):
        current = allowed_root
        st.session_state[browse_key] = str(current)
    path_key = f"_server_picker_path_{target_key}"
    pending_key = f"_server_picker_pending_{target_key}"
    navigation_key = f"_server_picker_next_path_{target_key}"
    if navigation_key in st.session_state:
        st.session_state[path_key] = str(st.session_state.pop(navigation_key))
    if target_key == "project_root_path":
        st.info("Choose the main folder for your study. Keep the reviewed sample sheet here and select the modality-specific data in its workspace. Your original files will not be moved or overwritten.")
    elif target_key == "annotation_csv":
        st.info("Choose your completed sample-sheet CSV. It needs biological_unit_id, specimen_id, condition, and tissue_source columns, with one row per physical specimen or section. TMA group_key values belong in the separate core map, not this sample sheet.")
    elif target_key.startswith("tma_core_map_path_"):
        st.info("Choose a reviewed core map linking the app's acquisition group_key to specimen_id in your sample sheet. Obtain the exact group keys by running Check inputs first. Optional slide_id, core_id, and section_id columns describe the TMA layout; they do not define independent biological replicates.")
    elif kind == "czi":
        st.info("Choose one native .czi source image. Its metadata records channel information and physical scale; sources remain read-only.")
    elif target_key.startswith("modality_folder_"):
        source_modality = target_key.removeprefix("modality_folder_")
        input_guidance = {
            "CycIF": "For raw images, choose a folder with one subfolder per slide containing its CZI acquisition rounds. For cell-feature analysis, choose the folder containing your segmented-cell table. The selected analysis type determines which inputs are read.",
            "HS": "Choose the folder containing the original hyperspectral CZI files. Native-file QC reads one selected CZI; the clinical workflow additionally requires its reviewed acquisition manifests.",
            "FLIM": "Choose the folder containing the original .sdt measurements. Raw count/decay QC does not require a lifetime calibration; full analysis also needs reviewed blocks and documented acquisition settings.",
            "H&E": "Choose a folder containing .svs, .czi, .tif, or .tiff tissue images. Whole-slide files and extracted core images are accepted for image intake/QC; loading a whole TMA slide does not automatically detect its cores.",
            "Visium": "Choose a Space Ranger outs folder, or a folder containing those bundles. The counts matrix and spatial positions must stay together. FASTQ files or a tissue image alone are not sufficient.",
        }
        st.info(input_guidance.get(source_modality, "Choose the folder containing the original acquisition files for this modality."))
        st.caption("The app reads these inputs and writes results to a separate output folder.")
    else:
        st.info("Choose the requested folder or CSV file. Sources are read-only; analysis outputs go to the configured output location.")
    st.caption(f"Current folder: `{current}`")
    st.caption("Browse locations under the workspace, or paste an existing path. Nothing is uploaded or changed.")
    path_text = st.text_input("Location", key=path_key)
    typed_path = _normalized_path(Path(path_text))
    if st.button("Open entered path", key=f"_server_picker_go_{target_key}"):
        valid_suffix = ".csv" if kind == "file" else ".czi"
        if not lexical_path_is_under(typed_path, allowed_root):
            st.error("Choose a path under the institute workspace.")
        elif kind in {"file", "czi"} and typed_path.is_file() and typed_path.suffix.lower() == valid_suffix:
            st.session_state[target_key] = str(typed_path)
            st.session_state[pending_key] = str(typed_path)
            st.session_state["_server_picker_open"] = False
            st.rerun()
        elif typed_path.is_dir():
            st.session_state[browse_key] = str(typed_path)
            st.session_state[navigation_key] = str(typed_path)
            st.rerun()
        else:
            expected = "a .czi file" if kind == "czi" else "a .csv file" if kind == "file" else "a directory"
            st.error(f"Enter an existing directory or {expected}.")
    if current != allowed_root and st.button("Parent folder", key=f"_server_picker_parent_{target_key}"):
        parent = str(current.parent)
        st.session_state[browse_key] = parent
        st.session_state[navigation_key] = parent
        st.rerun()
    show_hidden = st.checkbox("Show hidden folders", key="_server_picker_show_hidden", value=False)
    with st.container(height=320, border=True):
        directories = [
            path for path in child_directories(str(current))
            if show_hidden or not Path(path).name.startswith(".")
        ]
        if not directories:
            st.caption("No subfolders here.")
        for directory_text in directories:
            directory = Path(directory_text)
            if st.button(
                f"Open {directory.name}",
                key=f"_server_picker_open_{target_key}_{confirmation_key(str(directory))}",
                use_container_width=True,
            ):
                st.session_state[browse_key] = str(directory)
                st.session_state[navigation_key] = str(directory)
                st.rerun()
        if kind in {"file", "czi"}:
            suffix = ".czi" if kind == "czi" else ".csv"
            files = sorted(
                path for path in current.glob(f"*{suffix}")
                if path.is_file() and lexical_path_is_under(path, allowed_root)
            )
            for file_path in files:
                if st.button(
                    f"Choose {file_path.name}",
                    key=f"_server_picker_file_{target_key}_{confirmation_key(str(file_path))}",
                    use_container_width=True,
                ):
                    st.session_state[target_key] = str(file_path)
                    st.session_state["_server_picker_open"] = False
                    st.rerun()
    pending = str(st.session_state.get(pending_key, ""))
    if pending:
        st.caption(f"Selection: {pending}")
    if kind == "directory" and st.button("Choose this folder", key=f"_server_picker_choose_{target_key}"):
        st.session_state[pending_key] = str(current)
        st.session_state[target_key] = str(current)
        st.session_state["_server_picker_open"] = False
        st.rerun()
    cancel_col, _ = st.columns([1, 3])
    with cancel_col:
        if st.button("Cancel", key=f"_server_picker_cancel_{target_key}"):
            st.session_state["_server_picker_open"] = False
            st.rerun()
def project_samples_template() -> str:
    """Return a header-only project metadata template; never invent sample rows."""
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow((*PROJECT_SAMPLE_COLUMNS, "batch_or_block", "section_id"))
    return output.getvalue()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def confirmation_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def _signature_parameters(parameters: dict[str, Any]) -> str:
    snapshot = copy.deepcopy(parameters)
    core_map = snapshot.get("tma_core_map")
    if isinstance(core_map, dict) and core_map.get("path"):
        _, current_hash, errors = read_tma_core_map(str(core_map["path"]))
        core_map["current_file_sha256"] = current_hash if not errors else "unavailable"
    return json.dumps(snapshot, sort_keys=True, default=str)


def _example_job_pending(modality: str) -> bool:
    active_examples = st.session_state.get("active_example_run_dirs", {})
    run_text = active_examples.get(modality) if isinstance(active_examples, dict) else None
    if not isinstance(run_text, str) or not run_text:
        return False
    status = read_json(Path(run_text) / "status.json").get("status")
    return status in {"SUBMITTED", "QUEUED", "RUNNING"}


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


def _merge_missing(current: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Fill absent configuration from a parent without replacing entered values."""
    merged = dict(current)
    for key, value in source.items():
        existing = merged.get(key)
        if key not in merged:
            merged[key] = value
        elif isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_missing(existing, value)
    return merged

def read_tma_core_map(path_text: str) -> tuple[list[dict[str, str]], str, list[str]]:
    required = {"group_key", "specimen_id"}
    optional = {"slide_id", "core_id", "section_id"}
    path = _normalized_path(Path(path_text))
    if not path_text.strip():
        return [], "", []
    if not lexical_path_is_under(path, _normalized_path(Path(ALLOWED_ROOT))):
        return [], "", ["Choose a core-map CSV within the project workspace."]
    try:
        if path.stat().st_size > 1024 * 1024:
            return [], "", ["Core-map CSV exceeds the 1 MiB limit; choose the mapping table, not image or measurement data."]
        payload = path.read_bytes()
        reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")))
        fieldnames = reader.fieldnames or []
        headers = set(fieldnames)
        errors = []
        if len(fieldnames) != len(headers):
            errors.append("Core-map CSV column names must be unique.")
        if not required.issubset(headers):
            errors.append("Core-map CSV requires group_key and specimen_id columns.")
        rows = [
            {key: (row.get(key) or "").strip() for key in required | optional if key in headers}
            for row in reader
        ]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], "", [f"Unable to read core-map CSV: {exc}"]
    if not rows:
        errors.append("Core-map CSV must contain at least one reviewed mapping row.")
    group_keys = [row.get("group_key", "") for row in rows]
    specimen_ids = [row.get("specimen_id", "") for row in rows]
    if any(not value for value in group_keys) or len(group_keys) != len(set(group_keys)):
        errors.append("Core-map group_key values must be non-empty and unique.")
    if any(not value for value in specimen_ids):
        errors.append("Core-map specimen_id values must be non-empty.")
    return rows, hashlib.sha256(payload).hexdigest(), errors


def sample_map_template() -> str:
    return "group_key,specimen_id,slide_id,core_id,section_id\n"
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
        if field.name != "intake.expected_rounds":
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
        if len(choices) == 1:
            return choices[0], None
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
        if field.name == "intake.expected_rounds" and isinstance(default, (list, tuple)):
            choices = list(dict.fromkeys([*choices, *default]))
        selected_values = st.multiselect(
            label, choices, default=default or [], help=field.help_text, key=widget_key,
            accept_new_options=field.name == "intake.expected_rounds",
        )
        if field.required and not selected_values:
            return selected_values, f"{field.label} requires at least one explicit selection"
        return selected_values, None
    if field.kind in {"path", "config_file", "text"}:
        initial_text = str(default) if default is not None else ""
        if field.kind == "config_file":
            st.caption("Server-side configuration file path; browser uploads are disabled.")
        if workflow_id == "integration.serial_section_alignment":
            if st.button(f"Browse {field.label}…", key=f"browse_{widget_key}"):
                open_server_path_picker(
                    "czi", widget_key,
                    str(st.session_state.get("project_root_path", "")),
                )
        entered = st.text_input(label, value=initial_text, help=field.help_text, key=widget_key)
        if field.required and not entered.strip():
            return None, f"{field.label} is required"
        return entered.strip() or None, None
    return None, f"Unsupported parameter control kind: {field.kind}"


def _stage_parameters_enabled(
    descriptor: WorkflowDescriptor, field: ParameterField, parameters: dict[str, Any],
) -> bool:
    stage_id = field.stage_id
    if not stage_id:
        return True
    if descriptor.workflow_id == "cycif-raw-czi":
        activation = {
            "conversion": "conversion.enabled",
            "registration": "registration.enabled",
            "export": "registration.enabled",
            "segmentation": "segmentation.enabled",
        }
        flag = activation.get(stage_id)
        if flag is None:
            return True
        flag_value = _get_nested(parameters, flag)
        if flag_value is None:
            flag_value = next(
                (item.default for item in descriptor.parameter_fields if item.name == flag),
                False,
            )
        return flag_value is True
    if descriptor.modality == "FLIM":
        if stage_id == "flim_filter":
            return _get_nested(parameters, "filter_enabled") is True
        if stage_id == "flim_reference_calibration":
            return bool(str(_get_nested(parameters, "reference_calibration.path") or "").strip())
    if descriptor.workflow_id == "cycif.tabular_analysis":
        config_names = {
            "step06": "qc",
            "step07": "classification",
            "step08": "programs",
            "step09": "aggregate",
            "step10": "spatial_features",
            "step11": "spatial_null",
            "step12": "compare_groups",
        }
        field_name = config_names.get(stage_id)
        if field_name:
            return bool(str(_get_nested(parameters, field_name) or "").strip())
    return True


def validate_saved_parameters(
    descriptor: WorkflowDescriptor, parameters: dict[str, Any], discovered: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for field in descriptor.parameter_fields:
        if not _stage_parameters_enabled(descriptor, field, parameters):
            continue
        choices = field.choices
        if field.dynamic_choices_key:
            dynamic_choices = _get_nested(discovered, field.dynamic_choices_key)
            if not isinstance(dynamic_choices, list):
                continue
            choices = dynamic_choices
        value = _get_nested(parameters, field.name)
        if value is None:
            value = field.default
        missing = value is None
        if field.kind in {"text", "path", "config_file"}:
            missing = not str(value or "").strip()
        elif field.kind == "multiselect":
            missing = not isinstance(value, list) or not value
        elif field.kind == "select":
            missing = value is None or value == "(Select a value)"
        if field.required and missing:
            errors.append(f"{field.label} is required")
            continue
        if value is None:
            continue
        if field.kind in {"int", "float"}:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field.label} must be numeric")
                continue
            if field.min_value is not None and numeric_value < field.min_value:
                errors.append(f"{field.label} must be at least {field.min_value}")
            if field.max_value is not None and numeric_value > field.max_value:
                errors.append(f"{field.label} must be at most {field.max_value}")
        elif field.kind == "select" and choices and value not in choices:
            errors.append(f"{field.label} must be one of the available options")
        elif field.kind == "multiselect" and isinstance(value, list) and field.name != "intake.expected_rounds":
            if choices and any(choice not in choices for choice in value):
                errors.append(f"{field.label} contains an unavailable option")
    return errors


def render_parameter_fields(
    descriptor: WorkflowDescriptor,
    discovered: dict[str, Any],
    fields: list[ParameterField] | None = None,
    initial: dict[str, Any] | None = None,
    key_prefix: str = "",
    *,
    show_errors: bool = True,
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
        context = st.expander(f"Advanced — {section}", expanded=False) if advanced else st.container()
        with context:
            if not advanced:
                st.markdown(f"**{section}**")
            for field in section_fields:
                value, error = _render_parameter_field(field, descriptor.workflow_id, discovered, initial or {}, key_prefix)
                if value is not None:
                    _set_nested(parameters, field.name, value)
                else:
                    dynamic_choices = _get_nested(discovered, field.dynamic_choices_key) if field.dynamic_choices_key else None
                    dynamic_unavailable = field.dynamic_choices_key is not None and not isinstance(dynamic_choices, list)
                    available_choices = dynamic_choices if isinstance(dynamic_choices, list) else field.choices
                    no_choices = field.kind in {"select", "multiselect"} and not available_choices
                    if not dynamic_unavailable and not no_choices:
                        _set_nested(parameters, field.name, None)
                current_parameters = _merge_missing(parameters, initial or {})
                if error and _stage_parameters_enabled(descriptor, field, current_parameters):
                    errors.append(error)
                    if show_errors:
                        st.error(error)
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
def cycif_parent_runs(output_root: Path) -> list[Path]:
    run_root = output_root / "CycIF" / "runs"
    try:
        candidates = [path for path in run_root.iterdir() if path.is_dir() and (path / "run.json").is_file()]
    except OSError:
        return []
    return sorted(candidates, key=lambda path: path.name, reverse=True)


def cycif_checkpoint_boundaries(run_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Return boundaries backed by valid checkpoint receipts and successful stage events."""
    boundaries = {
        "conversion": ("intake",),
        "registration": ("intake",),
        "export": ("intake", "registration"),
        "segmentation": ("intake", "registration", "export"),
    }
    reasons: dict[str, str] = {}
    metadata = read_json(run_dir / "run.json")
    receipts = read_json(run_dir / "checkpoint_receipts.json")
    checkpoints = receipts.get("checkpoints", {})
    if metadata.get("workflow_id") != "cycif-raw-czi" or receipts.get("workflow_id") != "cycif-raw-czi":
        return [], {boundary: "Parent run is not a CyCIF raw-CZI checkpoint bundle." for boundary in boundaries}
    _, events, _ = run_files(run_dir)
    latest_status: dict[str, str] = {}
    for event in events:
        stage_id = event.get("stage_id")
        if isinstance(stage_id, str):
            latest_status[stage_id] = str(event.get("status", ""))
    eligible: list[str] = []
    for boundary, required_stages in boundaries.items():
        invalid: str | None = None
        for stage_id in required_stages:
            checkpoint = checkpoints.get(stage_id) if isinstance(checkpoints, dict) else None
            if not isinstance(checkpoint, dict) or checkpoint.get("workflow_id") != "cycif-raw-czi" or checkpoint.get("stage_id") != stage_id:
                invalid = f"Missing valid {stage_id} checkpoint receipt."
                break
            latest_stage_status = latest_status.get(stage_id, "missing")
            if latest_stage_status == "REUSED":
                latest_stage_status = str(checkpoint.get("stage_status", latest_stage_status))
            if latest_stage_status not in {"SUCCEEDED", "WARNING", "REUSED"}:
                invalid = f"{stage_id} latest checkpoint event is not terminal-successful or REUSED."
                break
            settings = checkpoint.get("settings")
            settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
            if not isinstance(settings, dict) or checkpoint.get("settings_sha256") != settings_hash:
                invalid = f"{stage_id} checkpoint settings receipt is missing or invalid."
                break
            from spatial_omics_toolkit.application.workflows.cycif import _tool_identity
            if checkpoint.get("tool_identity") != _tool_identity(stage_id):
                invalid = f"{stage_id} checkpoint runtime/tool identity differs from this application."
                break
            artifacts = checkpoint.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                invalid = f"{stage_id} checkpoint has no artifact hashes."
                break
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    invalid = f"{stage_id} checkpoint contains an invalid artifact receipt."
                    break
                artifact_path = Path(str(artifact.get("path", ""))).expanduser()
                digest = artifact.get("sha256")
                if not artifact_path.is_file() or not isinstance(digest, str) or len(digest) != 64:
                    invalid = f"{stage_id} checkpoint is missing an artifact or checksum: {artifact_path}."
                    break
                # Full content hashes run once in the allocated worker before reuse.
            if invalid:
                break
        if invalid:
            reasons[boundary] = invalid
        else:
            eligible.append(boundary)
    return eligible, reasons


def render_preflight(result: PreflightResult) -> None:
    if result.blockers:
        for blocker in result.blockers:
            st.error(blocker)
    if result.warnings:
        with st.expander(f"Notes to review ({len(result.warnings)})", expanded=False):
            for warning in result.warnings:
                st.warning(warning)
    if result.ready:
        st.success("Input checks passed.")
    with st.expander("Technical check details", expanded=False):
        st.json(result.resolved_parameters)




def _initialize_state() -> None:
    for legacy_key in (
        "active_workflow_id", "active_workflow_modality", "active_run_dir",
        "view_run_directory", "preflight", "selected_modality_folder",
    ):
        st.session_state.pop(legacy_key, None)
    defaults = {
        "project_root_path": "/home/groups/EmiliLab/pauli/",
        "output_root": "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_toolkit/runs",
        "annotation_csv": "",
        "active_run_dirs": {},
        "selected_run_dirs": {},
    }
    for key, value in defaults.items():
        if not str(st.session_state.get(key, "")).strip() and key not in {"active_run_dirs", "selected_run_dirs"}:
            st.session_state[key] = value
    for key in ("active_run_dirs", "selected_run_dirs"):
        if not isinstance(st.session_state.get(key), dict):
            st.session_state[key] = {}
    root = Path(str(st.session_state["project_root_path"])).expanduser()
    if not st.session_state.get("annotation_csv"):
        candidate = root / "project_samples.csv"
        st.session_state["annotation_csv"] = str(candidate) if candidate.is_file() else ""
    for modality in ("CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"):
        path_key = f"modality_folder_{modality}"
        if path_key not in st.session_state:
            st.session_state[path_key] = ""
        st.session_state[f"data_layout_{modality}"] = st.session_state.get(f"data_layout_{modality}", "Single sample")
        st.session_state[f"modality_output_{modality}"] = str(
            st.session_state.get(f"modality_output_{modality}") or st.session_state["output_root"]
        )
        st.session_state[f"example_output_{modality}"] = str(
            st.session_state.get(f"example_output_{modality}")
            or (Path(__file__).resolve().parents[2] / "runs" / "examples")
        )



def _render_example_description(example: ExampleSpec, *, compact: bool = False) -> None:
    introductions = {
        "CycIF": "Try one fluorescence-image acquisition from an extracted TMA core. This short example checks the file and acquisition naming, then builds an intake QC report. It does not run the full registration or cell-segmentation pipeline.",
        "HS": "Try a hyperspectral image, where each pixel contains measurements across wavelength bands. The example produces a total-intensity image and spectral-pattern diagrams called phasor plots.",
        "FLIM": "Try a measurement file from a fluorescence-lifetime microscope (.sdt). The example inspects every recorded block and plots photon counts and decay curves. It does not assign detector identities or calculate lifetimes without acquisition and calibration information.",
        "H&E": "Try a stained-tissue whole-slide image (.svs). The example reads its image metadata and creates a small thumbnail you can explore without loading or changing the full-resolution slide.",
        "Visium": "Try a small public Visium HD developer dataset. The example combines measured gene counts with their recorded spatial positions to produce count QC and a spatial preview. This downsampled teaching dataset is not intended for biological conclusions.",
        "Integration": "Try comparing an H&E image with the DAPI fluorescence channel of a candidate neighboring section. The example uses the recorded image scales to estimate a translation and produces a before-and-after overlay for inspection.",
    }
    st.markdown(f"**{example.title}**")
    st.write(introductions.get(example.modality, example.summary))
    if not compact:
        st.write("The input files and settings are already selected. You do not need to fill in Project or upload anything to try this example.")
        st.markdown("**What to do:** click **Run example**, follow its progress in **Run**, then open **Quality control** and **Results**. Use **Files** to download the recorded outputs.")
        if example.expected_outputs:
            st.markdown("**What you will get**")
            for output in example.expected_outputs:
                st.markdown(f"- {output}")
    with st.expander("About the example data and its limits", expanded=False):
        st.write(example.input_help)
        st.caption(f"File organization: {example.layout}")
        for source in example.source_paths:
            st.code(source.name, language=None)
        for limitation in example.limitations:
            st.write(limitation)
        st.caption("This example creates a separate run. It does not change your project metadata, own-data settings, or original files.")
    if example.modality == "Integration":
        st.warning("Run example confirms these displayed candidate source files for an exploratory, translation-only comparison. It does not confirm biological pairing, same-cell correspondence, or acceptance of the resulting alignment. Inspect the overlay before drawing any conclusion.")


def _render_project() -> None:
    st.title("Project")
    st.write("Set up your study folder, sample sheet and results location; examples do not require project setup.")
    with st.container(border=True, key="project_step_folder"):
        st.subheader("1. Choose the study folder")
        st.write("Choose the folder containing your study's data and sample sheet. The app reads your originals; it will not move, rename, or overwrite them.")
        if st.button("Browse project folder…", key="browse_project_root"):
            open_server_path_picker("directory", "project_root_path", str(st.session_state["project_root_path"]))
        root_widget = "project_root_widget"
        _restore_widget_state(root_widget, "project_root_path")
        st.session_state[root_widget] = str(st.session_state.get("project_root_path", ""))
        root_text = st.text_input(
            "Project root directory", key=root_widget,
            on_change=_persist_widget_state, args=(root_widget, "project_root_path"),
        )
        root = Path(root_text).expanduser()
        current_metadata = str(st.session_state.get("annotation_csv", ""))
        default_path = root / "project_samples.csv"
        old_default = str(Path(st.session_state.get("_annotation_default_root", root_text)) / "project_samples.csv")
        if not current_metadata or current_metadata == old_default:
            current_metadata = str(default_path) if default_path.is_file() else ""
            st.session_state["annotation_csv"] = current_metadata
        st.session_state["_annotation_default_root"] = str(root)
        metadata_widget = "annotation_csv_widget"
        st.session_state[metadata_widget] = current_metadata
    with st.container(border=True, key="project_step_samples"):
        st.subheader("2. Add the sample sheet")
        st.write("This CSV tells the app which physical specimens and independent biological units are represented. Download the blank template, fill it with reviewed study identifiers, and select the completed file.")
        with st.expander("How to fill in the sample sheet", expanded=False):
            st.markdown("- **specimen_id:** a unique physical specimen or section.\n- **biological_unit_id:** the independent patient, animal, or other biological unit. Reuse it for cores or sections from the same unit.\n- **condition:** the study group or treatment supplied with your data.\n- **tissue_source:** the tissue or source material supplied with your data.")
            st.write("For a TMA, several cores may belong to one biological unit. The core map is supplied in that modality's Data & settings; it links acquisitions to this sample sheet without treating cores as independent replicates.")
            st.caption("Use the literal unknown only when a source value is genuinely unavailable. Unmapped acquisitions can be used for technical QC, not biological comparisons.")
        if st.button("Browse metadata…", key="browse_project_metadata"):
            open_server_path_picker("file", "annotation_csv", root_text)
        st.text_input(
            "Project sample metadata (CSV)", key=metadata_widget,
            on_change=_persist_widget_state, args=(metadata_widget, "annotation_csv"),
        )
        st.download_button(
            "Download metadata template", data=project_samples_template(),
            file_name="project_samples.csv", mime="text/csv",
        )
        st.markdown("**Sample sheet review**")
        annotation = Path(str(st.session_state.get("annotation_csv", ""))).expanduser()
        if annotation.is_file():
            rows, errors = read_project_samples(annotation)
            for error in errors:
                st.error(error)
            if rows and not errors:
                st.success(f"Sample-sheet structure checked: {len(rows)} row(s).")
                st.caption("This check verifies the required fields, not the biological identities. Confirm those against your study records before assigning inputs to specimens.")
                st.dataframe(rows[:200], use_container_width=True, hide_index=True)
        else:
            st.info("No sample sheet is selected yet. Download the template, add your study's information, and choose the completed CSV. You can still run every included example without doing this.")
    with st.container(border=True, key="project_step_results"):
        st.subheader("3. Choose where results are saved")
        st.write("Choose a results folder outside the raw input folder. Each run gets a new subfolder, so earlier results remain intact. Each modality can keep its own output location.")
        output_widget = "project_output_root_widget"
        _restore_widget_state(output_widget, "output_root")
        st.text_input(
            "Default output root", key=output_widget,
            on_change=_persist_widget_state, args=(output_widget, "output_root"),
        )

def _modality_run_request(
    descriptor: WorkflowDescriptor, parameters: dict[str, Any], modality: str,
) -> tuple[RunRequest, PreflightResult]:
    project_root = Path(str(st.session_state["project_root_path"])).expanduser()
    modality_dir = Path(str(st.session_state.get(f"modality_folder_{modality}", ""))).expanduser()
    if descriptor.workflow_id == "integration.serial_section_alignment":
        source_text = str(parameters.get("moving_he_path", "")).strip()
        source_path = Path(source_text).expanduser()
        if source_text and not source_path.is_absolute():
            source_path = project_root / source_path
        modality_dir = source_path.resolve().parent if source_text else project_root
    return prepare_request(
        descriptor.workflow_id,
        project_root,
        Path(str(st.session_state.get("annotation_csv", ""))).expanduser(),
        modality_dir,
        Path(str(st.session_state[f"modality_output_{modality}"])).expanduser(),
        parameters,
    )


def _render_mapping(
    descriptor: WorkflowDescriptor, request_data: dict[str, Any], result_data: dict[str, Any],
) -> tuple[bool, bool]:
    """Render explicit identity or QC-only confirmations for a fresh preflight."""
    resolved = result_data.get("resolved_parameters", {})
    groups = resolved.get("acquisition_groups", []) if isinstance(resolved, dict) else []
    valid = isinstance(groups, list)
    confirmed = False
    specimen_mapping: dict[str, str] = {}
    core_mapping: dict[str, str] = {}
    use_specimen_column = False
    samples, sample_errors = read_project_samples(Path(request_data["annotation_csv"]))
    for error in sample_errors:
        st.error(error)
        valid = False
    specimen_ids = list(dict.fromkeys(
        (row.get("specimen_id") or "").strip() for row in samples
        if (row.get("specimen_id") or "").strip()
    ))
    tabular_cycif = descriptor.workflow_id == "cycif.tabular_analysis"
    manual = True
    metadata_hash = str(resolved.get("annotation_csv_sha256", "")) if isinstance(resolved, dict) else ""
    identity_key = confirmation_key(request_data["workflow_id"], request_data["modality_dir"], metadata_hash)
    parameters = request_data.get("parameters", {})
    tma_map = parameters.get("tma_core_map", {}) if isinstance(parameters, dict) else {}
    if parameters.get("data_layout") == "Tissue microarray (TMA)" and isinstance(groups, list) and groups:
        st.write("Match each detected acquisition to a specimen in your sample sheet. You may choose assignments below, or download this map, fill in the reviewed specimen IDs, and load it in Data & settings.")
        map_output = StringIO()
        map_writer = csv.writer(map_output)
        map_writer.writerow(("group_key", "specimen_id", "slide_id", "core_id", "section_id"))
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("group_key"), str):
                map_writer.writerow((group["group_key"], "UNMAPPED", "", "", ""))
        st.download_button(
            "Download core map for these inputs", data=map_output.getvalue(),
            file_name="core_map_for_review.csv", mime="text/csv",
            key=f"discovered_core_map_{descriptor.modality}",
        )
    if (
        isinstance(tma_map, dict)
        and parameters.get("data_layout") == "Tissue microarray (TMA)"
        and tma_map.get("path")
    ):
        _, current_hash, map_errors = read_tma_core_map(str(tma_map["path"]))
        discovered_keys = {
            item.get("group_key") for item in groups
            if isinstance(item, dict) and isinstance(item.get("group_key"), str)
        }
        imported_keys = {
            row.get("group_key") for row in tma_map.get("rows", [])
            if isinstance(row, dict)
        }
        if map_errors or current_hash != tma_map.get("sha256"):
            st.error("Core-map CSV changed or is no longer valid since setup. Re-select and review the file, then check inputs again.")
            valid = False
        if imported_keys != discovered_keys:
            st.error("Core-map group_key values must match the discovered acquisition groups exactly. Correct the CSV and check inputs again.")
            valid = False
    if descriptor.workflow_id == "integration.serial_section_alignment":
        st.info("The moving H&E and reference fluorescence/HS files remain UNMAPPED. Choose the two source files and explicitly confirm them in Data & settings; this confirms candidate inputs for exploratory QC only, not biological pairing or an accepted alignment.")
        confirmed = parameters.get("candidate_sources_confirmed") is True
        if len(groups) != 2:
            st.error("Integration requires exactly the two explicitly selected source files; identity mapping is not performed.")
            valid = False
        specimen_mapping = {
            group.get("group_key"): "UNMAPPED"
            for group in groups
            if isinstance(group, dict) and isinstance(group.get("group_key"), str)
        }
    elif tabular_cycif and bool(resolved.get("has_specimen_id")):
        source = st.radio(
            "Feature-table mapping source",
            ["Use existing specimen_id column", "Map exact core values to project specimen_id"],
            key=f"cycif_mapping_source_{identity_key}",
        )
        manual = source == "Map exact core values to project specimen_id"
    if descriptor.workflow_id == "integration.serial_section_alignment":
        pass
    elif tabular_cycif and not manual and bool(resolved.get("has_specimen_id")):
        confirmed = st.checkbox(
            "I confirm the existing specimen_id column is authoritative and matches project metadata.",
            key=f"cycif_existing_specimen_confirmed_{confirmation_key(request_data['workflow_id'], request_data['annotation_csv'], metadata_hash, *specimen_ids)}",
        )
        use_specimen_column = confirmed
        if not confirmed:
            st.info("Confirm the existing specimen_id column before starting this run.")
    elif groups:
        seen: set[str] = set()
        imported_rows = tma_map.get("rows", []) if isinstance(tma_map, dict) else []
        imported_by_group = {
            row.get("group_key"): row.get("specimen_id")
            for row in imported_rows if isinstance(row, dict)
        }
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("group_key"), str) or not group["group_key"].strip():
                st.error("An acquisition group has no valid group key; mapping is blocked.")
                valid = False
                continue
            key = group["group_key"]
            if key in seen:
                st.error(f"Duplicate acquisition group key: {key}")
                valid = False
                continue
            seen.add(key)
            paths = group.get("input_paths")
            if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
                st.error(f"Group {key} has invalid input paths.")
                valid = False
                paths = []
            columns = st.columns([2, 1])
            with columns[0]:
                st.markdown(f"**{group.get('label', key)}**")
                if parameters.get("data_layout") == "Tissue microarray (TMA)":
                    st.caption(f"Core-map group_key: `{key}`")
                with st.expander(f"Input paths · {group.get('label', key)}"):
                    for path in paths:
                        st.code(path, language=None)
            with columns[1]:
                options = ["UNMAPPED", *specimen_ids]
                imported_value = imported_by_group.get(key)
                selected_index = options.index(imported_value) if imported_value in options else 0
                specimen = st.selectbox(
                    "Project specimen",
                    options,
                    index=selected_index,
                    key=f"specimen_mapping_{request_data['modality']}_{confirmation_key(request_data['workflow_id'], key, str(tma_map.get('sha256', '')))}",
                )
            (core_mapping if tabular_cycif else specimen_mapping)[key] = specimen
        unmapped = [key for key, value in (core_mapping if tabular_cycif else specimen_mapping).items() if value == "UNMAPPED"]
        if unmapped:
            st.warning("Unmapped groups are eligible only for technical QC; specimen-level analysis remains blocked.")
        parts = [request_data["annotation_csv"], metadata_hash, str(tma_map.get("sha256", "")), *[f"{key}={value}" for key, value in sorted((core_mapping if tabular_cycif else specimen_mapping).items())]]
        confirmed = st.checkbox(
            "I confirm these explicit group-to-specimen mappings, including any QC-only unmapped groups.",
            key=f"specimen_mapping_confirmed_{request_data['modality']}_{confirmation_key(request_data['workflow_id'], *parts)}",
        )
    elif descriptor.workflow_id in {"he-image-intake", "cycif-raw-czi"}:
        st.info("No acquisition groups were reported. This is a bounded technical-QC/intake run only; no specimen identity or biological comparison will be inferred.")
        confirmed = st.checkbox(
            "I confirm this run is technical-QC-only and will not infer sample identity.",
            key=f"qc_only_confirmed_{request_data['modality']}_{identity_key}",
        )
    else:
        st.error("No acquisition groups were returned for explicit identity mapping. Submission is blocked.")
        valid = False
    parameters = request_data.setdefault("parameters", {})
    parameters.update({
        "specimen_mapping": specimen_mapping,
        "core_to_specimen": core_mapping,
        "use_specimen_id_column": use_specimen_column,
        "specimen_mapping_confirmed": confirmed,
    })
    resolved.update({
        "specimen_mapping": specimen_mapping,
        "core_to_specimen": core_mapping,
        "use_specimen_id_column": use_specimen_column,
        "specimen_mapping_confirmed": confirmed,
    })
    return valid, confirmed


def _acquisition_contract(result_data: dict[str, Any]) -> str | None:
    resolved = result_data.get("resolved_parameters")
    groups = resolved.get("acquisition_groups") if isinstance(resolved, dict) else None
    if not isinstance(groups, list):
        return None
    records: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        records.append({
            "group_key": group.get("group_key"),
            "label": group.get("label"),
            "input_paths": group.get("input_paths"),
        })
    records.sort(key=lambda item: str(item.get("group_key", "")))
    return json.dumps(records, sort_keys=True, default=str)


def _reviewed_preflight_changed(
    original_result: dict[str, Any], current_result: dict[str, Any],
) -> str | None:
    original_resolved = original_result.get("resolved_parameters", {})
    current_resolved = current_result.get("resolved_parameters", {})
    if not isinstance(original_resolved, dict) or not isinstance(current_resolved, dict):
        return "Preflight metadata is unavailable; run checks again."
    if "annotation_csv_sha256" in original_resolved and (
        current_resolved.get("annotation_csv_sha256") != original_resolved.get("annotation_csv_sha256")
    ):
        return "Project sample metadata changed after checks. Run checks again and review the new mapping."
    original_groups = _acquisition_contract(original_result)
    current_groups = _acquisition_contract(current_result)
    if original_groups is None or current_groups is None or original_groups != current_groups:
        return "The acquisition groups or source paths changed after checks. Run checks again before mapping."
    return None


def _render_cycif_continuation(descriptor: WorkflowDescriptor) -> dict[str, Any]:
    if descriptor.workflow_id != "cycif-raw-czi":
        return {}
    key = f"cycif_continuation_{confirmation_key(descriptor.workflow_id)}"
    state = st.session_state.get(key, {})
    if not isinstance(state, dict):
        state = {}
    mode = st.radio(
        "Run source", ["raw_start", "continue"],
        format_func=lambda value: "Start from raw data" if value == "raw_start" else "Continue from a verified run",
        horizontal=True, index=0 if state.get("mode", "raw_start") == "raw_start" else 1,
        key=f"cycif_mode_{confirmation_key(descriptor.workflow_id)}",
    )
    state["mode"] = mode
    state.setdefault("parent_run_dir", "")
    state.setdefault("start_stage", "")
    if mode == "continue":
        parents = cycif_parent_runs(Path(str(st.session_state[f"modality_output_{descriptor.modality}"])).expanduser())
        choices = ["(select a CycIF run)", *[str(path) for path in parents]]
        previous = state["parent_run_dir"]
        selected = st.selectbox(
            "Source run", choices, index=choices.index(previous) if previous in choices else 0,
            key=f"cycif_parent_{confirmation_key(descriptor.workflow_id)}",
        )
        state["parent_run_dir"] = "" if selected == choices[0] else selected
        eligible, reasons = cycif_checkpoint_boundaries(Path(selected)) if selected != choices[0] else ([], {})
        boundaries = ("conversion", "registration", "export", "segmentation")
        if eligible:
            prior = state.get("start_stage")
            state["start_stage"] = st.selectbox(
                "Continue at stage", eligible,
                index=eligible.index(prior) if prior in eligible else 0,
                key=f"cycif_boundary_{confirmation_key(descriptor.workflow_id, selected)}",
            )
        else:
            state["start_stage"] = ""
            st.info("No verified continuation boundary is available for this source run.")
        for boundary in boundaries:
            if boundary not in eligible:
                st.caption(f"{boundary}: {reasons.get(boundary, 'Requires verified source checkpoints.')}")
        if state["parent_run_dir"] and state["start_stage"]:
            parent = Path(state["parent_run_dir"])
            _, events, _ = run_files(parent)
            required = {
                "conversion": ("intake",), "registration": ("intake",),
                "export": ("intake", "registration"),
                "segmentation": ("intake", "registration", "export"),
            }.get(state["start_stage"], ())
            latest = {event.get("stage_id"): event for event in events}
            receipt = read_json(parent / "checkpoint_receipts.json").get("checkpoints", {})
            warning_ids: list[str] = []
            for stage_id in required:
                event = latest.get(stage_id, {})
                status = str(event.get("status", ""))
                checkpoint = receipt.get(stage_id, {}) if isinstance(receipt, dict) else {}
                if status == "REUSED" and isinstance(checkpoint, dict):
                    status = str(checkpoint.get("stage_status", status))
                if status == "WARNING":
                    warning_ids.append(stage_id)
            if warning_ids:
                st.warning(f"Source checkpoints contain QC warnings: {', '.join(warning_ids)}.")
                state["warning_review_confirmed"] = st.checkbox(
                    "I reviewed and accept reusing these warning checkpoints.",
                    value=bool(state.get("warning_review_confirmed", False)),
                    key=f"cycif_warning_review_{confirmation_key(descriptor.workflow_id, selected, state['start_stage'])}",
                )
            else:
                state["warning_review_confirmed"] = True
    else:
        state.update(parent_run_dir="", start_stage="", warning_review_confirmed=False)
    st.session_state[key] = state
    continuation = {"mode": mode}
    if mode == "continue":
        continuation.update({
            "parent_run_dir": state["parent_run_dir"],
            "start_stage": state["start_stage"],
            "warning_review_confirmed": state.get("warning_review_confirmed") is True,
        })
    parameters = st.session_state.setdefault(f"workflow_parameters_{descriptor.workflow_id}", {})
    parameters["continuation"] = continuation
    return continuation


def _render_modality(modality: str, descriptors: list[WorkflowDescriptor], *, read_only: bool = False) -> None:
    from spatial_omics_toolkit.application.ui.results import render_run, select_run
    from spatial_omics_toolkit.application.ui.presentation import MODALITY_LABELS, render_stage_flow
    from spatial_omics_toolkit.application.ui.implementation import render_step_sources

    label = MODALITY_LABELS.get(modality, modality)
    st.title(label)
    if not descriptors:
        st.info("No implemented workflow is currently available for this modality.")
        return
    example = None
    example_error = None
    example_descriptor = None
    if read_only:
        mode = "own"
    else:
        from spatial_omics_toolkit.application.examples import get_example

        try:
            example = get_example(modality)
            example_descriptor = next(
                (item for item in descriptors if item.workflow_id == example.workflow_id),
                None,
            )
            if example_descriptor is None:
                example_error = "The prepared example refers to a workflow that is not registered for this modality."
        except Exception as exc:
            example_error = str(exc)
        default_mode = "own" if example_error is not None else "example"
        mode_key = f"data_mode_{modality}"
        mode_state_key = f"{mode_key}_state"
        st.session_state.setdefault(mode_key, st.session_state.get(mode_state_key, default_mode))
        if st.session_state[mode_key] not in {"example", "own"}:
            st.session_state[mode_key] = default_mode
        mode = st.radio(
            "Input mode", ["example", "own"], horizontal=True, key=mode_key,
            format_func=lambda value: "Included example" if value == "example" else "Use my own data",
            on_change=_persist_widget_state, args=(mode_key, mode_state_key),
        )
        if example_error is not None:
            message = (
                f"Example unavailable: {example_error} "
                "Ask the toolkit administrator to prepare the file-backed example catalog and register its workflow; "
                "no placeholder data are used. Your own-data controls remain available."
            )
            if mode == "example":
                st.error(message)
                return
            st.info(message)
    workflow_key = f"selected_workflow_{confirmation_key(modality)}"
    workflow_state_key = f"selected_workflow_state_{modality}"
    if mode == "example":
        descriptor = example_descriptor
    else:
        titles = [item.title for item in descriptors]
        if workflow_key not in st.session_state or st.session_state[workflow_key] not in titles:
            saved_title = str(st.session_state.get(workflow_state_key, titles[0]))
            st.session_state[workflow_key] = saved_title if saved_title in titles else titles[0]
        if len(descriptors) > 1:
            short_titles = {
                "cycif-raw-czi": "Raw images",
                "cycif.tabular_analysis": "Cell-feature table",
            }
            title = st.selectbox(
                "Analysis type", titles, key=workflow_key,
                on_change=_persist_widget_state, args=(workflow_key, workflow_state_key),
                format_func=lambda value: next(
                    (short_titles[item.workflow_id] for item in descriptors
                     if item.title == value and item.workflow_id in short_titles),
                    value,
                ),
            )
        else:
            title = titles[0]
            st.session_state[workflow_key] = title
            st.session_state[workflow_state_key] = title
        descriptor = descriptors[titles.index(title)]
    if read_only:
        section = "Introduction"
    else:
        section_key = f"workspace_section_{modality}"
        section_state_key = f"workspace_section_state_{modality}"
        if section_key not in st.session_state:
            st.session_state[section_key] = str(st.session_state.get(section_state_key, "Introduction"))
        section = st.radio(
            "Workflow", ["Introduction", "Setup", "Run", "Quality control", "Results", "Files"],
            horizontal=True, key=section_key,
            on_change=_persist_widget_state, args=(section_key, section_state_key),
            format_func=lambda value: "Data & settings" if value == "Setup" else value,
            label_visibility="collapsed",
        )
    if section == "Introduction":
        st.subheader(descriptor.title)
        st.write(descriptor.description)
        st.subheader("Analysis stages")
        render_stage_flow(descriptor)
        st.subheader("Inputs")
        st.markdown(descriptor.input_description)
        st.subheader("Shared prerequisites for your own data")
        shared_inputs = [
            "Select an existing project folder and input folder in the approved workspace. "
            "The input folder can be outside the project folder.",
            "Place a reviewed `project_samples.csv` inside the project folder. It must contain at least one row "
            "and nonempty `biological_unit_id`, `specimen_id`, `condition`, and `tissue_source` values. "
            "Each `specimen_id` must be unique; biological-unit IDs may repeat across specimens.",
            "Choose an output folder in the approved workspace, outside the input folder. "
            "Select SLURM resources: 1–32 CPUs, 1–512 GB memory, and an 8h, 24h, 48h, 72h, or 168h time limit.",
        ]
        if descriptor.workflow_id == "cycif.tabular_analysis":
            shared_inputs.append(
                "Confirm specimen identities using either an existing specimen-ID column or a core-to-specimen map."
            )
        elif descriptor.workflow_id == "he-image-intake":
            shared_inputs.append("Image-to-specimen mapping is not required for this image-inventory workflow.")
        else:
            shared_inputs.append(
                "For every displayed image or acquisition group, confirm an exact sample-sheet `specimen_id`, "
                "or select `UNMAPPED` when its specimen is unknown. Unmapped groups are for technical QC only."
            )
        st.markdown("\n".join(f"- {item}" for item in shared_inputs))
        st.write(
            "These are requirements for submitted own-data runs. Included examples use their own fixed inputs "
            "and technical recipe; they do not require your project setup. Input checking does not guarantee "
            "successful execution or all described outputs."
        )
        st.subheader("Outputs")
        st.markdown(descriptor.output_description)
        st.subheader("Scope and limitations")
        if descriptor.limitations:
            st.markdown("\n".join(f"- {limitation}" for limitation in descriptor.limitations))
        else:
            st.caption("No additional workflow-specific limitations are declared.")
        render_step_sources(descriptor.workflow_id, key_prefix=f"introduction_{modality}")
        st.write(
            "This page describes the selected workflow. It does not check inputs, start a run, "
            "or change existing results."
        )
        return
    path_key = f"modality_folder_{modality}"
    if section == "Setup" and mode == "example":
        st.subheader("Start with the included example")
        _render_example_description(example)
        example_pending = _example_job_pending(modality)
        if example_pending:
            st.info("An example job is already submitted or running. Follow its status below before starting another.")
        st.button(
            "Run example", key=f"run_example_setup_{modality}", type="primary",
            disabled=example_pending,
            help="Submit the catalog's fixed technical recipe. This does not confirm biological pairing or accept candidate alignment.",
            on_click=_submit_example_from_ui, args=(modality,),
        )
        submission_error = st.session_state.get(f"example_submission_error_{modality}")
        if submission_error:
            st.error(submission_error)
        st.caption("Examples save their own run history. Your project files and own-data settings stay unchanged.")
        render_step_sources(descriptor.workflow_id, key_prefix=f"example_setup_{modality}")
        return
    if section == "Setup" and mode == "own":
        st.info("Use your own files below. Choose the project folder and sample metadata when requested; the source inputs are read-only and outputs are written separately.")
    if section == "Setup":
        layout_key = f"data_layout_{modality}"
        layout_state_key = f"{layout_key}_state"
        st.session_state.setdefault(layout_key, st.session_state.get(layout_state_key, "Single sample"))
        layout = st.selectbox(
            "Data layout", ["Single sample", "Multiple samples", "Tissue microarray (TMA)"],
            key=layout_key, on_change=_persist_widget_state, args=(layout_key, layout_state_key),
            help="Describes intake organization only; it does not infer specimen identity or biological replicates.",
        )
        tma_rows: list[dict[str, str]] = []
        tma_hash = ""
        tma_errors: list[str] = []
        tma_path = ""
        if layout == "Tissue microarray (TMA)":
            st.write("A TMA contains several cores on one slide. Whole-slide intake gives slide-level QC; per-core analysis needs extracted or core-indexed inputs. The app does not automatically identify cores or patient identities.")
            st.write("You can start without a core-map file. After Check inputs, download the map containing the detected group keys, enter reviewed specimen_id values from your sample sheet, and load it here. Several cores may point to the same specimen or biological unit.")
            st.caption("Required core-map columns: group_key, specimen_id. Optional: slide_id, core_id, section_id. Leave a group UNMAPPED only for technical QC.")
            st.download_button(
                "Download header-only core-map CSV", data=sample_map_template(),
                file_name="core_map_template.csv", mime="text/csv",
            )
            tma_path_state_key = f"tma_core_map_path_state_{modality}"
            tma_path_widget_key = f"tma_core_map_path_widget_{modality}"
            if tma_path_widget_key not in st.session_state:
                st.session_state[tma_path_widget_key] = st.session_state.get(tma_path_state_key, "")
            if st.button("Choose core-map CSV…", key=f"browse_tma_map_{modality}"):
                open_server_path_picker("file", tma_path_widget_key, str(st.session_state.get("project_root_path", "")))
            tma_path = st.text_input("Core-map CSV path (optional)", key=tma_path_widget_key)
            st.session_state[tma_path_state_key] = tma_path
            tma_rows, tma_hash, tma_errors = read_tma_core_map(tma_path)
            for error in tma_errors:
                st.error(error)
            if tma_rows and not tma_errors:
                st.dataframe(tma_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Single sample: review each discovered group-to-specimen assignment explicitly. "
            "Multiple samples: map every group; grouping does not infer replicate count."
        )
        st.subheader("Data & settings")
        st.caption("Choose the data and settings for this analysis. Your choices are kept as you move between sections.")
        if modality == "Integration":
            st.info("Integration uses two explicit native .czi source files, not a folder scan. Choose the moving H&E and reference fluorescence/HS files below; their channel and physical-scale metadata are checked before translation-only QC.")
        else:
            if st.button("Browse modality folder…", key=f"browse_modality_{modality}"):
                open_server_path_picker("directory", path_key, str(st.session_state.get(path_key, "")) or str(st.session_state["project_root_path"]))
            folder_widget = f"modality_folder_widget_{modality}"
            st.session_state[folder_widget] = str(st.session_state.get(path_key, ""))
            st.text_input(
                "Modality input folder", key=folder_widget,
                on_change=_persist_widget_state, args=(folder_widget, path_key),
            )
            st.caption("Choose the folder that contains the original modality files; do not rename files to invent sample identities. The toolkit reads these sources and writes a separate run bundle.")
            with st.expander("Supported inputs and an example folder pattern"):
                st.markdown(descriptor.input_description)
                st.write("Pattern: project folder → modality input folder → original acquisition files (or the workflow's documented table/image inputs). Keep original names and organization; use the review controls below to assign identity.")
        output_key = f"modality_output_{modality}"
        output_widget = f"modality_output_widget_{modality}"
        st.session_state[output_widget] = str(st.session_state.get(output_key, ""))
        with st.expander("Output location", expanded=False):
            st.text_input(
                "Output folder", key=output_widget,
                on_change=_persist_widget_state, args=(output_widget, output_key),
            )
        previous = st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {})
        if not isinstance(previous, dict):
            previous = {}
        panel_parameters: dict[str, Any] = {}
        panel_errors: list[str] = []
        if modality == "CycIF" and descriptor.workflow_id != "cycif.tabular_analysis":
            with st.expander("Marker panel and grouping confirmation", expanded=False):
                panel_widget = "cycif_panel_csv"
                _restore_widget_state(
                    panel_widget, "cycif_panel_csv_state",
                    str(_get_nested(previous, "registration.marker_panel_csv") or ""),
                )
                panel_text = st.text_input(
                    "Marker panel CSV path (optional)", key=panel_widget,
                    on_change=_persist_widget_state, args=(panel_widget, "cycif_panel_csv_state"),
                )
                panel_rows, panel_errors = preview_panel_csv(panel_text)
                for error in panel_errors:
                    st.error(error)
                if panel_rows:
                    st.dataframe(panel_rows, use_container_width=True, hide_index=True)
                group_widget = f"cycif_grouping_confirmed_{confirmation_key(modality, str(st.session_state.get('annotation_csv', '')))}"
                group_state = f"{group_widget}_state"
                _restore_widget_state(group_widget, group_state, False)
                panel_parameters["registration"] = {
                    "confirmed_grouping": st.checkbox(
                        "I confirm grouping does not infer biological identity or replicate structure.",
                        key=group_widget, on_change=_persist_widget_state, args=(group_widget, group_state),
                    ),
                    "marker_panel_confirmed": False,
                    "marker_panel_csv": None,
                }
                if panel_text:
                    panel_parts = [panel_text]
                    try:
                        stat = Path(panel_text).expanduser().stat()
                        panel_parts.extend((str(stat.st_size), str(stat.st_mtime_ns)))
                    except OSError:
                        pass
                    panel_parameters["registration"]["marker_panel_csv"] = str(Path(panel_text).expanduser())
                    panel_widget = f"cycif_panel_confirmed_{confirmation_key(*panel_parts)}"
                    panel_state = f"{panel_widget}_state"
                    _restore_widget_state(panel_widget, panel_state, False)
                    panel_parameters["registration"]["marker_panel_confirmed"] = st.checkbox(
                        "I reviewed this panel's zero-based channel-to-marker mapping.",
                        key=panel_widget, on_change=_persist_widget_state, args=(panel_widget, panel_state),
                    )
        continuation = _render_cycif_continuation(descriptor)
        previous = _merge_missing(panel_parameters, previous)
        if continuation:
            previous["continuation"] = continuation
        preflight_state = st.session_state.get(f"preflight_{modality}")
        discovered: dict[str, Any] = {}
        if isinstance(preflight_state, tuple) and len(preflight_state) == 3:
            result_candidate = preflight_state[1]
            if isinstance(result_candidate, dict) and isinstance(result_candidate.get("resolved_parameters"), dict):
                discovered = result_candidate["resolved_parameters"]
        parameter_fields = descriptor.parameter_fields
        if modality == "CycIF" and descriptor.workflow_id != "cycif.tabular_analysis":
            parameter_fields = [
                field for field in parameter_fields
                if field.name not in {
                    "registration.confirmed_grouping",
                    "registration.marker_panel_csv",
                    "registration.marker_panel_confirmed",
                }
            ]
            stages = descriptor.stages
            stage_ids = [stage.stage_id for stage in stages]
            stage_key = f"cycif_stage_{confirmation_key(descriptor.workflow_id)}"
            stage_state_key = f"cycif_stage_state_{confirmation_key(descriptor.workflow_id)}"
            if stage_key not in st.session_state or st.session_state[stage_key] not in stage_ids:
                saved_stage = str(st.session_state.get(stage_state_key, stage_ids[0]))
                st.session_state[stage_key] = saved_stage if saved_stage in stage_ids else stage_ids[0]
            selected_stage = st.radio(
                "CycIF stages", stage_ids, horizontal=True, key=stage_key,
                on_change=_persist_widget_state, args=(stage_key, stage_state_key),
                format_func=lambda value: next(stage.title for stage in stages if stage.stage_id == value),
            )
            stage = next(stage for stage in stages if stage.stage_id == selected_stage)
            st.caption(stage.purpose)
            parameter_fields = [field for field in parameter_fields if field.stage_id == selected_stage]
        parameters, errors = render_parameter_fields(
            descriptor, discovered, fields=parameter_fields, initial=previous,
            key_prefix=f"{confirmation_key(descriptor.workflow_id)}_",
            show_errors=False,
        )
        errors.extend(panel_errors)
        errors.extend(tma_errors)
        if layout == "Tissue microarray (TMA)" and tma_rows and not tma_errors:
            sample_rows, sample_read_errors = read_project_samples(Path(str(st.session_state.get("annotation_csv", ""))))
            errors.extend(sample_read_errors)
            for error in sample_read_errors:
                st.error(error)
            known_specimens = {
                (row.get("specimen_id") or "").strip() for row in sample_rows
                if (row.get("specimen_id") or "").strip()
            }
            unknown = sorted({
                row["specimen_id"] for row in tma_rows
                if row["specimen_id"] != "UNMAPPED" and row["specimen_id"] not in known_specimens
            })
            if unknown:
                message = "Core-map specimen_id values must match project samples or be UNMAPPED: " + ", ".join(unknown)
                errors.append(message)
                st.error(message)
        saved = _merge_missing(parameters, previous)
        saved["data_layout"] = layout
        if layout == "Tissue microarray (TMA)":
            saved["tma_core_map"] = {
                "path": str(Path(tma_path).expanduser()) if tma_path else "",
                "sha256": tma_hash,
                "rows": tma_rows,
            }
        else:
            saved.pop("tma_core_map", None)
        if continuation:
            saved["continuation"] = continuation
        st.session_state[f"workflow_parameters_{descriptor.workflow_id}"] = saved
        contract_errors = validate_saved_parameters(descriptor, saved, discovered)
        parameter_errors = [*errors, *contract_errors]
        st.session_state[f"workflow_parameter_errors_{descriptor.workflow_id}"] = parameter_errors
        st.session_state[f"workflow_parameters_valid_{descriptor.workflow_id}"] = not parameter_errors
        with st.expander("Workflow details and limitations"):
            st.write(descriptor.input_description)
            st.write(descriptor.output_description)
            for limitation in descriptor.limitations:
                st.warning(limitation)
        st.button(
            "Continue to Run", key=f"continue_to_run_{modality}", type="primary",
            on_click=_set_modality_section, args=(modality, "Run"),
        )
        render_step_sources(
            descriptor.workflow_id,
            stage_id=(
                str(st.session_state[f"cycif_stage_{confirmation_key(descriptor.workflow_id)}"])
                if descriptor.workflow_id == "cycif-raw-czi" else None
            ),
            key_prefix=f"own_setup_{modality}",
        )
        return

    if section == "Run" and mode == "example":
        st.subheader("Run")
        st.write("Run the prepared example below. Use Refresh status to check the background job, then open Quality control and Results when outputs appear. A rerun creates a new record rather than overwriting earlier results.")
        _render_example_description(example, compact=True)
        submission_notice = st.session_state.get(f"example_submission_notice_{modality}")
        if submission_notice:
            st.success(submission_notice)
        submission_error = st.session_state.get(f"example_submission_error_{modality}")
        if submission_error:
            st.error(submission_error)
        example_pending = _example_job_pending(modality)
        if example_pending:
            st.info("An example job is already submitted or running. Follow its status below before starting another.")
        st.button(
            "Run example / rerun", key=f"run_example_again_{modality}", type="primary",
            disabled=example_pending,
            help="Submit the fixed technical recipe only; this does not confirm biological pairing or accept an alignment.",
            on_click=_submit_example_from_ui, args=(modality,),
        )
        run_dir = select_run(
            modality, {example.workflow_id},
            Path(str(st.session_state[f"example_output_{modality}"])).expanduser(),
            state_scope="example",
        )
        if run_dir is not None:
            render_run(Path(run_dir), "run")
        else:
            st.info("No example run yet. Run the example above to see its live status and results.")
        return
    if section == "Run":
        st.subheader("Run")
        integration_workflow = descriptor.workflow_id == "integration.serial_section_alignment"
        st.write(
            "First checks inspect the complete input and workflow settings. Then review the selected source pair's explicit confirmation; no specimen identity is mapped. Start creates a background job, with progress and figures in this section."
            if integration_workflow else
            "First checks inspect the complete input and workflow settings. Then review each detected group-to-specimen assignment and confirm the analysis scope. Start creates a background job; its status and figures remain in this Run section."
        )
        if not integration_workflow and not str(st.session_state.get(path_key, "")).strip():
            st.info("Next: choose this modality's input folder in Data & settings.")
        parameters = st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {})
        if not isinstance(parameters, dict):
            parameters = {}
        parameter_valid = bool(st.session_state.get(f"workflow_parameters_valid_{descriptor.workflow_id}", False))
        if not parameter_valid:
            for error in st.session_state.get(f"workflow_parameter_errors_{descriptor.workflow_id}", []):
                st.error(str(error))
        with st.expander("Before you run", expanded=False):
            root_ready = Path(str(st.session_state.get("project_root_path", ""))).expanduser().is_dir()
            input_ready = (
                bool(parameters.get("moving_he_path") and parameters.get("reference_fluorescence_path"))
                if integration_workflow else
                Path(str(st.session_state.get(path_key, ""))).expanduser().is_dir()
            )
            metadata_ready = Path(str(st.session_state.get("annotation_csv", ""))).expanduser().is_file()
            output_ready = bool(str(st.session_state.get(f"modality_output_{modality}", "")).strip())
            for label_text, ready in (
                ("Project folder", root_ready),
                ("Project sample metadata", metadata_ready),
                ("Selected source files" if integration_workflow else "Modality input folder", input_ready),
                ("Settings", parameter_valid),
                ("Output location", output_ready),
            ):
                st.write(f"{'✓' if ready else '○'} {label_text}")
            st.caption("Fresh preflight verifies the full input and workflow contract; Integration checks explicit files without specimen mapping. Other workflows require reviewed mapping or QC-only scope confirmation.")
        if not metadata_ready or not root_ready:
            st.info("Add your project folder and sample metadata before starting an analysis.")
            st.button(
                "Open Project", key=f"open_project_from_{modality}", type="primary",
                on_click=lambda: st.session_state.__setitem__("spatial_page", "Project"),
            )
        if not integration_workflow and (not str(st.session_state.get(path_key, "")).strip() or not input_ready):
            st.button(
                "Choose input data", key=f"choose_inputs_from_{modality}",
                on_click=_set_modality_section, args=(modality, "Setup"),
            )
        if st.button(
            "Check inputs", key=f"check_inputs_{modality}",
            type="secondary" if st.session_state.get(f"preflight_{modality}") else "primary",
            disabled=not (
                root_ready and input_ready and metadata_ready and output_ready
                and (integration_workflow or bool(str(st.session_state.get(path_key, "")).strip()))
            ),
        ):
            st.session_state.pop(f"mapped_preflight_{modality}", None)
            try:
                if descriptor.workflow_id == "cycif-raw-czi" and parameters.get("continuation", {}).get("mode") == "continue" and (
                    not parameters.get("continuation", {}).get("parent_run_dir")
                    or not parameters.get("continuation", {}).get("start_stage")
                    or parameters.get("continuation", {}).get("warning_review_confirmed") is not True
                ):
                    st.error("Choose an eligible continuation boundary and acknowledge all source QC warnings.")
                elif descriptor.workflow_id == "cycif-raw-czi" and (
                    parameters.get("registration", {}).get("enabled") is True
                    and not parameters.get("registration", {}).get("confirmed_grouping")
                ):
                    st.error("Explicitly confirm acquisition grouping before enabling registration.")
                else:
                    request, result = _modality_run_request(descriptor, parameters, modality)
                    signature = confirmation_key(
                        descriptor.workflow_id, modality, str(st.session_state.get("project_root_path", "")),
                        str(st.session_state.get("annotation_csv", "")), str(st.session_state.get(path_key, "")),
                        str(st.session_state.get(f"modality_output_{modality}", "")), _signature_parameters(parameters),
                    )
                    st.session_state[f"preflight_{modality}"] = (
                        request.model_dump(mode="json"), result.model_dump(mode="json"), signature,
                    )
            except Exception as exc:
                st.error(f"Unable to preflight request: {exc}")
        preflight_state = st.session_state.get(f"preflight_{modality}")
        parameters = st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {})
        signature = confirmation_key(
            descriptor.workflow_id, modality, str(st.session_state.get("project_root_path", "")),
            str(st.session_state.get("annotation_csv", "")), str(st.session_state.get(path_key, "")),
            str(st.session_state.get(f"modality_output_{modality}", "")), _signature_parameters(parameters),
        )
        can_submit = False
        request_data: dict[str, Any] = {}
        if preflight_state:
            receipt_request, receipt_result, stored_signature = preflight_state
            if stored_signature != signature:
                st.warning("Inputs or settings changed after checks. Run Check inputs again.")
            else:
                original_result = copy.deepcopy(receipt_result)
                request_data = copy.deepcopy(receipt_request)
                result_data = copy.deepcopy(receipt_result)
                result = PreflightResult.model_validate(result_data)
                mapping_valid, mapping_confirmed = _render_mapping(descriptor, request_data, result_data)
                mapped_result_data: dict[str, Any] | None = None
                mapped_request_data: dict[str, Any] | None = None
                if mapping_valid and mapping_confirmed:
                    mapping_signature = confirmation_key(
                        stored_signature,
                        json.dumps(request_data.get("parameters", {}), sort_keys=True, default=str),
                    )
                    mapped_cache = st.session_state.get(f"mapped_preflight_{modality}")
                    if isinstance(mapped_cache, tuple) and len(mapped_cache) == 3 and mapped_cache[0] == mapping_signature:
                        _, mapped_request_data, mapped_result_data = mapped_cache
                    else:
                        try:
                            mapped_request, mapped_result = prepare_request(
                                request_data["workflow_id"], Path(request_data["project_root"]),
                                Path(request_data["annotation_csv"]), Path(request_data["modality_dir"]),
                                Path(request_data["output_root"]), request_data["parameters"],
                            )
                            mapped_request_data = mapped_request.model_dump(mode="json")
                            mapped_result_data = mapped_result.model_dump(mode="json")
                            changed = _reviewed_preflight_changed(original_result, mapped_result_data)
                            if changed:
                                st.warning(changed)
                                mapped_request_data = None
                                mapped_result_data = None
                                st.session_state.pop(f"mapped_preflight_{modality}", None)
                            else:
                                st.session_state[f"mapped_preflight_{modality}"] = (
                                    mapping_signature, mapped_request_data, mapped_result_data,
                                )
                        except Exception as exc:
                            st.error(f"Unable to validate the confirmed mapping: {exc}")
                            mapped_request_data = None
                            mapped_result_data = None
                if mapped_result_data is not None and mapped_request_data is not None:
                    mapped_result = PreflightResult.model_validate(mapped_result_data)
                    render_preflight(mapped_result)
                    request_data = mapped_request_data
                else:
                    render_preflight(result)
                parameter_errors = validate_saved_parameters(
                    descriptor,
                    st.session_state.get(f"workflow_parameters_{descriptor.workflow_id}", {}),
                    result.resolved_parameters,
                )
                parameter_valid = bool(st.session_state.get(f"workflow_parameters_valid_{descriptor.workflow_id}", False)) and not parameter_errors
                can_submit = (
                    mapping_valid and mapping_confirmed and parameter_valid
                    and mapped_result_data is not None and mapped_request_data is not None
                    and PreflightResult.model_validate(mapped_result_data).ready
                )
        if can_submit and st.button("Start run", type="primary", key=f"submit_{modality}"):
            try:
                request, final_result = prepare_request(
                    request_data["workflow_id"], Path(request_data["project_root"]),
                    Path(request_data["annotation_csv"]), Path(request_data["modality_dir"]),
                    Path(request_data["output_root"]), request_data["parameters"],
                )
                final_data = final_result.model_dump(mode="json")
                original_result = preflight_state[1] if preflight_state else {}
                changed = _reviewed_preflight_changed(original_result, final_data)
                render_preflight(final_result)
                if changed:
                    st.error(changed)
                elif not final_result.ready:
                    st.error("Final preflight is blocked; no job was submitted.")
                else:
                    job_id = submit_request(request, final_result)
                    active = dict(st.session_state["active_run_dirs"])
                    selected = dict(st.session_state["selected_run_dirs"])
                    active[modality] = str(request.run_dir)
                    selected[modality] = str(request.run_dir)
                    st.session_state["active_run_dirs"] = active
                    st.session_state["selected_run_dirs"] = selected
                    st.session_state[f"pending_run_selection_{modality}"] = str(request.run_dir)
                    st.session_state.pop(f"preflight_{modality}", None)
                    st.session_state.pop(f"mapped_preflight_{modality}", None)
                    st.success("Your run has started. Follow its progress below; your results will stay in this workspace.")
            except Exception as exc:
                st.error(f"Submission failed: {exc}")
        run_dir = select_run(
            modality, {item.workflow_id for item in descriptors},
            Path(str(st.session_state[f"modality_output_{modality}"])).expanduser(),
        )
        if run_dir is not None:
            selected_map = dict(st.session_state["selected_run_dirs"])
            selected_map[modality] = str(run_dir)
            st.session_state["selected_run_dirs"] = selected_map
            render_run(Path(run_dir), "run")
        return

    history_root_key = f"example_output_{modality}" if mode == "example" else f"modality_output_{modality}"
    history_workflows = {example.workflow_id} if mode == "example" else {item.workflow_id for item in descriptors}
    run_dir = select_run(
        modality, history_workflows,
        Path(str(st.session_state[history_root_key])).expanduser(),
        state_scope=mode,
    )
    if run_dir is not None:
        if mode == "own":
            selected_map = dict(st.session_state["selected_run_dirs"])
            selected_map[modality] = str(run_dir)
            st.session_state["selected_run_dirs"] = selected_map
        if section == "Quality control":
            st.caption("QC gathers recorded warnings, blocked steps and review decisions. Open an item to understand what needs attention; return to Run only when a workflow requests a confirmed decision.")
        elif section == "Results":
            st.caption("Results are generated from the selected run bundle. If the job is still running, check Run for progress; use the figures and tables here only after they appear.")
        render_run(Path(run_dir), {"Quality control": "qc", "Results": "results", "Files": "files"}[section])
    else:
        st.info(
            "No run is available here yet. Return to Data & settings to run the file-backed example, or switch to your own data and complete its checks."
            if mode == "example" else
            "No own-data run is available yet. Choose your files in Data & settings, then check inputs and confirm the group mappings."
        )
        st.button(
            "Go to Data & settings", key=f"setup_from_results_{modality}",
            on_click=_set_modality_section, args=(modality, "Setup"),
        )


def main() -> None:
    from spatial_omics_toolkit.application.ui.presentation import apply_theme, render_home

    apply_theme()
    workflows = list_workflows()
    modalities = ("CycIF", "HS", "FLIM", "H&E", "Visium", "Integration")
    by_modality = {name: [workflow for workflow in workflows if workflow.modality == name] for name in modalities}
    page = st.session_state.setdefault("spatial_page", "Home")
    try:
        settings = st.secrets.get("toolkit")
    except FileNotFoundError:
        settings = None
    read_only = False
    if settings is not None:
        if not hasattr(settings, "get"):
            st.error("Invalid [toolkit] settings: expected a table. The full workspace remains available.")
        else:
            mode = settings.get("mode", "hpc")
            if mode == "read_only":
                read_only = True
            elif mode != "hpc":
                st.error('Invalid [toolkit] mode: expected "hpc" or "read_only". The full workspace remains available.')
    if not read_only and shutil.which("sbatch") is None:
        st.info(
            "SLURM submission is unavailable: sbatch was not found on this server. "
            "The interface is available, but workflow jobs require an authorized SLURM backend "
            "and server-accessible input and output paths."
        )
    dataset_url = settings.get("datasets_url") if hasattr(settings, "get") else None
    if not read_only:
        _initialize_state()
    if read_only and page == "Home":
        st.info("This instance is view-only. Workflows execute on authorized HPC resources.")
    if not read_only and st.session_state.get("_server_picker_open"):
        server_path_picker_dialog()
    if page == "Home":
        render_home(
            lambda destination: st.session_state.__setitem__("spatial_page", destination),
            read_only=read_only,
            dataset_url=dataset_url if isinstance(dataset_url, str) else None,
        )
    else:
        with st.container(key="app_navigation"):
            if page == "Project":
                home_column, current_column = st.columns([1, 3], gap="small")
                project_column = None
            else:
                home_column, project_column, current_column = st.columns([1, 1, 2], gap="small")
            with home_column:
                st.button(
                    "Home", key="navigate_home", use_container_width=True,
                    on_click=lambda: st.session_state.__setitem__("spatial_page", "Home"),
                )
            if project_column is not None:
                with project_column:
                    st.button(
                        "Project", key="navigate_project", use_container_width=True,
                        on_click=lambda: st.session_state.__setitem__("spatial_page", "Project"),
                    )
            with current_column:
                st.caption(f"Current: {page}")
        if read_only:
            st.info("This instance is view-only. Workflows execute on authorized HPC resources.")
        if page == "Project":
            if read_only:
                st.title("Project")
                st.info("Project setup is available on authorized HPC resources, not in this view-only instance.")
            else:
                _render_project()
        else:
            _render_modality(page, by_modality[page], read_only=read_only)


main()
