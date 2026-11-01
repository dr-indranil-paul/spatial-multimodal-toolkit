from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import streamlit as st

from spatial_omics_toolkit.application.contracts import FunctionReference, StageDefinition
from spatial_omics_toolkit.application.workflows.registry import get_workflow

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_EXTERNAL_SOURCE_ROOT = (
    Path(__file__).resolve().parents[4]
    / ".."
    / "02_spatial_proteomics"
    / "03_analysis"
    / "01_exploratory_analysis"
    / "02_scripts"
).resolve()


def _source_path(source_file: str, module: Any) -> Path | None:
    """Resolve only package files or known canonical source scripts."""
    if source_file.startswith("external:"):
        filename = source_file.removeprefix("external:")
        if Path(filename).name != filename:
            return None
        directory = getattr(module, "_source_dir", None)
        source_root = Path(directory()) if callable(directory) else _EXTERNAL_SOURCE_ROOT
        candidate = (source_root / filename).resolve()
        return candidate if candidate.is_file() else None

    candidate = (_SOURCE_ROOT / source_file).resolve()
    if candidate.is_relative_to(_SOURCE_ROOT) and candidate.is_file():
        return candidate
    return None


def _function_line(path: Path, function_name: str) -> int | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    return min(lines) if lines else None


def _display_path(source_file: str, path: Path | None = None) -> str:
    if source_file.startswith("external:"):
        filename = source_file.removeprefix("external:")
        return str(path) if path is not None else filename
    return "src/" + source_file


def _render_reference(reference: FunctionReference, module: Any) -> None:
    path = _source_path(reference.source_file, module)
    line = _function_line(path, reference.function_name) if path is not None else None
    st.markdown(f"**{reference.role}** — `{reference.function_name}()`")
    if line is None:
        st.caption(f"Current source location unavailable · `{_display_path(reference.source_file)}`")
        return
    st.caption(f"Script: {path.name} · definition at line {line}")
    st.code(f"{path}:{line}", language=None)


def _render_recorded_versions(recorded_versions: dict[str, str] | None) -> None:
    if recorded_versions is None:
        return
    if not recorded_versions:
        st.caption("Software or source identifiers are unavailable in this view; the Files section holds the complete records.")
        return
    st.markdown("**Recorded by the saved run**")
    st.caption("These are the execution-time identifiers, not a snapshot of the current function definitions.")
    st.dataframe(
        [{"Source or tool": name, "Recorded identifier": value} for name, value in recorded_versions.items()],
        hide_index=True, use_container_width=True,
    )


def render_step_sources(
    workflow_id: str,
    stage_id: str | None = None,
    recorded_versions: dict[str, str] | None = None,
    key_prefix: str = "",
) -> None:
    """Show current source definitions separately from optional saved-run hashes."""
    with st.expander("Methods & source", expanded=False):
        try:
            module, descriptor = get_workflow(workflow_id)
        except (KeyError, ImportError, ValueError):
            st.caption("Workflow source context is unavailable.")
            _render_recorded_versions(recorded_versions)
            return

        stages: list[StageDefinition] = descriptor.stages
        selected: StageDefinition | None = None
        if stage_id is not None and stage_id.startswith("preview_"):
            selected = StageDefinition(
                stage_id=stage_id, title="PDF page previews",
                purpose="Creates bounded page images from an existing PDF report. This is a display step, not a new analysis; the original report's step describes its analytical methods.",
                implementations=[FunctionReference(
                    source_file="spatial_omics_toolkit/application/worker.py",
                    function_name="_render_pdf_previews",
                    role="Render bounded PDF page previews",
                )],
            )
        elif stage_id is not None:
            selected = next((stage for stage in stages if stage.stage_id == stage_id or stage_id in stage.event_stage_ids), None)
            if selected is None:
                st.caption("This stage has no declared source mapping; showing workflow-level context instead.")
        elif stages:
            selected_title = st.selectbox(
                "Workflow step",
                options=[stage.title for stage in stages],
                format_func=lambda title: title,
                key=f"{key_prefix}implementation_stage_{workflow_id}",
            )
            selected = next((stage for stage in stages if stage.title == selected_title), None)

        references = selected.implementations if selected is not None else [
            reference for stage in stages for reference in stage.implementations
        ]
        st.caption("Current code locations are shown below. Your settings determine which steps run; a saved run may have used earlier code.")
        if selected is not None:
            st.caption(selected.purpose)
        elif stage_id is None:
            st.caption(descriptor.description)
        else:
            st.caption(f"{descriptor.title}: {descriptor.description}")

        reference_index: dict[tuple[str, str], FunctionReference] = {}
        for reference in references:
            reference_index.setdefault((reference.source_file, reference.function_name), reference)
        unique_references = list(reference_index.values())
        if unique_references:
            index = 0
            if len(unique_references) > 1:
                index = st.selectbox(
                    "Processing function", options=list(range(len(unique_references))),
                    format_func=lambda item: unique_references[item].function_name,
                    key=f"{key_prefix}implementation_function_{workflow_id}_{selected.stage_id if selected else stage_id}",
                    help="Select a processing or plotting function to see its purpose and exact source location.",
                )
            _render_reference(unique_references[index], module)
        else:
            st.caption("No function-level source mapping is available for this workflow context.")
        _render_recorded_versions(recorded_versions)
