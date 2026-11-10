from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from spatial_omics_toolkit.application.acquisition import (
    METADATA_COLUMNS,
    FIELD_UNITS,
    METADATA_FIELDS,
    RELATIONSHIP_COLUMNS,
    discover_native_metadata,
    read_acquisition_metadata,
    resolve_source_path,
    read_acquisition_relationships,
    write_acquisition_metadata,
    write_acquisition_relationships,
)
from spatial_omics_toolkit.application.config import containing_workspace_root
from spatial_omics_toolkit.application.intake import lexical_path_is_under, path_is_under


def _csv_text(rows: Sequence[dict[str, str]], columns: Sequence[str]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _import_csv(content: bytes, columns: Sequence[str]) -> list[dict[str, str]]:
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("CSV exceeds the 5 MiB import limit.")
    try:
        reader = csv.DictReader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(columns):
            raise ValueError("CSV must have exactly these distinct columns: " + ", ".join(columns))
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV contains a row with the wrong number of fields.")
            rows.append({column: str(row[column]) for column in columns})
        return rows
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc


def _digest(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} must be a regular project file, not a link or directory.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conflicts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    values: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        key = (row.get("acquisition_id", ""), row.get("field", ""))
        if all(key) and row.get("value", "").strip().lower() not in {"", "unknown"}:
            values[key].add((row.get("value", ""), row.get("unit", "")))
    return [
        {"acquisition_id": acquisition, "field": field,
         "values": "; ".join(f"{value} {unit}".strip() for value, unit in sorted(choices))}
        for (acquisition, field), choices in sorted(values.items()) if len(choices) > 1
    ]


def _supported_pair(moving: str, reference: str) -> bool:
    return moving == "H&E" and reference in {"CycIF", "HS"}


def _modality_acquisitions(rows: list[dict[str, str]], modality: str) -> list[str]:
    return sorted({row["acquisition_id"] for row in rows if row.get("modality") == modality and row.get("acquisition_id")})


def _choose_conflict(rows: list[dict[str, str]], acquisition: str, field: str, chosen_index: int) -> list[dict[str, str]]:
    if (not 0 <= chosen_index < len(rows) or rows[chosen_index].get("acquisition_id") != acquisition
            or rows[chosen_index].get("field") != field
            or rows[chosen_index].get("value", "").strip().lower() in {"", "unknown"}):
        raise ValueError("The selected evidence no longer matches this conflict.")
    return [
        {**row, "reviewed": "true" if index == chosen_index else "false"}
        if row.get("acquisition_id") == acquisition and row.get("field") == field else dict(row)
        for index, row in enumerate(rows)
    ]


def _merge_discovered(
    rows: list[dict[str, str]], discovered: list[dict[str, str]], root: Path, workspace: Path,
) -> list[dict[str, str]]:
    if not discovered:
        return [dict(row) for row in rows]
    first = discovered[0]
    matches = [row for row in rows if row.get("acquisition_id") == first["acquisition_id"]]
    if matches:
        identities = {(row.get("modality", ""), row.get("source_path", ""), row.get("specimen_id", "")) for row in matches}
        if len(identities) != 1:
            raise ValueError("Existing acquisition identity is inconsistent; correct the draft first.")
        modality, source, specimen = next(iter(identities))
        if modality != first["modality"] or resolve_source_path(root, workspace, source) != resolve_source_path(root, workspace, first["source_path"]):
            raise ValueError("This acquisition ID already identifies a different modality or source.")
    else:
        source, specimen = first["source_path"], first["specimen_id"]
    merged = [dict(row) for row in rows]
    for item in discovered:
        candidate = {**item, "source_path": source, "specimen_id": specimen}
        if not any(all(existing.get(column, "") == candidate.get(column, "") for column in METADATA_COLUMNS if column != "reviewed") for existing in merged):
            merged.append(candidate)
    return merged


def _guidance_issues(
    rows: list[dict[str, str]], relationships: list[dict[str, str]], moving: str, reference: str,
    root: Path, workspace: Path,
) -> list[str]:
    issues: list[str] = []
    for label, acquisition, expected in (("Moving", moving, {"H&E"}), ("Reference", reference, {"CycIF", "HS"})):
        selected = [row for row in rows if row.get("acquisition_id") == acquisition]
        if not selected:
            issues.append(f"{label}: select a recorded acquisition.")
            continue
        identities = {(row.get("modality", ""), row.get("source_path", ""), row.get("specimen_id", "")) for row in selected}
        if len(identities) != 1:
            issues.append(f"{label}: conflicting acquisition identity/source; resolve the records.")
            continue
        modality, source, _ = next(iter(identities))
        if modality not in expected:
            issues.append(f"{label}: modality {modality!r} is unsupported for this preview.")
        try:
            path = resolve_source_path(root, workspace, source)
            if not path.is_file() or path.suffix.lower() != ".czi":
                issues.append(f"{label}: an existing native CZI file is required.")
        except (OSError, ValueError, RuntimeError):
            issues.append(f"{label}: source is missing, inaccessible, or outside the approved workspace.")
        for field, description in (("pixel_size_x", "native X scale"), ("pixel_size_y", "native Y scale"), *(([("channel_names", "reference channel names")] if label == "Reference" else []))):
            matching = [row for row in selected if row.get("field") == field and row.get("value") not in {"", "unknown"}]
            reviewed = [row for row in matching if row.get("reviewed") == "true"]
            choices = {(row.get("value", ""), row.get("unit", "")) for row in matching}
            reviewed_choices = {(row.get("value", ""), row.get("unit", "")) for row in reviewed}
            if not matching:
                issues.append(f"{label}: {description} is missing.")
            elif not reviewed or len(reviewed_choices) != 1:
                issues.append(f"{label}: {description} is unreviewed or has no unique reviewed choice.")
            elif len(choices) > 1:
                issues.append(f"{label}: {description} has competing values; verify the reviewed choice and provenance against the source.")
            if matching:
                issues.append(f"{label}: verify {description} against the native source; CSV provenance is a declaration, not independent verification.")
    relevant = [row for row in relationships if {row.get("acquisition_id"), row.get("related_acquisition_id")} == {moving, reference} and moving != reference]
    if len(relevant) != 1:
        issues.append("Relationship: missing or conflicting declaration/evidence for these acquisitions.")
    else:
        relation = relevant[0]
        if relation.get("relationship") != "serial_section" or relation.get("reviewed") != "true" or not relation.get("evidence", "").strip():
            issues.append("Relationship: a reviewed serial-section declaration with evidence is missing or unreviewed.")
    issues.append("Source authorization and exact reference channel index/name must be confirmed in Integration setup; this page cannot verify them.")
    return issues


def _editor(
    root: Path, workspace: Path, scope: str, name: str, columns: Sequence[str],
    loaded: list[dict[str, str]], errors: list[str],
    save: Callable[[Path, Path, list[dict[str, str]]], None],
) -> list[dict[str, str]]:
    prefix = f"acquisition_{scope}_{name}"
    draft_key = f"{prefix}_draft"
    generation_key = f"{prefix}_generation"
    digest_key = f"{prefix}_digest"
    baseline_key = f"{prefix}_baseline"
    file_path = root / f"acquisition_{name}.csv"
    try:
        current_digest = _digest(file_path)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return []
    if draft_key not in st.session_state:
        st.session_state[draft_key] = loaded
        st.session_state[digest_key] = current_digest
        st.session_state[generation_key] = 0
    for error in errors:
        st.warning(error)
    if current_digest != st.session_state[digest_key]:
        st.warning("The saved file changed since this draft was loaded. Reload it before saving to avoid overwriting another edit.")
        if st.button("Reload saved file", key=f"{prefix}_reload"):
            st.session_state[draft_key] = loaded
            st.session_state[digest_key] = current_digest
            st.session_state[generation_key] += 1
            st.rerun()
    upload = st.file_uploader("Import structured CSV into draft", type=["csv"], key=f"{prefix}_upload")
    if st.button("Replace draft with imported CSV", key=f"{prefix}_import", disabled=upload is None):
        try:
            st.session_state[draft_key] = _import_csv(upload.getvalue(), columns)
            st.session_state[generation_key] += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    generation = st.session_state[generation_key]
    editor_key = f"{prefix}_editor_{generation}"
    # Streamlit applies widget edits to the original input on every rerun.
    # Keep that input stable until the widget is replaced or cleaned up.
    if editor_key not in st.session_state:
        st.session_state[baseline_key] = [dict(row) for row in st.session_state[draft_key]]
    frame = pd.DataFrame(st.session_state[baseline_key], columns=list(columns)).fillna("").astype(str)
    edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, use_container_width=True,
                            key=editor_key)
    rows = [{column: str(row[column]) for column in columns} for row in edited.fillna("").to_dict(orient="records")]
    st.session_state[draft_key] = rows
    st.download_button("Download current draft", data=_csv_text(rows, columns),
                       file_name=f"acquisition_{name}.csv", mime="text/csv", key=f"{prefix}_download")
    st.download_button("Download blank template", data=_csv_text([], columns),
                       file_name=f"acquisition_{name}_template.csv", mime="text/csv", key=f"{prefix}_template")
    if st.button("Validate and save project file", key=f"{prefix}_save", type="primary"):
        try:
            if _digest(file_path) != st.session_state[digest_key]:
                raise ValueError("Saved file changed since the draft was loaded; reload before saving.")
            save(root, workspace, rows)
            st.session_state[draft_key] = rows
            st.session_state[digest_key] = _digest(file_path)
            st.session_state[generation_key] += 1
            st.success(f"Saved {file_path.name} after validation.")
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Not saved: {exc}")
    return rows


def render_acquisition_information(
    project_root: Path, picker: Callable[[str, str, str], None], annotation_csv: str,
) -> None:
    st.divider()
    st.header("Acquisition information")
    st.write("Optional, project-level records describe acquisitions and their evidence. The existing project_samples.csv remains the authority for specimen and biological-unit identity. These records do not prevent independent modality QC.")
    try:
        workspace = containing_workspace_root(project_root)
    except ValueError:
        st.info("Choose an existing project folder inside an approved workspace root to review acquisition information.")
        return
    if not project_root.is_dir():
        st.info("Choose an existing project folder inside an approved workspace root to review acquisition information.")
        return
    root = project_root.resolve()
    scope = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    annotation = Path(annotation_csv).expanduser() if annotation_csv.strip() else None
    try:
        metadata, metadata_errors = read_acquisition_metadata(root, workspace, annotation_csv=annotation)
        relationships, relationship_errors = read_acquisition_relationships(root, workspace, annotation_csv=annotation)
    except (OSError, ValueError) as exc:
        st.error(f"Unable to read acquisition information: {exc}")
        return
    st.info("Native headers can describe instrument settings, channels and scale, but never establish specimen identity or biological pairing. Review provenance, values and conflicts against your records; missing information stays unknown.")
    metadata_tab, relationship_tab, guidance_tab = st.tabs(["Acquisition metadata", "Relationships", "Integration guidance"])
    with metadata_tab:
        st.caption("One row per acquisition and metadata field. Choose provenance native, user, or unknown; provide source_detail for native/user and preserve units. Set reviewed to true only after checking its source. Assign specimen_id only from the selected sample sheet.")
        with st.expander("Supported fields and units"):
            st.write("Use the exact standard field names and units listed below. For an unknown value enter the literal `unknown`; do not guess a scale or channel identity.")
            st.write("For other contextual metadata, use `custom:<name>` (for example, `custom:objective_model`). The name must start with an ASCII letter, followed only by ASCII letters, digits, underscores, hyphens, or dots. Enter a nonempty string value, or `unknown` if unavailable; the unit is optional and may be arbitrary. Custom fields follow the same provenance, source-detail, conflict, and review rules as standard fields. They document context only and are never passed to workflow adapters as parameters.")
            st.dataframe([{"field": field, "allowed units": ", ".join(FIELD_UNITS.get(field, ())) or "(blank)"} for field in METADATA_FIELDS], hide_index=True, use_container_width=True)
        rows = _editor(root, workspace, scope, "metadata", METADATA_COLUMNS, metadata, metadata_errors,
                       partial(write_acquisition_metadata, annotation_csv=annotation))
        conflicts = _conflicts(rows)
        if conflicts:
            st.warning("Conflicting values were found for the same acquisition and field. An explicit reviewed choice is required before relying on a value; preserve the competing evidence and its provenance.")
            st.dataframe(conflicts, hide_index=True, use_container_width=True)
            for conflict in conflicts:
                acquisition, field = conflict["acquisition_id"], conflict["field"]
                indices = [index for index, row in enumerate(rows) if row.get("acquisition_id") == acquisition and row.get("field") == field]
                known_indices = [index for index in indices if rows[index].get("value", "").strip().lower() not in {"", "unknown"}]
                st.markdown(f"**{acquisition} · {field}**")
                st.dataframe([{"row": index + 1, **rows[index]} for index in indices], hide_index=True, use_container_width=True)
                choice_key = hashlib.sha256(f"{scope}:{acquisition}:{field}".encode()).hexdigest()[:16]
                choice = st.selectbox(
                    "Choose the evidence you have reviewed", known_indices, index=None,
                    format_func=lambda index: f"Row {index + 1}: {rows[index].get('value', '')} {rows[index].get('unit', '')} · {rows[index].get('provenance', '')} · {rows[index].get('source_detail', '')}",
                    key=f"{choice_key}_choice",
                )
                if st.button("Apply reviewed choice to draft", key=f"{choice_key}_apply", disabled=choice is None):
                    prefix = f"acquisition_{scope}_metadata"
                    st.session_state[f"{prefix}_draft"] = _choose_conflict(rows, acquisition, field, choice)
                    st.session_state[f"{prefix}_generation"] += 1
                    st.rerun()
            st.caption("Applying a choice marks the selected evidence reviewed and other values unreviewed; it preserves every row. Validate and save the draft to persist it.")
        with st.expander("Inspect a native file's metadata", expanded=False):
            st.caption("Metadata inspection is read-only and does not inspect full pixel data. Extracted rows are an unreviewed draft; no identity is inferred.")
            modality = st.selectbox("Modality", ["CycIF", "HS", "FLIM", "H&E", "Visium"], key=f"{scope}_native_modality")
            acquisition_id = st.text_input("Acquisition ID", key=f"{scope}_native_id")
            path_key = f"{scope}_native_source"
            st.text_input("Native source file", key=path_key)
            if st.button("Browse native file…", key=f"{scope}_native_browse"):
                picker("native", path_key, str(st.session_state.get(path_key, "")) or str(root))
            if st.button("Inspect metadata and add to draft", key=f"{scope}_native_inspect"):
                try:
                    source = str(st.session_state.get(path_key, "")).strip()
                    if not source or not acquisition_id.strip():
                        raise ValueError("Choose a source file and enter an acquisition ID.")
                    discovered = discover_native_metadata(root, workspace, source, modality, acquisition_id.strip())
                    if not discovered:
                        st.warning("No supported native metadata was found; enter documented values manually if available.")
                    else:
                        prefix = f"acquisition_{scope}_metadata"
                        st.session_state[f"{prefix}_draft"] = _merge_discovered(rows, discovered, root, workspace)
                        st.session_state[f"{prefix}_generation"] += 1
                        st.rerun()
                except (OSError, ValueError, ImportError) as exc:
                    st.error(f"Cannot inspect native metadata: {exc}")
    with relationship_tab:
        st.caption("Record explicit evidence for same_section, serial_section, or unknown relationships between acquisition IDs. A reviewed relationship is a user declaration, not an alignment or proof of cell correspondence.")
        _editor(root, workspace, scope, "relationships", RELATIONSHIP_COLUMNS, relationships, relationship_errors,
                partial(write_acquisition_relationships, annotation_csv=annotation))
    with guidance_tab:
        st.markdown("- Run each modality's technical QC independently, even when acquisition information is incomplete.\n- Use project_samples.csv and external study records to review specimen identity; filenames and native metadata are not biological pairing.\n- Declare same_section or serial_section relationships only with evidence. Serial sections are not the same cells. A reviewed relationship is not acceptance of a transform.")
        st.caption("This assessment uses saved, validated project records only, not unsaved edits. It does not enable or disable independent technical QC or the existing preview.")
        moving_modality = st.selectbox("Moving modality", ["H&E", "CycIF", "HS", "FLIM", "Visium"], key=f"{scope}_guide_moving_modality")
        reference_modality = st.selectbox("Reference modality", ["CycIF", "HS", "H&E", "FLIM", "Visium"], key=f"{scope}_guide_reference_modality")
        if not _supported_pair(moving_modality, reference_modality):
            st.warning(f"{moving_modality} → {reference_modality} is not supported for integration preview. HS–FLIM registration is not implemented; independent modality QC remains available.")
        if metadata_errors:
            st.warning("Saved acquisition metadata is invalid and is excluded from this guidance. Correct and save it before selecting acquisitions.")
        else:
            moving_choices = ["", *_modality_acquisitions(metadata, moving_modality)]
            reference_choices = ["", *_modality_acquisitions(metadata, reference_modality)]
            moving = st.selectbox("Moving acquisition ID", moving_choices, format_func=lambda value: value or "(select an acquisition)", key=f"{scope}_guide_moving_id_{moving_modality}")
            reference = st.selectbox("Reference acquisition ID", reference_choices, format_func=lambda value: value or "(select an acquisition)", key=f"{scope}_guide_reference_id_{reference_modality}")
            if moving and reference and moving == reference:
                st.warning("Choose two distinct acquisitions.")
            else:
                relevant_rows = [row for row in metadata if row.get("acquisition_id") in {moving, reference} and row.get("acquisition_id")]
                if relevant_rows:
                    st.dataframe(relevant_rows, hide_index=True, use_container_width=True)
                selected_conflicts = _conflicts(relevant_rows)
                if selected_conflicts:
                    st.warning("The selected acquisitions have competing metadata values; inspect their individual provenance and reviewed choices in Acquisition metadata.")
                    st.dataframe(selected_conflicts, hide_index=True, use_container_width=True)
                valid_relationships = relationships if not relationship_errors else []
                if relationship_errors:
                    st.warning("Saved relationships are invalid and excluded from guidance. Correct and save them before relying on any declaration.")
                pair_rows = [row for row in valid_relationships if moving and reference and {row.get("acquisition_id"), row.get("related_acquisition_id")} == {moving, reference}]
                if pair_rows:
                    st.dataframe(pair_rows, hide_index=True, use_container_width=True)
                st.markdown("**Missing, unreviewed, or independently unverified prerequisites**")
                for issue in _guidance_issues(metadata, valid_relationships, moving, reference, root, workspace):
                    st.warning(issue)
            st.warning("The integration adapter requires physical X and Y scale in each native CZI header. A CSV-declared scale, even if reviewed, cannot satisfy this requirement; the adapter must verify the native metadata.")
            st.caption("Only moving H&E and reference fluorescence/CycIF or HS CZI sources are supported; HS–FLIM and other pairs are not implemented. Preview is exploratory translation-only QC, not accepted registration, biological pairing, or same-cell correspondence. Select and authorize sources, exact reference channel index/name, and specimen mapping or UNMAPPED separately in Integration.")
