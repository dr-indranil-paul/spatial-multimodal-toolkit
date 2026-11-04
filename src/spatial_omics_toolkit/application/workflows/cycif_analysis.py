from __future__ import annotations

from datetime import datetime, timezone
import csv
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from spatial_omics_toolkit.application.contracts import (
    FunctionReference,
    ParameterField,
    PreflightResult,
    RunRequest,
    StageDefinition,
    StageEvent,
    ToolReference,
    WorkflowDescriptor,
    WorkflowEmitter,
    WorkflowResult,
)
from spatial_omics_toolkit.cycif.aggregate import step09_aggregate_per_core
from spatial_omics_toolkit.cycif.classify import step07_classify_cells, step08_score_programs
from spatial_omics_toolkit.cycif.compare import step12_compare_groups
from spatial_omics_toolkit.cycif.config import CellTypeMarkerConfig, ProgramDefinition
from spatial_omics_toolkit.cycif.qc import step06_qc_filter_intensities
from spatial_omics_toolkit.cycif.spatial_features import step10_spatial_features
from spatial_omics_toolkit.cycif.spatial_null import step11_spatial_null


LIMITATIONS = [
    "This workflow analyzes a selected, already-segmented MCMICRO cell-feature table; it does not inventory, register, or segment raw images.",
    "step13_pseudobulk is unavailable for continuous MCMICRO intensity values; this workflow does not treat intensities as raw counts.",
    "step10 spatial features are descriptive only. Spatial association inference requires an explicitly configured step11 spatial null.",
    "step11 raw p-values are always exploratory; the workflow does not rerun the full pipeline to adjust for selection.",
    "Spatial values use explicitly declared x/y coordinate units and frame; no scale conversion or coordinate transform is inferred.",
    "step12 comparisons use one value per independent biological_unit_id, aggregated directly from raw cells; cells and imaging cores are not treated as independent biological replicates.",
]


def _version() -> str:
    try:
        return importlib.metadata.version("spatial-omics-toolkit")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _limitations_for(params: dict[str, Any]) -> list[str]:
    limitations = list(LIMITATIONS)
    units = str(params.get("coordinate_units") or "").strip().lower()
    if units in {"pixel", "pixels", "px"}:
        limitations.append("Coordinates were declared in pixels; step10 radii and spatial outputs are in pixel-coordinate units only, with no micrometer conversion.")
    return limitations


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    compressed = suffixes[-1:] == [".gz"]
    data_suffix = suffixes[-2] if compressed and len(suffixes) > 1 else suffixes[-1] if suffixes else ""
    if data_suffix in {".parquet", ".pq"} and not compressed:
        return pd.read_parquet(path)
    if data_suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", compression="infer")
    if data_suffix == ".csv":
        return pd.read_csv(path, compression="infer")
    raise ValueError("Feature table must be CSV, TSV, or Parquet.")


def _load_config(raw: Any, label: str) -> dict[str, Any]:
    if isinstance(raw, (str, Path)):
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{label} config file does not exist: {path}")
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object or a path to one.")
    return dict(raw)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolved(request: RunRequest) -> tuple[dict[str, Any], list[str], list[str]]:
    p = request.parameters
    blockers: list[str] = []
    warnings: list[str] = []
    params: dict[str, Any] = {"acquisition_groups": [], "has_specimen_id": False}
    frame: pd.DataFrame | None = None
    feature_raw = p.get("feature_table_path")
    if not isinstance(feature_raw, str) or not feature_raw.strip():
        blockers.append("Select the exact upstream MCMICRO cell-feature table with feature_table_path.")
    else:
        feature_path = Path(feature_raw).expanduser().resolve()
        allowed = [request.modality_dir.resolve(), request.run_dir.resolve()]
        if not feature_path.is_file():
            blockers.append(f"Selected cell-feature table does not exist: {feature_path}")
        elif not any(_inside(feature_path, root) for root in allowed):
            blockers.append("feature_table_path must be under the selected CyCIF modality folder or this run's artifact directory.")
        else:
            try:
                frame = _read_table(feature_path)
                params["feature_table_path"] = str(feature_path)
                params["input_table_rows"] = int(len(frame))
                params["input_table_columns"] = list(map(str, frame.columns))
            except Exception as exc:
                blockers.append(f"Unable to read selected cell-feature table: {exc}")
                frame = None
    if not isinstance(p.get("columns"), dict):
        blockers.append("Provide columns mapping with acquisition_group, core_id, x, y, and intensity_columns.")
        columns = {}
    else:
        columns = dict(p["columns"])
    required_keys = ("acquisition_group", "core_id", "x", "y", "intensity_columns")
    missing_keys = [key for key in required_keys if key not in columns]
    if missing_keys:
        blockers.append("Column map is missing: " + ", ".join(missing_keys))
    if not isinstance(columns.get("intensity_columns"), list) or not columns.get("intensity_columns"):
        blockers.append("columns.intensity_columns must list one or more continuous marker intensity columns.")
    if frame is not None:
        required_cols = [columns.get(k) for k in ("acquisition_group", "core_id", "x", "y")]
        if isinstance(columns.get("intensity_columns"), list):
            required_cols.extend(columns["intensity_columns"])
        if columns.get("specimen_id") is None and "specimen_id" in frame.columns:
            columns["specimen_id"] = "specimen_id"
        if columns.get("specimen_id") is not None:
            required_cols.append(columns["specimen_id"])
        if any(not isinstance(c, str) or c not in frame.columns for c in required_cols):
            blockers.append("The column map references one or more columns absent from the selected feature table.")
        if not frame.empty and any(frame[c].isna().any() for c in [columns.get("acquisition_group"), columns.get("core_id") if isinstance(columns.get("core_id"), str) else ""] if c in frame.columns):
            blockers.append("Acquisition-group and core identifiers cannot contain missing values.")
        params["columns"] = columns
        params["has_specimen_id"] = isinstance(columns.get("specimen_id"), str) and columns["specimen_id"] in frame.columns
        if columns.get("core_id") in frame.columns and columns.get("acquisition_group") in frame.columns:
            groups = frame.groupby(columns["acquisition_group"], dropna=False)[columns["core_id"]].agg(lambda values: sorted(set(values.astype(str))))
            params["acquisition_groups"] = [
                {"group_key": str(core), "label": str(group), "input_paths": [params.get("feature_table_path", str(feature_raw))]}
                for group, cores in groups.items() for core in cores
            ]
    else:
        params["has_specimen_id"] = False
        params["acquisition_groups"] = []
    use_specimen_column = p.get("use_specimen_id_column") is True
    if use_specimen_column and not params["has_specimen_id"]:
        blockers.append("use_specimen_id_column=true requires columns.specimen_id to identify an existing specimen_id column.")
    if p.get("specimen_mapping_confirmed") is not True:
        blockers.append("Confirm the acquisition-group/core to project_samples specimen and biological-unit mapping (specimen_mapping_confirmed=true).")
    mapping = p.get("core_to_specimen")
    if isinstance(mapping, str):
        try:
            with Path(mapping).expanduser().open(encoding="utf-8") as stream:
                mapping = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"Unable to read core_to_specimen JSON mapping: {exc}")
            mapping = {}
    if not isinstance(mapping, dict):
        mapping = {}
    if not use_specimen_column and not mapping:
        blockers.append("Provide core_to_specimen keyed by exact feature-table core_id values, or explicitly confirm use_specimen_id_column=true.")
    if use_specimen_column and mapping:
        blockers.append("Use either the explicitly selected specimen_id column or core_to_specimen mapping, not both.")
    if not request.annotation_csv.is_file():
        blockers.append(f"project_samples.csv is missing: {request.annotation_csv}")
        sample_rows: list[dict[str, str]] = []
    else:
        try:
            with request.annotation_csv.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                needed = {"specimen_id", "biological_unit_id", "condition"}
                if not needed.issubset(reader.fieldnames or []):
                    blockers.append("project_samples.csv must contain specimen_id, biological_unit_id, and condition columns for this workflow.")
                    sample_rows = []
                else:
                    sample_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            blockers.append(f"Unable to read project_samples.csv: {exc}")
            sample_rows = []
    sample_by_specimen: dict[str, dict[str, str]] = {}
    for row in sample_rows:
        specimen = (row.get("specimen_id") or "").strip()
        if not specimen or not (row.get("biological_unit_id") or "").strip() or not (row.get("condition") or "").strip():
            blockers.append("project_samples.csv has an empty specimen_id, biological_unit_id, or condition value.")
            continue
        if specimen in sample_by_specimen:
            blockers.append(f"project_samples.csv contains duplicate specimen_id {specimen!r}; mapping is ambiguous.")
        sample_by_specimen[specimen] = row
    submitted_core_mapping: dict[str, str] = {}
    if not use_specimen_column:
        submitted_core_mapping = {str(k): str(v).strip() for k, v in mapping.items()}
    resolved_map: dict[str, dict[str, str]] = {}
    if frame is not None and isinstance(columns.get("core_id"), str) and columns["core_id"] in frame:
        observed_cores = set(frame[columns["core_id"]].dropna().astype(str))
        if use_specimen_column:
            specimen_col = columns["specimen_id"]
            specimen_values = frame.groupby(columns["core_id"])[specimen_col].agg(lambda values: set(values.dropna().astype(str)))
            for core, values in specimen_values.items():
                if len(values) != 1:
                    blockers.append(f"Core {str(core)!r} must have exactly one specimen_id in the selected table.")
                    continue
                specimen = next(iter(values))
                row = sample_by_specimen.get(specimen)
                if row is None:
                    blockers.append(f"specimen_id {specimen!r} for core {str(core)!r} is absent from project_samples.csv.")
                else:
                    resolved_map[str(core)] = {"specimen_id": specimen, "biological_unit_id": row["biological_unit_id"].strip(), "condition": row["condition"].strip()}
        else:
            if set(submitted_core_mapping) != observed_cores:
                blockers.append("core_to_specimen must map every and only observed core_id value in the selected table.")
            for core, specimen in submitted_core_mapping.items():
                row = sample_by_specimen.get(specimen)
                if row is None:
                    blockers.append(f"Mapped specimen_id {specimen!r} for core {core!r} is absent from project_samples.csv.")
                else:
                    resolved_map[core] = {"specimen_id": specimen, "biological_unit_id": row["biological_unit_id"].strip(), "condition": row["condition"].strip()}
    if frame is not None and isinstance(columns.get("core_id"), str) and columns["core_id"] in frame and isinstance(columns.get("acquisition_group"), str) and columns["acquisition_group"] in frame:
        core_groups = frame.groupby(columns["core_id"])[columns["acquisition_group"]].nunique(dropna=False)
        if (core_groups > 1).any():
            blockers.append("Each exact core_id value must belong to only one acquisition_group; core-to-specimen mapping would otherwise be ambiguous.")
    params["core_to_specimen"] = submitted_core_mapping
    params["resolved_core_to_specimen"] = resolved_map
    params["specimen_mapping_confirmed"] = p.get("specimen_mapping_confirmed") is True
    params["use_specimen_id_column"] = use_specimen_column
    params["annotation_csv"] = str(request.annotation_csv.resolve())
    params["pseudobulk"] = "not included: continuous MCMICRO intensity values are not raw counts"
    params["toolkit_version"] = _version()
    configs: dict[str, Any] = {}
    for key in ("qc", "classification", "programs", "aggregate", "spatial_features", "spatial_null", "compare_groups"):
        raw = p.get(key)
        if raw is None or raw is False:
            continue
        if raw is True:
            blockers.append(f"{key} was enabled without its required explicit configuration.")
            continue
        try:
            configs[key] = _load_config(raw, key)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(str(exc))
    if "classification" in configs:
        try:
            configs["classification"]["marker_config"] = CellTypeMarkerConfig(**configs["classification"]["marker_config"])
            marker_config = configs["classification"]["marker_config"]
            missing = set(marker_config.z_score_markers) - set(frame.columns if frame is not None else [])
            if missing:
                blockers.append("Classification config references missing marker columns: " + ", ".join(sorted(missing)))
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"Invalid classification marker_config: {exc}")
    if "programs" in configs:
        try:
            raw_programs = configs["programs"].get("definitions")
            if not isinstance(raw_programs, list) or not raw_programs:
                raise ValueError("programs.definitions must be a non-empty list of ProgramDefinition objects")
            configs["programs"]["definitions"] = [ProgramDefinition(**item) for item in raw_programs]
            missing = {m for program in configs["programs"]["definitions"] for m in program.marker_columns} - set(frame.columns if frame is not None else [])
            if missing:
                blockers.append("Program config references missing intensity columns: " + ", ".join(sorted(missing)))
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"Invalid programs config: {exc}")
    if "programs" in configs and "classification" not in configs and frame is not None:
        missing_z = {
            f"{marker}_z"
            for program in configs["programs"].get("definitions", [])
            for marker in program.marker_columns
            if f"{marker}_z" not in frame.columns
        }
        if missing_z:
            blockers.append("programs requires step07 classification or pre-existing marker z-score columns: " + ", ".join(sorted(missing_z)))
    for stage_key in ("aggregate", "spatial_features", "spatial_null"):
        if stage_key in configs and "classification" not in configs and frame is not None:
            type_col = configs[stage_key].get("type_column", "cell_type")
            if type_col not in frame.columns:
                blockers.append(f"{stage_key} requires an existing {type_col!r} column or explicitly enabled classification stage.")
    if "spatial_null" in configs and "classification" not in configs and frame is not None:
        type_col = configs["spatial_null"].get("type_column", "cell_type")
        available_types = set(frame[type_col].dropna().astype(str)) if type_col in frame.columns else set()
        for key in ("source_type", "target_type"):
            if configs["spatial_null"].get(key) not in available_types:
                blockers.append(f"spatial_null {key} must identify an observed cell type when classification is not enabled.")
    params["enabled_stages"] = list(configs)
    spatial_enabled = "spatial_features" in configs or "spatial_null" in configs
    coordinate_units = p.get("coordinate_units")
    coordinate_frame = p.get("coordinate_frame")
    if spatial_enabled:
        if not isinstance(coordinate_units, str) or not coordinate_units.strip():
            blockers.append("Spatial stages require an explicit coordinate_units declaration; no coordinate unit is inferred.")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            blockers.append("Spatial stages require an explicit coordinate_frame declaration (for example, native or registered).")
        if isinstance(coordinate_units, str) and coordinate_units.strip().lower() in {"um", "µm", "μm", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres"} and p.get("coordinate_scale_verified") is not True:
            blockers.append("Physical micrometer coordinates require explicit confirmation that the source scale/metadata was verified; the adapter does not infer or convert scale.")
        params["coordinate_units"] = coordinate_units.strip() if isinstance(coordinate_units, str) else None
        params["coordinate_frame"] = coordinate_frame.strip() if isinstance(coordinate_frame, str) else None
        params["coordinate_scale_verified"] = p.get("coordinate_scale_verified") is True
        if "spatial_features" in configs:
            radii = configs["spatial_features"].get("radii")
            if not isinstance(radii, list) or not radii or any(not isinstance(radius, (int, float)) or radius <= 0 for radius in radii):
                blockers.append("spatial_features.radii must explicitly list positive distances in the declared coordinate_units.")
    if "spatial_null" in configs:
        null_mask = configs["spatial_null"].get("tissue_mask_column")
        if null_mask is not None and (not isinstance(null_mask, str) or frame is None or null_mask not in frame.columns):
            blockers.append("spatial_null.tissue_mask_column must identify an existing input cell-table column when supplied.")
        prespecified = configs["spatial_null"].get("pre_specified")
        selected_after_viewing = configs["spatial_null"].get("selection_after_viewing_data")
        confirmed = configs["spatial_null"].get("pre_specification_confirmed")
        statement = configs["spatial_null"].get("pre_specification_statement")
        selection_statement = configs["spatial_null"].get("selection_statement")
        if not isinstance(prespecified, bool):
            blockers.append("spatial_null.pre_specified must explicitly be true or false.")
        if not isinstance(selected_after_viewing, bool):
            blockers.append("spatial_null.selection_after_viewing_data must explicitly be true or false.")
        if prespecified is True and selected_after_viewing is True:
            blockers.append("A pair selected after viewing results cannot be declared pre-specified.")
        if confirmed is not True:
            blockers.append("Explicitly confirm the spatial-null selection/pre-specification statement with pre_specification_confirmed=true.")
        if prespecified is True and (not isinstance(statement, str) or not statement.strip()):
            blockers.append("pre_specification_statement is required when pre_specified=true.")
        if not isinstance(selection_statement, str) or not selection_statement.strip():
            blockers.append("spatial_null.selection_statement must record why the pair/parameters were selected.")
        params["spatial_null_claim_label"] = "Exploratory"
        params["spatial_null_pre_specified"] = prespecified is True
        params["spatial_null_selection_after_viewing_data"] = selected_after_viewing is True
        params["spatial_null_pre_specification_confirmed"] = confirmed is True
        params["spatial_null_pre_specification_statement"] = statement.strip() if isinstance(statement, str) else None
        params["spatial_null_selection_statement"] = selection_statement.strip() if isinstance(selection_statement, str) else None
        params["spatial_null_selection_adjustment"] = "not implemented; all selected-pair outputs remain exploratory"
        params["spatial_null_region_assumption"] = (
            f"label permutations restricted within {null_mask!r}" if null_mask else
            "label permutations across the whole core; no tissue-mask restriction"
        )
        params["spatial_null_selected_pair"] = [configs["spatial_null"].get("source_type"), configs["spatial_null"].get("target_type")]
        params["spatial_null_k"] = configs["spatial_null"].get("k")
        params["spatial_null_permutations"] = configs["spatial_null"].get("n_permutations")
    params["stage_configs"] = {key: _jsonable(value) for key, value in configs.items()}
    if "spatial_null" in configs:
        null_config = configs["spatial_null"]
        if not all(k in null_config for k in ("source_type", "target_type", "k", "n_permutations")):
            blockers.append("spatial_null requires explicit source_type, target_type, k, and n_permutations.")
        if not isinstance(null_config.get("k"), int) or null_config.get("k", 0) < 1 or not isinstance(null_config.get("n_permutations"), int) or null_config.get("n_permutations", 0) < 1:
            blockers.append("spatial_null k and n_permutations must be positive integers.")
        if "spatial_features" not in configs:
            blockers.append("spatial_null requires an explicitly enabled spatial_features stage so the tested statistic is also available descriptively.")
        spatial_cfg = configs.get("spatial_features", {})
        selected_pairs = [tuple(pair) for pair in spatial_cfg.get("spatial_pairs", []) if isinstance(pair, (list, tuple)) and len(pair) == 2]
        if (null_config.get("source_type"), null_config.get("target_type")) not in selected_pairs:
            blockers.append("spatial_features.spatial_pairs must explicitly include the same source_type/target_type selected for spatial_null.")
        if null_config.get("k") not in spatial_cfg.get("knn_k_values", []):
            blockers.append("spatial_features.knn_k_values must explicitly include the k selected for spatial_null.")
    if "compare_groups" in configs:
        cfg = configs["compare_groups"]
        if "aggregate" not in configs:
            blockers.append("compare_groups requires aggregate enabled with level='biological_unit_id' to guarantee direct raw-cell aggregation.")
        elif configs["aggregate"].get("level") != "biological_unit_id":
            blockers.append("compare_groups is blocked: aggregation must be directly from raw cells to biological_unit_id, never core then patient.")
        if not cfg.get("metrics") or not isinstance(cfg.get("metrics"), list):
            blockers.append("compare_groups.metrics must explicitly list the numeric outcomes to test.")
        if len({v["condition"] for v in resolved_map.values()}) != 2:
            blockers.append("compare_groups requires exactly two mapped condition groups.")
        if frame is not None and resolved_map:
            unit_groups: dict[str, set[str]] = {}
            for core, entry in resolved_map.items():
                unit_groups.setdefault(entry["biological_unit_id"], set()).add(entry["condition"])
            if any(len(groups) != 1 for groups in unit_groups.values()):
                blockers.append("Each biological_unit_id must map to exactly one condition group.")
            mapped_units_by_group = {g: {u for u, gs in unit_groups.items() if g in gs} for g in {e["condition"] for e in resolved_map.values()}}
            if len(mapped_units_by_group) != 2 or any(len(units) < 2 for units in mapped_units_by_group.values()):
                blockers.append("Each of exactly two groups must contain at least two independent biological units for testing.")
    if not configs:
        warnings.append("No numerical analysis stage is enabled; select explicit stage configurations to run analysis.")
    if "aggregate" in configs and configs["aggregate"].get("level") not in {"core_id", "biological_unit_id"}:
        blockers.append("aggregate.level must explicitly be 'core_id' or 'biological_unit_id'.")
    return params, blockers, warnings


def _jsonable(value: Any) -> Any:
    if isinstance(value, CellTypeMarkerConfig) or isinstance(value, ProgramDefinition):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def describe() -> WorkflowDescriptor:
    tool = [ToolReference(name="spatial-omics-toolkit", version=_version(), role="Analyze cell-feature tables")]
    return WorkflowDescriptor(
        workflow_id="cycif.tabular_analysis",
        modality="CycIF",
        title="CyCIF cell-feature table analysis",
        description="Analyze a selected MCMICRO table of segmented-cell features using the optional analysis stages configured below.",
        stages=[
            StageDefinition(
                stage_id=f"step{n:02d}",
                title=title,
                purpose=purpose,
                tools=tool,
                implementations=[FunctionReference(source_file=source_file, function_name=function_name, role="Run the optional configured analysis step")],
            )
            for n, title, purpose, source_file, function_name in [
                (6, "Intensity quality control", "Correct background and assess marker signal-to-noise ratios.", "spatial_omics_toolkit/cycif/qc.py", "step06_qc_filter_intensities"),
                (7, "Classify cells", "Classify cells within each core using configured marker thresholds.", "spatial_omics_toolkit/cycif/classify.py", "step07_classify_cells"),
                (8, "Score marker programs", "Calculate per-cell program scores from configured marker definitions.", "spatial_omics_toolkit/cycif/classify.py", "step08_score_programs"),
                (9, "Summarize cells", "Summarize cells by imaging core or biological unit.", "spatial_omics_toolkit/cycif/aggregate.py", "step09_aggregate_per_core"),
                (10, "Describe spatial features", "Calculate descriptive spatial features without making inferential claims.", "spatial_omics_toolkit/cycif/spatial_features.py", "step10_spatial_features"),
                (11, "Test spatial associations", "Permute spatial labels for a selected pair of cell types; the resulting p-values are exploratory.", "spatial_omics_toolkit/cycif/spatial_null.py", "step11_spatial_null"),
                (12, "Compare groups", "Compare two groups using summaries calculated directly from cells for each biological unit.", "spatial_omics_toolkit/cycif/compare.py", "step12_compare_groups"),
            ]
        ],
        input_description=(
            "- Select an exact existing, already-segmented MCMICRO cell-feature table in the CycIF directory or current run artifacts: CSV, TSV/`.txt`, or Parquet/`.pq` (CSV and TSV/`.txt` may be gzip-compressed). This workflow does not process raw images or segment cells.\n"
            "- Map existing acquisition-group, core/imaging-field ID, x and y coordinate columns and one or more continuous marker-intensity columns. Group and core IDs must be present, and each core must belong to exactly one acquisition group; coordinates are required even without spatial analysis.\n"
            "- Explicitly confirm either a complete exact core-to-`specimen_id` mapping or use of an existing specimen-ID column, not both. Each core must resolve to one specimen in `project_samples.csv`, whose required fields include `specimen_id`, `biological_unit_id`, `condition`, and `tissue_source`.\n"
            "- Every numerical stage is optional and requires its own JSON-object configuration. Intensity QC uses background and signal percentiles; classification requires an explicit marker configuration; program scoring requires definitions and marker z-scores from classification or the input.\n"
            "- Aggregation requires a `core_id` or `biological_unit_id` level. Aggregation and spatial stages require classification or an existing configured cell-type column.\n"
            "- Spatial stages require declared coordinate units and frame; micrometer units require confirmation of verified source scale. Descriptive features require positive radii. Spatial null additionally requires a selected source/target pair, positive `k` and permutation count, selection declarations and confirmation, and matching pair and `k` in spatial features; an optional tissue-mask column restricts permutations.\n"
            "- Group comparison requires direct cell-to-biological-unit aggregation, an explicit metrics list, exactly two conditions, one condition per unit, and at least two independent biological units per condition."
        ),
        parameter_fields=[
            ParameterField(name="feature_table_path", label="MCMICRO cell-feature table", kind="path", required=True, section="Input", help_text="Select the exact existing CSV/TSV/Parquet table produced upstream by MCMICRO; no filename inference, raw-image processing, or segmentation is performed."),
            ParameterField(name="columns.acquisition_group", label="Acquisition-group column", kind="select", required=True, dynamic_choices_key="input_table_columns", section="Input", help_text="Exact source table column identifying acquisition group."),
            ParameterField(name="columns.core_id", label="Core/imaging-field column", kind="select", required=True, dynamic_choices_key="input_table_columns", section="Input", help_text="Exact source table column identifying the core used for mapping and per-core operations."),
            ParameterField(name="columns.specimen_id", label="Existing specimen_id column", kind="select", dynamic_choices_key="input_table_columns", section="Specimen mapping", help_text="Optional. Use only with the separate explicit use_specimen_id_column confirmation; otherwise map every exact core_id through core_to_specimen."),
            ParameterField(name="columns.x", label="Cell x-coordinate column", kind="select", required=True, dynamic_choices_key="input_table_columns", section="Input"),
            ParameterField(name="columns.y", label="Cell y-coordinate column", kind="select", required=True, dynamic_choices_key="input_table_columns", section="Input"),
            ParameterField(name="columns.intensity_columns", label="Continuous intensity columns", kind="multiselect", required=True, dynamic_choices_key="input_table_columns", section="Input", help_text="Select all marker intensity columns; these are continuous values, not raw counts."),
            ParameterField(name="coordinate_units", label="Coordinate units", kind="text", section="Spatial coordinates", help_text="Required whenever step10 or step11 is enabled. Declare actual source units (for example, pixels); micrometers require independently verified source scale metadata and explicit confirmation below."),
            ParameterField(name="coordinate_frame", label="Coordinate frame", kind="text", section="Spatial coordinates", help_text="Required whenever step10 or step11 is enabled; state whether x/y are native, registered, or another explicit frame. No coordinate transform is inferred."),
            ParameterField(name="coordinate_scale_verified", label="Confirm verified physical scale", kind="bool", default=False, section="Spatial coordinates", help_text="Required for micrometer declarations. Confirm source metadata/calibration was checked; this workflow performs no unit conversion."),
            ParameterField(name="specimen_mapping_confirmed", label="Confirm specimen mapping", kind="bool", required=True, default=False, section="Specimen mapping", help_text="Confirm the reviewed mapping of exact acquisition-group/core values to project_samples.csv specimen_id and biological_unit_id."),
            ParameterField(name="core_to_specimen", label="Core-to-specimen mapping JSON", kind="config_file", section="Specimen mapping", help_text="Optional JSON object mapping every exact source core_id value to a project_samples.csv specimen_id; required unless use_specimen_id_column=true."),
            ParameterField(name="use_specimen_id_column", label="Use existing specimen_id column", kind="bool", default=False, section="Specimen mapping", help_text="Enable only after confirming columns.specimen_id exists and correctly identifies specimens. Mutually exclusive with core_to_specimen."),
            ParameterField(name="qc", label="Step 06 intensity-QC configuration", kind="config_file", section="Optional analysis stages", stage_id="step06", help_text="Optional JSON object enables step06; supported values: background_percentile, signal_percentile, background_thresholds keyed by selected intensity column."),
            ParameterField(name="classification", label="Step 07 classification configuration", kind="config_file", section="Optional analysis stages", stage_id="step07", help_text="Optional JSON object enables step07; REQUIRED key marker_config is a serialized CellTypeMarkerConfig with epithelial_markers, mesenchymal_markers, immune_marker, myeloid_markers, antigen_presentation_marker, stromal_marker, endothelial_marker, proliferation_marker, z_score_markers, and optional thresholds. Every referenced marker must be a selected table column."),
            ParameterField(name="programs", label="Step 08 program definitions", kind="config_file", section="Optional analysis stages", stage_id="step08", help_text="Optional JSON object enables step08; REQUIRED definitions is a non-empty list of {name, marker_columns} ProgramDefinition values; referenced marker z-score columns must be produced by step07 or already exist."),
            ParameterField(name="aggregate", label="Step 09 aggregation configuration", kind="config_file", section="Optional analysis stages", stage_id="step09", help_text="Optional JSON object enables step09; REQUIRED level is exactly core_id or biological_unit_id. Comparisons require biological_unit_id for direct raw-cell aggregation."),
            ParameterField(name="spatial_features", label="Step 10 descriptive spatial configuration", kind="config_file", section="Optional analysis stages", stage_id="step10", unit="x/y coordinate units", help_text="Optional JSON object enables descriptive step10; optional spatial_pairs are [source_type,target_type] pairs; numeric radii values must use the same units as the selected x/y coordinate columns."),
            ParameterField(name="spatial_null", label="Step 11 spatial-null configuration", kind="config_file", section="Optional analysis stages", stage_id="step11", help_text="Optional JSON object enables step11; REQUIRED source_type, target_type, k, n_permutations, pre_specified, selection_after_viewing_data, pre_specification_confirmed=true, and selection_statement. pre_specification_statement is also required if pre_specified=true. The adapter records the statement and confirmation time, but because selection-adjusted full-pipeline reruns are not implemented, every selected-pair p-value remains Exploratory regardless of checkbox. Optional tissue_mask_column restricts swaps; absent means whole-core permutations."),
            ParameterField(name="compare_groups", label="Step 12 group-comparison configuration", kind="config_file", section="Optional analysis stages", stage_id="step12", help_text="Optional JSON object enables step12; REQUIRED metrics list. Runs only after direct raw-cell aggregation to biological_unit_id with exactly one row per independent unit, two mapped groups, complete project_samples.csv mapping, and at least two independent units per group."),
        ],
        output_description=(
            "- Run records under `cycif_analysis/` include resolved parameters, input mapping, stage statuses, and limitations; a blocked preflight records the available parameters, statuses, and limitations.\n"
            "- Only selected stages that execute successfully produce their products: QC-corrected cells and marker-SNR JSON, classified or program-scored cells, core or biological-unit aggregates, descriptive spatial features and coordinate metadata, spatial-null results and metadata, and biological-unit group comparisons. QC plots are collected when produced.\n"
            "- Spatial-null selected-pair p-values are exploratory, including for pre-specified pairs; selection-adjusted full-pipeline reruns are not implemented. Without a tissue mask, permutations span the whole core.\n"
            "- With no stages selected, the workflow warns and performs no numerical analysis. Continuous intensities are not raw counts, and pseudobulk is not offered."
        ),
        limitations=LIMITATIONS,
    )


def preflight(request: RunRequest) -> PreflightResult:
    params, blockers, warnings = _resolved(request)
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        out_root = request.run_dir / "cycif_analysis"
        out_root.mkdir(parents=True, exist_ok=True)
        event = StageEvent(stage_id="preflight", status="BLOCKED", message="; ".join(check.blockers), parameters=check.resolved_parameters)
        (out_root / "resolved_parameters.json").write_text(json.dumps(check.resolved_parameters, indent=2, sort_keys=True), encoding="utf-8")
        limitations = _limitations_for(check.resolved_parameters)
        (out_root / "limitations.json").write_text(json.dumps(limitations, indent=2), encoding="utf-8")
        (out_root / "stage_statuses.json").write_text(json.dumps([event.model_dump(mode="json")], indent=2), encoding="utf-8")
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], artifacts=[str(out_root / "resolved_parameters.json"), str(out_root / "limitations.json"), str(out_root / "stage_statuses.json")], warnings=check.warnings + check.blockers)
    params = check.resolved_parameters
    if "spatial_null" in params.get("enabled_stages", []):
        params["spatial_null_confirmation_recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    limitations = _limitations_for(params)
    out_root = request.run_dir / "cycif_analysis"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_parameters.json").write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    frame = _read_table(Path(params["feature_table_path"]))
    col = params["columns"]
    acquisition_col, core_col, x_col, y_col = (col[k] for k in ("acquisition_group", "core_id", "x", "y"))
    resolved_core_to_specimen = params["resolved_core_to_specimen"]
    frame["specimen_id"] = frame[core_col].astype(str).map(
        {key: value["specimen_id"] for key, value in resolved_core_to_specimen.items()}
    )
    frame["biological_unit_id"] = frame[core_col].astype(str).map(
        {key: value["biological_unit_id"] for key, value in resolved_core_to_specimen.items()}
    )
    frame["condition"] = frame[core_col].astype(str).map(
        {key: value["condition"] for key, value in resolved_core_to_specimen.items()}
    )
    if frame[["specimen_id", "biological_unit_id", "condition"]].isna().any().any():
        raise ValueError("Resolved core-to-specimen mapping failed to cover input rows.")
    mapping = {"source_table": params["feature_table_path"], "acquisition_group_column": acquisition_col, "core_column": core_col, "mapping_confirmation": True, "project_samples_csv": params["annotation_csv"]}
    # The explicit group field is preserved as source provenance; core mapping is never inferred from names.
    (out_root / "input_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    warnings = list(check.warnings) + ["step13_pseudobulk not run: continuous MCMICRO intensities are not raw counts.", "step10 spatial feature outputs are descriptive only."]
    if "spatial_null" in params.get("enabled_stages", []):
        warnings.append("All step11 selected-pair p-values are retained as Exploratory because selection-adjusted full-pipeline correction is not implemented, regardless of pre-specification confirmation.")
    stages: list[StageEvent] = []
    artifacts = [str(out_root / "resolved_parameters.json"), str(out_root / "input_mapping.json")]
    configs = params["stage_configs"]
    # Reconstruct typed config objects after the JSON-safe preflight record was built.
    
    def record(stage_id: str, status: str, message: str, stage_dir: Path, parameters: dict[str, Any] | None = None) -> None:
        files = sorted(str(p) for p in stage_dir.rglob("*") if p.is_file())
        event = StageEvent(stage_id=stage_id, status=status, message=message, parameters=parameters or {}, artifacts=files, tool_versions={"spatial-omics-toolkit": params["toolkit_version"]})
        stages.append(event)
        artifacts.extend(files)
        emit(event)
    try:
        if "qc" in configs:
            cfg = configs["qc"]
            stage_dir = out_root / "step06_qc"
            stage_dir.mkdir(exist_ok=True)
            intensity_cols = col["intensity_columns"]
            qc = step06_qc_filter_intensities(frame[intensity_cols].apply(pd.to_numeric, errors="raise"), background_percentile=cfg.get("background_percentile", 10.0), signal_percentile=cfg.get("signal_percentile", 90.0), background_thresholds=cfg.get("background_thresholds"), output_dir=stage_dir, verbose=False)
            frame.loc[:, intensity_cols] = qc[intensity_cols]
            frame.to_csv(stage_dir / "cell_features_qc.csv", index=False)
            snr = qc.attrs.get("marker_snr", {})
            (stage_dir / "marker_snr.json").write_text(json.dumps(snr, indent=2), encoding="utf-8")
            record("step06", "SUCCEEDED", "QC-corrected cell table written; marker SNR and plot recorded.", stage_dir, cfg)
        if "classification" in configs:
            cfg = configs["classification"]
            stage_dir = out_root / "step07_classification"
            stage_dir.mkdir(exist_ok=True)
            marker_config = CellTypeMarkerConfig(**cfg["marker_config"])
            frame = step07_classify_cells(frame, marker_config, core_col=cfg.get("core_column", core_col), output_dir=stage_dir, verbose=False)
            frame.to_csv(stage_dir / "cell_features_classified.csv", index=False)
            record("step07", "SUCCEEDED", "Classified cell table written using the explicitly supplied marker config.", stage_dir, {**cfg, "marker_config": asdict(marker_config)})
        if "programs" in configs:
            cfg = configs["programs"]
            stage_dir = out_root / "step08_programs"
            stage_dir.mkdir(exist_ok=True)
            definitions = [ProgramDefinition(**v) for v in cfg["definitions"]]
            frame = step08_score_programs(frame, definitions, z_suffix=cfg.get("z_suffix", "_z"), output_dir=stage_dir, verbose=False)
            frame.to_csv(stage_dir / "cell_features_program_scored.csv", index=False)
            record("step08", "SUCCEEDED", "Per-cell program score table written from explicitly supplied program definitions.", stage_dir, {"definitions": [asdict(v) for v in definitions], "z_suffix": cfg.get("z_suffix", "_z")})
        aggregate_frame = None
        if "aggregate" in configs:
            cfg = configs["aggregate"]
            stage_dir = out_root / "step09_aggregation"
            stage_dir.mkdir(exist_ok=True)
            level = cfg["level"]
            target_col = "biological_unit_id" if level == "biological_unit_id" else core_col
            if level == "biological_unit_id":
                # Direct raw-cell aggregation; never aggregate a core summary again.
                aggregate_input = frame
            else:
                aggregate_input = frame
            aggregate_frame = step09_aggregate_per_core(aggregate_input, core_col=target_col, type_col=cfg.get("type_column", "cell_type"), type_categories=cfg.get("type_categories"), marker_cols=cfg.get("marker_columns", col["intensity_columns"]), program_score_cols=cfg.get("program_score_columns"), extra_group_col="condition", output_dir=stage_dir, verbose=False)
            if level == "biological_unit_id":
                unit_to_specimen = frame.groupby("biological_unit_id")["specimen_id"].agg(lambda s: ";".join(sorted(set(s.astype(str)))))
                unit_to_condition = frame.groupby("biological_unit_id")["condition"].agg(lambda s: ";".join(sorted(set(s.astype(str)))))
                aggregate_frame["specimen_id"] = aggregate_frame[target_col].map(unit_to_specimen)
                aggregate_frame["condition"] = aggregate_frame[target_col].map(unit_to_condition)
            aggregate_frame.to_csv(stage_dir / ("biological_unit_features.csv" if level == "biological_unit_id" else "core_features.csv"), index=False)
            record("step09", "SUCCEEDED", f"Raw cell table aggregated directly by {target_col}; output layer is {level}.", stage_dir, cfg)
        if "spatial_features" in configs:
            cfg = configs["spatial_features"]
            stage_dir = out_root / "step10_spatial_features"
            stage_dir.mkdir(exist_ok=True)
            spatial_core = cfg.get("core_column", core_col)
            result = step10_spatial_features(frame, core_col=spatial_core, x_col=x_col, y_col=y_col, type_col=cfg.get("type_column", "cell_type"), broad_types=cfg.get("broad_types"), spatial_pairs=[tuple(pair) for pair in cfg["spatial_pairs"]] if "spatial_pairs" in cfg else [], program_cols=cfg.get("program_columns", []), morphology_cols=cfg.get("morphology_columns", []), grid_sizes=cfg.get("grid_sizes", [4, 8]), knn_k_values=cfg.get("knn_k_values", [5, 10]), radii=cfg["radii"], output_dir=stage_dir, verbose=False)
            result["coordinate_units"] = params["coordinate_units"]
            result["coordinate_frame"] = params["coordinate_frame"]
            result["radii_units"] = params["coordinate_units"]
            result.to_csv(stage_dir / "spatial_features_descriptive.csv", index=False)
            (stage_dir / "spatial_metadata.json").write_text(json.dumps({"coordinate_units": params["coordinate_units"], "coordinate_frame": params["coordinate_frame"], "coordinate_scale_verified": params["coordinate_scale_verified"], "radii": cfg["radii"], "radii_units": params["coordinate_units"], "spatial_pairs": cfg.get("spatial_pairs", []), "interpretation": "descriptive only"}, indent=2), encoding="utf-8")
            record("step10", "SUCCEEDED", f"Descriptive spatial features written in {params['coordinate_units']} ({params['coordinate_frame']} frame); no inferential claim.", stage_dir, {**cfg, "coordinate_units": params["coordinate_units"], "coordinate_frame": params["coordinate_frame"]})
        if "spatial_null" in configs:
            cfg = configs["spatial_null"]
            stage_dir = out_root / "step11_spatial_null"
            stage_dir.mkdir(exist_ok=True)
            result = step11_spatial_null(frame, source_type=cfg["source_type"], target_type=cfg["target_type"], core_col=cfg.get("core_column", core_col), x_col=x_col, y_col=y_col, type_col=cfg.get("type_column", "cell_type"), k=cfg["k"], n_permutations=cfg["n_permutations"], seed=cfg.get("seed", 0), tissue_mask_col=cfg.get("tissue_mask_column"), alternative=cfg.get("alternative", "two-sided"), output_dir=str(stage_dir), verbose=False)
            result["coordinate_units"] = params["coordinate_units"]
            result["coordinate_frame"] = params["coordinate_frame"]
            result["tested_pair"] = f"{cfg['source_type']} -> {cfg['target_type']}"
            result["k_neighbors"] = cfg["k"]
            result["permutation_region"] = params["spatial_null_region_assumption"]
            result["claim_label"] = "Exploratory"
            result["pre_specified"] = params["spatial_null_pre_specified"]
            result["selection_after_viewing_data"] = params["spatial_null_selection_after_viewing_data"]
            result["pre_specification_confirmed"] = params["spatial_null_pre_specification_confirmed"]
            result["pre_specification_statement"] = params["spatial_null_pre_specification_statement"]
            result["selection_statement"] = params["spatial_null_selection_statement"]
            result["confirmation_recorded_at_utc"] = params["spatial_null_confirmation_recorded_at_utc"]
            result.to_csv(stage_dir / "spatial_null_results.csv", index=False)
            (stage_dir / "spatial_metadata.json").write_text(json.dumps({"coordinate_units": params["coordinate_units"], "coordinate_frame": params["coordinate_frame"], "selected_pair": [cfg["source_type"], cfg["target_type"]], "k": cfg["k"], "n_permutations": cfg["n_permutations"], "tissue_mask_column": cfg.get("tissue_mask_column"), "permutation_region_assumption": params["spatial_null_region_assumption"], "claim_label": "Exploratory", "pre_specified": params["spatial_null_pre_specified"], "selection_after_viewing_data": params["spatial_null_selection_after_viewing_data"], "pre_specification_confirmed": params["spatial_null_pre_specification_confirmed"], "pre_specification_statement": params["spatial_null_pre_specification_statement"], "selection_statement": params["spatial_null_selection_statement"], "confirmation_recorded_at_utc": params["spatial_null_confirmation_recorded_at_utc"], "selection_adjustment": params["spatial_null_selection_adjustment"], "interpretation": "exploratory selected-pair label-swap spatial null; raw p-values are retained but not selection-adjusted"}, indent=2), encoding="utf-8")
            record("step11", "SUCCEEDED", f"EXPLORATORY spatial null for {cfg['source_type']} -> {cfg['target_type']} at k={cfg['k']} in {params['coordinate_units']} ({params['coordinate_frame']} frame); {params['spatial_null_region_assumption']}; no selection-adjusted claim.", stage_dir, {**cfg, "coordinate_units": params["coordinate_units"], "coordinate_frame": params["coordinate_frame"], "region_assumption": params["spatial_null_region_assumption"], "claim_label": "Exploratory", "pre_specification_statement": params["spatial_null_pre_specification_statement"], "selection_statement": params["spatial_null_selection_statement"], "confirmation_recorded_at_utc": params["spatial_null_confirmation_recorded_at_utc"], "selection_adjustment": params["spatial_null_selection_adjustment"]})
        if "compare_groups" in configs:
            cfg = configs["compare_groups"]
            stage_dir = out_root / "step12_group_comparison"
            stage_dir.mkdir(exist_ok=True)
            if aggregate_frame is None or configs["aggregate"].get("level") != "biological_unit_id":
                raise ValueError("step12 guard: comparison input is not the direct raw-cell-to-biological_unit_id aggregate.")
            unit_col = "biological_unit_id"
            if aggregate_frame[unit_col].duplicated().any() or aggregate_frame[unit_col].isna().any():
                raise ValueError("step12 guard: comparison requires exactly one aggregate row per biological unit.")
            conditions_per_unit = frame.groupby("biological_unit_id")["condition"].nunique()
            if (conditions_per_unit != 1).any() or aggregate_frame["condition"].nunique() != 2:
                raise ValueError("step12 guard: exactly two groups and one condition per independent biological unit are required.")
            metrics = cfg["metrics"]
            missing_metrics = set(metrics) - set(aggregate_frame.columns)
            if missing_metrics:
                raise ValueError("step12 metrics absent from direct aggregate: " + ", ".join(sorted(missing_metrics)))
            result = step12_compare_groups(aggregate_frame, group_col="condition", metrics=metrics, biological_unit_col=unit_col, output_dir=stage_dir, verbose=False)
            result.to_csv(stage_dir / "biological_unit_group_comparison.csv", index=False)
            record("step12", "SUCCEEDED", f"Two-group comparison executed at biological-unit n={aggregate_frame[unit_col].nunique()}.", stage_dir, {**cfg, "independent_biological_units": int(aggregate_frame[unit_col].nunique())})
    except Exception as exc:
        event = StageEvent(stage_id="analysis", status="FAILED", message=f"Analysis failed: {exc}", parameters=params)
        stages.append(event)
        emit(event)
        (out_root / "stage_statuses.json").write_text(json.dumps([s.model_dump(mode="json") for s in stages], indent=2), encoding="utf-8")
        (out_root / "limitations.json").write_text(json.dumps(limitations, indent=2), encoding="utf-8")
        artifacts.extend([str(out_root / "stage_statuses.json"), str(out_root / "limitations.json")])
        return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings + [str(exc)])
    if not stages:
        event = StageEvent(stage_id="analysis", status="WARNING", message="No analysis stages were enabled; no numerical analysis was performed.")
        stages.append(event)
        emit(event)
    statuses = [s.model_dump(mode="json") for s in stages]
    (out_root / "stage_statuses.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    (out_root / "limitations.json").write_text(json.dumps(limitations, indent=2), encoding="utf-8")
    artifacts.extend([str(out_root / "stage_statuses.json"), str(out_root / "limitations.json")])
    return WorkflowResult(status="WARNING" if not configs else "SUCCEEDED", stages=stages, artifacts=artifacts, warnings=warnings)
