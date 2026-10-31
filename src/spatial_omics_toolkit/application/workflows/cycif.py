from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import (
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


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _intake_options(parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate user-provided acquisition assumptions to the canonical intake config."""
    from spatial_omics_toolkit.cycif import RawImageIntakeConfig

    supplied = parameters.get("intake", {})
    if not isinstance(supplied, dict):
        raise ValueError("parameters.intake must be an object of RawImageIntakeConfig fields.")
    if "filename_pattern" not in supplied:
        raise ValueError("A user-reviewed intake.filename_pattern is required; acquisition naming is not inferred from PREVAIL defaults.")
    missing_assumptions = {"expected_rounds", "expected_channels"} - set(supplied)
    if missing_assumptions:
        raise ValueError(f"Explicit acquisition assumptions are required: {sorted(missing_assumptions)}; PREVAIL round/channel defaults are not applied.")
    try:
        if not supplied.get("expected_rounds") or int(supplied["expected_channels"]) < 1:
            raise ValueError("Explicit expected_rounds (non-empty) and expected_channels (positive) are required.")
    except (TypeError, ValueError) as exc:
        raise ValueError("Explicit expected_rounds (non-empty) and positive expected_channels are required.") from exc
    allowed = {"filename_pattern", "expected_rounds", "expected_channels", "raw_extension", "ome_xml_suffix", "expected_files_per_slide", "pixel_size_bounds_um"}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown intake configuration field(s): {', '.join(unknown)}")
    options = dict(supplied)
    if "expected_rounds" in options:
        options["expected_rounds"] = tuple(options["expected_rounds"])
    if "pixel_size_bounds_um" in options and options["pixel_size_bounds_um"] is not None:
        options["pixel_size_bounds_um"] = tuple(options["pixel_size_bounds_um"])
    try:
        compiled = re.compile(str(options["filename_pattern"]))
    except re.error as exc:
        raise ValueError(f"Invalid intake.filename_pattern: {exc}") from exc
    if not {"slide", "scene", "round"}.issubset(compiled.groupindex):
        raise ValueError("filename_pattern must define named groups 'slide', 'scene', and 'round' for stable acquisition grouping.")
    return {"config": RawImageIntakeConfig(**options)}


def _resolve_marker_panel(parameters: dict[str, Any]) -> dict[str, Any] | None:
    registration = parameters.get("registration")
    if not isinstance(registration, dict):
        return None
    panel_value = registration.get("marker_panel_csv")
    if not panel_value:
        return None
    if registration.get("marker_panel_confirmed") is not True:
        raise ValueError("Marker panel preview must be explicitly confirmed before export or segmentation.")
    panel_path = Path(str(panel_value)).expanduser().resolve()
    if not panel_path.is_file():
        raise ValueError(f"Selected marker_panel_metadata.csv does not exist: {panel_path}")
    import csv
    import hashlib

    with panel_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"round", "channel_index", "marker_name", "target_cell_compartment", "fluorophore", "notes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Marker panel CSV is missing required schema columns: {missing}")
        rows = list(reader)
    grouped: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        try:
            index = int(row["channel_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Marker panel has invalid zero-based channel_index {row.get('channel_index')!r}.") from exc
        marker_name = (row.get("marker_name") or "").strip()
        round_name = (row.get("round") or "").strip()
        if not round_name or not marker_name or index < 0:
            raise ValueError("Marker panel rows require non-empty round/marker_name and non-negative channel_index.")
        grouped.setdefault(round_name, []).append((index, marker_name))
    if not grouped:
        raise ValueError("Marker panel CSV contains no channel rows.")
    resolved: dict[str, list[str]] = {}
    for round_name, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        indices = [index for index, _ in entries]
        if indices != list(range(len(indices))):
            raise ValueError(f"Marker panel channel_index values for round {round_name!r} must be unique and contiguous from zero; received {indices}.")
        resolved[round_name] = [name for _, name in entries]
    panel = {"source_path": str(panel_path), "sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(), "marker_map": resolved}
    parameters["_resolved_marker_panel"] = panel
    return panel


def describe() -> WorkflowDescriptor:
    toolkit_version = _version("spatial-omics-toolkit")
    return WorkflowDescriptor(
        workflow_id="cycif-raw-czi",
        modality="CycIF",
        title="CyCIF raw CZI intake and staged image analysis",
        description="Run canonical notebook-derived CyCIF steps in one SLURM-allocated guarded Python worker. Intake scans raw files and OME metadata; downstream stages require explicit, complete configuration and preserve tool failures.",
        stages=[
            StageDefinition(stage_id="intake", title="Raw image intake", purpose="Inventory CZI files, parse configured acquisition names, validate cycle/scene coverage and OME metadata, and write the canonical QC report.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step01_scan_intake", version=toolkit_version, role="Canonical raw-CZI intake and OME metadata QC", documentation_url=None)]),
            StageDefinition(stage_id="conversion", title="CZI conversion", purpose="Convert raw CZI to OME-TIFF through the toolkit's SLURM-dispatch interface when explicitly configured.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step02_convert_raw_images", version=toolkit_version, role="Canonical CZI conversion", documentation_url=None)]),
            StageDefinition(stage_id="registration-export-segmentation", title="Registration, export, and segmentation", purpose="Run configured canonical registration/export and selected external segmentation backend; no result is claimed unless each stage completes.", tools=[ToolReference(name="spatial_omics_toolkit.cycif.step03_register_rounds / step04_export_registered / step05_segment_cells", version=toolkit_version, role="Canonical image analysis APIs; segmentation backend is externally versioned", documentation_url=None)]),
        ],
        parameter_fields=[
            ParameterField(name="intake.filename_pattern", label="Acquisition filename regex", kind="text", required=True, section="Intake", stage_id="intake", help_text="User-reviewed regex with named groups slide, scene, and round; PREVAIL defaults are not applied."),
            ParameterField(name="intake.expected_rounds", label="Expected rounds", kind="multiselect", required=True, default=[], dynamic_choices_key="available_rounds", section="Intake", stage_id="intake", help_text="Select expected round labels for coverage QC."),
            ParameterField(name="intake.raw_extension", label="Raw image extension", kind="text", default=".czi", section="Intake", stage_id="intake"),
            ParameterField(name="conversion.enabled", label="Retain converted OME-TIFF outputs", kind="bool", default=False, section="Conversion", stage_id="conversion", help_text="Optional; registration reads original CZI and does not consume conversion outputs."),
            ParameterField(name="conversion.config.generate_mip", label="Generate maximum-intensity projections", kind="bool", default=True, advanced=True, section="Conversion", stage_id="conversion"),
            ParameterField(name="conversion.config.generate_composite", label="Generate pseudocolor composites", kind="bool", default=True, advanced=True, section="Conversion", stage_id="conversion"),
            ParameterField(name="registration.enabled", label="Register and export scenes", kind="bool", default=False, section="Registration/export", stage_id="registration", help_text="Requires confirmation of parsed slide/scene/round grouping."),
            ParameterField(name="registration.confirmed_grouping", label="Confirm parsed acquisition grouping", kind="bool", default=False, section="Registration/export", stage_id="registration", help_text="Review acquisition_groups and confirm before registration."),
            ParameterField(name="registration.config.ref_round", label="Reference round", kind="select", required=True, dynamic_choices_key="available_rounds", section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.config.dapi_channel", label="Zero-based DAPI channel index", kind="int", default=0, min_value=0, section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.config.scale_factor", label="Registration scale factor", kind="float", default=1.0, min_value=0.01, max_value=1.0, step=0.01, section="Registration/export", stage_id="registration"),
            ParameterField(name="registration.marker_panel_csv", label="Marker panel metadata CSV", kind="config_file", section="Registration/export", stage_id="export", help_text="Preview and confirm its schema/map before setting marker_panel_confirmed."),
            ParameterField(name="registration.marker_panel_confirmed", label="Confirm marker/channel mapping", kind="bool", default=False, section="Registration/export", stage_id="export"),
            ParameterField(name="segmentation.backend", label="Segmentation backend", kind="select", choices=["MCMICRO", "PIPEX"], section="Segmentation", stage_id="segmentation", help_text="MCMICRO uses the centrally pinned runtime; PIPEX is blocked until a verified adapter is available."),
            ParameterField(name="segmentation.parameters.segmenter", label="MCMICRO segmenter", kind="select", choices=["mccellpose", "cellpose", "unmicst"], default="mccellpose", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.dapi_channel", label="MCMICRO DAPI channel (1-based)", kind="int", default=1, min_value=1, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_model", label="Cellpose model", kind="text", default="cyto3", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_diameter", label="Cellpose diameter", kind="int", default=30, min_value=1, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_expansion", label="Cell expansion", kind="int", default=10, min_value=0, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.use_gpu", label="Use GPU", kind="bool", default=False, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.enabled", label="Run segmentation", kind="bool", default=False, section="Segmentation", stage_id="segmentation"),
        ],
        input_description="One complete raw-CZI modality folder, project root, and project_samples.csv. Naming pattern, expected cycles/channels, and round/channel mapping are acquisition-specific parameters; no PREVAIL biological labels are assumed.",
        output_description="Run-scoped intake inventory, checks, report, and any explicitly configured stage outputs under run_dir.",
        limitations=["Conversion, registration, export, and segmentation are not implied by successful intake; they require explicit configured tools/resources.", "MCMICRO and PIPEX are mutually exclusive explicit segmentation choices and require their configured executable/configuration.", "No universal biological labels or cell-type configuration is assumed.", "Heavy stages run only when execute is called in the allocated Python worker."],
    )


def _acquisition_groups(root: Path, filename_pattern: str) -> list[dict[str, Any]]:
    pattern = re.compile(filename_pattern)
    groups: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".czi":
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        fields = match.groupdict()
        if not {"slide", "scene", "round"}.issubset(fields):
            continue
        slide = str(fields["slide"])
        scene = str(fields["scene"])
        group_key = f"slide={slide};scene={scene}"
        group = groups.setdefault(group_key, {"group_key": group_key, "label": f"{slide}_{scene}", "input_paths": []})
        group["input_paths"].append(str(path))
    return [groups[key] for key in sorted(groups)]


def _available_rounds(groups: list[dict[str, Any]], filename_pattern: str) -> list[str]:
    pattern = re.compile(filename_pattern)
    return sorted({str(match.group("round")) for group in groups for path in group["input_paths"] if (match := pattern.match(Path(path).name)) is not None})


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    warnings: list[str] = []
    conversion = request.parameters.get("conversion")
    params: dict[str, Any] = {"toolkit_version": _version("spatial-omics-toolkit"), "conversion_enabled": bool(isinstance(conversion, dict) and conversion.get("enabled")), "conversion_policy": "optional retained deliverable; registration reads original CZI"}
    intake_values = request.parameters.get("intake")
    if isinstance(intake_values, dict):
        params["acquisition_assumptions"] = {key: {"value": intake_values[key], "source": "explicit user configuration"} for key in ("filename_pattern", "expected_rounds", "expected_channels") if key in intake_values}
    if request.modality != "CycIF":
        blockers.append(f"Expected modality 'CycIF', received {request.modality!r}.")
    for label, path in (("project root", request.project_root), ("modality folder", request.modality_dir)):
        if not path.is_dir():
            blockers.append(f"Selected {label} is not a directory: {path}")
    if not request.annotation_csv.is_file():
        blockers.append(f"project_samples.csv is missing: {request.annotation_csv}")
    elif request.annotation_csv.name != "project_samples.csv":
        warnings.append(f"Annotation file is named {request.annotation_csv.name!r}; expected project_samples.csv.")
    if request.parameters.get("_preflight_mode") == "quick":
        try:
            _intake_options(request.parameters)
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))
        else:
            if request.modality_dir.is_dir():
                pattern = str(request.parameters["intake"]["filename_pattern"])
                groups = _acquisition_groups(request.modality_dir, pattern)
                params["acquisition_groups"] = groups
                params["available_rounds"] = _available_rounds(groups, pattern)
        return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)
    try:
        _intake_options(request.parameters)
    except (TypeError, ValueError) as exc:
        blockers.append(str(exc))
    if request.modality_dir.is_dir():
        czi_files = sorted(p for p in request.modality_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".czi")
        ome_files = sorted(p for p in request.modality_dir.rglob("*") if p.is_file() and p.name.lower().endswith(".ome.xml"))
        pattern_value = request.parameters.get("intake", {}).get("filename_pattern") if isinstance(request.parameters.get("intake"), dict) else None
        groups = _acquisition_groups(request.modality_dir, str(pattern_value)) if pattern_value else []
        if pattern_value:
            params.update({"acquisition_groups": groups, "available_rounds": _available_rounds(groups, str(pattern_value))})
        params.update({"czi_file_count": len(czi_files), "ome_xml_file_count": len(ome_files), "czi_files": [str(p) for p in czi_files], "ome_xml_files": [str(p) for p in ome_files]})
        if not czi_files:
            blockers.append("The selected modality folder contains no raw .czi files.")
        if not ome_files:
            warnings.append("No .ome.xml sidecars were found; step01_scan_intake will report metadata availability from the image files it can inspect.")
    segmentation = request.parameters.get("segmentation", {})
    if isinstance(segmentation, dict) and segmentation.get("enabled") is True:
        if not isinstance(segmentation, dict) or segmentation.get("backend") not in {"MCMICRO", "PIPEX"}:
            blockers.append("Configured segmentation must explicitly select backend 'MCMICRO' or 'PIPEX'.")
        elif segmentation["backend"] == "MCMICRO":
            software_root = os.environ.get("PAULI_FAST_SOFTWARES")
            if not software_root:
                blockers.append("MCMICRO runtime unavailable: PAULI_FAST_SOFTWARES is not set in the worker environment.")
            else:
                mcmicro_home = Path(software_root) / "envs" / "spatial-proteomics" / "mcmicro"
                mcmicro_bin = Path(software_root) / "bin" / "mcmicro"
                lock_file = mcmicro_home / "config" / "workflow-runtime.lock.sh"
                if not mcmicro_home.is_dir() or not (mcmicro_home / "main.nf").is_file():
                    blockers.append(f"Pinned MCMICRO workflow is unavailable at {mcmicro_home}.")
                if not mcmicro_bin.is_file() or not os.access(mcmicro_bin, os.X_OK):
                    blockers.append(f"Pinned MCMICRO wrapper is unavailable or not executable: {mcmicro_bin}.")
                if not lock_file.is_file():
                    blockers.append(f"Pinned MCMICRO runtime lock is missing: {lock_file}.")
                params["mcmicro_runtime"] = {"home": str(mcmicro_home), "wrapper": str(mcmicro_bin), "runtime_lock": str(lock_file)}
        else:
            binary = segmentation.get("binary")
            config = segmentation.get("config")
            if not binary or not config:
                blockers.append("PIPEX requires explicit binary and configuration paths.")
            elif not Path(str(binary)).is_file() or not Path(str(config)).is_file():
                blockers.append("Selected PIPEX binary or configuration file is unavailable.")
            elif not os.access(Path(str(binary)), os.X_OK):
                blockers.append(f"PIPEX binary is not executable: {binary}")
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        event = StageEvent(stage_id="intake", status="BLOCKED", message="CyCIF preflight blocked: " + "; ".join(check.blockers), parameters=check.resolved_parameters)
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=check.blockers + check.warnings)
    from spatial_omics_toolkit.cycif import step01_scan_intake

    stages: list[StageEvent] = []
    warnings = list(check.warnings)
    conversion = request.parameters.get("conversion")
    intake_dir = request.run_dir / "cycif" / "intake"
    artifacts: list[str] = []
    current_stage = "intake"
    try:
        options = _intake_options(request.parameters)
        emit(StageEvent(stage_id="intake", status="RUNNING", message="Scanning raw CZI inventory and validating naming, cycle/scene coverage, and OME metadata.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"modality_dir": str(request.modality_dir), **request.parameters.get("intake", {})}))
        result = step01_scan_intake(raw_dir=str(request.modality_dir), reports_dir=str(intake_dir), generate_html_report=True, generate_dapi_preview=False, strict=False, verbose=False, **options)
        artifacts = [str(p) for p in sorted(intake_dir.rglob("*")) if p.is_file()]
        checks = getattr(result, "checks", [])
        check_rows = checks.to_dict(orient="records") if hasattr(checks, "to_dict") else []
        failed_checks = [row for row in check_rows if str(row.get("Status", "")).strip() == "⚠"]
        intake_issues = {
            "check_warnings": failed_checks,
            "coverage_missing": [str(x) for x in getattr(result, "coverage_missing", []) or []],
            "unparsed": [str(x) for x in getattr(result, "unparsed", []) or []],
        }
        has_intake_issues = any(intake_issues.values())
        status = "WARNING" if has_intake_issues else "SUCCEEDED"
        message = "Canonical step01_scan_intake completed." if not has_intake_issues else "Canonical intake completed with QC warnings; review the complete warning rows and coverage evidence before downstream processing."
        event = StageEvent(stage_id="intake", status=status, message=message, tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters=intake_issues, artifacts=artifacts)
        emit(event)
        stages.append(event)
        if has_intake_issues:
            warnings.append(json.dumps(intake_issues, default=str))
            gate = StageEvent(stage_id="conversion", status="BLOCKED", message="Downstream imaging stages were not started because intake reported QC warnings.", parameters=intake_issues)
            emit(gate)
            stages.append(gate)
            return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings)
        current_stage = "conversion"
        if isinstance(conversion, dict) and conversion.get("enabled") is True:
            if not __import__("os").environ.get("SLURM_JOB_ID"):
                message = "CZI conversion is blocked outside an allocated SLURM worker."
                event = StageEvent(stage_id="conversion", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            from spatial_omics_toolkit.cycif import ImageConversionConfig, step02_convert_raw_images

            conversion_options = conversion.get("config", {})
            if not isinstance(conversion_options, dict):
                raise ValueError("parameters.conversion.config must be an object of ImageConversionConfig fields.")
            conversion_config = ImageConversionConfig(**conversion_options)
            czi_files = [Path(value) for value in check.resolved_parameters.get("czi_files", [])]
            input_folders = sorted({path.parent for path in czi_files})
            if not czi_files or not input_folders:
                raise RuntimeError("No raw CZI files were inventoried for conversion.")
            emit(StageEvent(stage_id="conversion", status="RUNNING", message=f"Converting {len(czi_files)} inventoried CZI file(s) across {len(input_folders)} physical folder(s) in the allocated SLURM worker.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"dispatch": "inprocess within allocated worker", "input_files": [str(path) for path in czi_files], **conversion_options}))
            conversion_outputs: list[str] = []
            conversion_results: dict[str, Any] = {}
            for input_folder in input_folders:
                relative = input_folder.relative_to(request.modality_dir)
                folder_key = relative.as_posix().replace("/", "__") or "root"
                conversion_dir = request.run_dir / "cycif" / "converted" / folder_key
                conversion_dir.mkdir(parents=True, exist_ok=True)
                expected_inputs = [str(path) for path in czi_files if path.parent == input_folder]
                conversion_results[str(input_folder)] = step02_convert_raw_images(str(input_folder), str(conversion_dir), dispatch="inprocess", conversion_config=conversion_config, strict=True, verbose=False)
                folder_outputs = [str(p) for p in sorted(conversion_dir.rglob("*")) if p.is_file()]
                if not folder_outputs:
                    raise RuntimeError(f"Canonical conversion produced no outputs for inventoried inputs: {expected_inputs}")
                conversion_outputs.extend(folder_outputs)
            if not conversion_outputs:
                raise RuntimeError("Canonical conversion returned without producing output files.")
            event = StageEvent(stage_id="conversion", status="SUCCEEDED", message=f"Canonical CZI conversion completed for {len(czi_files)} inventoried input file(s) in {len(input_folders)} folder(s).", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"results_by_input_folder": conversion_results}, artifacts=conversion_outputs)
            emit(event)
            stages.append(event)
        segmentation_completed = 0
        current_stage = "registration"
        registration = request.parameters.get("registration")
        registration_enabled = isinstance(registration, dict) and registration.get("enabled") is True
        if not registration_enabled:
            registration = None
        if registration:
            if not isinstance(registration, dict) or not registration.get("confirmed_grouping"):
                message = "Registration requires user-confirmed filename grouping in parameters.registration.confirmed_grouping=true."
                event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            import re
            from spatial_omics_toolkit.cycif import step03_register_rounds, step04_export_registered

            pattern = re.compile(str(request.parameters["intake"]["filename_pattern"]))
            grouped: dict[tuple[str, str], dict[str, str]] = {}
            for path in sorted(request.modality_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() != ".czi":
                    continue
                match = pattern.match(path.name)
                if match is None or not {"round", "slide", "scene"}.issubset(match.groupdict()):
                    continue
                key = (str(match.group("slide")), str(match.group("scene")))
                round_label = str(match.group("round"))
                if round_label in grouped.setdefault(key, {}):
                    message = f"Ambiguous duplicate CZI for slide={key[0]}, scene={key[1]}, round={round_label}."
                    event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                    emit(event)
                    stages.append(event)
                    return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
                grouped[key][round_label] = str(path)
            if not grouped:
                message = "No CZI files matched the user-configured filename pattern with named slide, scene, and round groups."
                event = StageEvent(stage_id="registration", status="BLOCKED", message=message)
                emit(event)
                stages.append(event)
                return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
            try:
                marker_panel = _resolve_marker_panel(request.parameters)
            except ValueError as exc:
                marker_panel = None
                warnings.append(str(exc))
            marker_map = marker_panel["marker_map"] if marker_panel is not None else None
            if marker_map is None:
                message = "Export/segmentation requires a user-selected and explicitly confirmed marker_panel_metadata.csv with the required schema."
                warnings.append(message)
            for (slide, scene), round_files in grouped.items():
                label = f"{slide}_{scene}"
                reg_dir = request.run_dir / "cycif" / "registered" / label
                reg_dir.mkdir(parents=True, exist_ok=True)
                emit(StageEvent(stage_id="registration", status="RUNNING", message=f"Registering {len(round_files)} configured rounds for {label}.", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"slide": slide, "scene": scene, "rounds": sorted(round_files)}))
                reg_result = step03_register_rounds(round_files, output_dir=str(reg_dir), verbose=False, **registration.get("config", {}))
                round_results = reg_result.get("rounds", {})
                bad_rounds = {name: info.get("status", "unregistered") for name, info in round_results.items() if not info.get("registered", False)}
                reg_artifacts = [str(p) for p in reg_dir.rglob("*") if p.is_file()]
                artifacts.extend(reg_artifacts)
                reg_status = "FAILED" if bad_rounds else "SUCCEEDED"
                reg_message = f"Registration transform failures preserved: {bad_rounds}" if bad_rounds else f"Registered scene {label}."
                event = StageEvent(stage_id="registration", status=reg_status, message=reg_message, tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"slide": slide, "scene": scene, "round_status": {name: info.get("status") for name, info in round_results.items()}}, artifacts=reg_artifacts)
                emit(event)
                stages.append(event)
                if bad_rounds:
                    return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings + [reg_message])
                if marker_map is None:
                    message = "Registration succeeded, but export is blocked until a marker-panel CSV is selected and explicitly confirmed."
                    event = StageEvent(stage_id="export", status="BLOCKED", message=message, parameters={"scene_label": label})
                    emit(event)
                    stages.append(event)
                    continue
                if set(marker_map) != set(round_files):
                    message = f"Confirmed marker panel rounds {sorted(marker_map)} do not match scene {label} rounds {sorted(round_files)}; registration QC is retained but export is blocked."
                    event = StageEvent(stage_id="export", status="BLOCKED", message=message, parameters={"scene_label": label, "marker_panel": marker_panel})
                    emit(event)
                    stages.append(event)
                    warnings.append(message)
                    continue
                current_stage = "export"
                export_dir = request.run_dir / "cycif" / "exported" / label
                export_dir.mkdir(parents=True, exist_ok=True)
                emit(StageEvent(stage_id="export", status="RUNNING", message=f"Exporting registered scene {label} with explicitly confirmed channel/marker mapping.", parameters={"slide": slide, "scene": scene, "marker_panel": marker_panel}))
                pipex_dir = request.run_dir / "cycif" / "pipex_channels"
                export_result = step04_export_registered(reg_result, label, str(export_dir), marker_map=marker_map, pipex_dir=str(pipex_dir), verbose=False, **registration.get("export_config", {}))
                export_artifacts = [str(p) for base in (export_dir, pipex_dir) for p in base.rglob("*") if p.is_file()]
                if not export_artifacts:
                    raise RuntimeError(f"step04_export_registered produced no artifacts for {label}: {export_result}")
                artifacts.extend(export_artifacts)
                segmentation = request.parameters.get("segmentation")
                if isinstance(segmentation, dict) and segmentation.get("enabled") is True and segmentation.get("backend") == "MCMICRO":
                    current_stage = "segmentation"
                    if not os.environ.get("SLURM_JOB_ID"):
                        message = "MCMICRO execution is blocked outside an allocated SLURM worker."
                        event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
                        emit(event)
                        stages.append(event)
                        return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
                    from spatial_omics_toolkit.cycif import MCMICROConfig, step05_segment_cells

                    runtime = check.resolved_parameters["mcmicro_runtime"]
                    version_process = subprocess.run([runtime["wrapper"], "-version"], check=True, capture_output=True, text=True, timeout=60)
                    mcmicro_version = (version_process.stdout or version_process.stderr).strip() or "version command returned no output"
                    config_values = segmentation.get("parameters", {})
                    if not isinstance(config_values, dict):
                        raise ValueError("segmentation.parameters must be an object of MCMICROConfig settings.")
                    mcmicro_config = MCMICROConfig(mcmicro_bin=runtime["wrapper"], mcmicro_nf=str(Path(runtime["home"]) / "main.nf"), **config_values)
                    project_dir = request.run_dir / "cycif" / "mcmicro" / label
                    composite_path = export_result.get("composite_path")
                    if not composite_path or not Path(str(composite_path)).is_file():
                        raise RuntimeError(f"step04_export_registered did not return a usable composite OME-TIFF for MCMICRO scene {label}.")
                    emit(StageEvent(stage_id="segmentation", status="RUNNING", message=f"Running the pinned MCMICRO backend for {label} inside the allocated SLURM worker.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"backend": "MCMICRO", "segmenter": mcmicro_config.segmenter, "marker_panel": marker_panel, "config": config_values, "project_dir": str(project_dir)}))
                    segmentation_result = step05_segment_cells("mcmicro", scene_label=label, composite_ometiff_path=str(composite_path), marker_map=marker_map, project_dir=str(project_dir), mcmicro_config=mcmicro_config, run=True, load_results=True, verbose=False)
                    segmentation_artifacts = [str(p) for p in project_dir.rglob("*") if p.is_file()]
                    if not segmentation_artifacts:
                        raise RuntimeError(f"MCMICRO returned without run-scoped artifacts for {label}: {segmentation_result}")
                    artifacts.extend(segmentation_artifacts)
                    event = StageEvent(stage_id="segmentation", status="SUCCEEDED", message=f"MCMICRO {mcmicro_version} completed for {label}.", tool_versions={"MCMICRO": mcmicro_version, "spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]}, parameters={"backend": "MCMICRO", "segmenter": mcmicro_config.segmenter, "marker_panel": marker_panel, "result": str(segmentation_result)}, artifacts=segmentation_artifacts)
                    emit(event)
                    segmentation_completed += 1
                    stages.append(event)
        segmentation = request.parameters.get("segmentation")
        if isinstance(segmentation, dict) and segmentation.get("enabled") is True and segmentation.get("backend") == "MCMICRO" and segmentation_completed == 0:
            message = "MCMICRO was selected but no scene reached segmentation; registration, export, and a confirmed marker map are required."
            event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
            emit(event)
            stages.append(event)
            return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
        if isinstance(segmentation, dict) and segmentation.get("enabled") is True and segmentation.get("backend") == "PIPEX":
            message = "PIPEX execution requires a verified PIPEX installation profile and PIPEXConfig; only MCMICRO is currently wired to the centrally pinned runtime."
            event = StageEvent(stage_id="segmentation", status="BLOCKED", message=message)
            emit(event)
            stages.append(event)
            return WorkflowResult(status="BLOCKED", stages=stages, artifacts=artifacts, warnings=warnings + [message])
        final_status = "BLOCKED" if any(stage.status == "BLOCKED" for stage in stages) else ("SUCCEEDED" if not failed_checks else "WARNING")
        return WorkflowResult(status=final_status, stages=stages, artifacts=artifacts, warnings=warnings)
    except Exception as exc:
        event = StageEvent(stage_id=current_stage, status="FAILED", message=f"{current_stage} stage failed: {type(exc).__name__}: {exc}", tool_versions={"spatial_omics_toolkit": check.resolved_parameters["toolkit_version"]})
        emit(event)
        stages.append(event)
        return WorkflowResult(status="FAILED", stages=stages, artifacts=artifacts, warnings=warnings + [event.message])
