from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import (
    FunctionReference,
    ParameterField,
    PreflightResult,
    RunRequest,
    StageDefinition,
    StageEvent,
    StageStatus,
    ToolReference,
    WorkflowDescriptor,
    WorkflowEmitter,
    WorkflowResult,
)


import hashlib
import platform
from datetime import datetime, timezone


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _stage_parameters(parameters: dict[str, Any], stage_id: str) -> dict[str, Any]:
    if stage_id == "intake":
        return {"intake": parameters.get("intake", {})}
    if stage_id == "conversion":
        return {"conversion": parameters.get("conversion", {})}
    if stage_id == "registration":
        return {"intake": parameters.get("intake", {}), "registration": {key: value for key, value in parameters.get("registration", {}).items() if key != "marker_panel_csv" and key != "marker_panel_confirmed"}}
    if stage_id == "export":
        panel = parameters.get("_resolved_marker_panel", {})
        return {"marker_panel_sha256": panel.get("sha256"), "marker_map": panel.get("marker_map"), "registration": parameters.get("registration", {}).get("export_config", {})}
    return {"segmentation": parameters.get("segmentation", {})}


def _input_fingerprint(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "inputs" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run input inventory is missing; checkpoint receipt cannot be created.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = sorted((str(entry.get("path")), str(entry.get("sha256"))) for entry in manifest.get("files", []) if entry.get("sha256"))
    if not hashes:
        raise ValueError("Run input inventory contains no hashed files.")
    return {"inventory_sha256": _canonical_hash(hashes), "annotation_csv_sha256": manifest.get("annotation_csv_sha256")}


def _source_code_identity(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in sorted(paths.items())}


def _tool_identity(stage_id: str) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    shared_sources = {
        "application/workflows/cycif.py": Path(__file__).resolve(),
        "application/worker.py": package_root / "application" / "worker.py",
        "application/intake.py": package_root / "application" / "intake.py",
        "application/runs.py": package_root / "application" / "runs.py",
        "cycif/__init__.py": package_root / "cycif" / "__init__.py",
        "cycif/config.py": package_root / "cycif" / "config.py",
    }
    stage_sources = {
        "intake": ("cycif/intake.py",),
        "conversion": ("cycif/convert.py",),
        "registration": ("cycif/register.py",),
        "export": ("cycif/register.py", "cycif/export_images.py"),
        "segmentation": ("cycif/export_images.py", "cycif/segment.py"),
    }
    if stage_id not in stage_sources:
        raise ValueError(f"Unsupported checkpoint stage for tool identity: {stage_id}")
    source_paths = {
        **shared_sources,
        **{
            name: package_root / name
            for name in stage_sources[stage_id]
        },
    }
    identity = {
        "spatial_omics_toolkit": _version("spatial-omics-toolkit"),
        "python": platform.python_version(),
        **{f"source:{name}": digest for name, digest in _source_code_identity(source_paths).items()},
    }
    for package in ("numpy", "opencv-python", "opencv-python-headless", "tifffile"):
        version = _version(package)
        if version != "not installed":
            identity[package] = version
    if stage_id == "segmentation":
        software_root = os.environ.get("PAULI_FAST_SOFTWARES")
        if software_root:
            mcmicro_home = Path(software_root) / "envs" / "spatial-proteomics" / "mcmicro"
            runtime_files = {
                "mcmicro_runtime_lock": Path(software_root) / "config" / "workflow-runtime.lock.sh",
                "mcmicro_main_nf": mcmicro_home / "main.nf",
            }
            identity.update({
                f"runtime:{name}": _sha256(path)
                for name, path in runtime_files.items()
                if path.is_file()
            })
    return identity


def _write_checkpoint_receipts(request: RunRequest, events: list[StageEvent]) -> None:
    fingerprint = _input_fingerprint(request.run_dir)
    receipts: dict[str, Any] = {}
    for stage_id in ("intake", "conversion", "registration", "export", "segmentation"):
        stage_events = [event for event in events if event.stage_id == stage_id]
        if not stage_events or any(event.status not in {"RUNNING", "SUCCEEDED", "WARNING"} for event in stage_events) or not any(event.status in {"SUCCEEDED", "WARNING"} for event in stage_events):
            continue
        if stage_id == "intake" and any(
            event.status == "WARNING"
            and any(event.parameters.get(key) for key in ("check_warnings", "coverage_missing", "unparsed"))
            for event in stage_events
        ):
            continue
        paths = sorted({Path(path).resolve() for event in stage_events for path in event.artifacts})
        if not paths or any(not path.is_file() for path in paths):
            continue
        artifacts = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
        settings = _stage_parameters(request.parameters, stage_id)
        tool_identity = _tool_identity(stage_id)
        for event in stage_events:
            if event.status in {"SUCCEEDED", "WARNING"}:
                tool_identity.update(event.tool_versions)
        receipts[stage_id] = {
            "workflow_id": request.workflow_id,
            "stage_id": stage_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_fingerprint": fingerprint,
            "settings_sha256": _canonical_hash(settings),
            "settings": settings,
            "tool_identity": tool_identity,
            "stage_status": "WARNING" if any(event.status == "WARNING" for event in stage_events) else "SUCCEEDED",
            "artifacts": artifacts,
        }
    if receipts:
        from spatial_omics_toolkit.application.runs import atomic_json

        # Conversion and registration are independent Nextflow branches.
        # Serialize only the small read/merge/write of their shared receipt.
        receipt_path = request.run_dir / "checkpoint_receipts.json"
        with (request.run_dir / ".checkpoint_receipts.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"workflow_id": request.workflow_id, "checkpoints": {}}
            existing["checkpoints"].update(receipts)
            atomic_json(receipt_path, existing)



def validate_continuation(request: RunRequest) -> tuple[list[str], list[dict[str, str]]]:
    """Validate immutable source receipts against the new run's current inventory."""
    selection = request.parameters.get("continuation", {})
    if not isinstance(selection, dict) or selection.get("mode") != "continue":
        return [], []
    source = Path(str(selection.get("parent_run_dir", ""))).expanduser().resolve()
    canonical_runs = (request.output_root / "CycIF" / "runs").resolve()
    if not source.is_relative_to(canonical_runs):
        return ["Selected source run must resolve beneath this project's CyCIF/runs directory."], []
    start_stage = selection.get("start_stage")
    required = {
        "conversion": ("intake",),
        "registration": ("intake",),
        "export": ("intake", "registration"),
        "segmentation": ("intake", "registration", "export"),
    }
    if start_stage not in required:
        return ["Choose a supported continuation boundary after intake, conversion, registration, or export."], []
    reasons: list[str] = []
    if not source.is_dir():
        return [f"Selected source run does not exist: {source}"], []
    try:
        metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        receipt = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))
        parent_inventory = json.loads((source / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        manifest_inventory = json.loads((request.run_dir / "inputs" / "manifest.json").read_text(encoding="utf-8"))
        child_metadata = json.loads((request.run_dir / "run.json").read_text(encoding="utf-8"))
        source_annotation = Path(child_metadata["project"]["annotation_csv"]).expanduser()
        from spatial_omics_toolkit.application.intake import sha256_file
        current_fingerprint = _input_fingerprint(request.run_dir)
        live_annotation_hash = sha256_file(source_annotation)
        if live_annotation_hash != manifest_inventory.get("annotation_csv_sha256"):
            return ["Current project_samples.csv changed after child-run submission; re-inventory and submit a fresh continuation."], []
        if live_annotation_hash != current_fingerprint.get("annotation_csv_sha256"):
            return ["project_samples.csv snapshot differs from the current project metadata."], []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"Source run is legacy or missing required checkpoint receipts/inventory: {exc}"], []
    if metadata.get("workflow_id") != "cycif-raw-czi" or receipt.get("workflow_id") != "cycif-raw-czi":
        reasons.append("Source workflow ID does not match cycif-raw-czi.")
    if source.parent.name != "runs" or source.parent.parent.name != "CycIF":
        reasons.append("Selected source is not a CyCIF run bundle under a canonical `CycIF/runs/<run_id>` path.")
    if str(metadata.get("run_id", "")) != source.name or metadata.get("modality") != "CycIF":
        reasons.append("Source run identity or modality does not match its run-bundle path.")
    if status.get("status") not in {"SUCCEEDED", "WARNING", "FAILED", "BLOCKED"}:
        reasons.append(f"Source run has no terminal workflow status: {status.get('status', 'missing')}.")
    if receipt.get("checkpoints") is not None and receipt.get("checkpoints", {}).get("intake", {}).get("input_fingerprint") != current_fingerprint:
        reasons.append("Current raw-CZI inventory or project_samples.csv fingerprint differs from the source run.")
    parent_hashes = sorted((str(entry.get("path")), str(entry.get("sha256"))) for entry in parent_inventory.get("files", []) if entry.get("sha256"))
    if _canonical_hash(parent_hashes) != current_fingerprint.get("inventory_sha256"):
        reasons.append("Current raw-CZI file hashes differ from the source run inventory.")
    checkpoints = receipt.get("checkpoints", {})
    try:
        source_events = [json.loads(line) for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        source_events = []
    latest_status: dict[str, str] = {}
    for source_event in source_events:
        stage_name = source_event.get("stage_id")
        if isinstance(stage_name, str):
            latest_status[stage_name] = str(source_event.get("status", ""))

    source_stage_status: dict[str, str] = {}
    warning_stages: list[str] = []
    for stage_id in required[start_stage]:
        checkpoint = checkpoints.get(stage_id)
        latest = latest_status.get(stage_id, "missing")
        if latest == "REUSED" and isinstance(checkpoint, dict):
            latest = str(checkpoint.get("stage_status", latest))
        source_stage_status[stage_id] = latest
        if latest == "WARNING":
            warning_stages.append(stage_id)
    if warning_stages and selection.get("warning_review_confirmed") is not True:
        reasons.append(f"Explicitly review the source QC report and confirm reuse of warning checkpoints: {warning_stages}.")
    reused: list[dict[str, str]] = []
    for stage_id in required[start_stage]:
        checkpoint = checkpoints.get(stage_id)
        if not isinstance(checkpoint, dict):
            reasons.append(f"Source run has no reusable {stage_id} checkpoint receipt.")
            continue
        stage_status = source_stage_status[stage_id]
        if stage_status not in {"SUCCEEDED", "WARNING", "REUSED"}:
            reasons.append(f"{stage_id} upstream stage is not complete and reusable; latest status is {stage_status}.")
        if checkpoint.get("workflow_id") != "cycif-raw-czi" or checkpoint.get("input_fingerprint") != current_fingerprint:
            reasons.append(f"{stage_id} checkpoint is bound to a different workflow or input fingerprint.")
            continue
        current_settings = _stage_parameters(request.parameters, stage_id)
        if checkpoint.get("settings_sha256") != _canonical_hash(current_settings):
            reasons.append(f"{stage_id} checkpoint settings differ from the current configuration.")
        expected_tools = _tool_identity(stage_id)
        if checkpoint.get("tool_identity") != expected_tools:
            reasons.append(f"{stage_id} checkpoint tool/runtime identity differs from the current worker.")
        artifacts = checkpoint.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            reasons.append(f"{stage_id} checkpoint has no artifact hashes.")
            continue
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or _sha256(path) != artifact.get("sha256"):
                reasons.append(f"{stage_id} checkpoint artifact is missing or modified: {path}")
            else:
                reused.append({"stage_id": stage_id, "path": str(path), "sha256": str(artifact["sha256"]), "source_status": stage_status})
    return reasons, reused


def _warning_reuse_messages(reused: list[dict[str, str]]) -> list[str]:
    warning_stages = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
    return [
        f"Reused {stage_id} from a WARNING checkpoint; the source QC limitations remain applicable."
        for stage_id in warning_stages
    ]





def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
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
        raise ValueError("A user-reviewed intake.filename_pattern is required; specify the acquisition naming pattern.")
    missing_assumptions = {"expected_rounds", "expected_channels"} - set(supplied)
    if missing_assumptions:
        raise ValueError(f"Explicit acquisition assumptions are required: {sorted(missing_assumptions)}; supply the expected rounds and channel count.")
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
    required_groups = {"round", "markers", "slide", "date", "acq_id", "scene"}
    missing_groups = sorted(required_groups - set(compiled.groupindex))
    if missing_groups:
        raise ValueError(
            "filename_pattern must define named groups 'round', 'markers', 'slide', 'date', 'acq_id', and 'scene'; "
            f"include explicit empty captures for fields absent from source filenames (missing: {missing_groups})."
        )
    return {"config": RawImageIntakeConfig(**options)}


def _resolve_marker_panel(parameters: dict[str, Any], project_root: Path) -> dict[str, Any] | None:
    registration = parameters.get("registration")
    if not isinstance(registration, dict):
        return None
    panel_value = registration.get("marker_panel_csv")
    if not panel_value:
        return None
    if registration.get("marker_panel_confirmed") is not True:
        raise ValueError("Marker panel preview must be explicitly confirmed before export or segmentation.")
    panel_path = Path(str(panel_value)).expanduser()
    if not panel_path.is_absolute():
        panel_path = project_root / panel_path
    panel_path = panel_path.resolve()
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
        title="CyCIF raw CZI image processing",
        description="Inventory raw CyCIF images and check their OME metadata in a SLURM-allocated Python worker. Optional conversion, registration, export, and segmentation require configuration; tool failures are reported.",
        stages=[
            StageDefinition(stage_id="intake", title="Image intake and quality checks", purpose="Inventory CZI files and check acquisition coverage and OME metadata.", depends_on=[], condition="Always; failed QC blocks downstream imaging", inputs=["Raw CZI files", "Acquisition assumptions"], outputs=["Intake inventory and QC report"], tools=[ToolReference(name="spatial_omics_toolkit.cycif.step01_scan_intake", version=toolkit_version, role="Inventory raw CZI images and check OME metadata")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/intake.py", function_name="step01_scan_intake", role="Inventory raw CZI images and check metadata")]),
            StageDefinition(stage_id="conversion", title="Convert images to OME-TIFF", purpose="Optionally retain converted CZI images; registration reads the original files.", depends_on=["intake"], condition="conversion.enabled", inputs=["Raw CZI files", "Intake QC"], outputs=["Converted OME-TIFF files"], tools=[ToolReference(name="spatial_omics_toolkit.cycif.step02_convert_raw_images", version=toolkit_version, role="Convert CZI images")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/convert.py", function_name="step02_convert_raw_images", role="Optionally convert CZI images")]),
            StageDefinition(stage_id="registration", title="Align imaging rounds", purpose="Register CZI imaging rounds to the selected reference round.", depends_on=["intake"], condition="registration.enabled and confirmed grouping", inputs=["Raw CZI files", "Intake QC", "Confirmed grouping"], outputs=["Registered TIFFs", "Registration handoffs"], tools=[ToolReference(name="spatial_omics_toolkit.cycif.step03_register_rounds", version=toolkit_version, role="Register imaging rounds")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/register.py", function_name="step03_register_rounds", role="Register imaging rounds")]),
            StageDefinition(stage_id="export", title="Export marker images", purpose="Export registered scenes using a confirmed marker-to-channel map.", depends_on=["registration"], condition="Registration enabled and marker panel confirmed", inputs=["Registration handoffs", "Confirmed marker panel"], outputs=["Composite OME-TIFF and metadata"], tools=[ToolReference(name="spatial_omics_toolkit.cycif.step04_export_registered", version=toolkit_version, role="Export registered images")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/export_images.py", function_name="step04_export_registered", role="Export images using the confirmed marker mapping")]),
            StageDefinition(stage_id="segmentation", title="Segment cells", purpose="Run the configured MCMICRO workflow on exported scenes in the allocated worker.", depends_on=["export"], condition="segmentation.enabled and MCMICRO selected", inputs=["Composite OME-TIFF", "Confirmed marker panel"], outputs=["MCMICRO segmentation files"], tools=[ToolReference(name="MCMICRO", version=toolkit_version, role="Cell segmentation")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/segment.py", function_name="step05_segment_cells", role="Run MCMICRO through the toolkit")]),
        ],
        parameter_fields=[
            ParameterField(name="intake.filename_pattern", label="Acquisition filename regex", kind="text", required=True, section="Intake", stage_id="intake", help_text="User-reviewed regex with named groups round, markers, slide, date, acq_id, and scene. Use explicit empty named captures for fields absent from source filenames."),
            ParameterField(name="intake.expected_rounds", label="Expected rounds", kind="multiselect", required=True, default=[], dynamic_choices_key="available_rounds", section="Intake", stage_id="intake", help_text="Select expected round labels for coverage QC."),
            ParameterField(name="intake.expected_channels", label="Expected channels per image", kind="int", required=True, min_value=1, step=1, section="Intake", stage_id="intake", help_text="Enter the channel count reviewed from the raw acquisition metadata; it is not inferred from filenames or an example panel."),
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
            ParameterField(name="segmentation.backend", label="Segmentation engine", kind="select", choices=["MCMICRO"], default="MCMICRO", section="Segmentation", stage_id="segmentation", help_text="The toolkit runs MCMICRO for cell segmentation."),
            ParameterField(name="segmentation.parameters.segmenter", label="MCMICRO segmenter", kind="select", choices=["mccellpose", "cellpose", "unmicst"], default="mccellpose", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.dapi_channel", label="MCMICRO DAPI channel (1-based)", kind="int", default=1, min_value=1, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_model", label="Cellpose model", kind="text", default="cyto3", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_diameter", label="Cellpose diameter", kind="int", default=30, min_value=1, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.cellpose_expansion", label="Cell expansion", kind="int", default=10, min_value=0, unit="pixels", section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.parameters.use_gpu", label="Use GPU", kind="bool", default=False, section="Segmentation", stage_id="segmentation", advanced=True),
            ParameterField(name="segmentation.enabled", label="Run segmentation", kind="bool", default=False, section="Segmentation", stage_id="segmentation"),
        ],
        input_description=(
            "- Select an existing project directory, CycIF image directory, and `project_samples.csv` with nonempty `biological_unit_id`, `specimen_id`, `condition`, and `tissue_source` values.\n"
            "- Place each slide's raw CZI files directly in its own immediate subfolder of the image directory; optional OME-XML sidecars belong alongside them. Recursive preflight discovery alone does not establish this intake layout.\n"
            "- Supply a reviewed filename regex with named captures `round`, `markers`, `slide`, `date`, `acq_id`, and `scene` (use empty captures for absent fields), a nonempty expected-round list, and a positive expected channel count.\n"
            "- Review and confirm every discovered `slide=<slide>;scene=<scene>` group, mapping each exactly to a project `specimen_id` or `UNMAPPED`.\n"
            "- Conversion is optional and requires an allocated SLURM worker. Registration reads original CZI files, requires confirmed grouping, a reference round, and configured zero-based DAPI channel and scale; duplicate round files block it.\n"
            "- Export requires a selected, confirmed marker-panel CSV with `round`, `channel_index`, `marker_name`, `target_cell_compartment`, `fluorophore`, and `notes` headers. Round names must match, and channel indices must be unique and contiguous from zero within each round.\n"
            "- Optional MCMICRO segmentation requires successful registered composite export, an allocated SLURM worker, and the configured `PAULI_FAST_SOFTWARES` runtime; its DAPI channel is one-based. Continuation from a prior stage requires matching validated checkpoints and explicit review of warning checkpoints."
        ),
        output_description=(
            "- Intake writes `cycif/intake/intake_inventory.csv`, `intake_checks.csv`, `intake_qc_report.html`, and report images when reached. Failed checks, missing coverage, or unparsed filenames block downstream imaging; unavailable metadata may instead be reported as unassessed with warnings.\n"
            "- If enabled and successful, conversion writes OME-TIFF outputs under `cycif/converted/`, with optional projections and pseudocolor composites; registration writes scene-specific artifacts under `cycif/registered/`.\n"
            "- Successful confirmed-panel export writes composite OME-TIFF and metadata under `cycif/exported/`, with masks and QC plots depending on settings; individual PIPEX-channel exports are disabled.\n"
            "- If enabled and successful, segmentation writes scene-specific MCMICRO files under `cycif/mcmicro/`. Intake success alone does not guarantee any downstream output."
        ),
        limitations=[
            "Conversion, registration, export, and segmentation run only when configured and the required software is available.",
            "This image-processing workflow does not classify cells into biological cell types.",
            "Image conversion and segmentation require a configured SLURM compute worker.",
        ],
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
        if segmentation.get("backend", "MCMICRO") != "MCMICRO":
            blockers.append("The CyCIF app supports MCMICRO segmentation only.")
        else:
            software_root = os.environ.get("PAULI_FAST_SOFTWARES")
            if not software_root:
                blockers.append("MCMICRO runtime unavailable: PAULI_FAST_SOFTWARES is not set in the worker environment.")
            else:
                mcmicro_home = Path(software_root) / "envs" / "spatial-proteomics" / "mcmicro"
                mcmicro_bin = Path(software_root) / "bin" / "mcmicro"
                lock_file = Path(software_root) / "config" / "workflow-runtime.lock.sh"
                if not mcmicro_home.is_dir() or not (mcmicro_home / "main.nf").is_file():
                    blockers.append(f"Pinned MCMICRO workflow is unavailable at {mcmicro_home}.")
                if not mcmicro_bin.is_file() or not os.access(mcmicro_bin, os.X_OK):
                    blockers.append(f"Pinned MCMICRO wrapper is unavailable or not executable: {mcmicro_bin}.")
                if not lock_file.is_file():
                    blockers.append(f"Pinned MCMICRO runtime lock is missing: {lock_file}.")
                params["mcmicro_runtime"] = {"home": str(mcmicro_home), "wrapper": str(mcmicro_bin), "runtime_lock": str(lock_file)}
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)


def _write_registration_handoff(result: dict[str, Any], scene_label: str, registration_dir: Path, dapi_channel: int) -> Path:
    handoff: dict[str, Any] = {
        "scene_label": scene_label,
        "reference_shape": list(result["reference_shape"]),
        "round_order": list(result["round_order"]),
        "pixel_size_um": result.get("pixel_size_um"),
        "params": result.get("params", {}),
        "rounds": {},
    }
    for round_name in result["round_order"]:
        item = result["rounds"][round_name]
        channels = item.get("channels")
        channel_path = registration_dir / f"{round_name}_registered.tif"
        if channels is not None and not channel_path.is_file():
            raise RuntimeError(f"Registration handoff is missing its durable TIFF stack: {channel_path}")
        handoff["rounds"][round_name] = {
            "registered": bool(item["registered"]),
            "status": str(item["status"]),
            "transform": item["transform"].tolist() if item.get("transform") is not None else None,
            "inliers": int(item.get("inliers", 0)),
            "matches": int(item.get("matches", 0)),
            "czi_path": str(item.get("czi_path", "")),
            "czi_size_bytes": int(item.get("czi_size_bytes", 0)),
            "mov_shape": list(item["mov_shape"]) if item.get("mov_shape") is not None else None,
            "channels_path": str(channel_path.resolve()) if channels is not None else None,
            "dapi_channel": int(dapi_channel),
        }
    handoff_path = registration_dir / f"{scene_label}_registration_handoff.json"
    handoff_path.write_text(json.dumps(handoff, sort_keys=True, indent=2), encoding="utf-8")
    return handoff_path


def _load_registration_handoff(path: Path) -> dict[str, Any]:
    import numpy as np
    import tifffile

    handoff = json.loads(path.read_text(encoding="utf-8"))
    rounds: dict[str, Any] = {}
    for round_name in handoff["round_order"]:
        item = handoff["rounds"][round_name]
        channels = tifffile.imread(item["channels_path"]) if item.get("channels_path") else None
        rounds[round_name] = {
            "channels": channels,
            "dapi": channels[int(item["dapi_channel"])] if channels is not None else None,
            "transform": np.asarray(item["transform"], dtype=np.float64) if item.get("transform") is not None else None,
            "inliers": int(item["inliers"]),
            "matches": int(item["matches"]),
            "registered": bool(item["registered"]),
            "status": str(item["status"]),
            "mov_shape": tuple(item["mov_shape"]) if item.get("mov_shape") is not None else None,
            "czi_path": item.get("czi_path"),
            "czi_size_bytes": int(item.get("czi_size_bytes", 0)),
        }
    return dict(
        rounds=rounds,
        round_order=list(handoff["round_order"]),
        reference_shape=tuple(handoff["reference_shape"]),
        pixel_size_um=handoff.get("pixel_size_um"),
        params=handoff.get("params", {}),
    )


def _continuation_stage(request: RunRequest, stage_id: str, emit: WorkflowEmitter) -> WorkflowResult | None:
    """Validate the complete source lineage before exposing any reused artifact."""
    selection = request.parameters.get("continuation", {})
    if not isinstance(selection, dict) or selection.get("mode") != "continue":
        return None
    start = selection.get("start_stage")
    required = {
        "conversion": {"intake"},
        "registration": {"intake"},
        "export": {"intake", "registration"},
        "segmentation": {"intake", "registration", "export"},
    }
    if start not in required:
        raise ValueError(f"Unsupported continuation boundary: {start!r}.")
    if start in {"conversion", "registration", "segmentation"}:
        config = request.parameters.get(str(start), {})
        if not isinstance(config, dict) or config.get("enabled") is not True:
            raise ValueError(f"{start.capitalize()} continuation requires {start}.enabled=true.")
    if start in {"export", "segmentation"}:
        _resolve_marker_panel(request.parameters, request.project_root)
    reasons, reused = validate_continuation(request)
    if reasons:
        raise ValueError("CyCIF continuation blocked: " + "; ".join(reasons))
    source = Path(selection["parent_run_dir"]).expanduser().resolve()
    if stage_id == "intake":
        from spatial_omics_toolkit.application.runs import atomic_json

        metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
        warning_ids = sorted({item["stage_id"] for item in reused if item.get("source_status") == "WARNING"})
        atomic_json(request.run_dir / "resume_receipt.json", {
            "parent_run_id": metadata.get("run_id"), "parent_run_path": str(source),
            "start_stage": start, "raw_input_fingerprint": _input_fingerprint(request.run_dir),
            "reused_checkpoints": reused, "warning_review_confirmed": selection.get("warning_review_confirmed") is True,
            "warning_stage_ids": warning_ids,
            "warning_review_recorded_at_utc": datetime.now(timezone.utc).isoformat() if warning_ids else None,
        })
    if stage_id == "conversion" and start != "conversion":
        event = StageEvent(stage_id=stage_id, status="SKIPPED", message="Conversion is independent of the selected continuation boundary.")
        emit(event)
        return WorkflowResult(status="SKIPPED", stages=[event])
    if stage_id not in required[start]:
        return None
    stage_items = [item for item in reused if item["stage_id"] == stage_id]
    artifacts = [item["path"] for item in stage_items]
    source_status = stage_items[0]["source_status"]
    event = StageEvent(stage_id=stage_id, status="REUSED", message=f"Reused verified {stage_id} checkpoint from parent run {source.name} (source status {source_status}).", parameters={"parent_run_dir": str(source), "start_stage": start, "source_status": source_status}, artifacts=artifacts)
    emit(event)
    from spatial_omics_toolkit.application.runs import atomic_json

    source_checkpoint = json.loads((source / "checkpoint_receipts.json").read_text(encoding="utf-8"))["checkpoints"][stage_id]
    receipt_path = request.run_dir / "checkpoint_receipts.json"
    with (request.run_dir / ".checkpoint_receipts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"workflow_id": request.workflow_id, "checkpoints": {}}
        child["checkpoints"][stage_id] = source_checkpoint
        atomic_json(receipt_path, child)
    state: dict[str, Any] = {}
    if stage_id == "registration":
        state["handoffs"] = sorted(path for path in artifacts if path.endswith("_registration_handoff.json"))
    if stage_id == "export":
        state["composites"] = {Path(path).name.removesuffix("_composite.ome.tif"): path for path in artifacts if path.endswith("_composite.ome.tif")}
        state["marker_map"] = source_checkpoint["settings"].get("marker_map")
    return WorkflowResult(status="REUSED", stages=[event], artifacts=artifacts, warnings=_warning_reuse_messages(stage_items), state=state)


def execute_stage(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
    """Execute exactly one imaging stage, communicating through durable files."""
    dependencies = {stage.stage_id: stage.depends_on for stage in describe().stages}
    if stage_id not in dependencies:
        raise ValueError(f"Unknown CyCIF stage: {stage_id}")
    events: list[StageEvent] = []
    artifacts: list[str] = []
    warnings: list[str] = []
    state: dict[str, Any] = {}

    def record(event: StageEvent) -> None:
        events.append(event)
        emit(event)

    def finish(status: StageStatus, message: str, parameters: dict[str, Any] | None = None) -> WorkflowResult:
        serializable_parameters = json.loads(json.dumps(parameters or {}, default=str))
        event = StageEvent(stage_id=stage_id, status=status, message=message, parameters=serializable_parameters, artifacts=artifacts, tool_versions={"spatial_omics_toolkit": _version("spatial-omics-toolkit")})
        record(event)
        if status in {"FAILED", "BLOCKED"}:
            warnings.append(message)
        if status in {"SUCCEEDED", "WARNING"}:
            try:
                _write_checkpoint_receipts(request, events)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return WorkflowResult(status=status, stages=events, artifacts=artifacts, warnings=warnings, state=state)

    for dependency in dependencies[stage_id]:
        result = upstream.get(dependency)
        if result is None or result.status not in {"SUCCEEDED", "WARNING", "REUSED", "SKIPPED"} or (dependency == "intake" and result.status == "SKIPPED"):
            return finish("BLOCKED", f"Required upstream stage {dependency} is missing or not cleared.")
    try:
        resumed = _continuation_stage(request, stage_id, record)
        if resumed is not None:
            return resumed
        registration = request.parameters.get("registration", {})
        segmentation = request.parameters.get("segmentation", {})
        conversion = request.parameters.get("conversion", {})
        registration_enabled = (isinstance(registration, dict) and registration.get("enabled") is True) or (stage_id == "export" and upstream["registration"].status == "REUSED")
        segmentation_enabled = isinstance(segmentation, dict) and segmentation.get("enabled") is True
        if stage_id == "conversion" and (not isinstance(conversion, dict) or conversion.get("enabled") is not True):
            return finish("SKIPPED", "Conversion was not enabled.")
        if stage_id in {"registration", "export"} and not registration_enabled:
            if stage_id == "export" and segmentation_enabled:
                return finish("BLOCKED", "Segmentation requires registration and export.")
            return finish("SKIPPED", "Registration and export were not enabled.")
        if stage_id == "segmentation" and not segmentation_enabled:
            return finish("SKIPPED", "Segmentation was not enabled.")
        if stage_id in {"export", "segmentation"} and upstream[dependencies[stage_id][0]].status == "SKIPPED":
            return finish("BLOCKED", "The required imaging stage was skipped.")

        if stage_id == "intake":
            check = preflight(request)
            if not check.ready:
                warnings.extend(check.blockers + check.warnings)
                return finish("BLOCKED", "CyCIF preflight blocked: " + "; ".join(check.blockers), check.resolved_parameters)
            from spatial_omics_toolkit.cycif import step01_scan_intake

            warnings.extend(check.warnings)
            intake_dir = request.run_dir / "cycif" / "intake"
            record(StageEvent(stage_id=stage_id, status="RUNNING", message="Scanning raw CZI inventory and validating acquisition metadata."))
            result = step01_scan_intake(raw_dir=str(request.modality_dir), reports_dir=str(intake_dir), generate_html_report=True, generate_dapi_preview=False, strict=False, verbose=False, **_intake_options(request.parameters))
            artifacts.extend(str(path) for path in sorted(intake_dir.rglob("*")) if path.is_file())
            checks = getattr(result, "checks", [])
            rows = checks.to_dict(orient="records") if hasattr(checks, "to_dict") else []
            failed = [row for row in rows if str(row.get("Status", "")).strip() == "⚠"]
            unassessed = [row for row in rows if str(row.get("Status", "")).strip() == "—" and str(row.get("Check", "")).startswith(("Channel count", "Pixel size"))]
            issues = {"check_warnings": failed, "checks_unassessed": unassessed, "coverage_missing": [str(x) for x in getattr(result, "coverage_missing", []) or []], "unparsed": [str(x) for x in getattr(result, "unparsed", []) or []], "preflight_warnings": check.warnings}
            if failed or issues["coverage_missing"] or issues["unparsed"]:
                return finish("BLOCKED", "Intake found failed QC checks; review the report. Downstream imaging stages are blocked.", issues)
            if unassessed or check.warnings:
                warnings.append(json.dumps({"checks_unassessed": unassessed, "preflight_warnings": check.warnings}, default=str))
                return finish("WARNING", "Intake completed with unassessed QC checks; review the report and warning details.", issues)
            return finish("SUCCEEDED", "Canonical step01_scan_intake completed.", issues)

        if stage_id == "conversion":
            if not os.environ.get("SLURM_JOB_ID"):
                return finish("BLOCKED", "CZI conversion is blocked outside an allocated SLURM worker.")
            from spatial_omics_toolkit.cycif import ImageConversionConfig, step02_convert_raw_images

            options = conversion.get("config", {})
            if not isinstance(options, dict):
                raise ValueError("parameters.conversion.config must be an object of ImageConversionConfig fields.")
            config = ImageConversionConfig(**options)
            files = sorted(path for path in request.modality_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".czi")
            if not files:
                raise RuntimeError("No raw CZI files were inventoried for conversion.")
            results: dict[str, Any] = {}
            for folder in sorted({path.parent for path in files}):
                key = folder.relative_to(request.modality_dir).as_posix().replace("/", "__") or "root"
                destination = request.run_dir / "cycif" / "converted" / key
                destination.mkdir(parents=True, exist_ok=True)
                record(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Converting inventoried CZI files in {folder}."))
                results[str(folder)] = step02_convert_raw_images(str(folder), str(destination), dispatch="inprocess", conversion_config=config, strict=True, verbose=False)
                outputs = [str(path) for path in sorted(destination.rglob("*")) if path.is_file()]
                if not outputs:
                    raise RuntimeError(f"Canonical conversion produced no outputs for {folder}.")
                artifacts.extend(outputs)
            return finish("SUCCEEDED", f"Canonical CZI conversion completed for {len(files)} inventoried files.", {"results_by_input_folder": results})

        if stage_id == "registration":
            if not registration.get("confirmed_grouping"):
                return finish("BLOCKED", "Registration requires user-confirmed filename grouping in parameters.registration.confirmed_grouping=true.")
            from spatial_omics_toolkit.cycif import step03_register_rounds

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
                    return finish("BLOCKED", f"Ambiguous duplicate CZI for slide={key[0]}, scene={key[1]}, round={round_label}.")
                grouped[key][round_label] = str(path)
            if not grouped:
                return finish("BLOCKED", "No CZI files matched the configured slide, scene, and round pattern.")
            handoffs: list[str] = []
            for (slide, scene), round_files in grouped.items():
                label = f"{slide}_{scene}"
                destination = request.run_dir / "cycif" / "registered" / label
                destination.mkdir(parents=True, exist_ok=True)
                record(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Registering {len(round_files)} rounds for {label}.", parameters={"slide": slide, "scene": scene, "rounds": sorted(round_files)}))
                result = step03_register_rounds(round_files, output_dir=str(destination), verbose=False, **registration.get("config", {}))
                rounds = result.get("rounds", {})
                bad = {name: info.get("status", "unregistered") for name, info in rounds.items() if not info.get("registered", False)}
                if not bad:
                    handoffs.append(
                        str(_write_registration_handoff(
                            result, label, destination,
                            int(registration.get("config", {}).get("dapi_channel", 0)),
                        ))
                    )
                artifacts.extend(str(path) for path in destination.rglob("*") if path.is_file())
                if bad:
                    return finish("FAILED", f"Registration transform failures preserved: {bad}")
            state["handoffs"] = handoffs
            return finish("SUCCEEDED", f"Registered {len(handoffs)} scenes.")

        if stage_id == "export":
            from spatial_omics_toolkit.cycif import step04_export_registered

            panel = _resolve_marker_panel(request.parameters, request.project_root)
            if panel is None:
                return finish("BLOCKED", "Registration succeeded, but export is blocked until a marker-panel CSV is selected and explicitly confirmed.")
            marker_map = panel["marker_map"]
            handoffs = upstream["registration"].state.get("handoffs", [])
            if not handoffs:
                return finish("BLOCKED", "Registration checkpoint has no durable per-scene registration handoff.")
            composites: dict[str, str] = {}
            for name in handoffs:
                handoff = Path(name)
                metadata = json.loads(handoff.read_text(encoding="utf-8"))
                label = str(metadata["scene_label"])
                result = _load_registration_handoff(handoff)
                if set(marker_map) != set(result["round_order"]):
                    return finish("BLOCKED", f"Confirmed marker panel rounds {sorted(marker_map)} do not match scene {label} rounds {sorted(result['round_order'])}; registration QC is retained.")
                destination = request.run_dir / "cycif" / "exported" / label
                destination.mkdir(parents=True, exist_ok=True)
                record(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Exporting registered scene {label}.", parameters={"marker_panel": panel}))
                exported = step04_export_registered(result, label, str(destination), marker_map=marker_map, write_pipex_channels=False, verbose=False, **registration.get("export_config", {}))
                outputs = [str(path) for path in destination.rglob("*") if path.is_file()]
                if not outputs:
                    raise RuntimeError(f"step04_export_registered produced no artifacts for {label}: {exported}")
                artifacts.extend(outputs)
                composite = exported.get("composite_path")
                if composite:
                    composites[label] = str(composite)
            state.update({"composites": composites, "marker_map": marker_map})
            return finish("SUCCEEDED", f"Exported {len(handoffs)} scenes with the confirmed marker map.", {"marker_panel": panel})

        if segmentation.get("backend", "MCMICRO") != "MCMICRO":
            return finish("BLOCKED", "The CyCIF app supports MCMICRO segmentation only.")
        if not os.environ.get("SLURM_JOB_ID"):
            return finish("BLOCKED", "MCMICRO execution is blocked outside an allocated SLURM worker.")
        check = preflight(request)
        if not check.ready:
            return finish("BLOCKED", "MCMICRO preflight blocked: " + "; ".join(check.blockers))
        from spatial_omics_toolkit.cycif import MCMICROConfig, step05_segment_cells

        runtime = check.resolved_parameters["mcmicro_runtime"]
        version = subprocess.run([runtime["wrapper"], "-version"], check=True, capture_output=True, text=True, timeout=60)
        mcmicro_version = (version.stdout or version.stderr).strip() or "version command returned no output"
        options = segmentation.get("parameters", {})
        if not isinstance(options, dict):
            raise ValueError("segmentation.parameters must be an object of MCMICROConfig settings.")
        # The canonical default is the cluster-submitting exacloud profile.
        # Nested execution must instead stay inside the current allocation.
        if options.get("profile", "singularity") != "singularity" or options.get("extra_args"):
            return finish("BLOCKED", "MCMICRO must use the allocation-local singularity profile without extra Nextflow arguments.")
        if options.get("use_gpu") and not any(os.environ.get(key) for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")):
            return finish("BLOCKED", "MCMICRO GPU execution requires an allocated SLURM GPU.")
        from spatial_omics_toolkit.application.nextflow import _allocated_resources

        cpus, memory_gb = _allocated_resources(request)
        memory_mb = max(1, memory_gb * 1024 - 1024)
        config_path = request.run_dir / "cycif" / "mcmicro-local.config"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "process.executor = 'local'\n"
            f"executor.cpus = {cpus}\nexecutor.memory = '{memory_mb} MB'\nexecutor.queueSize = 1\n"
            f"process.resourceLimits = [cpus: {cpus}, memory: '{memory_mb} MB']\n"
            "process.maxForks = 1\n"
            "process {\n  withName: 'segmentation:worker' {\n"
            f"    cpus = {cpus}\n    memory = '{memory_mb} MB'\n"
            "  }\n}\n",
            encoding="utf-8",
        )
        options = {**options, "profile": "singularity", "extra_args": f"-c {shlex.quote(str(config_path))}"}
        config = MCMICROConfig(mcmicro_bin=runtime["wrapper"], mcmicro_nf=str(Path(runtime["home"]) / "main.nf"), **options)
        exported_state = upstream["export"].state
        composites = exported_state.get("composites", {})
        marker_map = exported_state.get("marker_map")
        if not composites or not marker_map:
            return finish("BLOCKED", "MCMICRO requires exported composites and a confirmed marker map.")
        for label, composite in composites.items():
            if not Path(composite).is_file():
                raise RuntimeError(f"No usable composite OME-TIFF for MCMICRO scene {label}.")
            project_dir = request.run_dir / "cycif" / "mcmicro" / label
            record(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Running MCMICRO for {label} in the allocated worker.", tool_versions={"MCMICRO": mcmicro_version}, parameters={"project_dir": str(project_dir), "composite_path": composite}))
            result = step05_segment_cells("mcmicro", scene_label=label, composite_ometiff_path=composite, marker_map=marker_map, project_dir=str(project_dir), mcmicro_config=config, run=True, load_results=True, verbose=False)
            outputs = [str(path) for path in project_dir.rglob("*") if path.is_file()]
            if not outputs:
                raise RuntimeError(f"MCMICRO returned without run-scoped artifacts for {label}: {result}")
            artifacts.extend(outputs)
        return finish("SUCCEEDED", f"MCMICRO {mcmicro_version} completed for {len(composites)} scenes.", {"backend": "MCMICRO", "segmenter": config.segmenter, "version": mcmicro_version})
    except ValueError as exc:
        return finish("BLOCKED", f"{stage_id} stage blocked: {exc}")
    except Exception as exc:
        return finish("FAILED", f"{stage_id} stage failed: {type(exc).__name__}: {exc}")




