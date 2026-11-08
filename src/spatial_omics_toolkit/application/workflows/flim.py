from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import math
from pathlib import Path
from typing import Any

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

WORKFLOW_ID = "flim.raw_sdt"
_CANONICAL_SCRIPTS = Path(
    "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/"
    "01_exploratory_analysis/02_scripts"
)


def _source_dir() -> Path:
    configured = os.environ.get("SPATIAL_TOOLKIT_FLIM_SOURCE_DIR")
    return Path(configured).expanduser().resolve() if configured else _CANONICAL_SCRIPTS


def _load_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _sources() -> tuple[Any, Any]:
    directory = _source_dir()
    flim_path = directory / "clinical_roi_flim.py"
    plots_path = directory / "clinical_roi_flim_plots.py"
    missing = [str(path) for path in (flim_path, plots_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Canonical FLIM source is unavailable: " + ", ".join(missing))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return _load_module("clinical_roi_flim", flim_path), _load_module("clinical_roi_flim_plots", plots_path)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"canonical_workflow": "clinical_roi_flim.py (source-bound)"}
    for package in ("numpy", "pandas", "phasorpy", "scipy", "threadpoolctl", "matplotlib", "sdtfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    versions["clinical_roi_flim.py"] = _sha256(_source_dir() / "clinical_roi_flim.py")
    versions["clinical_roi_flim_plots.py"] = _sha256(_source_dir() / "clinical_roi_flim_plots.py")
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe() -> WorkflowDescriptor:
    source = ToolReference(
        name="clinical_roi_flim.py",
        version="SHA-256 recorded in each run receipt",
        role="Inventory raw SDT data, check quality, compute phasors and apparent lifetimes, and apply masks, filtering, and measured-reference calibration",
        documentation_url=None,
    )
    plots = ToolReference(
        name="clinical_roi_flim_plots.py",
        version="SHA-256 recorded in each run receipt",
        role="Create quality-control and analysis plots linked to processing records",
        documentation_url=None,
    )
    return WorkflowDescriptor(
        workflow_id=WORKFLOW_ID,
        modality="FLIM",
        title="Raw FLIM SDT QC and phasor analysis",
        description="Analyze SDT data in a FLIM-only job. Raw quality control stops after intensity and decay diagnostics; full analysis continues to frequency-dependent phasors. The workflow does not pair hyperspectral data or perform cross-modal analysis.",
        stages=[
            StageDefinition(stage_id="flim_inventory", title="SDT inventory", purpose="Hash and inspect all supported SDT blocks while retaining source block and acquisition metadata.", tools=[source], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="inspect_flim_inputs", role="Inspect source SDT inventory")], event_stage_ids=["flim_block_review", "flim_block_selection_policy"], depends_on=[], condition='Always; pauses for review unless all-block technical QC is authorized', inputs=['SDT files, review decision'], outputs=['Inventory and selected block metadata']),
            StageDefinition(stage_id="flim_raw_qc", title="Raw decay QC", purpose="Summarize recorded counts and decay data using the source decoder in chunks.", tools=[source, plots], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="select_flim_blocks", role="Select reviewed blocks"), FunctionReference(source_file="external:clinical_roi_flim.py", function_name="summarize_flim_qc", role="Summarize raw count/decay data"), FunctionReference(source_file="external:clinical_roi_flim_plots.py", function_name="plot_flim_qc", role="Render raw QC diagnostics")], event_stage_ids=["flim_raw_qc_plots"], depends_on=['flim_inventory'], condition='After reviewed inventory', inputs=['Selected inventory'], outputs=['Raw QC arrays and plots']),
            StageDefinition(stage_id="flim_phasor", title="Phasor analysis", purpose="Compute phasors using the recorded time axis and an explicitly documented acquisition frequency.", tools=[source, plots], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="compute_flim_phasors", role="Compute physical-time phasors"), FunctionReference(source_file="external:clinical_roi_flim_plots.py", function_name="plot_flim_phasors", role="Render phasor diagnostics")], event_stage_ids=["flim_phasor_plots"], depends_on=['flim_inventory'], condition='Full analysis only', inputs=['Selected inventory, acquisition frequency'], outputs=['Phasor arrays and plots']),
            StageDefinition(stage_id="flim_count_mask", title="Count mask", purpose="Apply the specified minimum total count threshold to select pixels for analysis.", tools=[source, plots], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="mask_flim_phasors", role="Apply declared count mask"), FunctionReference(source_file="external:clinical_roi_flim_plots.py", function_name="plot_flim_mask", role="Render mask diagnostics")], event_stage_ids=["flim_count_mask_plots"], depends_on=['flim_phasor'], condition='Full analysis only', inputs=['Phasors, count threshold'], outputs=['Masked phasors']),
            StageDefinition(stage_id="flim_filter", title="Optional phasor filter", purpose="Optionally apply the source workflow's median-coordinate filter.", tools=[source, plots], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="filter_flim_phasors", role="Optionally filter phasors"), FunctionReference(source_file="external:clinical_roi_flim_plots.py", function_name="plot_flim_filter", role="Plot filter diagnostics")], event_stage_ids=["flim_filter_plots"], depends_on=['flim_count_mask'], condition='Full analysis and filter enabled', inputs=['Masked phasors'], outputs=['Filtered phasors']),
            StageDefinition(stage_id="flim_lifetimes", title="Apparent lifetime conversion", purpose="Convert final phasors to apparent phase/modulation lifetimes; calibrated status is recorded only after measured-reference calibration.", tools=[source, plots], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="compute_flim_lifetimes", role="Compute apparent lifetimes"), FunctionReference(source_file="external:clinical_roi_flim_plots.py", function_name="plot_flim_lifetimes", role="Render lifetime diagnostics")], event_stage_ids=["flim_lifetime_plots"], depends_on=['flim_count_mask', 'flim_filter', 'flim_reference_calibration'], condition='Full analysis only', inputs=['Final phasors'], outputs=['Lifetime arrays, plots and manifest']),
            StageDefinition(stage_id="flim_reference_inventory", title="Measured-reference inventory", purpose="Inspect the explicitly selected reference SDT block and verify its identity.", tools=[source], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="inspect_flim_inputs", role="Inspect measured-reference SDT")], depends_on=["flim_inventory"], condition="Full analysis with measured reference", inputs=["Reference SDT"], outputs=["Reference inventory"]),
            StageDefinition(stage_id="flim_reference_raw_qc", title="Measured-reference raw QC", purpose="Summarize recorded reference counts and decay data; this alone does not calibrate the target.", tools=[source], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="summarize_flim_qc", role="Summarize reference count/decay data")], depends_on=["flim_reference_inventory"], condition="Full analysis with measured reference", inputs=["Reference inventory"], outputs=["Reference raw QC"]),
            StageDefinition(stage_id="flim_reference_phasor", title="Measured-reference phasors", purpose="Compute phasors for the supplied measured standard using the specified acquisition frequency.", tools=[source], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="compute_flim_phasors", role="Compute measured-reference phasors")], depends_on=["flim_reference_inventory"], condition="Full analysis with measured reference", inputs=["Reference inventory, acquisition frequency"], outputs=["Reference phasors"]),
            StageDefinition(stage_id="flim_reference_calibration", title="Optional measured-reference calibration", purpose="Apply phasor calibration only from a supplied measured SDT reference with explicit block, lifetime, provenance, and matching-settings confirmation.", tools=[source], implementations=[FunctionReference(source_file="external:clinical_roi_flim.py", function_name="calibrate_flim_phasors", role="Apply explicitly confirmed measured-reference calibration")], depends_on=['flim_count_mask', 'flim_filter', 'flim_reference_phasor'], condition='Full analysis with measured reference', inputs=['Target and reference phasors'], outputs=['Calibrated phasors']),
        ],
        input_description=(
            "- A complete modality folder containing one or more native `.sdt` files, searched recursively; all discovered SDTs are included. Other files are reported, not converted. The canonical `clinical_roi_flim.py` and `clinical_roi_flim_plots.py` scripts are required.\n"
            "- The shared sample CSV and confirmed acquisition-to-specimen mapping. SDT data must also satisfy the native decoder's supported layout and timing/count checks.\n"
            "- Choose raw QC or full analysis. Normally, review the worker's block inventory and confirm unique nonnegative block indices present in every SDT; the review is bound to the run and inventory. For raw QC only, explicitly authorize all-recorded-block technical QC instead of selecting blocks. Block selection does not establish detector or biological identity.\n"
            "- Full analysis requires a positive acquisition frequency in MHz and written provenance, plus a nonnegative count threshold. Optional median filtering requires 2-D data, a positive odd size, and a positive repeat count.\n"
            "- Optional measured-reference calibration requires a separate reference `.sdt` outside the target folder, one explicit reference block, a nonnegative known lifetime, provenance, and confirmation of matching detector, timing, and optical settings. An optional existing HS–FLIM pairing manifest is recorded only."
        ),
        output_description=(
            "- The initial inventory records SDT sources and blocks. Unless all-block raw QC is explicitly authorized, the run pauses at `QC_PENDING` for block selection and review; no subsequent analysis has run.\n"
            "- After selection or authorized all-block policy, raw QC produces count and decay arrays, summaries, diagnostic PDFs, and processing records, then stops without phasors, masks, calibration, or lifetimes.\n"
            "- Full analysis continues, if earlier stages succeed, to physical-time phasors, count masks, apparent phase and modulation lifetimes, their diagnostic plots, a result manifest, and processing records. Median-filter outputs and measured-reference QC, phasors, and calibration are produced only when requested.\n"
            "- Without a measured reference, lifetimes are uncalibrated. A timing-period mismatch normally stops lifetime conversion; the explicit mismatch option retains phasors but writes NaN lifetimes for mismatched blocks. Failures can leave partial outputs; no HS pairing or cross-modal analysis is performed."
        ),
        parameter_fields=[
            ParameterField(name="analysis_scope", label="Analysis scope", kind="select", default="full", choices=["raw_qc", "full"], section="Analysis", help_text="raw_qc performs raw intensity/decay technical QC and stops before frequency-dependent phasor or lifetime calculation. Use explicit reviewed block selection, or opt into all recorded blocks with both technical-QC authorization fields. No detector/cell identity is inferred. full preserves the complete workflow."),
            ParameterField(name="block_indices", label="Reviewed SDT blocks", kind="multiselect", dynamic_choices_key="flim_block_indices", section="SDT inventory", stage_id="flim_inventory", help_text="For selected-block mode, inspect the worker inventory and choose blocks explicitly; no default-all selection is made. All-block raw QC is a separate explicit technical-QC policy."),
            ParameterField(name="raw_qc_all_blocks", label="Inspect all recorded blocks for technical QC", kind="bool", default=False, section="SDT inventory", stage_id="flim_inventory", help_text="Opt in only for analysis_scope=raw_qc. Processes all blocks recorded in the native SDT inventory for technical count/decay QC; no detector identity is assumed and no phasors/lifetimes are computed."),
            ParameterField(name="technical_qc_authorized", label="Authorize all-block technical QC", kind="bool", default=False, section="SDT inventory", stage_id="flim_inventory", help_text="Required together with raw_qc_all_blocks. Explicitly authorizes raw count/decay inspection of all recorded blocks, not detector selection or biological analysis."),
            ParameterField(name="blocks_review_confirmed", label="Confirm reviewed block selections", kind="bool", default=False, section="SDT inventory", stage_id="flim_inventory", help_text="Required on resume after reviewing the worker-generated block inventory."),
            ParameterField(name="frequency_mhz", label="Acquisition frequency", kind="float", required=False, min_value=0.000001, unit="MHz", section="Phasor analysis", stage_id="flim_phasor", help_text="Required only for analysis_scope=full after block review. Use an explicitly documented acquisition frequency; it is never inferred from decay-window length."),
            ParameterField(name="frequency_provenance", label="Frequency source / provenance", kind="text", required=False, section="Phasor analysis", stage_id="flim_phasor", help_text="Required only for analysis_scope=full after block review; record acquisition settings source and rationale."),
            ParameterField(name="minimum_total_counts", label="Minimum total counts per pixel", kind="float", required=False, default=0.0, min_value=0, unit="counts", section="Count mask", stage_id="flim_count_mask", help_text="Used only by full phasor analysis; raw_qc does not apply a count mask."),
            ParameterField(name="filter_enabled", label="Apply median phasor-coordinate filter", kind="bool", default=False, section="Filter", stage_id="flim_filter", help_text="Optional canonical filter; does not replace or alter raw source data."),
            ParameterField(name="filter_size", label="Median filter size", kind="int", default=3, min_value=1, section="Filter", stage_id="flim_filter", advanced=True, help_text="Must be odd; used only when filtering is enabled."),
            ParameterField(name="filter_repeat", label="Median filter repeats", kind="int", default=1, min_value=1, section="Filter", stage_id="flim_filter", advanced=True, help_text="Used only when filtering is enabled."),
            ParameterField(name="reference_calibration.path", label="Measured reference SDT file", kind="path", section="Measured reference calibration", stage_id="flim_reference_calibration", help_text="Leave blank for uncalibrated apparent lifetimes. Calibration is enabled only when an actual measured reference file is supplied."),
            ParameterField(name="reference_calibration.block_index", label="Reference SDT block index", kind="int", min_value=0, section="Measured reference calibration", stage_id="flim_reference_calibration", help_text="Explicitly select the measured reference block."),
            ParameterField(name="reference_calibration.known_lifetime_ns", label="Measured reference lifetime", kind="float", min_value=0, unit="ns", section="Measured reference calibration", stage_id="flim_reference_calibration"),
            ParameterField(name="reference_calibration.provenance", label="Reference measurement provenance", kind="text", section="Measured reference calibration", stage_id="flim_reference_calibration"),
            ParameterField(name="reference_calibration.instrument_settings_match_confirmed", label="Confirm matching instrument settings", kind="bool", default=False, section="Measured reference calibration", stage_id="flim_reference_calibration", help_text="Required confirmation that detector, timing, and optical settings are applicable to targets."),
            ParameterField(name="period_tolerance_bins", label="TCSPC period tolerance", kind="float", default=1.0, min_value=0, unit="time bins", section="Lifetime conversion", stage_id="flim_lifetimes", advanced=True),
            ParameterField(name="allow_period_mismatch", label="Retain phasors with lifetime outputs undefined on period mismatch", kind="bool", default=False, section="Lifetime conversion", stage_id="flim_lifetimes", advanced=True, help_text="When enabled, mismatched blocks retain phasors but receive NaN lifetime values."),
            ParameterField(name="hs_flim_pairing_manifest", label="Explicit HS+FLIM pairing manifest (record only)", kind="config_file", section="Provenance", stage_id="flim_inventory", help_text="Optional for this FLIM-only adapter; records supplied pair evidence without inference or cross-modal output."),
        ],
        limitations=[
            "SDT is the only supported raw input; unsupported formats are reported, not converted.",
            "A laser/acquisition frequency is never inferred from the recorded decay window.",
            "Uncalibrated outputs are apparent lifetimes and are never labelled calibrated.",
            "Measured reference calibration is available only when a separate reference SDT and measured lifetime are supplied with matching-settings confirmation.",
            "The adapter does not infer HS+FLIM pairs, publish a pairing manifest, or perform cross-modal analysis.",
            "The inventory normally pauses at QC_PENDING; selected-block analyses require a reviewer decision linked to the inventory record. Only the opt-in raw_qc inspection of all recorded blocks bypasses individual block review and remains limited to counts and decay data.",
            "Scientific calculations are performed by the source workflow, not reimplemented by this adapter.",
        ],
    )


def _scan(request: RunRequest) -> tuple[list[Path], list[str]]:
    root = request.modality_dir.expanduser().resolve()
    supported: list[Path] = []
    unsupported: list[str] = []
    if not root.is_dir():
        return supported, [f"Modality folder does not exist: {root}"]
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.suffix.lower() == ".sdt":
                supported.append(path.resolve())
            else:
                unsupported.append(str(path.resolve()))
    return supported, unsupported

def _stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

def _validate_review_decision(
    review_decision_path: Path,
    inventory: dict[str, Any],
    *,
    selected_blocks: list[int],
    run_id: str,
    review_attempt: int,
) -> None:
    if not review_decision_path.is_file():
        raise ValueError("Review confirmation artifact is missing; block selection cannot proceed.")
    review_decision = json.loads(review_decision_path.read_text())
    reviewed_blocks = review_decision.get("block_indices")
    if review_decision.get("blocks_review_confirmed") is not True or not isinstance(reviewed_blocks, list) or sorted(reviewed_blocks) != sorted(selected_blocks):
        raise ValueError("Review decision does not match the explicitly selected SDT blocks.")
    if review_decision.get("flim_inventory_receipt_sha256") != _sha256(Path(inventory["receipt_path"])):
        raise ValueError("Review decision is not bound to the canonical FLIM inventory receipt.")
    reviewer = review_decision.get("reviewer") or review_decision.get("user")
    review_timestamp = review_decision.get("timestamp") or review_decision.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(review_timestamp, str) or not review_timestamp.strip():
        raise ValueError("Review decision must record the reviewer/user and timestamp.")
    if review_decision.get("run_id") != run_id:
        raise ValueError("Review decision belongs to a different workflow run.")
    if review_decision.get("review_attempt") != review_attempt:
        raise ValueError("Review decision belongs to a different review attempt.")



def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    warnings: list[str] = []
    inputs, unsupported = _scan(request)
    if not inputs:
        blockers.append("No supported .sdt input files were found in the complete modality folder.")
    source_directory = _source_dir()
    missing_sources = [name for name in ("clinical_roi_flim.py", "clinical_roi_flim_plots.py") if not (source_directory / name).is_file()]
    if missing_sources:
        blockers.append("Canonical FLIM source is unavailable: " + ", ".join(str(source_directory / name) for name in missing_sources))
    parameters = request.parameters
    analysis_scope = parameters.get("analysis_scope", "full")
    if analysis_scope not in {"raw_qc", "full"}:
        blockers.append("analysis_scope must be 'raw_qc' or 'full'.")
    raw_qc_all_blocks = parameters.get("raw_qc_all_blocks", False)
    technical_qc_authorized = parameters.get("technical_qc_authorized", False)
    if not isinstance(raw_qc_all_blocks, bool) or not isinstance(technical_qc_authorized, bool):
        blockers.append("raw_qc_all_blocks and technical_qc_authorized must be booleans.")
        raw_qc_all_blocks = False
        technical_qc_authorized = False
    if raw_qc_all_blocks and analysis_scope != "raw_qc":
        blockers.append("raw_qc_all_blocks is permitted only with analysis_scope='raw_qc'.")
    if raw_qc_all_blocks and not technical_qc_authorized:
        blockers.append("All-block raw QC requires explicit technical_qc_authorized=true.")
    if technical_qc_authorized and not raw_qc_all_blocks:
        blockers.append("technical_qc_authorized applies only when raw_qc_all_blocks=true.")
    raw_block_indices = parameters.get("block_indices")
    selected_blocks: list[int] | None = None
    if isinstance(raw_block_indices, str):
        try:
            parsed = [int(value.strip()) for value in raw_block_indices.split(",") if value.strip()]
            selected_blocks = parsed or None
        except ValueError:
            blockers.append("block_indices must be comma-separated nonnegative integers.")
    elif isinstance(raw_block_indices, list):
        selected_blocks = raw_block_indices or None
    if selected_blocks is not None and (not selected_blocks or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in selected_blocks) or len(set(selected_blocks)) != len(selected_blocks)):
        blockers.append("block_indices must be unique nonnegative integer SDT block indices.")
    if raw_qc_all_blocks and selected_blocks is not None:
        blockers.append("Do not combine raw_qc_all_blocks with an explicit block_indices selection.")
    if selected_blocks is not None:
        frequency = parameters.get("frequency_mhz")
        frequency_source = parameters.get("frequency_provenance")
        if analysis_scope == "full":
            if not isinstance(frequency, (int, float)) or isinstance(frequency, bool) or not math.isfinite(float(frequency)) or frequency <= 0:
                blockers.append("A positive acquisition frequency in MHz is required for full analysis after block review; it is never inferred from decay-window timing.")
            if not isinstance(frequency_source, str) or not frequency_source.strip():
                blockers.append("frequency_provenance is required for full analysis after block review.")
    else:
        frequency = parameters.get("frequency_mhz")
        frequency_source = parameters.get("frequency_provenance")
    threshold = parameters.get("minimum_total_counts", 0)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(float(threshold)) or threshold < 0:
        blockers.append("minimum_total_counts must be a finite nonnegative number.")
    if parameters.get("filter_enabled", False):
        size, repeat = parameters.get("filter_size", 3), parameters.get("filter_repeat", 1)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or size % 2 != 1:
            blockers.append("filter_size must be a positive odd integer.")
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            blockers.append("filter_repeat must be a positive integer.")
    calibration = parameters.get("reference_calibration")
    if calibration is None and parameters.get("reference_calibration.path"):
        calibration = {
            "path": parameters.get("reference_calibration.path"),
            "block_indices": [parameters.get("reference_calibration.block_index")],
            "known_lifetime_ns": parameters.get("reference_calibration.known_lifetime_ns"),
            "provenance": parameters.get("reference_calibration.provenance"),
            "instrument_settings_match_confirmed": parameters.get("reference_calibration.instrument_settings_match_confirmed", False),
        }
    if isinstance(calibration, dict) and "block_indices" not in calibration and isinstance(calibration.get("block_index"), int):
        calibration = calibration | {"block_indices": [calibration["block_index"]]}
    if isinstance(calibration, dict) and not calibration.get("path"):
        calibration = None
    if calibration is not None:
        if not isinstance(calibration, dict):
            blockers.append("reference_calibration must be a mapping for an actually supplied measured reference.")
        else:
            reference = calibration.get("path")
            reference_path = Path(reference).expanduser() if isinstance(reference, str) else None
            if reference_path is not None and not reference_path.is_absolute():
                reference_path = request.project_root / reference_path
            if reference_path is None or not reference_path.is_file() or reference_path.suffix.lower() != ".sdt":
                blockers.append("Reference calibration requested but its separate reference .sdt file is unavailable.")
            if reference_path is not None:
                reference_path = reference_path.resolve()
                calibration = calibration | {"path": str(reference_path)}
            if reference_path is not None and reference_path in inputs:
                blockers.append("Measured reference SDT must be separate from the target FLIM modality input folder.")
            if not isinstance(calibration.get("known_lifetime_ns"), (int, float)) or isinstance(calibration.get("known_lifetime_ns"), bool) or not math.isfinite(float(calibration.get("known_lifetime_ns", -1))) or calibration.get("known_lifetime_ns", -1) < 0:
                blockers.append("Measured reference calibration requires known_lifetime_ns >= 0.")
            reference_blocks = calibration.get("block_indices")
            if not isinstance(reference_blocks, list) or len(reference_blocks) != 1 or not isinstance(reference_blocks[0], int) or isinstance(reference_blocks[0], bool) or reference_blocks[0] < 0:
                blockers.append("Measured reference calibration requires one explicitly selected reference SDT block index.")
            if not isinstance(calibration.get("provenance"), str) or not calibration.get("provenance", "").strip():
                blockers.append("Measured reference calibration requires explicit provenance.")
            if calibration.get("instrument_settings_match_confirmed") is not True:
                blockers.append("Measured reference calibration requires explicit matching-settings confirmation.")
    period_tolerance = parameters.get("period_tolerance_bins", 1.0)
    if not isinstance(period_tolerance, (int, float)) or isinstance(period_tolerance, bool) or not math.isfinite(float(period_tolerance)) or period_tolerance < 0:
        blockers.append("period_tolerance_bins must be finite and nonnegative.")
    pairing = parameters.get("hs_flim_pairing_manifest")
    if pairing == "":
        pairing = None
    if isinstance(pairing, str):
        pairing_path = Path(pairing).expanduser()
        if not pairing_path.is_absolute():
            pairing_path = request.project_root / pairing_path
        pairing = str(pairing_path.resolve())
    if pairing is not None and (not isinstance(pairing, str) or not Path(pairing).is_file()):
        blockers.append("The supplied HS+FLIM pairing manifest does not exist.")
    if unsupported:
        warnings.append(f"Unsupported files ({len(unsupported)}) were found and will not be decoded.")
    warnings.append("This adapter is FLIM-only: an optional pairing manifest is recorded as provenance, never used to infer pairs or create cross-modal outputs.")
    review_attempt = parameters.get("review_attempt", 0)
    if not isinstance(review_attempt, int) or isinstance(review_attempt, bool) or review_attempt < 0:
        blockers.append("review_attempt must be a nonnegative integer.")
    resolved = {
        "analysis_scope": analysis_scope,
        "input_files": [str(path) for path in inputs],
        "acquisition_groups": [
            {
                "group_key": str(path),
                "label": path.name,
                "input_paths": [str(path)],
            }
            for path in inputs
        ],
        "input_stat": {str(path): _stat_signature(path) for path in inputs} if inputs else {},
        "unsupported_files": unsupported,
        "block_indices": selected_blocks,
        "raw_qc_all_blocks": raw_qc_all_blocks,
        "technical_qc_authorized": technical_qc_authorized,
        "blocks_review_confirmed": parameters.get("blocks_review_confirmed", False),
        "frequency_mhz": frequency,
        "frequency_provenance": frequency_source,
        "minimum_total_counts": threshold,
        "filter_enabled": bool(parameters.get("filter_enabled", False)),
        "filter_size": parameters.get("filter_size", 3),
        "filter_repeat": parameters.get("filter_repeat", 1),
        "reference_calibration_supplied": calibration is not None,
        "reference_calibration": calibration,
        "hs_flim_pairing_manifest": str(Path(pairing).expanduser().resolve()) if isinstance(pairing, str) and Path(pairing).expanduser().is_file() else None,
        "hs_flim_pairing_manifest_stat": _stat_signature(Path(pairing).expanduser().resolve()) if isinstance(pairing, str) and Path(pairing).expanduser().is_file() else None,
        "reference_calibration_stat": _stat_signature(Path(calibration["path"]).expanduser().resolve()) if isinstance(calibration, dict) and isinstance(calibration.get("path"), str) and Path(calibration["path"]).expanduser().is_file() else None,
        "review_attempt": review_attempt,
    }
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=resolved)


def execute_stage(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
    definitions = {stage.stage_id: stage for stage in describe().stages}
    if stage_id not in definitions:
        raise ValueError(f"Unknown FLIM stage: {stage_id}")
    for dependency in definitions[stage_id].depends_on:
        parent = upstream.get(dependency)
        if parent is None or parent.status not in {"SUCCEEDED", "REUSED", "SKIPPED"}:
            status = parent.status if parent is not None and parent.status in {"QC_PENDING", "BLOCKED", "FAILED"} else "BLOCKED"
            event = StageEvent(stage_id=stage_id, status=status, message=f"Dependency {dependency} is not complete")
            emit(event)
            return WorkflowResult(status=status, stages=[event], review_request=parent.review_request if parent else None)
    check = preflight(request)
    warnings = list(check.warnings)
    events: list[StageEvent] = []
    if not check.ready:
        blocked = StageEvent(stage_id="flim_preflight", status="BLOCKED", message="; ".join(check.blockers), parameters=check.resolved_parameters)
        emit(blocked)
        return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + check.blockers)
    try:
        flim, plots = _sources()
    except Exception as exc:
        message = f"Canonical FLIM source/runtime unavailable in worker: {type(exc).__name__}: {exc}"
        blocked = StageEvent(stage_id="flim_runtime", status="BLOCKED", message=message)
        emit(blocked)
        return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
    root = request.run_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    objects = root / "objects"
    tables = root / "tables"
    diagnostics = root / "diagnostics"
    objects.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    params = check.resolved_parameters
    for name in params["input_files"]:
        path = Path(name)
        if _stat_signature(path) != params["input_stat"][name]:
            message = f"Input changed after preflight; submit a fresh run: {name}"
            blocked = StageEvent(stage_id="flim_input_integrity", status="BLOCKED", message=message)
            emit(blocked)
            return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
    params["input_sha256"] = {name: _sha256(Path(name)) for name in params["input_files"]}
    if any(_stat_signature(Path(name)) != params["input_stat"][name] for name in params["input_files"]):
        message = "One or more SDT inputs changed while worker-side hashing was in progress."
        blocked = StageEvent(stage_id="flim_input_integrity", status="BLOCKED", message=message)
        emit(blocked)
        return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
    pairing_path = params["hs_flim_pairing_manifest"]
    if pairing_path is not None:
        if _stat_signature(Path(pairing_path)) != params["hs_flim_pairing_manifest_stat"]:
            message = "HS+FLIM pairing manifest changed after preflight."
            blocked = StageEvent(stage_id="flim_pairing_integrity", status="BLOCKED", message=message)
            emit(blocked)
            return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
        params["hs_flim_pairing_manifest_sha256"] = _sha256(Path(pairing_path))
        if _stat_signature(Path(pairing_path)) != params["hs_flim_pairing_manifest_stat"]:
            message = "HS+FLIM pairing manifest changed while worker-side hashing was in progress."
            blocked = StageEvent(stage_id="flim_pairing_integrity", status="BLOCKED", message=message)
            emit(blocked)
            return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
    pairing_evidence = {"path": pairing_path, "sha256": params["hs_flim_pairing_manifest_sha256"]} if pairing_path is not None else None
    calibration_request = params.get("reference_calibration")
    if calibration_request is not None:
        reference_path = Path(calibration_request["path"]).expanduser().resolve()
        if _stat_signature(reference_path) != params["reference_calibration_stat"]:
            message = "Measured reference SDT changed after preflight."
            blocked = StageEvent(stage_id="flim_reference_integrity", status="BLOCKED", message=message)
            emit(blocked)
            return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
        params["reference_calibration_sha256"] = _sha256(reference_path)
        if _stat_signature(reference_path) != params["reference_calibration_stat"]:
            message = "Measured reference SDT changed while worker-side hashing was in progress."
            blocked = StageEvent(stage_id="flim_reference_integrity", status="BLOCKED", message=message)
            emit(blocked)
            return WorkflowResult(status="BLOCKED", stages=[blocked], warnings=warnings + [message])
    calibration = params.get("reference_calibration")
    if calibration is not None:
        calibration_path = str(Path(calibration["path"]).expanduser().resolve())
        params["reference_calibration"] = calibration | {"path": calibration_path, "source_sha256": params["reference_calibration_sha256"]}
    tools = _versions()
    artifacts: list[str] = []

    def run_stage(stage_id: str, title: str, operation: Any, stage_params: dict[str, Any]) -> Any:
        running = StageEvent(stage_id=stage_id, status="RUNNING", message=title, tool_versions=tools, parameters=stage_params)
        emit(running)
        value = operation()
        receipt = Path(value["receipt_path"]).resolve()
        outputs = [str(receipt)]
        for item in value.get("detectors", {}).values():
            object_path = item.get("object_path")
            if object_path:
                outputs.append(str(Path(object_path).resolve()))
        for plot_path in value.get("pdf_paths", []):
            outputs.append(str(Path(plot_path).resolve()))
        if value.get("receipt_path"):
            outputs.append(str(Path(value["receipt_path"]).resolve()))
        summary = value.get("summary")
        if summary is not None and hasattr(summary, "to_csv"):
            summary_path = tables / f"{stage_id}_summary.csv"
            summary.to_csv(summary_path, index=False)
            outputs.append(str(summary_path))
        event = StageEvent(stage_id=stage_id, status="SUCCEEDED", message=title, tool_versions=tools, parameters=stage_params, artifacts=outputs)
        emit(event)
        events.append(event)
        artifacts.extend(outputs)
        return value
    def record_plots(stage_id: str, title: str, result: dict[str, Any]) -> None:
        plot_artifacts = [str(Path(path).resolve()) for path in result.get("pdf_paths", [])]
        if result.get("receipt_path"):
            plot_artifacts.append(str(Path(result["receipt_path"]).resolve()))
        event = StageEvent(stage_id=stage_id, status="SUCCEEDED", message=title, tool_versions=tools, parameters={"artifact_kind": "diagnostic_plot"}, artifacts=plot_artifacts)
        emit(event)
        events.append(event)
        artifacts.extend(plot_artifacts)
    def load(stage: str, key: str = "result") -> dict[str, Any]:
        import pandas as pd
        state_path = Path(upstream[stage].state[key])
        data = json.loads(state_path.read_text())
        if data.get("summary_path"):
            data["summary"] = pd.read_csv(data.pop("summary_path"))
        if "records" in data:
            data["manifest"] = pd.DataFrame(data["records"])
        return data

    def save(value: dict[str, Any], name: str) -> str:
        data = {key: val for key, val in value.items() if key not in {"summary", "manifest"}}
        if "summary" in value:
            summary_path = tables / f"{stage_id}_{name}_state_summary.csv"
            value["summary"].to_csv(summary_path, index=False)
            data["summary_path"] = str(summary_path)
        path = objects / f"{stage_id}_{name}_state.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return str(path)

    try:
        if stage_id == "flim_inventory":
            prior_inventory_path = root / "input_inventory.json"
            inventory = run_stage("flim_inventory", "Inspecting complete SDT block inventory", lambda: flim.inspect_flim_inputs(params["input_files"], cache_dir=objects / "inventory", detector_indices=None), {"selected_blocks": params["block_indices"], "input_sha256": params["input_sha256"]})
            observed_hashes = {row["path"]: row["source_sha256"] for row in inventory["records"]}
            if observed_hashes != params["input_sha256"]:
                raise ValueError("Canonical inventory source hashes differ from worker-side hashes; inputs changed or source identity mismatched.")
            if prior_inventory_path.is_file():
                prior_inventory = json.loads(prior_inventory_path.read_text())
                prior_hashes = {row["path"]: row["source_sha256"] for row in prior_inventory["records"]}
                if prior_hashes != observed_hashes:
                    raise ValueError("Source SDT hashes differ from the previously reviewed inventory; block review is invalidated.")
                if prior_inventory.get("pairing_manifest") != pairing_evidence:
                    raise ValueError("HS+FLIM pairing manifest changed since inventory review.")
                if prior_inventory.get("inventory_receipt_sha256") != _sha256(Path(inventory["receipt_path"])):
                    raise ValueError("Canonical inventory receipt changed since block review.")
            else:
                prior_inventory_path.write_text(json.dumps({"records": inventory["records"], "unsupported_files": params["unsupported_files"], "source_sha256": observed_hashes, "inventory_receipt_path": str(Path(inventory["receipt_path"]).resolve()), "inventory_receipt_sha256": _sha256(Path(inventory["receipt_path"])), "pairing_manifest": pairing_evidence}, indent=2, default=str))
                artifacts.append(str(prior_inventory_path))
            if not params["raw_qc_all_blocks"] and (params["block_indices"] is None or params.get("blocks_review_confirmed") is not True):
                by_block: dict[int, list[dict[str, Any]]] = {}
                for row in inventory["records"]:
                    by_block.setdefault(int(row["block_index"]), []).append(row)
                choices = []
                for block_index, rows in sorted(by_block.items()):
                    choices.append({"value": block_index, "label": f"Block {block_index} ({len(rows)} acquisition record(s))", "records": [{"path": row["path"], "source_sha256": row["source_sha256"], "detector_index": row.get("detector_index"), "selection_kind": row.get("selection_kind"), "shape": row.get("shape"), "time_step_seconds": row.get("time_step_seconds"), "window_seconds": row.get("window_seconds"), "acquisition_frequency_mhz": row.get("frequency_mhz")} for row in rows]})
                review_request_path = root / "review_request.json"
                available_blocks = [choice["value"] for choice in choices]
                canonical_receipt_sha256 = _sha256(Path(inventory["receipt_path"]))
                review_request = {
                    "status": "QC_PENDING",
                    "run_id": str(request.run_id),
                    "review_attempt": params.get("review_attempt"),
                    "dynamic_choices_key": "flim_block_indices",
                    "block_indices": available_blocks,
                    "choices": {"block_indices": available_blocks},
                    "block_details": choices,
                    "requested_block_indices": params["block_indices"],
                    "inventory_artifact": str(prior_inventory_path),
                    "inventory_receipt_path": str(Path(inventory["receipt_path"]).resolve()),
                    "flim_inventory_receipt_sha256": canonical_receipt_sha256,
                    "source_sha256": observed_hashes,
                    "pairing_manifest": pairing_evidence,
                    "requires_explicit_selection": True,
                    "requires_reviewer_confirmation": True,
                    "selection_scope": "selected block indices are applied to every listed SDT source",
                }
                pending = StageEvent(stage_id="flim_block_review", status="WARNING", message="QC inventory complete; awaiting explicit user selection and review confirmation before scientific analysis.", parameters={"dynamic_choices_key": "flim_block_indices", "choice_count": len(choices)}, artifacts=[str(prior_inventory_path), str(review_request_path), str(Path(inventory["receipt_path"]).resolve())])
                emit(pending)
                events.append(pending)
                return WorkflowResult(status="QC_PENDING", stages=events, artifacts=sorted(set(artifacts)), warnings=warnings + ["No scientific analysis ran; explicit block selection and review are required."], review_request=review_request)
            if params["raw_qc_all_blocks"]:
                selected_blocks = sorted({int(row["block_index"]) for row in inventory["records"]})
                if not selected_blocks:
                    raise ValueError("Canonical SDT inventory contains no recorded blocks for technical QC.")
                inventory_receipt_path = Path(inventory["receipt_path"]).resolve()
                policy_path = root / "selection_policy.json"
                policy = {
                    "schema": "flim_raw_qc_selection_policy_v1",
                    "policy": "all recorded blocks; no detector identity assumed",
                    "analysis_scope": "raw_qc",
                    "technical_qc_authorized": True,
                    "block_indices": selected_blocks,
                    "source_sha256": observed_hashes,
                    "inventory_receipt_path": str(inventory_receipt_path),
                    "inventory_receipt_sha256": _sha256(inventory_receipt_path),
                    "selection_semantics": "select every unique native block index recorded by canonical inventory across the exact input SDT files",
                    "scope_limit": "raw count/decay technical QC only; no phasor, lifetime, calibration, or biological detector interpretation",
                }
                policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
                artifacts.append(str(policy_path))
                policy_event = StageEvent(stage_id="flim_block_selection_policy", status="SUCCEEDED", message="Authorized all-recorded-block technical QC policy bound to source hashes and canonical inventory receipt", tool_versions=tools, parameters={"policy": policy["policy"], "block_indices": selected_blocks}, artifacts=[str(policy_path)])
                emit(policy_event)
                events.append(policy_event)
            else:
                selected_blocks = params["block_indices"]
                review_decision_path = root / "review_decision.json"
                _validate_review_decision(review_decision_path, inventory, selected_blocks=selected_blocks, run_id=str(request.run_id), review_attempt=params.get("review_attempt"))
                artifacts.append(str(review_decision_path))
            inventory = flim.select_flim_blocks(inventory, selected_blocks)
            state = {"result": save(inventory, "inventory")}
        else:
            calibration = params.get("reference_calibration")
            optional_reference = stage_id in {"flim_reference_inventory", "flim_reference_raw_qc", "flim_reference_phasor", "flim_reference_calibration"}
            if (params["analysis_scope"] == "raw_qc" and stage_id != "flim_raw_qc") or (optional_reference and calibration is None) or (stage_id == "flim_filter" and not params["filter_enabled"]):
                event = StageEvent(stage_id=stage_id, status="SKIPPED", message="Stage not requested for this analysis")
                emit(event)
                return WorkflowResult(status="SKIPPED", stages=[event])
            from spatial_omics_toolkit.application.plotting import ensure_arial
            state: dict[str, Any] = {}
            if stage_id == "flim_raw_qc":
                inventory = load("flim_inventory")
                value = run_stage(stage_id, "Summarizing raw counts and decays", lambda: flim.summarize_flim_qc(inventory, object_dir=objects / "raw_qc", table_dir=tables), {})
                ensure_arial()
                record_plots("flim_raw_qc_plots", "Raw count and decay diagnostics", plots.plot_flim_qc(value, diagnostics / "raw_qc"))
            elif stage_id == "flim_phasor":
                inventory = load("flim_inventory")
                value = run_stage(stage_id, "Computing physical-time phasors", lambda: flim.compute_flim_phasors(inventory, object_dir=objects / "phasors", table_dir=tables, frequency_mhz=float(params["frequency_mhz"]), frequency_provenance=str(params["frequency_provenance"])), {"frequency_mhz": params["frequency_mhz"], "frequency_provenance": params["frequency_provenance"]})
                ensure_arial()
                record_plots("flim_phasor_plots", "Physical-time phasor diagnostics", plots.plot_flim_phasors(value, diagnostics / "phasors"))
            elif stage_id == "flim_count_mask":
                parent = load("flim_phasor")
                value = run_stage(stage_id, "Applying declared total-count support mask", lambda: flim.mask_flim_phasors(parent, object_dir=objects / "count_mask", table_dir=tables, minimum_total_counts=float(params["minimum_total_counts"])), {"minimum_total_counts": params["minimum_total_counts"]})
                ensure_arial()
                record_plots("flim_count_mask_plots", "Count-mask diagnostics", plots.plot_flim_mask(parent, value, diagnostics / "count_mask"))
            elif stage_id == "flim_filter":
                parent = load("flim_count_mask")
                value = run_stage(stage_id, "Applying canonical median filter", lambda: flim.filter_flim_phasors(parent, object_dir=objects / "filter", table_dir=tables, size=int(params["filter_size"]), repeat=int(params["filter_repeat"])), {"size": params["filter_size"], "repeat": params["filter_repeat"]})
                ensure_arial()
                record_plots("flim_filter_plots", "Median-filter diagnostics", plots.plot_flim_filter(parent, value, diagnostics / "filter"))
            elif stage_id == "flim_reference_inventory":
                value = run_stage(stage_id, "Inspecting measured-reference SDT", lambda: flim.inspect_flim_inputs([calibration["path"]], cache_dir=objects / "reference_inventory", detector_indices=calibration.get("block_indices")), {"reference_path": calibration["path"], "reference_sha256": params["reference_calibration_sha256"]})
                reference_hashes = {row["path"]: row["source_sha256"] for row in value["records"]}
                if reference_hashes.get(calibration["path"]) != params["reference_calibration_sha256"]:
                    raise ValueError("Measured reference source hash differs from worker-side integrity hash.")
            elif stage_id == "flim_reference_raw_qc":
                reference_inventory = load("flim_reference_inventory")
                value = run_stage(stage_id, "Summarizing measured-reference raw counts and decays", lambda: flim.summarize_flim_qc(reference_inventory, object_dir=objects / "reference_qc", table_dir=tables), {"reference_path": calibration["path"], "reference_sha256": params["reference_calibration_sha256"]})
            elif stage_id == "flim_reference_phasor":
                reference_inventory = load("flim_reference_inventory")
                value = run_stage(stage_id, "Computing measured-reference phasors", lambda: flim.compute_flim_phasors(reference_inventory, object_dir=objects / "reference_phasors", table_dir=tables, frequency_mhz=float(params["frequency_mhz"]), frequency_provenance=str(params["frequency_provenance"])), {"frequency_mhz": params["frequency_mhz"]})
            elif stage_id == "flim_reference_calibration":
                parent = load("flim_filter" if upstream["flim_filter"].status != "SKIPPED" else "flim_count_mask")
                reference = load("flim_reference_phasor")
                keys = [key for key, metadata in reference["detectors"].items() if metadata.get("block_index") == calibration["block_indices"][0]]
                if len(keys) != 1:
                    raise ValueError(f"Selected measured reference block resolves to {len(keys)} detector records; explicit unambiguous reference block is required.")
                value = run_stage(stage_id, "Calibrating against the supplied measured reference", lambda: flim.calibrate_flim_phasors(parent, reference, object_dir=objects / "calibration", table_dir=tables, reference_detector_key=keys[0], known_lifetime_ns=float(calibration["known_lifetime_ns"]), calibration_provenance=str(calibration["provenance"]), detector_keys=list(parent["detectors"]), instrument_settings_match_confirmed=True), {"reference_path": calibration["path"], "reference_sha256": params["reference_calibration_sha256"], "known_lifetime_ns": calibration["known_lifetime_ns"], "provenance": calibration["provenance"], "instrument_settings_match_confirmed": True})
            elif stage_id == "flim_lifetimes":
                source_stage = "flim_reference_calibration" if upstream["flim_reference_calibration"].status != "SKIPPED" else ("flim_filter" if upstream["flim_filter"].status != "SKIPPED" else "flim_count_mask")
                parent = load(source_stage)
                value = run_stage(stage_id, "Computing apparent phase and modulation lifetimes", lambda: flim.compute_flim_lifetimes(parent, object_dir=objects / "lifetimes", table_dir=tables, period_tolerance_bins=float(params.get("period_tolerance_bins", 1.0)), allow_period_mismatch=bool(params.get("allow_period_mismatch", False))), {"period_tolerance_bins": params.get("period_tolerance_bins", 1.0), "allow_period_mismatch": bool(params.get("allow_period_mismatch", False)), "interpretation": "apparent; calibrated only with supplied measured reference" if calibration else "apparent, uncalibrated"})
                ensure_arial()
                record_plots("flim_lifetime_plots", "Apparent lifetime diagnostics", plots.plot_flim_lifetimes(value, diagnostics / "lifetimes"))
                exported = flim.export_flim_results(value, destination=root / "flim_result_manifest.csv")
                artifacts.extend([str(exported["manifest_path"]), str(exported["receipt_path"])])
            state["result"] = save(value, "result")
            if stage_id == "flim_lifetimes" or (stage_id == "flim_raw_qc" and params["analysis_scope"] == "raw_qc"):
                receipt = root / "adapter_receipt.json"
                receipt.write_text(json.dumps({"workflow_id": WORKFLOW_ID, "run_id": str(request.run_id), "analysis_scope": params["analysis_scope"], "source_sha256": {name: _sha256(_source_dir() / name) for name in ("clinical_roi_flim.py", "clinical_roi_flim_plots.py")}, "parameters": params, "tool_versions": tools, "pairing_manifest_recorded_only": params["hs_flim_pairing_manifest"], "artifacts": sorted(set(artifacts))}, indent=2, default=str))
                artifacts.append(str(receipt))
        return WorkflowResult(status="SUCCEEDED", stages=events, artifacts=sorted(set(artifacts)), warnings=warnings, state=state)
    except Exception as exc:
        failed = StageEvent(stage_id=stage_id, status="FAILED", message=f"{type(exc).__name__}: {exc}", tool_versions=tools, parameters=params)
        emit(failed)
        events.append(failed)
        return WorkflowResult(status="FAILED", stages=events, artifacts=sorted(set(artifacts)), warnings=warnings + [failed.message])
