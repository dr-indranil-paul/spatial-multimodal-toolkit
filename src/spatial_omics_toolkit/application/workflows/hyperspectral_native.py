from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
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

WORKFLOW_ID = "hs.native_spectral_qc"
_CANONICAL_SCRIPTS = Path(
    "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/"
    "01_exploratory_analysis/02_scripts"
)


def _source_dir() -> Path:
    configured = os.environ.get("SPATIAL_TOOLKIT_HS_SOURCE_DIR")
    return Path(configured).expanduser().resolve() if configured else _CANONICAL_SCRIPTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _reader() -> Any:
    directory = _source_dir()
    source = directory / "clinical_roi_hs_flim.py"
    missing = [str(path) for path in (source, directory / "clinical_roi_hyperspectral.py") if not path.is_file()]
    if missing:
        raise FileNotFoundError("Canonical native HS source is unavailable: " + ", ".join(missing))
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return _load_module("clinical_roi_hs_flim", source)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "canonical_workflow": "clinical_roi_hs_flim.py:read_native_hs_moments (source-bound)",
        "clinical_roi_hs_flim.py": _sha256(_source_dir() / "clinical_roi_hs_flim.py"),
        "clinical_roi_hyperspectral.py": _sha256(_source_dir() / "clinical_roi_hyperspectral.py"),
    }
    for package in ("numpy", "pandas", "matplotlib", "aicspylibczi", "phasorpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def describe() -> WorkflowDescriptor:
    reader = ToolReference(
        name="clinical_roi_hs_flim.py",
        version="SHA-256 recorded in each run receipt",
        role="Check CZI layout and spectral bands, and read two-harmonic spectral moments",
    )
    metadata = ToolReference(
        name="clinical_roi_hyperspectral.py",
        version="SHA-256 recorded in each run receipt",
        role="Validate raw hyperspectral metadata and spectral bands",
    )
    return WorkflowDescriptor(
        workflow_id=WORKFLOW_ID,
        modality="HS",
        title="Hyperspectral technical QC",
        description="Read one selected raw spectral CZI and produce intensity and spectral-phasor diagnostics. These outputs are for technical quality control only, not biological inference, unmixing, or group comparisons.",
        stages=[
            StageDefinition(stage_id="hs_native_moments", title="Read and audit native spectral moments", purpose="In one canonical read, validate the source layout and spectral bands, record metadata and compute five moment maps on the original grid.", tools=[reader, metadata], implementations=[FunctionReference(source_file="external:clinical_roi_hs_flim.py", function_name="read_native_hs_moments", role="Validate and read the native spectral moments"), FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="_layout", role="Validate the original channel layout"), FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="inspect_raw_metadata", role="Inspect raw metadata"), FunctionReference(source_file="external:clinical_roi_hyperspectral.py", function_name="_spectral_channel_grid", role="Validate the spectral channel grid")], depends_on=[], condition="One selected native CZI passes preflight", inputs=["Selected native CZI", "Working-memory limit"], outputs=["Source metadata JSON", "Spectral-band audit CSV", "Spectral moment maps NPZ"]),
            StageDefinition(stage_id="hs_native_qc_plots", title="Plot technical quality checks", purpose="Plot intensity and first- and second-harmonic spectral phasor densities without biological interpretation.", tools=[ToolReference(name="matplotlib", version="Version recorded in each run receipt", role="Render technical QC PDFs")], implementations=[FunctionReference(source_file="spatial_omics_toolkit/application/workflows/hyperspectral_native.py", function_name="execute_stage", role="Create quality-control figures from saved spectral moments")], depends_on=["hs_native_moments"], condition="Native spectral moments succeeded", inputs=["Spectral moment maps NPZ", "Source metadata JSON"], outputs=["Intensity QC PDF", "Spectral phasor density QC PDF", "Adapter receipt JSON"]),
        ],
        input_description=(
            "- A modality folder searched recursively for native spectral `.czi` files. One file is processed per run: a sole CZI is selected automatically; with multiple files, supply the exact `source_czi` path. Other files are reported, not decoded.\n"
            "- The canonical `clinical_roi_hs_flim.py` and `clinical_roi_hyperspectral.py` scripts, and a positive working-memory limit (default 4 GiB).\n"
            "- The selected CZI must satisfy the native reader's image-layout, uint16 channel-plane, memory, and spectral-band checks, including at least five uniformly spaced, contiguous, equal-width bands.\n"
            "- The shared sample CSV and confirmed acquisition mapping are still required; the mapping may be `UNMAPPED`. No clinical selection ledger is required."
        ),
        output_description=(
            "- On successful native spectral QC: source metadata JSON, spectral-band audit CSV, and a compressed file containing intensity and first- and second-harmonic spectral moment maps.\n"
            "- Intensity and spectral-phasor density QC PDFs, plus processing and provenance JSON.\n"
            "- These describe recorded spectral patterns for technical QC, not molecular identity, unmixing, calibrated lifetime, or biological comparisons. Unsupported data, changed inputs, memory limits, or processing errors can stop the run and leave partial outputs."
        ),
        parameter_fields=[
            ParameterField(name="source_czi", label="Exact source CZI (when multiple)", kind="path", section="Input", help_text="Optional when the folder contains exactly one .czi; otherwise select one exact .czi in the modality folder."),
            ParameterField(name="max_working_bytes", label="Maximum native working memory", kind="int", default=4 * 1024**3, min_value=1, unit="bytes", section="Resources", help_text="Hard bound passed to the canonical native reader; jobs exceeding this memory estimate fail rather than silently expanding the limit."),
        ],
        limitations=[
            "One raw CZI is processed per run; multiple sources require separate runs with an explicit selection for each.",
            "Spectral phasors summarize the recorded order of contiguous, equal-width bands; they are not calibrated lifetimes, unmixing results, or molecular identities.",
            "Plots and outputs are for technical quality control and exploration; they do not support biological or cohort-level claims.",
            "The original-data reader enforces its band, layout, and memory requirements; this workflow does not resample or change the raw input.",
        ],
    )


def _scan(request: RunRequest) -> tuple[list[Path], list[str]]:
    root = request.modality_dir.expanduser().resolve()
    if not root.is_dir():
        return [], [f"Modality folder does not exist: {root}"]
    supported: list[Path] = []
    unsupported: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.suffix.lower() == ".czi":
                supported.append(path.resolve())
            else:
                unsupported.append(str(path.resolve()))
    return supported, unsupported


def _stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    warnings: list[str] = []
    candidates, unsupported = _scan(request)
    selector = request.parameters.get("source_czi")
    selected: Path | None = None
    if selector:
        candidate_selector = Path(str(selector)).expanduser()
        if not candidate_selector.is_absolute():
            candidate_selector = request.modality_dir.expanduser().resolve() / candidate_selector
        try:
            selected = candidate_selector.resolve(strict=True)
        except (OSError, RuntimeError):
            blockers.append("source_czi does not resolve to an existing file.")
        if selected is not None and selected.suffix.lower() != ".czi":
            blockers.append("source_czi must be a native .czi file.")
        if selected is not None and selected not in candidates:
            blockers.append("source_czi must be one of the .czi files inventoried under the modality folder.")
    elif len(candidates) == 1:
        selected = candidates[0]
    elif not candidates:
        blockers.append("No native .czi input files were found in the modality folder.")
    else:
        blockers.append("Multiple native .czi files were found; provide the exact source_czi selector.")
    source_directory = _source_dir()
    required_sources = ("clinical_roi_hs_flim.py", "clinical_roi_hyperspectral.py")
    missing = [name for name in required_sources if not (source_directory / name).is_file()]
    if missing:
        blockers.append("Canonical native HS source is unavailable: " + ", ".join(str(source_directory / name) for name in missing))
    memory_limit = request.parameters.get("max_working_bytes", 4 * 1024**3)
    if not isinstance(memory_limit, int) or isinstance(memory_limit, bool) or memory_limit < 1:
        blockers.append("max_working_bytes must be a positive integer.")
    if unsupported:
        warnings.append(f"Unsupported files ({len(unsupported)}) were found and will not be decoded.")
    if selected is not None and selected.is_file():
        stat = _stat_signature(selected)
        resolved = {
            "input_files": [str(selected)],
            "inventory_input_paths": [str(selected)],
            "acquisition_groups": [{
                "group_key": str(selected),
                "label": selected.name,
                "input_paths": [str(selected)],
                "group_type": "native_technical_source_file",
                "metadata": {"format": "CZI", "technical_file_grouping_only": True},
            }],
            "input_stat": {str(selected): stat},
            "unsupported_files": unsupported,
            "max_working_bytes": memory_limit,
        }
    else:
        resolved = {"input_files": [], "inventory_input_paths": [], "acquisition_groups": [], "unsupported_files": unsupported, "max_working_bytes": memory_limit}
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=resolved)


def execute_stage(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
    if stage_id not in {"hs_native_moments", "hs_native_qc_plots"}:
        raise ValueError(f"Unknown native hyperspectral stage: {stage_id}")
    warnings: list[str] = []
    params: dict[str, Any] = {}
    root = request.run_dir.expanduser().resolve()
    objects, tables, diagnostics = root / "objects", root / "tables", root / "diagnostics"
    if stage_id == "hs_native_qc_plots":
        dependency = upstream.get("hs_native_moments")
        if dependency is None or dependency.status not in {"SUCCEEDED", "REUSED"}:
            reason = "Native spectral moments have not succeeded."
            event = StageEvent(stage_id=stage_id, status="BLOCKED", message=reason)
            emit(event)
            return WorkflowResult(status="BLOCKED", stages=[event], warnings=[reason])
        params = dependency.state.get("parameters", {})
        warnings = list(dependency.warnings)
    else:
        check = preflight(request)
        warnings = list(check.warnings)
        params = check.resolved_parameters
        if not check.ready:
            event = StageEvent(stage_id=stage_id, status="BLOCKED", message="; ".join(check.blockers), parameters=params)
            emit(event)
            return WorkflowResult(status="BLOCKED", stages=[event], warnings=warnings + check.blockers)
    running = StageEvent(stage_id=stage_id, status="RUNNING", message="Running native spectral technical QC stage", parameters=params)
    emit(running)
    try:
        import numpy as np

        if stage_id == "hs_native_moments":
            for directory in (root, objects, tables):
                directory.mkdir(parents=True, exist_ok=True)
            source_path = Path(params["input_files"][0])
            source_dir = _source_dir()
            if _stat_signature(source_path) != params["input_stat"][str(source_path)]:
                raise ValueError("Native CZI changed after preflight; submit a fresh run.")
            source_sha256 = _sha256(source_path)
            if _stat_signature(source_path) != params["input_stat"][str(source_path)]:
                raise ValueError("Native CZI changed while source hashing was in progress.")
            params["input_sha256"] = {str(source_path): source_sha256}
            tools = _versions()
            moments, bands, native_metadata = _reader().read_native_hs_moments(source_path, max_working_bytes=params["max_working_bytes"])
            metadata_path = root / "native_source_metadata.json"
            metadata = {
                "workflow_id": WORKFLOW_ID,
                "source_path": str(source_path),
                "source_sha256": source_sha256,
                "source_stat": params["input_stat"][str(source_path)],
                "native_metadata": native_metadata,
                "moment_names": ["intensity_sum", "h1_real_weighted_sum", "h1_imag_weighted_sum", "h2_real_weighted_sum", "h2_imag_weighted_sum"],
                "interpretation": "Technical QC only. No biological identity, cohort aggregation, unmixing, calibrated lifetime, or inferential result.",
                "canonical_source_sha256": {name: _sha256(source_dir / name) for name in ("clinical_roi_hs_flim.py", "clinical_roi_hyperspectral.py")},
                "tool_versions": tools,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
            bands_path = tables / "native_spectral_band_audit.csv"
            bands.to_csv(bands_path, index=False)
            moments_path = objects / "native_spectral_moments.npz"
            np.savez_compressed(moments_path, moments=moments)
            artifacts = [str(metadata_path), str(bands_path), str(moments_path)]
            event = StageEvent(stage_id=stage_id, status="SUCCEEDED", message="Native source metadata, band audit, and canonical spectral moment maps saved", tool_versions=tools, parameters=params, artifacts=artifacts)
            emit(event)
            return WorkflowResult(status="SUCCEEDED", stages=[event], artifacts=artifacts, warnings=warnings, state={"parameters": params, "metadata_path": str(metadata_path), "moments_path": str(moments_path)})

        diagnostics.mkdir(parents=True, exist_ok=True)
        metadata = json.loads(Path(dependency.state["metadata_path"]).read_text())
        with np.load(Path(dependency.state["moments_path"]), allow_pickle=False) as saved:
            moments = saved["moments"]
        tools = metadata["tool_versions"]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from spatial_omics_toolkit.application.plotting import ensure_arial
        ensure_arial()
        plt.rcParams.update({"font.family": "Arial", "font.size": 5, "font.weight": "normal", "axes.labelsize": 5, "axes.labelweight": "normal", "axes.titlesize": 5, "axes.linewidth": 0.5, "xtick.labelsize": 5, "ytick.labelsize": 5, "lines.linewidth": 0.5, "grid.linewidth": 0.5, "grid.color": "#E5E7EB", "grid.linestyle": "-", "pdf.fonttype": 42, "ps.fonttype": 42})
        total = moments[0]
        intensity_pdf = diagnostics / "native_intensity_qc.pdf"
        fig, ax = plt.subplots(figsize=(90 / 25.4, 72 / 25.4))
        image = ax.imshow(np.log1p(total), cmap="magma", interpolation="nearest", alpha=0.92, zorder=2)
        ax.set_xlabel("Native X pixel")
        ax.set_ylabel("Native Y pixel")
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both", color="#E5E7EB", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=5, length=2, pad=1, width=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("log(1 + native spectral intensity sum)", fontsize=5)
        colorbar.ax.tick_params(labelsize=5, width=0.5, length=2)
        fig.savefig(intensity_pdf)
        plt.close(fig)
        phasor_pdf = diagnostics / "native_spectral_phasor_density_qc.pdf"
        fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 72 / 25.4))
        for harmonic, ax in enumerate(axes, start=1):
            real = np.divide(moments[2 * harmonic - 1], total, out=np.zeros_like(total), where=total > 0)
            imag = np.divide(moments[2 * harmonic], total, out=np.zeros_like(total), where=total > 0)
            positive = total > 0
            density, _, _ = np.histogram2d(real[positive], imag[positive], bins=160, range=[[-1.05, 1.05], [-1.05, 1.05]])
            masked = np.ma.masked_less_equal(density.T, 0)
            ax.pcolormesh(np.linspace(-1.05, 1.05, 161), np.linspace(-1.05, 1.05, 161), masked, norm=LogNorm(vmin=1, vmax=max(2, float(density.max()))), shading="flat", cmap="viridis", rasterized=True)
            ax.set_xlabel(f"H{harmonic} real")
            ax.set_ylabel(f"H{harmonic} imaginary")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_axisbelow(True)
            ax.grid(True, which="major", axis="both", color="#E5E7EB", linewidth=0.5)
            ax.tick_params(width=0.5, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        fig.savefig(phasor_pdf)
        plt.close(fig)
        artifacts = [str(intensity_pdf), str(phasor_pdf)]
        receipt_path = root / "adapter_receipt.json"
        metadata["artifacts"] = sorted(set(dependency.artifacts + artifacts))
        metadata["parameters"] = params
        receipt_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
        artifacts.append(str(receipt_path))
        event = StageEvent(stage_id=stage_id, status="SUCCEEDED", message="Technical intensity and H1/H2 phasor-density PDFs rendered from source-bound native moments", tool_versions=tools, parameters={"plot_interpretation": "technical QC only", "density_bins_per_axis": 160}, artifacts=artifacts)
        emit(event)
        return WorkflowResult(status="SUCCEEDED", stages=[event], artifacts=artifacts, warnings=warnings)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        failed = StageEvent(stage_id=stage_id, status="FAILED", message=message, parameters=params)
        emit(failed)
        return WorkflowResult(status="FAILED", stages=[failed], warnings=warnings + [message])
