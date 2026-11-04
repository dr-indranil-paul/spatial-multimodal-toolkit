from __future__ import annotations

import importlib.metadata
import json
import os
import warnings as python_warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from spatial_omics_toolkit.application.contracts import (
    FunctionReference,
    PreflightResult,
    ParameterField,
    RunRequest,
    StageDefinition,
    StageEvent,
    ToolReference,
    WorkflowDescriptor,
    WorkflowEmitter,
    WorkflowResult,
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _discover_bundles(root: Path, bin_size: int | None) -> tuple[list[dict[str, Any]], list[str]]:
    bundles: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not root.is_dir():
        return bundles, blockers
    candidates = {root}
    visited: set[Path] = set()
    for current, directories, filenames in os.walk(root, followlinks=True):
        current_path = Path(current)
        resolved_path = current_path.resolve()
        if resolved_path in visited:
            directories[:] = []
            continue
        visited.add(resolved_path)
        if "outs" in directories:
            directories[:] = ["outs"]
        if current_path.name == "outs":
            candidates.add(current_path)
            directories[:] = []
            continue
        file_names = set(filenames)
        directory_names = set(directories)
        positions_dir = current_path / "spatial"
        has_positions = any(
            (positions_dir / name).is_file()
            for name in (
                "tissue_positions.csv",
                "tissue_positions_list.csv",
                "tissue_positions.parquet",
            )
        )
        has_matrix = (
            "filtered_feature_bc_matrix.h5" in file_names
            or "filtered_feature_bc_matrix" in directory_names
        )
        has_binned_outputs = "binned_outputs" in directory_names
        if has_positions or has_matrix or has_binned_outputs:
            candidates.add(current_path)
            directories[:] = []
    for candidate in sorted(candidates):
        relative_parts = candidate.relative_to(root).parts
        inside_binned_output = "binned_outputs" in relative_parts
        positions_dir = candidate / "spatial"
        classic_positions = any(
            (positions_dir / name).is_file()
            for name in ("tissue_positions.csv", "tissue_positions_list.csv")
        )
        classic_matrix = (candidate / "filtered_feature_bc_matrix.h5").is_file()
        if not inside_binned_output and (classic_positions or classic_matrix) and not (
            candidate / "binned_outputs"
        ).is_dir():
            if classic_positions and classic_matrix:
                bundles.append(
                    {
                        "bundle_key": candidate.relative_to(root).as_posix() or ".",
                        "bundle_root": str(candidate),
                        "platform": "visium",
                        "bin_size": None,
                    }
                )
            else:
                blockers.append(
                    f"Incomplete Visium bundle at {candidate}: filtered matrix and spatial tissue positions are both required."
                )
        binned_root = candidate / "binned_outputs"
        if binned_root.is_dir():
            if bin_size is None:
                blockers.append(f"Visium HD bundle at {candidate} requires explicit parameters.bin_size (2, 8, or 16).")
                continue
            selected = binned_root / f"square_{bin_size:03d}um"
            hd_positions = selected / "spatial" / "tissue_positions.parquet"
            hd_scale_factors = selected / "spatial" / "scalefactors_json.json"
            hd_matrix = (selected / "filtered_feature_bc_matrix.h5").is_file()
            if hd_positions.is_file() and hd_matrix and hd_scale_factors.is_file():
                bundles.append({"bundle_key": candidate.relative_to(root).as_posix() or ".", "bundle_root": str(candidate), "platform": "visium_hd", "bin_size": bin_size})
            else:
                blockers.append(
                    f"Incomplete Visium HD bundle at {candidate}: selected square_{bin_size:03d}um requires "
                    "filtered_feature_bc_matrix.h5, tissue_positions.parquet, and scalefactors_json.json."
                )
    return bundles, blockers

def _available_hd_bin_sizes(root: Path) -> list[int]:
    resolution_sets: list[set[int]] = []
    visited: set[Path] = set()
    for current, directories, filenames in os.walk(root, followlinks=True):
        current_path = Path(current)
        resolved_path = current_path.resolve()
        if resolved_path in visited:
            directories[:] = []
            continue
        visited.add(resolved_path)
        if "outs" in directories:
            directories[:] = ["outs"]
        directory_names = set(directories)
        if "binned_outputs" in directory_names:
            binned_root = current_path / "binned_outputs"
            sizes: set[int] = set()
            if len(_visium_hd_feature_slice_files(current_path)) == 1:
                for bin_root in binned_root.glob("square_*um"):
                    numeric = bin_root.name.removeprefix("square_").removesuffix("um")
                    if not bin_root.is_dir() or not numeric.isdigit():
                        continue
                    size = int(numeric)
                    spatial_root = bin_root / "spatial"
                    if (
                        size in {2, 8, 16}
                        and (bin_root / "filtered_feature_bc_matrix.h5").is_file()
                        and (spatial_root / "tissue_positions.parquet").is_file()
                        and (spatial_root / "scalefactors_json.json").is_file()
                    ):
                        sizes.add(size)
            resolution_sets.append(sizes)
            directories[:] = []
        elif (
            "filtered_feature_bc_matrix.h5" in filenames
            or "filtered_feature_bc_matrix" in directory_names
            or "spatial" in directory_names
        ):
            directories[:] = []
    if not resolution_sets:
        return []
    common_sizes = set.intersection(*resolution_sets)
    return [size for size in (2, 8, 16) if size in common_sizes]


def _visium_hd_feature_slice_files(bundle_root: Path) -> list[Path]:
    return sorted(
        path
        for path in bundle_root.glob("*feature_slice.h5")
        if path.is_file()
    )
def _is_bounded_lowres_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > 20 * 1024 * 1024:
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.width > 2048 or image.height > 2048:
                return False
            image.verify()
    except (ImportError, OSError, SyntaxError, ValueError):
        return False
    return True

def _visium_hd_lowres_image_path(bundle_root: Path, bin_size: int) -> Path | None:
    candidates = (
        bundle_root / "spatial" / "tissue_lowres_image.png",
        bundle_root
        / "binned_outputs"
        / f"square_{bin_size:03d}um"
        / "spatial"
        / "tissue_lowres_image.png",
    )
    return next(
        (path for path in candidates if _is_bounded_lowres_image(path)),
        None,
    )





def _link_reader_source(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required Visium HD reader input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _build_visium_hd_reader_view(bundle_root: Path, bin_size: int, reader_root: Path) -> None:
    feature_slice_files = _visium_hd_feature_slice_files(bundle_root)
    if len(feature_slice_files) != 1:
        raise ValueError(
            f"Expected exactly one Visium HD feature_slice.h5 file in {bundle_root}; "
            f"found {len(feature_slice_files)}."
        )
    _link_reader_source(feature_slice_files[0], reader_root / feature_slice_files[0].name)
    selected_bin = f"square_{bin_size:03d}um"
    source_bin_root = bundle_root / "binned_outputs" / selected_bin
    reader_bin_root = reader_root / "binned_outputs" / selected_bin
    _link_reader_source(
        source_bin_root / "filtered_feature_bc_matrix.h5",
        reader_bin_root / "filtered_feature_bc_matrix.h5",
    )
    source_spatial = source_bin_root / "spatial"
    reader_spatial = reader_bin_root / "spatial"
    positions_source = source_spatial / "tissue_positions.parquet"
    if not positions_source.is_file():
        raise FileNotFoundError(
            f"SpatialData-IO 0.6.0 requires selected-bin tissue_positions.parquet: {positions_source}"
        )
    _link_reader_source(positions_source, reader_spatial / positions_source.name)
    scale_factors_source = source_spatial / "scalefactors_json.json"
    _link_reader_source(scale_factors_source, reader_spatial / scale_factors_source.name)
    lowres_source = _visium_hd_lowres_image_path(bundle_root, bin_size)
    if lowres_source is not None:
        _link_reader_source(lowres_source, reader_root / "spatial" / lowres_source.name)
    root_image_scale_source = bundle_root / "spatial" / "scalefactors_json.json"
    if lowres_source is not None and lowres_source.parent == source_spatial and scale_factors_source.is_file():
        image_scale_source = scale_factors_source
    elif root_image_scale_source.is_file():
        image_scale_source = root_image_scale_source
    else:
        image_scale_source = scale_factors_source
    if image_scale_source.is_file():
        _link_reader_source(
            image_scale_source,
            reader_root / "segmented_outputs" / "spatial" / image_scale_source.name,
        )


def _acquisition_input_paths(bundle: dict[str, Any]) -> list[str]:
    bundle_root = Path(bundle["bundle_root"])
    if bundle["platform"] == "visium_hd":
        spatial_dir = bundle_root / "binned_outputs" / f"square_{int(bundle['bin_size']):03d}um" / "spatial"
        matrix_dir = bundle_root / "binned_outputs" / f"square_{int(bundle['bin_size']):03d}um"
        positions_candidates = [spatial_dir / "tissue_positions.parquet"]
    else:
        spatial_dir = bundle_root / "spatial"
        matrix_dir = bundle_root
        positions_candidates = [
            spatial_dir / "tissue_positions.csv",
            spatial_dir / "tissue_positions_list.csv",
        ]
    matrix_candidates = [matrix_dir / "filtered_feature_bc_matrix.h5"]
    paths = [str(bundle_root)]
    if bundle["platform"] == "visium_hd":
        paths.extend(str(path) for path in _visium_hd_feature_slice_files(bundle_root))
    paths.extend(str(path) for path in matrix_candidates if path.is_file())
    paths.extend(str(path) for path in positions_candidates if path.is_file())
    image_dir = bundle_root / "spatial"
    lowres_path = (
        _visium_hd_lowres_image_path(bundle_root, int(bundle["bin_size"]))
        if bundle["platform"] == "visium_hd"
        else image_dir / "tissue_lowres_image.png"
    )
    display_paths = [
        spatial_dir / "scalefactors_json.json",
        image_dir / "scalefactors_json.json",
    ]
    if lowres_path is not None:
        display_paths.append(lowres_path)
    paths.extend(
        str(path)
        for path in display_paths
        if path.is_file() and str(path) not in paths
    )
    return paths


def _write_count_preview(coordinates: Any, counts: Any, path: Path, max_points: int) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(coordinates)
    count_values = np.asarray(counts).ravel()
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] != count_values.size:
        raise ValueError("Native spatial coordinates do not match the measured count observations.")
    indices = np.arange(values.shape[0]) if values.shape[0] <= max_points else np.linspace(0, values.shape[0] - 1, max_points, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 6))
    points = ax.scatter(
        values[indices, 0],
        values[indices, 1],
        c=np.log1p(count_values[indices]),
        s=1,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("native x coordinate")
    ax.set_ylabel("native y coordinate")
    fig.colorbar(points, ax=ax, label="log1p measured total counts")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {
        "artifact_type": "bounded_spatial_count_preview",
        "data_status": "derived display-only view of measured counts",
        "artifact": str(path),
        "native_coordinate_system": "reader-provided obsm['spatial']",
        "display_transform": "native x unchanged; native y inverted for image-coordinate orientation; color=log1p measured total counts",
        "observations_plotted": int(indices.size),
        "observations_total": int(values.shape[0]),
        "point_cap": max_points,
        "subsampling": "evenly spaced by reader row order" if indices.size < values.shape[0] else "none",
    }




def describe() -> WorkflowDescriptor:
    return WorkflowDescriptor(
        workflow_id="visium-native-qc",
        modality="Visium",
        title="Space Ranger Visium counts and quality checks",
        description="Read complete Space Ranger Visium or Visium HD output bundles with Scanpy and SpatialData-IO, preserve the original counts, report spot or bin geometry and image scale metadata, and calculate count-based quality metrics. For HD, quality checks use only the explicitly selected `square_<three-digit-size>um` count table; cell and nucleus segmentation tables are not analyzed.",
        stages=[
            StageDefinition(
                stage_id="bundle-intake",
                title="Check Space Ranger input files",
                purpose="Validate the original count matrix, tissue coordinates, tissue image and scale metadata.",
                tools=[
                    ToolReference(
                        name="Scanpy read_visium",
                        version=_version("scanpy"),
                        role="Read Space Ranger Visium output",
                        documentation_url="https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_visium.html",
                    ),
                    ToolReference(
                        name="spatialdata-io visium_hd",
                        version=_version("spatialdata-io"),
                        role="Read Space Ranger Visium HD output",
                        documentation_url="https://spatialdata.scverse.org/projects/io/en/latest/generated/spatialdata_io.visium_hd.html",
                    ),
                ],
                implementations=[
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/visium.py", function_name="_discover_bundles", role="Find supported Visium bundles"),
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/visium.py", function_name="_build_visium_hd_reader_view", role="Prepare the selected original HD inputs for the reader"),
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/visium.py", function_name="execute", role="Load the original Visium or HD bundle with the appropriate reader"),
                ],
            ),
            StageDefinition(
                stage_id="count-qc",
                title="Count quality checks",
                purpose="Summarize measured total counts and detected features from the selected original spot or bin count table, and create a size-limited preview. Segmentation-derived cell tables are not used.",
                tools=[
                    ToolReference(
                        name="Scanpy",
                        version=_version("scanpy"),
                        role="Read count matrices and calculate quality metrics",
                        documentation_url="https://scanpy.readthedocs.io/",
                    ),
                    ToolReference(
                        name="Matplotlib",
                        version=_version("matplotlib"),
                        role="Create a size-limited spatial preview",
                        documentation_url="https://matplotlib.org/",
                    ),
                ],
                implementations=[
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/visium.py", function_name="execute", role="Compute count quality metrics"),
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/visium.py", function_name="_write_count_preview", role="Render a size-limited spatial count preview"),
                ],
            ),
        ],
        input_description=(
            "- Select a folder containing complete Space Ranger output bundles. Classic Visium "
            "requires `filtered_feature_bc_matrix.h5` and `spatial/tissue_positions.csv` "
            "or `spatial/tissue_positions_list.csv` in each bundle; incomplete discovered bundles block preflight.\n"
            "- For Visium HD, select 2, 8, or 16 µm. Each selected `binned_outputs/square_*um` "
            "bundle needs `filtered_feature_bc_matrix.h5`, `spatial/tissue_positions.parquet`, "
            "and `spatial/scalefactors_json.json`; its bundle root needs exactly one "
            "`*feature_slice.h5`. Available choices are complete sizes common to all HD bundles.\n"
            "- Classic Visium requires Scanpy; HD requires spatialdata-io. The selected reader "
            "must load a nonempty matrix during execution.\n"
            "- Confirm a mapping for every displayed bundle key (the selected root is `.`) "
            "to an exact `project_samples.csv` specimen_id or explicitly select `UNMAPPED` "
            "if unknown. Folder names do not establish identity. The worker requires confirmation "
            "even if workflow preflight only warns that mapping is unconfirmed.\n"
            "- Optional display limit: 1,000–100,000 spots or bins (default 50,000); "
            "the count matrix is not subsampled."
        ),
        output_description=(
            "- Once processing runs, `visium_native_qc/bundle_inventory.json` lists discovered "
            "bundles, including bundles that fail individually.\n"
            "- For each successfully processed bundle, `spot_bin_qc.csv` records original "
            "barcodes, measured total counts and detected features, with original x/y coordinates "
            "when usable; `bundle_receipt.json` records sources, mapping, reader, scales and counts.\n"
            "- `measured_counts_spatial_preview.png` is created only when usable coordinates "
            "are available. A low-resolution tissue image is copied as `tissue_image_lowres.png` "
            "only if valid, nonempty, at most 20 MiB and at most 2048 pixels per side.\n"
            "- Original count matrices remain in their source folders. QC is unnormalized and "
            "per spot or bin; HD segmentation tables, deconvolution, differential testing and "
            "spatial enrichment are not analyzed."
        ),
        parameter_fields=[
            ParameterField(
                name="bin_size",
                label="Visium HD bin size",
                kind="select",
                required=False,
                default=None,
                choices=[2, 8, 16],
                dynamic_choices_key="visium_hd_bin_sizes",
                unit="µm",
                section="Data resolution",
                stage_id="bundle-intake",
                help_text="Required for detected Visium HD bundles; choices are restricted to complete common resolutions under the selected modality folder. Classic Visium does not use this value.",
            ),
            ParameterField(
                name="preview_max_points",
                label="Maximum points in spatial preview",
                kind="int",
                default=50000,
                min_value=1000,
                max_value=100000,
                step=5000,
                unit="bins or spots",
                section="Display",
                stage_id="count-qc",
                help_text="Limits displayed coordinate points only; measured counts and original matrices are not subsampled or changed.",
            ),
        ],
        limitations=[
            "Spot or bin quality checks are descriptive; spots and bins are not independent biological replicates.",
            "Visium HD bins are platform-native bins, not cells; segmentation-derived cell or nucleus tables are not used.",
            "No cell-type deconvolution, differential testing, or spatial enrichment is performed.",
            "Count previews display at most 100,000 coordinates, as chosen by the user. Only a validated low-resolution tissue image may enter the HD reader; full-resolution/hires images and segmentation tables remain in the source files and are not loaded.",
        ],
    )


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    warnings: list[str] = []
    if request.modality != "Visium":
        blockers.append(f"Expected modality 'Visium', received {request.modality!r}.")
    scanpy_version = _version("scanpy")
    spatialdata_io_version = _version("spatialdata-io")
    available_hd_bin_sizes = _available_hd_bin_sizes(request.modality_dir)
    preview_max_points = request.parameters.get("preview_max_points", 50000)
    if isinstance(preview_max_points, bool) or not isinstance(preview_max_points, int) or not 1000 <= preview_max_points <= 100000:
        blockers.append("preview_max_points must be an integer from 1000 through 100000.")
        preview_max_points = 50000
    raw_bin_size = request.parameters.get("bin_size")
    bin_size: int | None = None
    if raw_bin_size is not None:
        try:
            bin_size = int(raw_bin_size)
        except (TypeError, ValueError):
            blockers.append("Visium HD bin_size must be one of the Space Ranger resolutions 2, 8, or 16 micrometers.")
        if bin_size not in {2, 8, 16}:
            blockers.append("Visium HD bin_size must be one of the Space Ranger resolutions 2, 8, or 16 micrometers.")
            bin_size = None
    bundles, bundle_blockers = _discover_bundles(request.modality_dir, bin_size)
    blockers.extend(bundle_blockers)
    if not bundles:
        blockers.append("No complete authentic Space Ranger Visium/Visium HD count-and-position bundle was found.")
    has_hd = any(bundle["platform"] == "visium_hd" for bundle in bundles)
    has_visium = any(bundle["platform"] == "visium" for bundle in bundles)
    if has_hd:
        for bundle_info in (bundle for bundle in bundles if bundle["platform"] == "visium_hd"):
            feature_slice_files = _visium_hd_feature_slice_files(Path(bundle_info["bundle_root"]))
            if len(feature_slice_files) != 1:
                blockers.append(
                    f"Visium HD bundle {bundle_info['bundle_key']!r} must contain exactly one "
                    f"feature_slice.h5 metadata file; found {len(feature_slice_files)}."
                )
    if has_hd and spatialdata_io_version == "not installed":
        blockers.append("Visium HD requires spatialdata-io's published visium_hd reader; it is not installed.")
    if has_visium and scanpy_version == "not installed":
        blockers.append("Scanpy is not installed; its published Space Ranger Visium reader is required.")
    raw_mapping = request.parameters.get("specimen_mapping")
    raw_confirmed = request.parameters.get("specimen_mapping_confirmed", False)
    if not isinstance(raw_confirmed, bool):
        blockers.append("parameters.specimen_mapping_confirmed must be a boolean review confirmation.")
    mapping_confirmed = raw_confirmed is True
    requested_mapping: dict[str, str] = {}
    specimen_mapping: dict[str, str] = {}
    known_keys = {str(bundle["bundle_key"]) for bundle in bundles}
    if raw_mapping is not None and not isinstance(raw_mapping, dict):
        blockers.append("parameters.specimen_mapping must map exact acquisition-group keys to project_samples specimen IDs or 'UNMAPPED'.")
    elif isinstance(raw_mapping, dict):
        for raw_key, raw_specimen_id in raw_mapping.items():
            key = str(raw_key)
            key_path = Path(key)
            if key_path.is_absolute() or ".." in key_path.parts:
                blockers.append(f"Specimen mapping key must be an exact modality-relative acquisition-group key: {key!r}.")
            elif key not in known_keys:
                blockers.append(f"Specimen mapping refers to an undiscovered acquisition group: {key!r}.")
            elif not isinstance(raw_specimen_id, str) or not raw_specimen_id.strip():
                blockers.append(f"Specimen mapping for acquisition group {key!r} must contain a non-empty specimen ID or 'UNMAPPED'.")
            else:
                requested_mapping[key] = raw_specimen_id.strip()
    if mapping_confirmed:
        missing_mapping = known_keys - requested_mapping.keys()
        if missing_mapping:
            blockers.append(
                "Confirmed specimen mapping must explicitly include every discovered bundle; missing: "
                + ", ".join(sorted(missing_mapping))
            )
    mapped_specimens = {
        key: specimen_id
        for key, specimen_id in requested_mapping.items()
        if specimen_id != "UNMAPPED"
    }
    if mapped_specimens:
        from spatial_omics_toolkit.application.intake import read_project_samples

        sample_rows, sample_blockers = read_project_samples(request.annotation_csv)
        blockers.extend(sample_blockers)
        valid_specimen_ids = {row["specimen_id"].strip() for row in sample_rows if row.get("specimen_id")}
        for key, specimen_id in mapped_specimens.items():
            if specimen_id not in valid_specimen_ids:
                blockers.append(
                    f"Mapped specimen_id {specimen_id!r} for acquisition group {key!r} is absent from selected project_samples.csv."
                )
    if mapping_confirmed:
        specimen_mapping = requested_mapping
    else:
        warnings.append("Specimen mapping is not confirmed; all groups remain UNMAPPED and only per-bundle QC is available.")
    unmapped_keys = {
        key for key in known_keys
        if specimen_mapping.get(key, "UNMAPPED") == "UNMAPPED"
    }
    if mapping_confirmed and unmapped_keys:
        warnings.append(
            "Acquisition groups without a confirmed project_samples mapping remain UNMAPPED: "
            + ", ".join(sorted(unmapped_keys))
            + ". No sample-level aggregation/comparison is permitted."
        )
    groups = [
        {
            "group_key": bundle["bundle_key"],
            "label": bundle["bundle_key"],
            "input_paths": _acquisition_input_paths(bundle),
        }
        for bundle in bundles
    ]
    inventory_input_paths = sorted(
        {
            path
            for group in groups
            for path in group["input_paths"]
        }
    )
    params: dict[str, Any] = {
        "scanpy_version": scanpy_version,
        "spatialdata_io_version": spatialdata_io_version,
        "visium_hd_bin_sizes": available_hd_bin_sizes,
        "bundles": bundles,
        "inventory_input_paths": inventory_input_paths,
        "acquisition_groups": groups,
        "bundle_audit_blockers": bundle_blockers,
        "bin_size": bin_size,
        "preview_max_points": preview_max_points,
        "specimen_mapping": specimen_mapping,
        "specimen_mapping_confirmed": mapping_confirmed,
    }
    return PreflightResult(ready=not blockers, blockers=blockers, warnings=warnings, resolved_parameters=params)


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        event = StageEvent(
            stage_id="bundle-intake",
            status="BLOCKED",
            message="Visium bundle preflight blocked.",
            parameters=check.resolved_parameters,
        )
        emit(event)
        return WorkflowResult(
            status="BLOCKED",
            stages=[event],
            warnings=check.blockers + check.warnings,
        )

    import numpy as np

    request.run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = request.run_dir / "visium_native_qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    events: list[StageEvent] = []
    warnings = list(check.warnings)
    bundles = check.resolved_parameters["bundles"]
    for index, bundle_info in enumerate(bundles, start=1):
        bundle = Path(bundle_info["bundle_root"])
        is_hd = bundle_info["platform"] == "visium_hd"
        platform = f"visium_hd:{bundle_info['bin_size']}" if is_hd else "visium"
        reader_version = str(
            check.resolved_parameters["spatialdata_io_version"]
            if is_hd
            else check.resolved_parameters["scanpy_version"]
        )
        reader_name = "spatialdata_io.visium_hd" if is_hd else "scanpy.read_visium"
        tool_versions = {
            "scanpy": _version("scanpy"),
            "matplotlib": _version("matplotlib"),
        }
        if is_hd:
            tool_versions["spatialdata-io"] = reader_version
        sample_output = output_dir / f"{index:03d}_{bundle.name or 'bundle'}"
        sample_warning_start = len(warnings)
        sample_output.mkdir(parents=True, exist_ok=True)
        emit(
            StageEvent(
                stage_id="bundle-intake",
                status="RUNNING",
                message=f"Reading Space Ranger bundle {bundle}.",
                tool_versions=tool_versions,
                parameters={
                    "platform": platform,
                    "reader": reader_name,
                    "bundle_root": str(bundle),
                    "reader_scope": "requested native bin, required metadata/coordinates/scales, and bounded low-resolution image only" if is_hd else "selected classic Visium bundle",
                },
            )
        )
        selected_table_name = "filtered_feature_bc_matrix.h5"
        reader_table_names = [selected_table_name]
        ignored_reader_table_names: list[str] = []
        reader_succeeded = False
        try:
            if is_hd:
                from spatialdata_io import visium_hd

                selected_table_name = f"square_{int(bundle_info['bin_size']):03d}um"
                reader_notices: list[str] = []
                with TemporaryDirectory(prefix="visium-reader-", dir=sample_output) as reader_directory:
                    reader_root = Path(reader_directory)
                    _build_visium_hd_reader_view(
                        bundle,
                        int(bundle_info["bin_size"]),
                        reader_root,
                    )
                    with python_warnings.catch_warnings(record=True) as reader_warnings:
                        python_warnings.simplefilter("always")
                        sdata = visium_hd(
                            path=reader_root,
                            bin_size=int(bundle_info["bin_size"]),
                            load_segmentations_only=False,
                            load_nucleus_segmentations=False,
                            load_all_images=False,
                        )
                    for warning in reader_warnings:
                        message = str(warning.message)
                        if (
                            "No full resolution image found" in message
                            or "No image path found containing the hires image" in message
                        ):
                            reader_notices.append(
                                "Full-resolution and hires images are intentionally excluded from "
                                f"the bounded QC reader view. SpatialData-IO notice: {message}"
                            )
                        else:
                            warnings.append(f"{reader_name} warning: {message}")
                    if selected_table_name not in sdata.tables:
                        available_tables = ", ".join(str(name) for name in sdata.tables)
                        raise ValueError(
                            f"spatialdata-io did not return the selected binned count table "
                            f"{selected_table_name!r}; available tables: {available_tables}"
                        )
                    adata = sdata.tables[selected_table_name]
                    reader_table_names = [str(name) for name in sdata.tables]
                    ignored_reader_table_names = [
                        name for name in reader_table_names if name != selected_table_name
                    ]
                    del sdata
                if ignored_reader_table_names:
                    warnings.append(
                        "SpatialData-IO exposed additional table(s) "
                        f"{ignored_reader_table_names}; only {selected_table_name} was used for bin QC."
                    )
            else:
                import scanpy as sc

                adata = sc.read_visium(path=str(bundle), count_file="filtered_feature_bc_matrix.h5")
            if adata.n_obs == 0 or adata.n_vars == 0:
                raise ValueError("The filtered count matrix has zero observations or features.")
            reader_succeeded = True
            intake_event = StageEvent(
                stage_id="bundle-intake",
                status="SUCCEEDED",
                message=f"Loaded {adata.n_obs} native spots/bins and {adata.n_vars} features with {reader_name}.",
                tool_versions=tool_versions,
                parameters={
                    "platform": platform,
                    "reader": reader_name,
                    "count_table_name": selected_table_name if is_hd else "filtered_feature_bc_matrix.h5",
                    "bundle_key": bundle_info["bundle_key"],
                },
            )
            events.append(intake_event)
            emit(intake_event)
            counts = adata.X
            adata.obs["total_counts_measured"] = np.asarray(counts.sum(axis=1)).ravel()
            adata.obs["features_detected_measured"] = np.asarray((counts > 0).sum(axis=1)).ravel()
            qc_path = sample_output / "spot_bin_qc.csv"
            qc_frame = adata.obs[["total_counts_measured", "features_detected_measured"]].copy()
            spatial_coordinates = adata.obsm.get("spatial")
            preview_path: Path | None = None
            preview_receipt: dict[str, Any] | None = None
            if spatial_coordinates is not None:
                coordinate_values = np.asarray(spatial_coordinates)
                if coordinate_values.ndim == 2 and coordinate_values.shape[0] == adata.n_obs and coordinate_values.shape[1] >= 2:
                    qc_frame["spatial_x_native"] = coordinate_values[:, 0]
                    qc_frame["spatial_y_native"] = coordinate_values[:, 1]
                    preview_path = sample_output / "measured_counts_spatial_preview.png"
                    preview_receipt = _write_count_preview(
                        coordinate_values,
                        adata.obs["total_counts_measured"].to_numpy(),
                        preview_path,
                        int(check.resolved_parameters["preview_max_points"]),
                    )
                else:
                    warnings.append(f"Reader returned unusable spatial coordinates for {bundle}; count QC was retained without a spatial preview.")
            else:
                warnings.append(f"Reader returned no native spatial coordinate mapping for {bundle}; count QC was retained without a spatial preview.")
            qc_frame.index.name = "native_barcode"
            qc_frame.to_csv(qc_path)
            image_dir = bundle / "spatial"
            image_directories = [image_dir]
            if is_hd:
                image_directories.append(
                    bundle
                    / "binned_outputs"
                    / f"square_{int(bundle_info['bin_size']):03d}um"
                    / "spatial"
                )
            source_image_paths = [
                image_directory / image_name
                for image_directory in image_directories
                for image_name in (
                    "tissue_lowres_image.png",
                    "tissue_hires_image.png",
                    "tissue_hires_image.jpg",
                    "tissue_fullres_image.tif",
                )
                if (image_directory / image_name).is_file()
            ]
            native_image_sources = [
                {"path": str(path), "size_bytes": path.stat().st_size}
                for path in source_image_paths
            ]
            tissue_image = (
                _visium_hd_lowres_image_path(bundle, int(bundle_info["bin_size"]))
                if is_hd
                else image_dir / "tissue_lowres_image.png"
            )
            if tissue_image is None:
                tissue_image = next(
                    (image_directory / "tissue_lowres_image.png" for image_directory in image_directories
                     if (image_directory / "tissue_lowres_image.png").is_file()),
                    None,
                )
            tissue_image_copy: Path | None = None
            if tissue_image is not None:
                image_size = tissue_image.stat().st_size
                if image_size == 0:
                    warnings.append(f"Low-resolution tissue preview is empty and will not be copied: {tissue_image}.")
                elif image_size > 20 * 1024 * 1024:
                    warnings.append(f"Low-resolution tissue preview exceeds the bounded copy size; source remains at {tissue_image}.")
                else:
                    try:
                        from PIL import Image

                        with Image.open(tissue_image) as image:
                            if image.width > 2048 or image.height > 2048:
                                raise ValueError("preview exceeds 2048 × 2048 pixels")
                            image.verify()
                    except (ImportError, OSError, SyntaxError, ValueError) as exc:
                        warnings.append(f"Low-resolution tissue preview is invalid or exceeds preview limits and will not be copied: {tissue_image}: {exc}")
                    else:
                        import shutil

                        tissue_image_copy = sample_output / "tissue_image_lowres.png"
                        shutil.copy2(tissue_image, tissue_image_copy)
            elif source_image_paths:
                warnings.append(f"No Space Ranger low-resolution tissue preview is available; source image(s) remain in {image_dir}.")
            image_scale_path = image_dir / "scalefactors_json.json"
            scale_path = (
                bundle
                / "binned_outputs"
                / f"square_{int(bundle_info['bin_size']):03d}um"
                / "spatial"
                / "scalefactors_json.json"
                if is_hd
                else image_scale_path
            )
            if is_hd and tissue_image is not None and tissue_image.parent == scale_path.parent:
                image_scale_path = scale_path
            scale_factors = json.loads(scale_path.read_text(encoding="utf-8")) if scale_path.is_file() else {}
            if image_scale_path.is_file():
                image_scale_factors = json.loads(image_scale_path.read_text(encoding="utf-8"))
                image_scale_source_path: Path | None = image_scale_path
            else:
                image_scale_factors = scale_factors
                image_scale_source_path = scale_path if scale_path.is_file() else None
            if is_hd and not scale_path.is_file():
                warnings.append(f"No scale factors were found for the selected Visium HD bin table at {scale_path}.")
            input_paths = _acquisition_input_paths(bundle_info)
            count_matrix_path = next((path for path in input_paths if "filtered_feature_bc_matrix" in path), None)
            count_matrix_size = (
                Path(count_matrix_path).stat().st_size
                if count_matrix_path is not None and Path(count_matrix_path).is_file()
                else None
            )
            coordinate_sources = [
                {"path": path, "size_bytes": Path(path).stat().st_size}
                for path in input_paths
                if "tissue_positions" in Path(path).name and Path(path).is_file()
            ]
            tissue_image_asset = (
                {
                    "path": str(tissue_image_copy),
                    "artifact_type": "bounded_spaceranger_lowres_tissue_preview",
                    "data_status": "display-only source image",
                    "source_path": str(tissue_image),
                    "size_bytes": tissue_image_copy.stat().st_size,
                    "scale_factor": image_scale_factors.get("tissue_lowres_scalef"),
                }
                if tissue_image_copy is not None and tissue_image is not None
                else None
            )
            display_assets = [asset for asset in (preview_receipt, tissue_image_asset) if asset is not None]
            segmentation_root = bundle / "segmented_outputs"
            segmentation_source_paths = [
                str(path)
                for path in (
                    segmentation_root / "filtered_feature_cell_matrix.h5",
                    segmentation_root / "cell_segmentations.geojson",
                )
                if path.is_file()
            ]
            receipt_path = sample_output / "bundle_receipt.json"
            receipt = {
                "reader": reader_name,
                "tool_version": reader_version,
                "platform": platform,
                "source_bundle": str(bundle),
                "bundle_key": bundle_info["bundle_key"],
                "specimen_id": check.resolved_parameters["specimen_mapping"].get(bundle_info["bundle_key"], "UNMAPPED"),
                "specimen_mapping_confirmed": check.resolved_parameters["specimen_mapping_confirmed"],
                "mapping_source_file": str(request.annotation_csv),
                "mapping_source_note": "The selected project_samples.csv is recorded; specimen identity is used only after explicit mapping confirmation, never inferred from bundle names.",
                "source_count_matrix": count_matrix_path,
                "source_count_matrix_size_bytes": count_matrix_size,
                "native_observations": int(adata.n_obs),
                "features": int(adata.n_vars),
                "count_layer": "Measured filtered counts read from the source bundle; matrix is referenced, not copied.",
                "native_coordinate_sources": coordinate_sources,
                "coordinate_table": str(qc_path),
                "coordinate_columns": ["spatial_x_native", "spatial_y_native"] if preview_receipt is not None else [],
                "coordinate_system": "reader-provided obsm['spatial'] as native x/y; no coordinate transform applied in QC table",
                "native_image_sources": native_image_sources,
                "tissue_lowres_preview": tissue_image_asset,
                "scale_factors": scale_factors,
                "scale_factors_source_path": str(scale_path) if scale_path.is_file() else None,
                "image_scale_factors": image_scale_factors,
                "image_scale_factors_source_path": str(image_scale_source_path) if image_scale_source_path is not None else None,
                "selected_resolution_um": int(bundle_info["bin_size"]) if is_hd else None,
                "count_table_name": selected_table_name if is_hd else "filtered_feature_bc_matrix.h5",
                "reader_table_names": reader_table_names,
                "ignored_reader_table_names": ignored_reader_table_names,
                "reader_input_policy": (
                    "A temporary reader view contains only the requested bin count matrix, feature-slice "
                    "metadata, selected bin positions and scales, and bounded low-resolution image. "
                    "Full-resolution/hires images and cell/nucleus segmentation outputs are not loaded."
                ) if is_hd else "Published Scanpy Space Ranger reader uses the selected Visium bundle.",
                "reader_notices": reader_notices if is_hd else [],
                "segmentation_source_files_not_loaded": segmentation_source_paths if is_hd else [],
                "display_assets": display_assets,
                "interpretation": "Rows are native spots or platform bins, never cells or independent biological replicates.",
            }
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
            sample_artifacts = [str(qc_path), str(receipt_path)]
            if preview_path is not None:
                sample_artifacts.append(str(preview_path))
            if tissue_image_copy is not None:
                sample_artifacts.append(str(tissue_image_copy))
            artifacts.extend(sample_artifacts)
            event = StageEvent(
                stage_id="count-qc",
                status="WARNING" if len(warnings) > sample_warning_start else "SUCCEEDED",
                message=f"Read {adata.n_obs} native spots/bins and {adata.n_vars} features from {bundle}.",
                tool_versions=tool_versions,
                parameters={
                    "platform": platform,
                    "normalization": "none",
                    "count_table_name": selected_table_name if is_hd else "filtered_feature_bc_matrix.h5",
                    "reader_table_names": reader_table_names,
                    "scale_factors_source_path": str(scale_path) if scale_path.is_file() else None,
                    "ignored_reader_table_names": ignored_reader_table_names,
                    "bin_is_cell": False,
                    "preview_max_points": int(check.resolved_parameters["preview_max_points"]),
                    "bundle_root": str(bundle),
                    "bundle_key": bundle_info["bundle_key"],
                    "specimen_id": check.resolved_parameters["specimen_mapping"].get(bundle_info["bundle_key"], "UNMAPPED"),
                    "specimen_mapping_confirmed": check.resolved_parameters["specimen_mapping_confirmed"],
                },
                artifacts=sample_artifacts,
            )
        except Exception as exc:
            event = StageEvent(
                stage_id="count-qc" if reader_succeeded else "bundle-intake",
                status="FAILED",
                message=f"Published Visium reader or QC failed for {bundle}: {type(exc).__name__}: {exc}",
                tool_versions=tool_versions,
                parameters={"platform": platform, "bundle_root": str(bundle)},
            )
            warnings.append(event.message)
        events.append(event)
        emit(event)
    inventory_path = output_dir / "bundle_inventory.json"
    inventory_path.write_text(json.dumps(bundles, indent=2, default=str) + "\n", encoding="utf-8")
    artifacts.append(str(inventory_path))
    failed = any(event.status == "FAILED" for event in events)
    return WorkflowResult(
        status="FAILED" if failed else ("WARNING" if warnings else "SUCCEEDED"),
        stages=events,
        artifacts=artifacts,
        warnings=warnings,
    )
