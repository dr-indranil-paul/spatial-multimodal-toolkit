from __future__ import annotations

import importlib.metadata
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from spatial_omics_toolkit.application.intake import ALLOWED_ROOT, lexical_path_is_under, path_is_under

import numpy as np
from PIL import Image

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

_WORKFLOW_ID = "integration.serial_section_alignment"
_MAX_DIMENSION = 512
_MAX_NATIVE_READ_BYTES = 128 * 1024 * 1024


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"




def _source_path(request: RunRequest, parameter: str) -> Path | None:
    value = request.parameters.get(parameter)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else request.project_root / candidate).resolve()


def describe() -> WorkflowDescriptor:
    return WorkflowDescriptor(
        workflow_id=_WORKFLOW_ID,
        modality="Integration",
        title="Exploratory serial-section H&E–fluorescence alignment",
        description=(
            "Estimate and visualize an exploratory translation-only alignment between a selected H&E CZI and "
            "a selected fluorescence or hyperspectral CZI channel. This technical quality check does not establish "
            "biological pairing, same-cell registration, or biological identity."
        ),
        stages=[
            StageDefinition(
                stage_id="authorization", title="Source and channel authorization",
                purpose="Validate explicit candidate sources, channel selection and technical-QC authorization.",
                tools=[ToolReference(name="Spatial toolkit preflight", version="internal", role="Validate source and authorization parameters")],
                implementations=[FunctionReference(source_file="spatial_omics_toolkit/application/workflows/integration.py", function_name="preflight", role="Check source paths and authorization")],
                depends_on=[], condition="Explicit candidate sources and channel are authorized.",
                inputs=["Selected CZI paths", "Candidate confirmation and channel selection"], outputs=["Validated source parameters"],
            ),
            StageDefinition(
                stage_id="metadata-scale", title="Metadata and physical scale",
                purpose="Read native geometry and validate channel identity and physical pixel scale.",
                tools=[ToolReference(name="aicspylibczi", version=_version("aicspylibczi"), role="Read CZI metadata")],
                implementations=[FunctionReference(source_file="spatial_omics_toolkit/application/workflows/integration.py", function_name="_native_metadata", role="Read CZI geometry and scale")],
                depends_on=["authorization"], condition="Authorization passed.",
                inputs=["Validated source parameters", "CZI metadata"], outputs=["Validated native metadata"],
            ),
            StageDefinition(
                stage_id="bounded-previews", title="Bounded previews",
                purpose="Read bounded source planes and compose the physical-scale reference canvas.",
                tools=[ToolReference(name="aicspylibczi", version=_version("aicspylibczi"), role="Read bounded CZI planes"), ToolReference(name="scikit-image", version=_version("scikit-image"), role="Sample onto reference grid")],
                implementations=[FunctionReference(source_file="spatial_omics_toolkit/application/workflows/integration.py", function_name="_read_bounded", role="Read source images within the size limit"), FunctionReference(source_file="spatial_omics_toolkit/application/workflows/integration.py", function_name="_physical_he_canvas", role="Compose physical-scale canvas")],
                depends_on=["metadata-scale"], condition="Channel and physical scale validated.",
                inputs=["Validated native metadata", "CZI source planes"], outputs=["Bounded source previews", "Reference-grid canvas", "Source inventory"],
            ),
            StageDefinition(
                stage_id="translation-registration", title="Translation-only registration",
                purpose="Estimate the exploratory translation on bounded reference-grid images.",
                tools=[ToolReference(name="scikit-image", version=_version("scikit-image"), role="Translation registration")],
                implementations=[FunctionReference(source_file="spatial_omics_toolkit/cycif/register.py", function_name="register_he_to_fluorescence", role="Register images using translation only")],
                depends_on=["bounded-previews"], condition="Bounded previews are available.",
                inputs=["Reference-grid canvas", "Bounded reference plane"], outputs=["Translation and warped image"],
            ),
            StageDefinition(
                stage_id="review", title="Metrics and human review",
                purpose="Save overlays and descriptive metrics for visual review; do not accept alignment automatically.",
                tools=[ToolReference(name="Matplotlib", version=_version("matplotlib"), role="Create review figures")],
                implementations=[FunctionReference(source_file="spatial_omics_toolkit/application/workflows/integration.py", function_name="execute_stage", role="Generate review outputs")],
                depends_on=["translation-registration"], condition="Registration completed; human review is required.",
                inputs=["Translation and warped image", "Bounded previews and source inventory"], outputs=["Overlay PDF and PNG", "Alignment QC report"],
                event_stage_ids=["workflow_output"],
            ),
        ],
        input_description=(
            "- Select existing moving H&E and reference fluorescence or hyperspectral `.czi` "
            "files under the approved workspace; relative paths resolve against the project root.\n"
            "- Explicitly choose `serial_section_candidate` and confirm the candidate sources. "
            "This authorizes exploratory technical QC, not biological pairing.\n"
            "- Supply a nonnegative reference channel index and its exact CZI metadata name. "
            "The reference must resolve to one grayscale plane; H&E must be native BGR24 at C=0. "
            "The first recorded Z and T indices are used.\n"
            "- Each source needs positive physical X and Y pixel sizes isotropic within 1%, "
            "one dimension block, at most one scene and valid mosaic bounds. Non-mosaic planes "
            "must use a supported pixel type and fit the 128 MiB per-source native read limit. "
            "These source details are checked during execution.\n"
            "- The translation search radius is 1–64 reference preview pixels (default 16). "
            "Runtime requires aicspylibczi, NumPy, SciPy, scikit-image, Matplotlib and Pillow.\n"
            "- Confirm a mapping entry for both `integration:moving_he` and "
            "`integration:reference` to a project_samples.csv specimen_id or explicitly "
            "`UNMAPPED` if unknown. The worker requires this even though the workflow reports "
            "the comparison as unmapped and does not infer pairing."
        ),
        output_description=(
            "- If execution completes, `integration_serial_section_alignment/` contains "
            "`source_inventory_and_native_metadata.json`, `moving_he_source_preview.png`, "
            "`reference_channel_source_preview.png`, `before_after_alignment_overlay.pdf`, "
            "`before_after_alignment_overlay.png`, and `alignment_qc_report.json`. "
            "A failure can leave a partial output set.\n"
            "- The result is a WARNING and requires visual review: translation and correlation "
            "are descriptive, with no acceptance threshold. The alignment is not accepted and "
            "does not establish same-cell registration or biological identity."
        ),
        parameter_fields=[
            ParameterField(name="moving_he_path", label="Moving H&E CZI", kind="path", required=True, section="Sources", stage_id="authorization", help_text="Exact H&E CZI path; must be explicitly supplied."),
            ParameterField(name="reference_fluorescence_path", label="Reference fluorescence/HS CZI", kind="path", required=True, section="Sources", stage_id="authorization", help_text="Exact reference CZI path; must be explicitly supplied."),
            ParameterField(name="linkage", label="Section relationship", kind="select", required=True, default="serial_section_candidate", choices=["serial_section_candidate"], section="Review", stage_id="authorization", help_text="A candidate adjacent-section relationship only; this does not confirm biological pairing."),
            ParameterField(name="candidate_sources_confirmed", label="I confirm these are the two candidate source images to compare", kind="bool", required=True, default=False, section="Review", stage_id="authorization", review_required=True, help_text="Explicitly authorizes these selected sources for exploratory technical QC as a serial-section candidate; it does not confirm biological pairing or accepted alignment."),
            ParameterField(name="reference_channel_index", label="Reference channel index (C)", kind="int", required=True, section="Reference image", stage_id="authorization", help_text="Choose the fluorescence/HS channel from the source metadata. Channel identity is never inferred."),
            ParameterField(name="reference_channel_name", label="Reference channel name confirmation", kind="text", required=True, section="Reference image", stage_id="authorization", help_text="Enter the metadata channel name corresponding to the chosen index; it must match source metadata exactly."),
            ParameterField(name="search_radius", label="Translation search radius", kind="int", default=16, min_value=1, max_value=64, unit="reference preview pixels", section="Registration", stage_id="translation-registration", help_text="Search extent for translation-only alignment, measured in pixels of the reduced reference image."),
        ],
        limitations=[
            "This exploratory serial-section alignment uses translation only; it does not model deformation or establish same-cell registration or biological identity.",
            "Confirming the candidate sources authorizes only exploratory comparison; it does not establish biological pairing or accepted alignment. Visual review remains required.",
            "Both CZI files need physical X/Y pixel-size metadata; missing or anisotropic scale metadata blocks registration rather than assuming a scale.",
            "Correlation is descriptive and has no calibrated acceptance threshold; inspect every output visually. Computing a transform does not mean the alignment is accepted.",
        ],
    )


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    if request.workflow_id != _WORKFLOW_ID:
        blockers.append(f"Expected workflow {_WORKFLOW_ID!r}, received {request.workflow_id!r}.")
    if request.modality != "Integration":
        blockers.append(f"Expected modality 'Integration', received {request.modality!r}.")
    linkage = request.parameters.get("linkage")
    if linkage != "serial_section_candidate":
        blockers.append("linkage must explicitly be 'serial_section_candidate'.")
    if request.parameters.get("candidate_sources_confirmed") is not True:
        blockers.append("candidate_sources_confirmed must be explicitly true to authorize exploratory technical QC of these selected candidate sources.")
    channel_index = request.parameters.get("reference_channel_index")
    if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
        blockers.append("reference_channel_index must be an explicit non-negative integer.")
    channel_name = request.parameters.get("reference_channel_name")
    if not isinstance(channel_name, str) or not channel_name.strip():
        blockers.append("reference_channel_name must explicitly identify the selected channel as named in source metadata.")
    radius = request.parameters.get("search_radius", 16)
    if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 64:
        blockers.append("search_radius must be an integer from 1 through 64 preview pixels.")
        radius = 16
    he_path = _source_path(request, "moving_he_path")
    fluor_path = _source_path(request, "reference_fluorescence_path")
    source_paths = [path for path in (he_path, fluor_path) if path is not None]
    for label, path in (("Moving H&E CZI", he_path), ("Reference fluorescence/HS CZI", fluor_path)):
        if path is None:
            blockers.append(f"{label} must be explicitly supplied.")
            continue
        if not lexical_path_is_under(path, ALLOWED_ROOT) or not path_is_under(path, ALLOWED_ROOT):
            blockers.append(f"{label} must resolve under {ALLOWED_ROOT}: {path}")
        if not path.is_file():
            blockers.append(f"{label} does not exist: {path}")
        elif path.suffix.lower() != ".czi":
            blockers.append(f"{label} must be a .czi file: {path}")
    input_paths = [str(path) for path in source_paths]
    acquisition_groups = [
        {
            "group_key": group_key,
            "label": path.name,
            "input_paths": [str(path)],
            "specimen_mapping": "UNMAPPED",
            "mapping_status": "UNMAPPED",
        }
        for group_key, path in (("integration:moving_he", he_path), ("integration:reference", fluor_path))
        if path is not None
    ]
    versions = {name: _version(name) for name in ("aicspylibczi", "numpy", "scipy", "scikit-image", "matplotlib", "Pillow")}
    missing = [name for name, version in versions.items() if version == "not installed"]
    if missing:
        blockers.append("Required runtime packages are unavailable: " + ", ".join(missing))
    return PreflightResult(
        ready=not blockers,
        blockers=blockers,
        warnings=["All acquisition groups remain UNMAPPED; this run is restricted to technical quality checks of the explicitly selected images."],
        resolved_parameters={
            "moving_he_path": str(he_path) if he_path is not None else "",
            "reference_fluorescence_path": str(fluor_path) if fluor_path is not None else "",
            "inventory_input_paths": input_paths,
            "acquisition_groups": acquisition_groups,
            "linkage": linkage,
            "candidate_sources_confirmed": request.parameters.get("candidate_sources_confirmed") is True,
            "reference_channel_index": channel_index,
            "reference_channel_name": channel_name,
            "search_radius": radius,
            "max_dimension": _MAX_DIMENSION,
            "acquisition_group_mapping": "UNMAPPED",
            "required_runtime_packages": versions,
            "runtime_prerequisites": [
                "aicspylibczi CZI reader", "NumPy", "SciPy", "scikit-image", "Matplotlib", "Pillow",
                "CZI physical X/Y scaling metadata for both inputs", "explicit channel index/name confirmation",
                f"Non-mosaic reads are limited to {_MAX_NATIVE_READ_BYTES} estimated native bytes per source before bounded resizing.",
                "explicit serial_section_candidate linkage and candidate_sources_confirmed authorization for exploratory technical QC",
            ],
        },
    )


def _czi_reader() -> Any:
    from aicspylibczi import CziFile

    return CziFile


def _native_metadata(path: Path) -> dict[str, Any]:
    reader = _czi_reader()(str(path))
    blocks = reader.get_dims_shape()
    if len(blocks) != 1:
        raise ValueError(f"Multiple CZI dimension blocks require explicit scene selection; found {len(blocks)} blocks.")
    dims = {str(axis): [int(bounds[0]), int(bounds[1])] for axis, bounds in blocks[0].items()}
    if "X" not in dims or "Y" not in dims:
        raise ValueError(f"CZI native X/Y dimensions are missing: {dims}")
    if "S" in dims and dims["S"][1] - dims["S"][0] > 1:
        raise ValueError(f"Multiple CZI scenes require explicit scene selection; found S={dims['S']}.")
    mosaic_bbox_xywh: list[int] | None = None
    if reader.is_mosaic():
        box = reader.get_mosaic_bounding_box()
        raw_box = (box.x, box.y, box.w, box.h)
        if any(not math.isfinite(float(value)) or int(value) != value for value in raw_box):
            raise ValueError(f"Non-integral or non-finite native mosaic bounding box: {raw_box}.")
        mosaic_bbox_xywh = [int(value) for value in raw_box]
        if mosaic_bbox_xywh[2] <= 0 or mosaic_bbox_xywh[3] <= 0:
            raise ValueError(f"Invalid native mosaic bounding box: {mosaic_bbox_xywh}.")
    channels = []
    for node in reader.meta.findall(".//Information/Image/Dimensions/Channels/Channel"):
        channels.append(node.findtext("Name", default=node.get("Name", "")))
    scaling: dict[str, float] = {}
    for node in reader.meta.findall(".//Scaling/Items/Distance"):
        axis = node.findtext("Id", default=node.get("Id", ""))
        value = node.findtext("Value", default="")
        if axis in {"X", "Y"} and value:
            try:
                pitch_um = float(value) * 1e6
            except ValueError:
                continue
            if math.isfinite(pitch_um) and pitch_um > 0:
                scaling[axis] = pitch_um
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "dimensions": dims,
        "dimension_ranges_are_tile_ranges_for_mosaic": bool(reader.is_mosaic()),
        "mosaic_bbox_xywh": mosaic_bbox_xywh,
        "mosaic_width": mosaic_bbox_xywh[2] if mosaic_bbox_xywh is not None else None,
        "mosaic_height": mosaic_bbox_xywh[3] if mosaic_bbox_xywh is not None else None,
        "channels": channels,
        "pixel_type": str(reader.pixel_type),
        "is_mosaic": bool(reader.is_mosaic()),
        "pixel_size_um": scaling,
        "scale_source": "CZI Scaling/Items/Distance Value (meters converted to micrometers)",
        "xml_metadata": ET.tostring(reader.meta, encoding="unicode"),
    }


def _read_bounded(path: Path, metadata: dict[str, Any], *, channel: int | None) -> tuple[np.ndarray, float, str]:
    reader = _czi_reader()(str(path))
    dims = metadata["dimensions"]
    if reader.is_mosaic():
        region = metadata["mosaic_bbox_xywh"]
        if region is None:
            raise ValueError("Native mosaic bounding box is unavailable.")
        _, _, native_width, native_height = region
    else:
        region = None
        x0, x1 = dims["X"]
        y0, y1 = dims["Y"]
        native_width, native_height = x1 - x0, y1 - y0
    scale_factor = min(1.0, _MAX_DIMENSION / max(native_width, native_height))
    if not reader.is_mosaic():
        bytes_per_pixel = {"bgr24": 3, "gray8": 1, "gray16": 2, "gray32": 4, "gray32float": 4}.get(
            metadata["pixel_type"].lower()
        )
        if bytes_per_pixel is None:
            raise ValueError(f"Cannot bound non-mosaic native read for pixel type {metadata['pixel_type']!r}.")
        estimated_native_bytes = (x1 - x0) * (y1 - y0) * bytes_per_pixel
        if estimated_native_bytes > _MAX_NATIVE_READ_BYTES:
            raise ValueError(
                f"Non-mosaic native plane requires {estimated_native_bytes} bytes, above the "
                f"{_MAX_NATIVE_READ_BYTES}-byte explicit per-source read budget."
            )
    selectors: dict[str, int] = {}
    for axis in ("C", "Z", "T"):
        if axis in dims:
            start, stop = dims[axis]
            if axis == "C" and channel is not None:
                if not start <= channel < stop:
                    raise ValueError(f"Requested C={channel} is outside metadata range [{start}, {stop}).")
                selectors[axis] = channel
            elif axis == "C":
                if not start <= 0 < stop:
                    raise ValueError(f"H&E C=0 is outside metadata range [{start}, {stop}).")
                selectors[axis] = 0
            else:
                selectors[axis] = start
    if reader.is_mosaic():
        pixels = reader.read_mosaic(region=tuple(region), scale_factor=scale_factor, **selectors)
        axes = f"mosaic; region={region}; selectors=" + json.dumps(selectors, sort_keys=True)
    else:
        pixels, axes = reader.read_image(**selectors)
        pixels = np.squeeze(pixels)
        if max(pixels.shape[:2]) > _MAX_DIMENSION:
            from skimage.transform import resize

            original_dtype = pixels.dtype
            output_h = max(1, round(pixels.shape[0] * scale_factor))
            output_w = max(1, round(pixels.shape[1] * scale_factor))
            target_shape = (output_h, output_w, pixels.shape[2]) if pixels.ndim == 3 else (output_h, output_w)
            pixels = resize(pixels, target_shape, preserve_range=True, anti_aliasing=True)
            if np.issubdtype(original_dtype, np.integer):
                pixels = np.clip(pixels, 0, np.iinfo(original_dtype).max)
            pixels = pixels.astype(original_dtype)
            axes = f"{axes}; native-read-bytes={estimated_native_bytes}; resized_to={target_shape}"
    pixels = np.squeeze(pixels)
    if channel is None:
        if str(reader.pixel_type) != "bgr24" or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError(f"Expected an RGB H&E BGR24 mosaic at C=0, got {reader.pixel_type} {pixels.shape}.")
        pixels = np.ascontiguousarray(pixels[..., ::-1])
    elif pixels.ndim != 2:
        raise ValueError(f"Expected one grayscale reference channel, got shape {pixels.shape}.")
    if pixels.ndim not in (2, 3) or max(pixels.shape[:2]) > _MAX_DIMENSION:
        raise ValueError(f"Bounded CZI reader returned unsupported or oversized shape {pixels.shape}.")
    actual_scale = math.sqrt((pixels.shape[0] / native_height) * (pixels.shape[1] / native_width))
    return pixels, actual_scale, str(axes)


def _physical_he_canvas(he: np.ndarray, target_shape: tuple[int, int], scale: float) -> np.ndarray:
    from skimage.transform import AffineTransform, warp

    if he.ndim != 3 or he.shape[2] != 3:
        raise ValueError(f"Expected an RGB H&E preview, got shape {he.shape}.")
    if len(target_shape) != 2 or max(target_shape) > _MAX_DIMENSION or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid bounded reference canvas or physical resampling scale: {target_shape}, {scale}.")
    # Output reference pixel -> input H&E preview pixel, preserving the metadata-derived scale.
    inverse_map = AffineTransform(scale=(1.0 / scale, 1.0 / scale))
    warped = warp(
        he,
        inverse_map=inverse_map,
        output_shape=(*target_shape, 3),
        order=1,
        mode="constant",
        cval=255,
        preserve_range=True,
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def _bounded_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array
    low, high = np.percentile(array, (1, 99))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        normalized = (np.clip((array.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return np.repeat(normalized[..., None], 3, axis=2)


def _save_source_preview(array: np.ndarray, destination: Path) -> None:
    if array.ndim == 3:
        Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB").save(destination)
    else:
        low, high = np.percentile(array, (1, 99))
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
            display = np.zeros(array.shape, dtype=np.uint8)
        else:
            display = (np.clip((array.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(display, mode="L").save(destination)


def execute_stage(request: RunRequest, stage_id: str, upstream: dict[str, WorkflowResult], emit: WorkflowEmitter) -> WorkflowResult:
    dependencies = {stage.stage_id: stage.depends_on for stage in describe().stages}
    if stage_id not in dependencies:
        raise ValueError(f"Unknown integration stage: {stage_id}")
    for dependency in dependencies[stage_id]:
        result = upstream.get(dependency)
        if result is None or result.status not in {"SUCCEEDED", "REUSED", "WARNING"}:
            event = StageEvent(stage_id=stage_id, status="BLOCKED", message=f"Required stage {dependency} did not complete successfully.")
            emit(event)
            return WorkflowResult(status="BLOCKED", stages=[event], warnings=[event.message])
    if stage_id == "authorization":
        check = preflight(request)
        status = "SUCCEEDED" if check.ready else "BLOCKED"
        event = StageEvent(stage_id=stage_id, status=status, message="Source authorization passed." if check.ready else "Integration preflight blocked.", parameters=check.resolved_parameters)
        emit(event)
        return WorkflowResult(status=status, stages=[event], warnings=check.warnings if check.ready else check.blockers, state={"params": check.resolved_parameters} if check.ready else {})

    state = dict(upstream[dependencies[stage_id][0]].state)
    params = state["params"]
    output_dir = request.run_dir / "integration_serial_section_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    emit(StageEvent(stage_id=stage_id, status="RUNNING", message=f"Running integration stage {stage_id}.", parameters=params))
    try:
        if stage_id == "metadata-scale":
            he_path = Path(params["moving_he_path"])
            fluor_path = Path(params["reference_fluorescence_path"])
            he_meta = _native_metadata(he_path)
            fluor_meta = _native_metadata(fluor_path)
            if he_meta["pixel_type"] != "bgr24":
                raise ValueError(f"H&E source must be a native BGR24 CZI preview, got {he_meta['pixel_type']}.")
            if fluor_meta["channels"][params["reference_channel_index"]] != params["reference_channel_name"]:
                raise ValueError(
                    "Chosen reference channel name does not exactly match CZI metadata at the selected index: "
                    f"C={params['reference_channel_index']} metadata={fluor_meta['channels']}"
                )
            for label, metadata in (("H&E", he_meta), ("reference", fluor_meta)):
                if set(metadata["pixel_size_um"]) != {"X", "Y"}:
                    raise ValueError(f"{label} CZI requires physical X and Y pixel-size metadata; found {metadata['pixel_size_um']}.")
                x_um, y_um = metadata["pixel_size_um"]["X"], metadata["pixel_size_um"]["Y"]
                if not math.isclose(x_um, y_um, rel_tol=0.01):
                    raise ValueError(f"{label} CZI has anisotropic physical pixel size ({x_um}, {y_um}) um; scalar registration scale is not justified.")
            metadata_path = output_dir / "native_metadata.json"
            metadata_path.write_text(json.dumps({"he": he_meta, "fluor": fluor_meta}, indent=2) + "\n", encoding="utf-8")
            state["metadata_path"] = str(metadata_path)
            artifacts = [str(metadata_path)]
        elif stage_id == "bounded-previews":
            metadata = json.loads(Path(state["metadata_path"]).read_text(encoding="utf-8"))
            he_meta, fluor_meta = metadata["he"], metadata["fluor"]
            he_path, fluor_path = Path(params["moving_he_path"]), Path(params["reference_fluorescence_path"])
            he, he_downsample, he_axes = _read_bounded(he_path, he_meta, channel=None)
            fluor, fluor_downsample, fluor_axes = _read_bounded(fluor_path, fluor_meta, channel=params["reference_channel_index"])
            he_pitch = (he_meta["pixel_size_um"]["X"] + he_meta["pixel_size_um"]["Y"]) / 2
            fluor_pitch = (fluor_meta["pixel_size_um"]["X"] + fluor_meta["pixel_size_um"]["Y"]) / 2
            # Convert source-pixel pitch to the actual bounded-preview pixel pitch before registration.
            he_scale_to_fluor = (he_pitch / he_downsample) / (fluor_pitch / fluor_downsample)
            if not math.isfinite(he_scale_to_fluor) or he_scale_to_fluor <= 0:
                raise ValueError(f"Computed non-positive/non-finite preview scale relation: {he_scale_to_fluor}.")
            he_fov_reference_px = {
                "width": he.shape[1] * he_scale_to_fluor,
                "height": he.shape[0] * he_scale_to_fluor,
            }
            fov_composition = {
                "he_preview_field_of_view_in_reference_pixels": he_fov_reference_px,
                "reference_preview_field_of_view_pixels": {"width": fluor.shape[1], "height": fluor.shape[0]},
                "he_fov_cropped_at_right_or_bottom_px": {
                    "width": max(0.0, he_fov_reference_px["width"] - fluor.shape[1]),
                    "height": max(0.0, he_fov_reference_px["height"] - fluor.shape[0]),
                },
                "reference_canvas_padding_at_right_or_bottom_px": {
                    "width": max(0.0, fluor.shape[1] - he_fov_reference_px["width"]),
                    "height": max(0.0, fluor.shape[0] - he_fov_reference_px["height"]),
                },
                "method": "Direct inverse-affine sampling from bounded H&E preview into bounded reference grid; no oversized scaled intermediate is allocated.",
            }
            he_canvas = _physical_he_canvas(he, fluor.shape, he_scale_to_fluor)
            source_inventory = {
                "workflow_id": _WORKFLOW_ID,
                "linkage": params["linkage"],
                "candidate_sources_confirmed": params["candidate_sources_confirmed"],
                "acquisition_group_mapping": "UNMAPPED",
                "moving_he": {**he_meta, "bounded_preview_shape_yxc": list(he.shape), "reader_axes": he_axes, "preview_scale_factor": he_downsample, "reference_grid_canvas_shape_yxc": list(he_canvas.shape)},
                "reference_fluorescence": {**fluor_meta, "selected_channel_index": params["reference_channel_index"], "selected_channel_name": params["reference_channel_name"], "bounded_preview_shape_yx": list(fluor.shape), "reader_axes": fluor_axes, "preview_scale_factor": fluor_downsample},
                "scale_relation": {
                    "he_to_fluor_preview_pixel_resolution": he_scale_to_fluor,
                    "formula": "(H&E native um/px / H&E preview factor) / (reference native um/px / reference preview factor)",
                    "orientation": "native CZI X/Y axes as returned by the CZI reader; no rotations/reflections are inferred",
                    "reference_fov_composition": fov_composition,
                    "transform_coordinates": "translation in bounded reference preview pixels after direct metadata-scale sampling onto the reference canvas",
                },
            }
            inventory_path = output_dir / "source_inventory_and_native_metadata.json"
            inventory_path.write_text(json.dumps(source_inventory, indent=2, default=str) + "\n", encoding="utf-8")
            he_preview_path = output_dir / "moving_he_source_preview.png"
            fluor_preview_path = output_dir / "reference_channel_source_preview.png"
            _save_source_preview(he, he_preview_path)
            _save_source_preview(fluor, fluor_preview_path)
            arrays_path = output_dir / "bounded_arrays.npz"
            np.savez(arrays_path, he_canvas=he_canvas, fluor=fluor)
            state.update({"arrays_path": str(arrays_path), "inventory_path": str(inventory_path), "he_preview_path": str(he_preview_path), "fluor_preview_path": str(fluor_preview_path), "he_scale_to_fluor": he_scale_to_fluor, "fluor_downsample": fluor_downsample, "fov_composition": fov_composition})
            artifacts = [str(inventory_path), str(he_preview_path), str(fluor_preview_path)]
        elif stage_id == "translation-registration":
            from spatial_omics_toolkit.cycif.register import register_he_to_fluorescence
            with np.load(state["arrays_path"], allow_pickle=False) as arrays:
                he_canvas, fluor = arrays["he_canvas"], arrays["fluor"]
            result = register_he_to_fluorescence(he_canvas, fluor, he_scale_to_fluor=1.0, search_radius=params["search_radius"])
            warped = result["warped_he"]
            if warped.shape[:2] != fluor.shape[:2]:
                raise ValueError(f"Canonical registration returned warped H&E shape {warped.shape}, reference shape {fluor.shape}.")
            registration_path = output_dir / "registration_arrays.npz"
            np.savez(registration_path, warped=warped, tx=float(result["tx"]), ty=float(result["ty"]), correlation=float(result["correlation_score"]))
            state.update({"registration_path": str(registration_path), "method_status": result["status"]})
            artifacts = [str(registration_path)]
        else:
            with np.load(state["arrays_path"], allow_pickle=False) as arrays:
                he_canvas, fluor = arrays["he_canvas"], arrays["fluor"]
            with np.load(state["registration_path"], allow_pickle=False) as registration:
                warped = registration["warped"]
                result = {"tx": float(registration["tx"]), "ty": float(registration["ty"]), "correlation_score": float(registration["correlation"]), "status": state["method_status"]}
            inventory_path, he_preview_path, fluor_preview_path = (Path(state[key]) for key in ("inventory_path", "he_preview_path", "fluor_preview_path"))
            he_scale_to_fluor, fluor_downsample, fov_composition = state["he_scale_to_fluor"], state["fluor_downsample"], state["fov_composition"]
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from spatial_omics_toolkit.application.plotting import ensure_arial
            ensure_arial()

            plt.rcParams.update({"font.family": "Arial", "font.size": 5, "font.weight": "normal", "axes.labelsize": 5, "axes.titlesize": 5, "xtick.labelsize": 5, "ytick.labelsize": 5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5, "lines.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
            fluor_display = np.asarray(fluor, dtype=np.float32)
            low, high = np.percentile(fluor_display, (1, 99))
            fluor_display = np.clip((fluor_display - low) / (high - low), 0, 1) if high > low else np.zeros_like(fluor_display)
            figure, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 8 / 2.54), dpi=150)
            axes[0].imshow(fluor_display, cmap="gray", interpolation="nearest")
            axes[0].imshow(_bounded_rgb(he_canvas), alpha=0.38, interpolation="nearest")
            axes[0].set_xlabel("Reference X (preview px)")
            axes[0].set_ylabel("Reference Y (preview px)")
            axes[1].imshow(fluor_display, cmap="gray", interpolation="nearest")
            axes[1].imshow(warped, alpha=0.38, interpolation="nearest")
            axes[1].set_xlabel("Reference X (preview px)")
            axes[1].set_ylabel("Reference Y (preview px)")
            for axis in axes:
                axis.tick_params(width=0.5, labelsize=5)
            figure.text(0.28, 0.94, "Before translation", ha="center", va="center", fontsize=5)
            figure.text(0.73, 0.94, "After translation", ha="center", va="center", fontsize=5)
            figure.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.88, wspace=0.16)
            overlay_pdf = output_dir / "before_after_alignment_overlay.pdf"
            overlay_png = output_dir / "before_after_alignment_overlay.png"
            figure.savefig(overlay_pdf, dpi=300)
            figure.savefig(overlay_png, dpi=150)
            plt.close(figure)

            tx, ty = float(result["tx"]), float(result["ty"])
            correlation = float(result["correlation_score"])
            metric_diagnostic = not math.isfinite(correlation) or correlation <= 0.0
            result_invalid = result["status"] != "ok" or not all(math.isfinite(value) for value in (tx, ty))
            tx_record = tx if math.isfinite(tx) else None
            ty_record = ty if math.isfinite(ty) else None
            correlation_record = correlation if math.isfinite(correlation) else None
            qc = {
                "status": "WARNING",
                "method_status": result["status"],
                "alignment_accepted": False,
                "review_required": True,
                "method": "canonical register_he_to_fluorescence; translation-only exploratory serial-section alignment",
                "translation_reference_preview_pixels": {"tx": tx_record, "ty": ty_record},
                "translation_native_reference_pixel_equivalent": {
                    "tx": tx / fluor_downsample if math.isfinite(tx) else None,
                    "ty": ty / fluor_downsample if math.isfinite(ty) else None,
                },
                "correlation_score": correlation_record,
                "reference_fov_composition": fov_composition,
                "nonpositive_or_nonfinite_metric": metric_diagnostic,
                "interpretation": "Metric is descriptive only; no acceptance threshold is defined. Alignment is not accepted; human visual review is required regardless of score.",
                "limitations": ["No same-cell registration", "No biological identity or pairing claim", "No non-rigid correction", "Translation coordinates are in bounded preview/reference image axes"],
                "source_inventory": str(inventory_path),
                "overlays": {"pdf": str(overlay_pdf), "png": str(overlay_png)},
                "source_previews": {"moving_he": str(he_preview_path), "reference_channel": str(fluor_preview_path)},
            }
            qc_path = output_dir / "alignment_qc_report.json"
            qc_path.write_text(json.dumps(qc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            artifacts = [str(inventory_path), str(he_preview_path), str(fluor_preview_path), str(overlay_pdf), str(overlay_png), str(qc_path)]
            status = "WARNING"
            warnings = ["Translation-only exploratory serial-section alignment; inspect the saved overlay visually. No same-cell registration or biological identity is established."]
            if metric_diagnostic or result_invalid:
                warnings.append("The metric is non-positive/non-finite or the method result is invalid; human visual review is required, and alignment is not accepted.")
            event = StageEvent(stage_id=stage_id, status=status, message="Alignment outputs and descriptive metrics were generated from the selected raw CZI sources; human visual review is required.", parameters={**params, "he_to_fluor_preview_scale": he_scale_to_fluor, "tx": tx_record, "ty": ty_record, "correlation_score": correlation_record, "nonpositive_or_nonfinite_metric": metric_diagnostic, "alignment_accepted": False, "review_required": True}, artifacts=artifacts)
            emit(event)
            return WorkflowResult(status=status, stages=[event], artifacts=artifacts, warnings=warnings, state=state, review_request={"required": True, "reason": "Inspect source previews and before/after overlay; this exploratory translation is not validated biological registration."})
        event = StageEvent(stage_id=stage_id, status="SUCCEEDED", message=f"Integration stage {stage_id} completed.", artifacts=artifacts)
        emit(event)
        return WorkflowResult(status="SUCCEEDED", stages=[event], artifacts=artifacts, state=state)
    except Exception as error:
        event = StageEvent(stage_id=stage_id, status="FAILED", message=f"Integration execution failed: {type(error).__name__}: {error}", parameters=params)
        emit(event)
        return WorkflowResult(status="FAILED", stages=[event], warnings=[event.message])
