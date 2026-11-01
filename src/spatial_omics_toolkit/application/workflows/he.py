from __future__ import annotations

import csv
import importlib.metadata
import json
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

_IMAGE_SUFFIXES = {".svs", ".czi", ".tif", ".tiff"}


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and (
            path.suffix.lower() in _IMAGE_SUFFIXES
            or path.name.lower().endswith((".ome.tif", ".ome.tiff"))
        )
    )


def describe() -> WorkflowDescriptor:
    return WorkflowDescriptor(
        workflow_id="he-image-intake",
        modality="H&E",
        title="H&E source-preserving image intake and QC",
        description=(
            "Inventory supported H&E image files, record readable image metadata, "
            "and create bounded previews where a qualified reader is available. "
            "No biological pairing or image registration is inferred."
        ),
        stages=[
            StageDefinition(
                stage_id="image-intake",
                title="Image inventory and preview",
                purpose=(
                    "Record source files and metadata without modifying source images; "
                    "render reduced previews when supported."
                ),
                tools=[
                    ToolReference(
                        name="OpenSlide",
                        version=_version("openslide-python"),
                        role="Pyramidal SVS metadata and thumbnail reader",
                        documentation_url="https://openslide.org/api/python/",
                    ),
                    ToolReference(
                        name="aicspylibczi",
                        version=_version("aicspylibczi"),
                        role="CZI metadata reader",
                        documentation_url="https://github.com/AllenCellModeling/aicsimageio",
                    ),
                    ToolReference(
                        name="tifffile",
                        version=_version("tifffile"),
                        role="TIFF/OME-TIFF metadata reader",
                        documentation_url="https://github.com/cgohlke/tifffile",
                    ),
                ],
                implementations=[
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/he.py", function_name="_image_files", role="Discover supported source files"),
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/he.py", function_name="_inspect_image", role="Read image metadata and create per-image preview"),
                    FunctionReference(source_file="spatial_omics_toolkit/application/workflows/he.py", function_name="_render_czi_preview", role="Dispatch CZI preview rendering"),
                    FunctionReference(source_file="external:image_previews.py", function_name="czi_rgb_preview", role="Canonical CZI RGB preview rendering"),
                ],
                event_stage_ids=["workflow_output"],
            )
        ],
        input_description=(
            "User-selected H&E folder containing supported .svs, .czi, TIFF, "
            "or OME-TIFF images."
        ),
        output_description=(
            "Per-run JSON and CSV inventories, reader metadata and optional "
            "reduced preview images under run_dir."
        ),
        parameter_fields=[
            ParameterField(
                name="preview_max_dimension",
                label="Maximum preview dimension",
                kind="int",
                default=2048,
                min_value=64,
                max_value=8192,
                step=64,
                unit="pixels",
                section="Display",
                stage_id="image-intake",
                help_text="Upper bound for bounded SVS/CZI previews. This does not change source pixels; the canonical CZI JPEG helper may apply a smaller output cap.",
            ),
            ParameterField(
                name="czi_preview_scale_factor",
                label="CZI mosaic preview scale factor",
                kind="float",
                default=0.12,
                min_value=0.001,
                max_value=0.12,
                step=0.01,
                unit="fraction of source dimensions",
                section="Display",
                stage_id="image-intake",
                help_text="Display-only downsampling scale for the canonical CZI RGB mosaic preview; raw source data remain unchanged.",
            ),
        ],
        limitations=[
            "Inventory and image appearance do not establish biological pairing, same-section identity, or cross-modality alignment.",
            "This adapter does not register images; no qualified VALIS route is enabled here.",
            "Reader support is runtime-dependent; unreadable files retain inventory evidence and fail the workflow when no format metadata could be read.",
        ],
    )


def preflight(request: RunRequest) -> PreflightResult:
    blockers: list[str] = []
    if request.modality != "H&E":
        blockers.append(f"Expected modality 'H&E', received {request.modality!r}.")
    if not request.modality_dir.is_dir():
        blockers.append(f"H&E input folder does not exist: {request.modality_dir}")
    readers = {
        "openslide": _version("openslide-python"),
        "aicspylibczi": _version("aicspylibczi"),
        "tifffile": _version("tifffile"),
    }
    if not any(version != "not installed" for version in readers.values()):
        blockers.append("No supported image reader is installed (OpenSlide, aicspylibczi, or tifffile).")
    preview_dimension = request.parameters.get("preview_max_dimension", 2048)
    if isinstance(preview_dimension, bool) or not isinstance(preview_dimension, int) or not 64 <= preview_dimension <= 8192:
        blockers.append("preview_max_dimension must be an integer from 64 through 8192 pixels.")
        preview_dimension = 2048
    czi_scale = request.parameters.get("czi_preview_scale_factor", 0.12)
    if isinstance(czi_scale, bool) or not isinstance(czi_scale, (int, float)) or not 0.001 <= czi_scale <= 0.12:
        blockers.append("czi_preview_scale_factor must be between 0.001 and 0.12.")
        czi_scale = 0.12
    return PreflightResult(
        ready=not blockers,
        blockers=blockers,
        warnings=["File discovery and image metadata inspection run in the allocated worker task."],
        resolved_parameters={
            "readers": readers,
            "preview_max_dimension": preview_dimension,
            "czi_preview_scale_factor": float(czi_scale),
        },
    )


def _czi_preview_helper_path() -> Path:
    toolkit_root = Path(__file__).resolve().parents[4]
    return (
        toolkit_root.parent
        / "02_spatial_proteomics"
        / "03_analysis"
        / "01_exploratory_analysis"
        / "02_scripts"
        / "image_previews.py"
    )


def _render_czi_preview(path: Path, preview_path: Path, scale_factor: float) -> tuple[int, int]:
    import importlib.util
    import sys
    from io import BytesIO
    from PIL import Image

    helper_path = _czi_preview_helper_path()
    if not helper_path.is_file():
        raise FileNotFoundError(f"Canonical image preview helper is unavailable: {helper_path}")
    module_name = "spatial_toolkit_canonical_image_previews"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical preview helper at {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    jpeg_bytes = module.czi_rgb_preview(path, scale_factor=scale_factor)
    preview_path.write_bytes(jpeg_bytes)
    with Image.open(BytesIO(jpeg_bytes)) as image:
        return image.size


def _inspect_image(path: Path, preview_path: Path, preview_size: int, czi_scale_factor: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lower(),
        "metadata": {},
        "preview": None,
        "warnings": [],
    }
    suffix = path.suffix.lower()
    if suffix == ".svs":
        try:
            import openslide

            slide = openslide.OpenSlide(str(path))
            dimensions = list(slide.dimensions)
            thumbnail = slide.get_thumbnail((preview_size, preview_size))
            thumbnail.save(preview_path)
            result["metadata"] = {
                "dimensions_level0": dimensions,
                "level_dimensions": [list(level_dimensions) for level_dimensions in slide.level_dimensions],
                "level_count": slide.level_count,
                "properties": dict(slide.properties),
                "coordinate_system": "SVS level-0 pixel coordinates",
                "preview_scale_xy": [thumbnail.width / dimensions[0], thumbnail.height / dimensions[1]],
            }
            slide.close()
            result["preview"] = str(preview_path)
        except Exception as exc:
            result["warnings"].append(f"OpenSlide could not read this SVS: {exc}")
    elif suffix == ".czi":
        try:
            from aicspylibczi import CziFile

            czi = CziFile(str(path))
            dimension_shapes = czi.get_dims_shape()
            first_shape = dimension_shapes[0] if dimension_shapes else {}
            x_shape = first_shape.get("X", (0, 0))
            y_shape = first_shape.get("Y", (0, 0))
            dimensions = [int(x_shape[1]), int(y_shape[1])]
            is_mosaic = czi.is_mosaic()
            physical_size_reader = getattr(czi, "get_physical_pixel_size", None)
            physical_pixel_size = physical_size_reader() if callable(physical_size_reader) else None
            result["metadata"] = {
                "dimensions": czi.get_dims(),
                "coordinate_system": "CZI pixel coordinates",
                "image_dimensions_xy": dimensions,
                "physical_pixel_size": physical_pixel_size,
                "is_mosaic": is_mosaic,
            }
            if not is_mosaic:
                raise ValueError("Canonical CZI RGB preview supports mosaic acquisitions only.")
            scale_factor = min(czi_scale_factor, preview_size / max(dimensions or [preview_size]))
            preview_dimensions = _render_czi_preview(path, preview_path, scale_factor)
            result["preview"] = str(preview_path)
            result["metadata"].update(
                {
                    "preview_scale_factor": scale_factor,
                    "preview_dimensions_xy": list(preview_dimensions),
                    "preview_source_bounds_xy": [0, 0, dimensions[0], dimensions[1]],
                    "display_transform": "canonical czi_rgb_preview; C=0 BGR24-to-RGB display only; no registration",
                }
            )
        except Exception as exc:
            result["warnings"].append(f"Bounded canonical CZI preview unavailable: {exc}")
    else:
        try:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                result["metadata"] = {
                    "series": [
                        {"shape": list(series.shape), "dtype": str(series.dtype), "axes": series.axes}
                        for series in tif.series
                    ],
                    "pages": len(tif.pages),
                }
            result["warnings"].append(
                "TIFF metadata recorded; preview omitted to avoid decoding a potentially whole-slide image into memory."
            )
        except Exception as exc:
            result["warnings"].append(f"TIFF metadata reader failed: {exc}")
    result["full_resolution_source"] = {
        "path": str(path),
        "size_bytes": result["size_bytes"],
        "browser_render": False,
    }
    if result["preview"] is not None:
        result["preview_artifact"] = {
            "path": result["preview"],
            "artifact_type": "bounded_image_preview",
            "data_status": "derived display-only image",
            "source_path": str(path),
            "scale_metadata": {
                key: value
                for key, value in result["metadata"].items()
                if "scale" in key or "coordinate" in key or "bounds" in key or key in {"physical_pixel_size", "properties"}
            },
        }
    return result


def execute(request: RunRequest, emit: WorkflowEmitter) -> WorkflowResult:
    check = preflight(request)
    if not check.ready:
        event = StageEvent(
            stage_id="image-intake",
            status="BLOCKED",
            message="H&E intake preflight blocked.",
            parameters=check.resolved_parameters,
        )
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=check.blockers)

    request.run_dir.mkdir(parents=True, exist_ok=True)
    output_dir = request.run_dir / "he_image_intake"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _image_files(request.modality_dir)
    if not files:
        event = StageEvent(
            stage_id="image-intake",
            status="BLOCKED",
            message="No supported H&E image files were found in the selected input folder.",
            tool_versions=check.resolved_parameters["readers"],
        )
        emit(event)
        return WorkflowResult(status="BLOCKED", stages=[event], warnings=[event.message])
    preview_size = int(check.resolved_parameters["preview_max_dimension"])
    czi_scale_factor = float(check.resolved_parameters["czi_preview_scale_factor"])
    readers = check.resolved_parameters["readers"]
    emit(
        StageEvent(
            stage_id="image-intake",
            status="RUNNING",
            message=f"Inspecting {len(files)} source image(s); source files are not modified.",
            tool_versions=readers,
            parameters={
                "preview_max_dimension": preview_size,
                "czi_preview_scale_factor": czi_scale_factor,
            },
        )
    )
    records = [
        _inspect_image(
            path,
            output_dir / f"{index:04d}_{path.stem[:80]}_preview.jpg",
            preview_size,
            czi_scale_factor,
        )
        for index, path in enumerate(files, start=1)
    ]
    inventory_json = output_dir / "image_inventory.json"
    inventory_csv = output_dir / "image_inventory.csv"
    inventory_json.write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    with inventory_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "size_bytes", "format", "metadata", "preview", "preview_artifact", "full_resolution_source", "warnings"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "metadata": json.dumps(record["metadata"], default=str),
                    "preview_artifact": json.dumps(record.get("preview_artifact"), default=str),
                    "full_resolution_source": json.dumps(record["full_resolution_source"], default=str),
                    "warnings": "; ".join(record["warnings"]),
                }
            )
    artifacts = [
        str(inventory_json),
        str(inventory_csv),
        *[str(record["preview"]) for record in records if record["preview"]],
    ]
    warnings = [warning for record in records for warning in record["warnings"]]
    failed = any(not record["metadata"] for record in records)
    if failed:
        warnings.append("One or more files could not be inspected with an installed format-specific reader; see the per-file inventory errors.")
    status = "FAILED" if failed else ("WARNING" if warnings else "SUCCEEDED")
    event = StageEvent(
        stage_id="image-intake",
        status=status,
        message=(
            f"Recorded inventory for {len(records)} input image(s); "
            "no pairing or alignment was inferred."
        ),
        tool_versions=readers,
        parameters={"image_count": len(files)},
        artifacts=artifacts,
    )
    emit(event)
    return WorkflowResult(status=status, stages=[event], artifacts=artifacts, warnings=warnings)
