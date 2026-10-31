"""PLA microscopy preprocessing with autofluorescence correction and visual QC."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, segmentation


@dataclass(frozen=True)
class PLAProcessingConfig:
    """Parameters for one CZI PLA preprocessing run."""

    dapi_channel: int = 0
    autofluorescence_channel: int = 1
    pla_channel: int = 2
    dapi_z: int | None = None
    scene_indices: tuple[int, ...] | None = None
    mosaic_scale: float = 1.0
    percentile_sample_stride: int = 8
    tissue_percentile: float = 20.0
    regression_percentile: float = 85.0
    pla_threshold_percentile: float = 99.8
    spot_min_size: int = 2
    spot_max_size: int = 250
    dapi_low_percentile: float = 1.0
    dapi_high_percentile: float = 99.8
    dapi_unsharp_sigma: float = 2.0
    dapi_unsharp_amount: float = 1.5
    dot_display_boost: float = 6.0
    dot_display_dilation: int = 0
    preview_downsample: int = 8

    def __post_init__(self) -> None:
        if not 0 < self.mosaic_scale <= 1:
            raise ValueError("mosaic_scale must be in (0, 1]")
        if self.percentile_sample_stride < 1 or self.preview_downsample < 1:
            raise ValueError("sampling strides must be positive")
        if not 0 < self.regression_percentile < 100:
            raise ValueError("regression_percentile must be in (0, 100)")
        if not 0 < self.pla_threshold_percentile < 100:
            raise ValueError("pla_threshold_percentile must be in (0, 100)")
        if self.dot_display_boost <= 0 or self.dot_display_dilation < 0:
            raise ValueError("invalid dot display enhancement parameters")
        if self.spot_min_size < 1 or self.spot_max_size < self.spot_min_size:
            raise ValueError("invalid spot size bounds")


def _masked_percentile(
    image: np.ndarray, percentile: float, mask: np.ndarray | None = None
) -> float:
    values = image[mask] if mask is not None else image.ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0
    return float(np.percentile(values, percentile))


def estimate_af_scale(
    autofluorescence: np.ndarray,
    pla: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    regression_percentile: float = 85.0,
) -> tuple[float, float]:
    """Robustly fit ``PLA = intercept + scale * AF`` on non-spot pixels."""
    af = np.asarray(autofluorescence, dtype=np.float32)
    signal = np.asarray(pla, dtype=np.float32)
    valid = np.isfinite(af) & np.isfinite(signal)
    if tissue_mask is not None:
        valid &= tissue_mask
    if not np.any(valid):
        return 0.0, 0.0
    cutoff = _masked_percentile(signal, regression_percentile, valid)
    background = valid & (signal <= cutoff)
    x = af[background].astype(np.float64)
    y = signal[background].astype(np.float64)
    if x.size < 2 or np.ptp(x) == 0:
        return 0.0, float(np.median(y)) if y.size else 0.0
    x0 = float(np.median(x))
    y0 = float(np.median(y))
    denom = float(np.dot(x - x0, x - x0))
    scale = max(0.0, float(np.dot(x - x0, y - y0) / denom)) if denom else 0.0
    intercept = y0 - scale * x0
    return scale, intercept


def subtract_autofluorescence(
    autofluorescence: np.ndarray,
    pla: np.ndarray,
    scale: float,
    intercept: float,
) -> np.ndarray:
    corrected = np.asarray(pla, dtype=np.float32) - (
        float(intercept) + float(scale) * np.asarray(autofluorescence, dtype=np.float32)
    )
    return np.maximum(corrected, 0.0)


def estimate_local_background_stack(
    autofluorescence_stack: np.ndarray,
    pla_stack: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    method: str = "morphological_opening",
    background_radius: int = 12,
    gaussian_sigma: float = 12.0,
    use_af488: bool = True,
    regression_percentile: float = 85.0,
    sample_stride: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate smooth background independently on every AF647 Z-plane.

    AF488 supplies a structured-background predictor. A broad local estimator then
    models the remaining slowly varying AF647 baseline without preserving compact
    PLA-shaped peaks.
    """
    af = np.asarray(autofluorescence_stack, dtype=np.float32)
    pla = np.asarray(pla_stack, dtype=np.float32)
    mask = np.asarray(tissue_mask, dtype=bool)
    if af.shape != pla.shape or af.ndim != 3:
        raise ValueError("AF488 and AF647 stacks must have matching (Z, Y, X) shapes")
    if mask.shape != pla.shape[1:]:
        raise ValueError("tissue_mask must match the stack YX shape")
    if method not in {"morphological_opening", "gaussian"}:
        raise ValueError("method must be 'morphological_opening' or 'gaussian'")
    if background_radius < 1 or gaussian_sigma <= 0 or sample_stride < 1:
        raise ValueError("background scale and sampling stride must be positive")

    sample_mask = np.broadcast_to(
        mask[::sample_stride, ::sample_stride],
        (pla.shape[0], *mask[::sample_stride, ::sample_stride].shape),
    ).reshape(-1)
    if use_af488:
        af_scale, af_intercept = estimate_af_scale(
            af[:, ::sample_stride, ::sample_stride].reshape(-1),
            pla[:, ::sample_stride, ::sample_stride].reshape(-1),
            sample_mask,
            regression_percentile,
        )
        structured = np.maximum(af_intercept + af_scale * af, 0.0)
    else:
        af_scale, af_intercept = 0.0, 0.0
        structured = np.zeros_like(pla)

    unexplained = np.maximum(pla - structured, 0.0)
    local = np.empty_like(unexplained)
    footprint = morphology.disk(background_radius)
    for z_index, plane in enumerate(unexplained):
        if method == "morphological_opening":
            local[z_index] = morphology.opening(plane, footprint)
        else:
            local[z_index] = ndi.gaussian_filter(plane, gaussian_sigma)

    background = np.minimum(structured + local, pla)
    corrected = np.maximum(pla - background, 0.0)
    background[:, ~mask] = 0.0
    corrected[:, ~mask] = 0.0
    metrics = {
        "method": method,
        "background_radius": int(background_radius),
        "gaussian_sigma": float(gaussian_sigma),
        "use_af488": bool(use_af488),
        "af_scale": float(af_scale),
        "af_intercept": float(af_intercept),
        "regression_percentile": float(regression_percentile),
        "sample_stride": int(sample_stride),
    }
    return background, corrected, metrics


def detect_psf_peaks(
    corrected_image: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    min_sigma: float = 0.8,
    max_sigma: float = 2.5,
    num_sigma: int = 8,
    threshold: float = 0.02,
    overlap: float = 0.5,
) -> np.ndarray:
    """Detect compact approximately Gaussian peaks with Laplacian-of-Gaussian."""
    image = np.asarray(corrected_image, dtype=np.float32)
    mask = np.asarray(tissue_mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError("corrected_image and tissue_mask must have matching shapes")
    positive = image[mask & (image > 0)]
    if not positive.size:
        return np.empty((0, 3), dtype=np.float64)
    low, high = np.percentile(positive, (1.0, 99.9))
    if high <= low:
        return np.empty((0, 3), dtype=np.float64)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    normalized[~mask] = 0.0
    return feature.blob_log(
        normalized,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=True,
    )


def detect_pla_spots(
    corrected_mip: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    threshold_percentile: float = 99.8,
    min_size: int = 2,
    max_size: int = 250,
) -> tuple[np.ndarray, float]:
    """Threshold the corrected MIP and retain compact connected components."""
    image = np.asarray(corrected_mip, dtype=np.float32)
    positive_mask = image > 0
    if tissue_mask is not None:
        positive_mask &= tissue_mask
    threshold = _masked_percentile(image, threshold_percentile, positive_mask)
    mask = (image >= threshold) & positive_mask
    if tissue_mask is not None:
        mask &= tissue_mask
    labels = measure.label(mask)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for region in measure.regionprops(labels):
        if min_size <= region.area <= max_size:
            keep[region.label] = True
    filtered = keep[labels]
    return measure.label(filtered).astype(np.uint32), threshold


def enhance_dapi(
    dapi: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
    unsharp_sigma: float = 2.0,
    unsharp_amount: float = 1.5,
) -> np.ndarray:
    """Create a bright, contrast-stretched, unsharp-masked uint16 DAPI image."""
    image = np.asarray(dapi, dtype=np.float32)
    nonzero = image[image > 0]
    if not nonzero.size:
        return np.zeros(image.shape, dtype=np.uint16)
    lo, hi = np.percentile(nonzero, (low_percentile, high_percentile))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint16)
    normalized = exposure.rescale_intensity(
        image, in_range=(lo, hi), out_range=(0.0, 1.0)
    )
    blurred = ndi.gaussian_filter(normalized, unsharp_sigma)
    sharpened = np.clip(normalized + unsharp_amount * (normalized - blurred), 0.0, 1.0)
    return np.round(sharpened * np.iinfo(np.uint16).max).astype(np.uint16)


def segment_dapi_cells(
    enhanced_dapi: np.ndarray,
    threshold_scale: float = 1.0,
    min_nucleus_area: int = 40,
    fill_holes: bool = True,
    distance_smoothing_sigma: float = 1.0,
    min_peak_distance: int = 5,
    min_peak_height: float = 2.0,
) -> np.ndarray:
    """Segment nuclei with explicitly tunable threshold and watershed controls."""
    if threshold_scale <= 0:
        raise ValueError("threshold_scale must be positive")
    if min_nucleus_area < 1 or min_peak_distance < 1:
        raise ValueError("area and peak-distance parameters must be positive")
    if distance_smoothing_sigma < 0 or min_peak_height < 0:
        raise ValueError("smoothing and peak-height parameters cannot be negative")
    image = np.asarray(enhanced_dapi)
    positive = image[image > 0]
    if not positive.size:
        return np.zeros(image.shape, dtype=np.uint32)
    threshold = filters.threshold_otsu(positive) * threshold_scale
    mask = morphology.remove_small_objects(
        image > threshold,
        max_size=max(0, min_nucleus_area - 1),
    )
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    distance = ndi.distance_transform_edt(mask)
    smoothed = ndi.gaussian_filter(distance, distance_smoothing_sigma)
    peak_values = ndi.maximum_filter(smoothed, size=2 * min_peak_distance + 1)
    peaks = (smoothed == peak_values) & (smoothed >= min_peak_height) & mask
    markers = measure.label(peaks)
    if not markers.max():
        return measure.label(mask).astype(np.uint32)
    return segmentation.watershed(-smoothed, markers, mask=mask).astype(np.uint32)


def _scene_indices(czi, requested: Sequence[int] | None) -> tuple[int, ...]:
    available = tuple(range(len(czi.get_dims_shape())))
    if requested is None:
        return available
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(f"scene indices outside CZI bounds: {invalid}")
    return tuple(requested)


def _channel_names(czi) -> list[str]:
    nodes = czi.meta.findall(".//Information/Image/Dimensions/Channels/Channel")
    names = [node.get("Name") or f"Ch{i}" for i, node in enumerate(nodes)]
    return names[: czi.get_dims_shape()[0]["C"][1]]


def _available_z(czi, scene: int, channel: int) -> tuple[int, ...]:
    coords = {
        int(info.dimension_coordinates["Z"])
        for info in czi.get_all_mosaic_tile_bounding_boxes(S=scene)
        if int(info.dimension_coordinates["C"]) == channel
    }
    if not coords:
        raise ValueError(f"scene {scene} channel {channel} has no readable planes")
    return tuple(sorted(coords))


def _read_mosaic_plane(
    czi, scene: int, channel: int, z: int, scale: float
) -> np.ndarray:
    box = czi.get_mosaic_scene_bounding_box(scene)
    plane = czi.read_mosaic(
        C=channel,
        Z=z,
        scale_factor=scale,
        region=(box.x, box.y, box.w, box.h),
    )
    return np.squeeze(plane).astype(np.float32, copy=False)


def _tissue_mask(autofluorescence_mip: np.ndarray, percentile: float) -> np.ndarray:
    positive = autofluorescence_mip[autofluorescence_mip > 0]
    if not positive.size:
        return np.zeros(autofluorescence_mip.shape, dtype=bool)
    threshold = float(np.percentile(positive, percentile))
    return autofluorescence_mip > threshold


def _display_uint8(
    image: np.ndarray, low: float = 1.0, high: float = 99.8
) -> np.ndarray:
    values = image[image > 0]
    if not values.size:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(values, (low, high))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = exposure.rescale_intensity(image, in_range=(lo, hi), out_range=(0, 255))
    return scaled.astype(np.uint8)


def inspect_pla_czi(czi_path: str | Path) -> dict:
    """Return channel, scene, and Z-plane metadata without loading image pixels."""
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise RuntimeError(
            "Install the imaging extra: pip install -e '.[imaging]'"
        ) from exc
    path = Path(czi_path)
    czi = CziFile(path)
    channels = _channel_names(czi)
    scenes = []
    for scene in range(len(czi.get_dims_shape())):
        scenes.append(
            {
                "scene": scene,
                "bounding_box": tuple(
                    getattr(czi.get_mosaic_scene_bounding_box(scene), key)
                    for key in ("x", "y", "w", "h")
                ),
                "z_by_channel": {
                    name: list(_available_z(czi, scene, index))
                    for index, name in enumerate(channels)
                },
            }
        )
    return {"path": str(path.resolve()), "channels": channels, "scenes": scenes}


def extract_czi_acquisition_metadata(czi_path: str | Path) -> list[dict]:
    """Extract intensity-critical channel acquisition settings from one CZI."""
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise RuntimeError(
            "Install the imaging extra: pip install -e '.[imaging]'"
        ) from exc
    path = Path(czi_path)
    czi = CziFile(path)
    rows = []
    channel_nodes = czi.meta.findall(".//Information/Image/Dimensions/Channels/Channel")
    for index, node in enumerate(channel_nodes[: czi.get_dims_shape()[0]["C"][1]]):
        exposure_ns = _xml_number(node, "ExposureTime")
        intensity_text = node.findtext(
            "./LightSourcesSettings/LightSourceSettings/Intensity"
        )
        rows.append(
            {
                "file": path.name,
                "path": str(path.resolve()),
                "channel_index": index,
                "channel_name": node.get("Name") or f"Ch{index}",
                "exposure_ms": (
                    exposure_ns / 1_000_000.0 if exposure_ns is not None else None
                ),
                "illumination_percent": _leading_number(intensity_text),
                "binning": node.findtext("./DetectorSettings/Binning"),
                "detector_id": _xml_ref_id(node, "./DetectorSettings/Detector"),
                "component_bit_count": _xml_int(node, "ComponentBitCount"),
                "pixel_type": node.findtext("PixelType"),
                "acquisition_mode": node.findtext("AcquisitionMode"),
                "illumination_type": node.findtext("IlluminationType"),
                "excitation_nm": _xml_number(node, "ExcitationWavelength"),
                "emission_nm": _xml_number(node, "EmissionWavelength"),
                "filter_set_id": _xml_ref_id(node, "FilterSetRef"),
                "light_source_id": _xml_ref_id(
                    node, "./LightSourcesSettings/LightSourceSettings/LightSource"
                ),
                "fluor": node.findtext("Fluor"),
            }
        )
    return rows


def audit_czi_acquisition_consistency(paths: Iterable[str | Path]) -> dict:
    """Compare intensity-critical CZI settings within each channel across files."""
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("No CZI paths supplied")
    rows = [row for path in files for row in extract_czi_acquisition_metadata(path)]
    critical = (
        "exposure_ms",
        "illumination_percent",
        "binning",
        "detector_id",
        "component_bit_count",
        "pixel_type",
        "acquisition_mode",
        "excitation_nm",
        "emission_nm",
        "filter_set_id",
        "light_source_id",
    )
    channels = sorted({row["channel_name"] for row in rows})
    comparisons = []
    for channel in channels:
        channel_rows = [row for row in rows if row["channel_name"] == channel]
        for setting in critical:
            values = [row[setting] for row in channel_rows]
            normalized = {_stable_metadata_value(value) for value in values}
            comparisons.append(
                {
                    "channel_name": channel,
                    "setting": setting,
                    "consistent": len(normalized) == 1 and None not in normalized,
                    "unique_values": sorted(
                        (value for value in normalized if value is not None),
                        key=str,
                    ),
                    "missing_count": sum(value is None for value in values),
                }
            )
    expected_rows = len(files) * len(channels)
    complete_matrix = len(rows) == expected_rows
    comparable = complete_matrix and all(item["consistent"] for item in comparisons)
    return {
        "file_count": len(files),
        "channel_count": len(channels),
        "complete_file_channel_matrix": complete_matrix,
        "raw_intensity_comparable": comparable,
        "rows": rows,
        "comparisons": comparisons,
    }


def _xml_number(node, path: str) -> float | None:
    text = node.findtext(path)
    return float(text) if text not in (None, "") else None


def _xml_int(node, path: str) -> int | None:
    value = _xml_number(node, path)
    return int(value) if value is not None else None


def _xml_ref_id(node, path: str) -> str | None:
    child = node.find(path)
    return child.get("Id") if child is not None else None


def _leading_number(text: str | None) -> float | None:
    if text is None:
        return None
    token = text.strip().split()[0]
    return float(token) if token else None


def _stable_metadata_value(value):
    return round(value, 9) if isinstance(value, float) else value


def load_pla_scene(
    czi_path: str | Path,
    scene: int,
    config: PLAProcessingConfig = PLAProcessingConfig(),
) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Load one DAPI plane and matched AF/PLA Z stacks from a mosaic CZI scene."""
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise RuntimeError(
            "Install the imaging extra: pip install -e '.[imaging]'"
        ) from exc
    czi = CziFile(Path(czi_path))
    af_z = _available_z(czi, scene, config.autofluorescence_channel)
    pla_z = _available_z(czi, scene, config.pla_channel)
    common_z = tuple(sorted(set(af_z) & set(pla_z)))
    if not common_z:
        raise ValueError(f"scene {scene} has no common AF/PLA Z planes")
    dapi_zs = _available_z(czi, scene, config.dapi_channel)
    dapi_z = config.dapi_z if config.dapi_z is not None else dapi_zs[len(dapi_zs) // 2]
    if dapi_z not in dapi_zs:
        raise ValueError(
            f"scene {scene} DAPI Z={dapi_z} is unavailable; found {dapi_zs}"
        )
    return {
        "dapi_raw": _read_mosaic_plane(
            czi, scene, config.dapi_channel, dapi_z, config.mosaic_scale
        ),
        "af_stack": np.stack(
            [
                _read_mosaic_plane(
                    czi, scene, config.autofluorescence_channel, z, config.mosaic_scale
                )
                for z in common_z
            ]
        ),
        "pla_stack": np.stack(
            [
                _read_mosaic_plane(
                    czi, scene, config.pla_channel, z, config.mosaic_scale
                )
                for z in common_z
            ]
        ),
        "dapi_z": dapi_z,
        "z_planes": common_z,
    }


def make_tissue_mask(
    autofluorescence_mip: np.ndarray,
    percentile: float = 20.0,
) -> np.ndarray:
    """Create a tissue mask from positive AF pixels for notebook inspection."""
    return _tissue_mask(autofluorescence_mip, percentile)


def enhance_dot_display(
    filtered_mip: np.ndarray,
    boost: float = 6.0,
    dilation: int = 0,
) -> np.ndarray:
    """Create a display-only dot image; quantitative pixels remain unchanged."""
    return _enhance_dot_display(filtered_mip, boost, dilation)


def _save_scene_qc(
    path: Path,
    dapi: np.ndarray,
    af_mip: np.ndarray,
    raw_mip: np.ndarray,
    corrected_mip: np.ndarray,
    spots: np.ndarray,
    threshold: float,
    scale: float,
    downsample: int,
) -> None:
    sl = np.s_[::downsample, ::downsample]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = (
        (dapi[sl], "gray", "Enhanced DAPI"),
        (af_mip[sl], "Greens", "AF488 MIP"),
        (raw_mip[sl], "Reds", "Raw AF647 MIP"),
        (corrected_mip[sl], "magma", "AF-corrected PLA MIP"),
        ((spots > 0)[sl], "gray", "Retained PLA dots"),
    )
    for ax, (image, cmap, title) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    overlay = np.zeros((*dapi[sl].shape, 3), dtype=np.float32)
    overlay[..., 2] = _display_uint8(dapi[sl]) / 255.0
    overlay[..., 0] = _display_uint8(corrected_mip[sl]) / 255.0
    axes.flat[5].imshow(overlay)
    axes.flat[5].set_title("DAPI + corrected PLA")
    axes.flat[5].axis("off")
    fig.suptitle(f"AF scale={scale:.4g}; PLA cutoff={threshold:.4g}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_profile_qc(
    path: Path,
    raw_values: np.ndarray,
    corrected_values: np.ndarray,
    threshold: float,
) -> None:
    raw = raw_values[np.isfinite(raw_values)]
    corrected = corrected_values[np.isfinite(corrected_values)]
    quantiles = np.linspace(0.0, 1.0, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        np.quantile(raw, quantiles), quantiles, label="raw AF647", linewidth=0.8
    )
    axes[0].plot(
        np.quantile(corrected, quantiles),
        quantiles,
        label="AF-corrected",
        linewidth=0.8,
    )
    axes[0].axvline(
        threshold, color="black", linestyle="--", linewidth=0.8, label="cutoff"
    )
    axes[0].set_xscale("symlog", linthresh=max(1.0, threshold / 100.0))
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Empirical cumulative fraction")
    axes[0].legend()
    positive = corrected[corrected > 0]
    axes[1].hist(positive, bins=128, log=True, color="#b91c1c")
    axes[1].axvline(threshold, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Corrected PLA intensity")
    axes[1].set_ylabel("Pixel count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _enhance_dot_display(
    filtered_mip: np.ndarray,
    boost: float,
    dilation: int,
) -> np.ndarray:
    """Create a display-only uint16 dot image without changing quantitative outputs."""
    display = _display_uint8(filtered_mip).astype(np.float32) / 255.0
    display = np.clip(display * boost, 0.0, 1.0)
    if dilation:
        display = morphology.dilation(display, morphology.disk(dilation))
    return np.round(display * np.iinfo(np.uint16).max).astype(np.uint16)


def process_pla_czi(
    czi_path: str | Path,
    output_dir: str | Path,
    config: PLAProcessingConfig = PLAProcessingConfig(),
) -> dict:
    """Process selected CZI scenes and save TIFFs, diagnostics, and provenance."""
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise RuntimeError(
            "Install the imaging extra: pip install -e '.[imaging]'"
        ) from exc

    input_path = Path(czi_path)
    root = Path(output_dir) / input_path.stem
    root.mkdir(parents=True, exist_ok=True)
    czi = CziFile(input_path)
    names = _channel_names(czi)
    scenes = _scene_indices(czi, config.scene_indices)
    results: list[dict] = []

    for scene in scenes:
        scene_dir = root / f"scene_{scene:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        af_z = _available_z(czi, scene, config.autofluorescence_channel)
        pla_z = _available_z(czi, scene, config.pla_channel)
        common_z = tuple(sorted(set(af_z) & set(pla_z)))
        if not common_z:
            raise ValueError(f"scene {scene} has no common AF/PLA Z planes")
        dapi_zs = _available_z(czi, scene, config.dapi_channel)
        dapi_z = (
            config.dapi_z if config.dapi_z is not None else dapi_zs[len(dapi_zs) // 2]
        )
        if dapi_z not in dapi_zs:
            raise ValueError(
                f"scene {scene} DAPI Z={dapi_z} is unavailable; found {dapi_zs}"
            )

        dapi_raw = _read_mosaic_plane(
            czi, scene, config.dapi_channel, dapi_z, config.mosaic_scale
        )
        enhanced = enhance_dapi(
            dapi_raw,
            config.dapi_low_percentile,
            config.dapi_high_percentile,
            config.dapi_unsharp_sigma,
            config.dapi_unsharp_amount,
        )
        nuclei = segment_dapi_cells(enhanced)

        af_planes = []
        raw_planes = []
        for z in common_z:
            af_planes.append(
                _read_mosaic_plane(
                    czi, scene, config.autofluorescence_channel, z, config.mosaic_scale
                )
            )
            raw_planes.append(
                _read_mosaic_plane(
                    czi, scene, config.pla_channel, z, config.mosaic_scale
                )
            )
        af_stack = np.stack(af_planes)
        raw_stack = np.stack(raw_planes)
        af_mip = np.max(af_stack, axis=0)
        raw_mip = np.max(raw_stack, axis=0)
        tissue = _tissue_mask(af_mip, config.tissue_percentile)

        stride = config.percentile_sample_stride
        sample_mask = tissue[::stride, ::stride]
        scale, intercept = estimate_af_scale(
            af_stack[:, ::stride, ::stride].reshape(-1),
            raw_stack[:, ::stride, ::stride].reshape(-1),
            np.broadcast_to(sample_mask, (len(common_z), *sample_mask.shape)).reshape(
                -1
            ),
            config.regression_percentile,
        )
        corrected_stack = subtract_autofluorescence(
            af_stack, raw_stack, scale, intercept
        )
        corrected_mip = np.max(corrected_stack, axis=0)
        spots, threshold = detect_pla_spots(
            corrected_mip,
            tissue,
            config.pla_threshold_percentile,
            config.spot_min_size,
            config.spot_max_size,
        )
        filtered_mip = np.where(spots > 0, corrected_mip, 0)
        display_dots = _enhance_dot_display(
            filtered_mip,
            config.dot_display_boost,
            config.dot_display_dilation,
        )
        display_composite = np.zeros((*enhanced.shape, 3), dtype=np.uint8)
        display_composite[..., 2] = _display_uint8(enhanced)
        dot_8bit = (display_dots / 257).astype(np.uint8)
        display_composite[..., 0] = dot_8bit
        display_composite[..., 2] = np.maximum(display_composite[..., 2], dot_8bit)

        tifffile.imwrite(
            scene_dir / "dapi_enhanced.ome.tif", enhanced, metadata={"axes": "YX"}
        )
        tifffile.imwrite(
            scene_dir / "nuclei_labels.ome.tif", nuclei, metadata={"axes": "YX"}
        )
        tifffile.imwrite(
            scene_dir / "af488_mip.ome.tif",
            af_mip.astype(np.uint16),
            metadata={"axes": "YX"},
        )
        tifffile.imwrite(
            scene_dir / "af647_raw_mip.ome.tif",
            raw_mip.astype(np.uint16),
            metadata={"axes": "YX"},
        )
        tifffile.imwrite(
            scene_dir / "pla_corrected_mip.ome.tif",
            corrected_mip.astype(np.float32),
            metadata={"axes": "YX"},
        )
        tifffile.imwrite(
            scene_dir / "pla_dots_filtered.ome.tif",
            filtered_mip.astype(np.float32),
            metadata={"axes": "YX"},
        )
        tifffile.imwrite(
            scene_dir / "pla_spot_labels.ome.tif", spots, metadata={"axes": "YX"}
        )
        tifffile.imwrite(
            scene_dir / "pla_dots_display.ome.tif",
            display_dots,
            metadata={"axes": "YX"},
        )
        tifffile.imwrite(scene_dir / "dapi_pla_display.png", display_composite)
        _save_scene_qc(
            scene_dir / "processing_qc.png",
            enhanced,
            af_mip,
            raw_mip,
            corrected_mip,
            spots,
            threshold,
            scale,
            config.preview_downsample,
        )
        sampled_raw = raw_mip[::stride, ::stride][sample_mask]
        sampled_corrected = corrected_mip[::stride, ::stride][sample_mask]
        _save_profile_qc(
            scene_dir / "intensity_profiles.png",
            sampled_raw,
            sampled_corrected,
            threshold,
        )

        metrics = {
            "scene": scene,
            "dapi_z": dapi_z,
            "z_planes": list(common_z),
            "af_scale": scale,
            "af_intercept": intercept,
            "pla_threshold": threshold,
            "spot_count": int(spots.max()),
            "tissue_pixels": int(tissue.sum()),
            "channel_names": names,
        }
        (scene_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        results.append(metrics)

    manifest = {
        "input": str(input_path.resolve()),
        "output": str(root.resolve()),
        "config": asdict(config),
        "scenes": results,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def process_pla_batch(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    config: PLAProcessingConfig = PLAProcessingConfig(),
) -> list[dict]:
    """Process multiple CZI files, failing loudly on the first unreadable input."""
    paths = [Path(path) for path in inputs]
    if not paths:
        raise ValueError("no CZI inputs supplied")
    return [process_pla_czi(path, output_dir, config) for path in paths]
