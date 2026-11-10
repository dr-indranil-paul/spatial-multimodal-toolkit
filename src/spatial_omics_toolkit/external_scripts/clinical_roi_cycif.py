"""Clinical CycIF processing, QC, round registration, and continuous marker export.

Segmentation-free quantitative pipeline aligning cyclic immunofluorescence rounds
(R1-R5, R5Q) to R1 reference DAPI coordinates, verifying physical scale calibration,
producing common-grid continuous marker images with explicit validity/tissue support,
and recording complete cryptographic provenance receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import importlib
import importlib.metadata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def _toolkit_registration() -> Any:
    """Resolve the existing sibling toolkit for direct helper and notebook use."""
    source = Path(__file__).resolve().parents[4] / "02_spatial_toolkit" / "src"
    if not (source / "spatial_omics_toolkit").is_dir():
        raise FileNotFoundError(f"Required spatial toolkit source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module("spatial_omics_toolkit.cycif.register")


# ---------------------------------------------------------------------------
# Constants and Schema Specifications
# ---------------------------------------------------------------------------

PREVAIL_BARRETTS_CZI_FILENAME_PATTERN = r"^(R\d+Q?)_([A-Za-z0-9._-]+)_(hBE-TMA[12]-1)_(\d{4}_\d{2}_\d{2})__(\d+)-scene([A-Za-z0-9]+)\.czi$"

# Standard rounds in biological progression order
STANDARD_ROUNDS = ("R1", "R2", "R3", "R4", "R5", "R5Q")

# Biological marker rounds (R5Q is a background quench acquisition, never exported as an antibody marker)
BIOLOGICAL_ROUNDS = ("R1", "R2", "R3", "R4", "R5")

# Marker export table contract required by Notebook 02 and downstream consumers
CYCIF_MARKER_TABLE_COLUMNS = [
    "assigned_tma",
    "confirmed_core_id",
    "marker",
    "image_path",
    "support_path",
    "tissue_mask_path",
    "reference_image_path",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "source_path",
    "source_sha256",
    "receipt_path",
    "receipt_sha256",
    "registration_driver",
    "section_relationship",
    "section_evidence",
    "group_id",
    "round",
    "channel_idx",
    "exposure_ms",
    "correction_status",
    "background_source_path",
    "background_source_sha256",
    "background_exposure_ms",
    "background_scale",
    "raw_registered_image_path",
    "intensity_units",
    "reference_bbox_xywh",
    "saturated_native_pixels",
    "native_pixel_count",
]


# ---------------------------------------------------------------------------
# Filename & Path Parsing
# ---------------------------------------------------------------------------


def parse_cycif_czi_filename(
    filename: str,
    pattern: str = PREVAIL_BARRETTS_CZI_FILENAME_PATTERN,
) -> Optional[dict[str, str]]:
    """Parse a CycIF CZI filename into constituent acquisition metadata."""
    m = re.match(pattern, filename)
    if not m:
        return None
    round_label = m.group(1)
    markers_str = m.group(2)
    slide = m.group(3)
    date = m.group(4)
    acq_id = m.group(5)
    scene = m.group(6)

    tma = "TMA1" if "TMA1" in slide else ("TMA2" if "TMA2" in slide else slide)
    return {
        "round": round_label,
        "panel_markers": markers_str,
        "slide": slide,
        "tma": tma,
        "date": date,
        "acq_id": acq_id,
        "scene": scene,
        "core_id": scene,
    }


def parse_panel_markers(panel_str: str) -> list[str]:
    """Parse the biological antibody marker names declared in the filename panel segment."""
    return [m.strip() for m in panel_str.split(".") if m.strip()]


def round_sort_key(label: str) -> tuple[int, int]:
    """Sort key for biological CycIF rounds: R1 < R2 < R3 < R4 < R5 < R5Q."""
    m = re.match(r"^R(\d+)(Q?)$", label)
    if m:
        return (int(m.group(1)), 1 if m.group(2) == "Q" else 0)
    return (999, 0)


# ---------------------------------------------------------------------------
# Cryptographic Provenance & Hash Caching
# ---------------------------------------------------------------------------


def compute_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute complete SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_array_sha256(arr: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of an in-memory NumPy array."""
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(arr).data)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# CZI Metadata & Exposure Inspection
# ---------------------------------------------------------------------------


def extract_czi_metadata(czi_path: Union[str, Path]) -> dict[str, Any]:
    """Extract physical pixel sizes, channel definitions, and bounding boxes from CZI XML.

    Deduplicates channels by Id, checks child elements as well as attributes for Channel Name,
    and extracts DistanceX/DistanceY in micrometers.
    """
    path_obj = Path(czi_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CZI file does not exist: {path_obj}")

    import aicspylibczi

    czi = aicspylibczi.CziFile(str(path_obj))
    meta = czi.meta
    root = ET.fromstring(meta) if isinstance(meta, str) else meta

    # Physical pixel sizes (meters -> micrometers)
    pixel_size_x_um: Optional[float] = None
    pixel_size_y_um: Optional[float] = None
    for dist in root.iter():
        if dist.tag.split("}")[-1] == "Distance":
            dist_id = dist.get("Id")
            if dist_id not in {"X", "Y"}:
                continue
            val_elem = None
            for child in dist:
                if child.tag.split("}")[-1] == "Value":
                    val_elem = child
                    break
            if val_elem is not None and val_elem.text:
                try:
                    val_float = float(val_elem.text) * 1e6
                    if val_float <= 0 or not np.isfinite(val_float):
                        raise ValueError(
                            f"{path_obj.name}: non-positive or non-finite pixel size {val_float}"
                        )
                    if dist_id == "X":
                        if pixel_size_x_um is not None and not np.isclose(
                            pixel_size_x_um, val_float
                        ):
                            raise ValueError("Conflicting X physical pixel calibrations")
                        pixel_size_x_um = val_float
                    elif dist_id == "Y":
                        if pixel_size_y_um is not None and not np.isclose(
                            pixel_size_y_um, val_float
                        ):
                            raise ValueError("Conflicting Y physical pixel calibrations")
                        pixel_size_y_um = val_float
                except ValueError as e:
                    raise ValueError(f"{path_obj.name}: invalid Distance XML text: {e}") from e

    if pixel_size_x_um is None or pixel_size_y_um is None:
        raise ValueError(
            f"{path_obj.name}: missing physical pixel calibration (DistanceX or DistanceY)"
        )

    # Channel details deduplicated by channel Id
    seen_ids: set[str] = set()
    channels: list[dict[str, Any]] = []

    for ch in root.iter():
        if ch.tag.split("}")[-1] != "Channel":
            continue
        ch_id = ch.get("Id", "")
        if ch_id and ch_id in seen_ids:
            continue
        if ch_id:
            seen_ids.add(ch_id)

        # Name may be attribute or child element <Name>
        ch_name = ch.get("Name", "")
        fluor = ch.get("Fluor", "")
        for child in ch:
            child_tag = child.tag.split("}")[-1]
            if child_tag == "Name" and child.text and not ch_name:
                ch_name = child.text.strip()
            elif child_tag == "Fluor" and child.text and not fluor:
                fluor = child.text.strip()

        exposures_ms: list[float] = []
        for node in ch.iter():
            if node.tag.split("}")[-1] == "ExposureTime" and node.text:
                try:
                    v = float(node.text)
                    norm_v = v / 1_000_000.0 if v >= 1_000_000 else v
                    if norm_v > 0 and np.isfinite(norm_v):
                        exposures_ms.append(norm_v)
                except ValueError:
                    pass

        if exposures_ms:
            if not np.allclose(exposures_ms, exposures_ms[0]):
                raise ValueError(
                    f"{path_obj.name}: conflicting exposure times for channel {ch_id}: {exposures_ms}"
                )
            exp_ms: Optional[float] = exposures_ms[0]
        else:
            exp_ms = None

        channels.append(
            {
                "channel_id": ch_id,
                "name": ch_name,
                "fluor": fluor,
                "exposure_ms": exp_ms,
            }
        )

    # Prefer the numbered image-channel definitions. The same acquisition also
    # stores anonymous hardware-channel copies; those are corroborating metadata.
    numbered = [
        channel for channel in channels if re.fullmatch(r"Channel:\d+", channel["channel_id"])
    ]
    if numbered:
        numbered.sort(key=lambda channel: int(channel["channel_id"].split(":")[-1]))
        for channel in numbered:
            copies = [
                item["exposure_ms"]
                for item in channels
                if item["name"] == channel["name"] and item["exposure_ms"] is not None
            ]
            if copies and not np.allclose(copies, copies[0]):
                raise ValueError(f"Conflicting duplicated exposure metadata: {path_obj}")
        channels = numbered

    # Detect bounding box and dims
    try:
        bbox = czi.get_mosaic_bounding_box()
    except Exception:
        bbox = None

    dims_shape = czi.get_dims_shape()
    if len(dims_shape) != 1:
        raise ValueError("Use per-scene CZI inputs with one unambiguous dimension layout")
    axis_counts: dict[str, int] = {}
    for ax, (start, stop) in dims_shape[0].items():
        count = int(stop) - int(start)
        if count < 1 or (ax in {"S", "T", "Z"} and count != 1):
            raise ValueError(f"Unsupported multi-plane CZI axis {ax}: {(start, stop)}")
        if ax == "C" and start != 0:
            raise ValueError("Nonzero channel origin needs explicit channel mapping")
        axis_counts[ax] = count

    total_channels = axis_counts.get("C", len(channels))
    if len(channels) != total_channels:
        raise ValueError(f"{path_obj.name}: ambiguous image-channel metadata")
    if numbered and [channel["channel_id"] for channel in channels] != [
        f"Channel:{index}" for index in range(total_channels)
    ]:
        raise ValueError(f"{path_obj.name}: image-channel indices are not contiguous")
    if bbox is not None:
        bbox = (int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h))
        channel_tile_boxes = {
            str(index): [
                (int(box.x), int(box.y), int(box.w), int(box.h))
                for box in czi.get_all_mosaic_tile_bounding_boxes(C=index).values()
            ]
            for index in range(total_channels)
        }
        tile_boxes = channel_tile_boxes["0"]
        all_boxes = [box for boxes in channel_tile_boxes.values() for box in boxes]
        x0 = min(box[0] for box in all_boxes)
        y0 = min(box[1] for box in all_boxes)
        x1 = max(box[0] + box[2] for box in all_boxes)
        y1 = max(box[1] + box[3] for box in all_boxes)
        if x0 < bbox[0] or y0 < bbox[1] or x1 > bbox[0] + bbox[2] or y1 > bbox[1] + bbox[3]:
            raise ValueError("Acquired channel tiles extend outside the reported mosaic bounds")
    else:
        tile_boxes = []
        channel_tile_boxes = {}

    return {
        "pixel_size_x_um": pixel_size_x_um,
        "pixel_size_y_um": pixel_size_y_um,
        "channels": channels,
        "num_channels": total_channels,
        "bbox": bbox,
        "axis_counts": axis_counts,
        "tile_boxes_xywh": tile_boxes,
        "tile_boxes_by_channel": channel_tile_boxes,
    }


def audit_czi_panel_channels(
    czi_path: Union[str, Path],
    declared_panel_markers: list[str],
    metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Audit CZI channel metadata against declared panel markers.

    Verifies that Channel 0 is confirmed DAPI and that subsequent 4 channels
    correspond to declared biological markers, confirming exactly 5 total channels.
    """
    meta = metadata if metadata is not None else extract_czi_metadata(czi_path)
    channels = meta["channels"]
    num_channels = meta["num_channels"]

    if num_channels != 5:
        raise ValueError(
            f"{Path(czi_path).name}: expected exactly 5 acquisition channels (DAPI + 4 markers), found {num_channels}"
        )

    # Verify Channel 0 is confirmed DAPI (metadata text must explicitly contain DAPI)
    ch0 = channels[0] if channels else {}
    ch0_name = str(ch0.get("name", "")).upper()
    ch0_fluor = str(ch0.get("fluor", "")).upper()
    if "DAPI" not in ch0_name and "DAPI" not in ch0_fluor:
        raise ValueError(
            f"{Path(czi_path).name}: Channel 0 is not confirmed DAPI in metadata (name='{ch0.get('name')}', fluor='{ch0.get('fluor')}'). Refusing 405 assumption without metadata confirmation."
        )

    if len(declared_panel_markers) != 4:
        raise ValueError(
            f"{Path(czi_path).name}: declared panel must contain exactly 4 markers, found {len(declared_panel_markers)}: {declared_panel_markers}"
        )

    assigned: list[dict[str, Any]] = [
        {
            "channel_idx": 0,
            "marker": "DAPI",
            "channel_name": ch0.get("name", "DAPI"),
            "exposure_ms": ch0.get("exposure_ms"),
            "is_dapi": True,
        }
    ]

    for idx, marker in enumerate(declared_panel_markers, start=1):
        ch_info = channels[idx] if idx < len(channels) else {}
        assigned.append(
            {
                "channel_idx": idx,
                "marker": marker,
                "channel_name": ch_info.get("name", marker),
                "exposure_ms": ch_info.get("exposure_ms"),
                "is_dapi": False,
            }
        )

    return assigned


# ---------------------------------------------------------------------------
# Background Subtraction & Quench Correction
# ---------------------------------------------------------------------------


def channel_exposure_ms(czi_path: Union[str, Path], channel_name: str) -> float:
    """Read one acquisition exposure in milliseconds from CZI metadata."""
    meta = extract_czi_metadata(czi_path)
    for ch in meta["channels"]:
        if ch.get("name") == channel_name:
            exp = ch.get("exposure_ms")
            if exp is not None and exp > 0:
                return float(exp)
    if channel_name.isdigit():
        idx = int(channel_name)
        if idx < len(meta["channels"]):
            exp = meta["channels"][idx].get("exposure_ms")
            if exp is not None and exp > 0:
                return float(exp)
    raise RuntimeError(
        f"{Path(czi_path).name}: no positive exposure metadata found for channel '{channel_name}'"
    )


def backsub_correct(
    signal: np.ndarray,
    background: np.ndarray,
    signal_exposure_ms: float,
    background_exposure_ms: float,
) -> tuple[np.ndarray, float]:
    """Apply Backsub exposure-scaled background subtraction formula: max(0, sig - scale * bg)."""
    # Import from existing script cycif_zeb1_segment_quantify
    try:
        from cycif_zeb1_segment_quantify import backsub_correct as _backsub

        return _backsub(signal, background, signal_exposure_ms, background_exposure_ms)
    except ImportError:
        if signal_exposure_ms <= 0 or background_exposure_ms <= 0:
            raise ValueError("Backsub exposure values must be positive")
        exposure_scale = signal_exposure_ms / background_exposure_ms
        corrected = np.maximum(
            0.0,
            signal.astype(np.float32)
            - np.float32(exposure_scale) * background.astype(np.float32),
        )
        return corrected.astype(np.float32), float(exposure_scale)


# ---------------------------------------------------------------------------
# Tissue Mask & Support Masks
# ---------------------------------------------------------------------------


def compute_tissue_mask(
    image: np.ndarray, percentile: float = 60.0, closing_radius: int = 8
) -> np.ndarray:
    """Reuse the existing structural tissue mask; do not substitute another recipe."""
    from guided_serial_registration import tissue_mask

    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Tissue masking requires a finite 2D structural image")
    if not np.any(image > 0):
        return np.zeros(image.shape, dtype=bool)
    return tissue_mask(image, percentile=percentile, closing_radius=closing_radius).astype(bool)


# ---------------------------------------------------------------------------
# Core Inventory & Discovery
# ---------------------------------------------------------------------------


def inventory_cycif_sources(
    raw_root: Union[str, Path],
    eligible_cores: Optional[Sequence[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Scan raw CycIF CZI files under raw_root, parse names, and group rounds per core."""
    raw_path = Path(raw_root)
    if not raw_path.exists():
        raise FileNotFoundError(f"CycIF raw root not found: {raw_path}")

    records = []
    for czi in raw_path.glob("**/*.czi"):
        if not czi.is_file():
            continue
        parsed = parse_cycif_czi_filename(czi.name)
        if not parsed:
            continue
        records.append(
            {
                "assigned_tma": parsed["tma"],
                "confirmed_core_id": parsed["core_id"],
                "round": parsed["round"],
                "panel_markers": parsed["panel_markers"],
                "path": str(czi.resolve()),
                "filename": czi.name,
                "file_size_bytes": czi.stat().st_size,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    grouped = df.groupby(["assigned_tma", "confirmed_core_id"])
    core_summary = []
    for (tma, core), grp in grouped:
        rounds_present = set(grp["round"])
        missing_rounds = [r for r in STANDARD_ROUNDS if r not in rounds_present]
        is_complete = len(missing_rounds) == 0
        has_duplicate_rounds = len(grp["round"]) > len(rounds_present)

        is_eligible = is_complete and not has_duplicate_rounds
        if eligible_cores is not None:
            is_eligible = is_eligible and ((tma, core) in eligible_cores)

        core_summary.append(
            {
                "assigned_tma": tma,
                "confirmed_core_id": core,
                "rounds_count": len(grp),
                "rounds_present": ";".join(sorted(rounds_present, key=round_sort_key)),
                "missing_rounds": ";".join(missing_rounds),
                "is_complete": is_complete,
                "has_duplicate_rounds": has_duplicate_rounds,
                "is_eligible": is_eligible,
            }
        )

    summary_df = pd.DataFrame(core_summary)
    merged = df.merge(summary_df, on=["assigned_tma", "confirmed_core_id"], how="left")
    return merged.sort_values(
        ["assigned_tma", "confirmed_core_id", "round"],
        key=lambda col: col.map(round_sort_key) if col.name == "round" else col,
    )


# ---------------------------------------------------------------------------
# SIFT + RANSAC Registration & Support Computation
# ---------------------------------------------------------------------------


def register_moving_round_to_reference(
    ref_dapi: np.ndarray,
    mov_dapi: np.ndarray,
    scale_factor: float = 1.0,
) -> tuple[Optional[np.ndarray], int, int, str]:
    """Register moving round DAPI image to R1 reference DAPI image using SIFT+RANSAC."""
    M, inliers, matches, status = _toolkit_registration().register_round_pair(
        ref_dapi, mov_dapi
    )
    return M, inliers, matches, status


def warp_image_to_reference(
    image: np.ndarray,
    M: Optional[np.ndarray],
    reference_shape: tuple[int, int],
) -> np.ndarray:
    """Warp moving channel image to reference shape using 2x3 affine matrix M."""
    if M is None:
        if image.shape != reference_shape:
            raise ValueError(
                f"Cannot align images without registration transform M when shapes differ: {image.shape} vs reference {reference_shape}"
            )
        return image.copy()

    return _toolkit_registration().warp_image(image.astype(np.float32), M, reference_shape)


def compute_registered_support(
    ref_shape: tuple[int, int],
    mov_shape: tuple[int, int],
    M: Optional[np.ndarray],
    native_support: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute boolean validity mask indicating valid field-of-view support in reference frame."""
    if M is None:
        if mov_shape != ref_shape:
            raise ValueError(
                f"Cannot compute support without registration transform M when shapes differ: {mov_shape} vs reference {ref_shape}"
            )
        return (
            np.ones(ref_shape, dtype=bool) if native_support is None else native_support.copy()
        )

    import cv2

    mov_ones = (
        np.ones(mov_shape, dtype=np.float32)
        if native_support is None
        else native_support.astype(np.float32)
    )
    warped_mask = cv2.warpAffine(
        mov_ones,
        np.asarray(M, dtype=np.float64),
        (ref_shape[1], ref_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # A quantitative pixel is eligible only when every interpolation contributor
    # came from acquired tile coverage, excluding partially padded edge pixels.
    return warped_mask >= 1.0 - np.finfo(np.float32).eps


def native_observed_support(
    metadata: dict[str, Any], shape: tuple[int, int], channel_idx: int = 0
) -> np.ndarray:
    """Mark acquired mosaic tiles, independently of signal and tissue intensity."""
    boxes = metadata.get("tile_boxes_by_channel", {}).get(
        str(channel_idx), metadata.get("tile_boxes_xywh", [])
    )
    if not boxes:
        return np.ones(shape, dtype=bool)
    origin_x, origin_y, width, height = metadata["bbox"]
    if shape != (height, width):
        raise ValueError("Decoded image does not match its native mosaic bounding box")
    support = np.zeros(shape, dtype=bool)
    for tile_x, tile_y, tile_width, tile_height in boxes:
        x0, y0 = tile_x - origin_x, tile_y - origin_y
        support[
            max(0, y0) : min(height, y0 + tile_height), max(0, x0) : min(width, x0 + tile_width)
        ] = True
    return support


# ---------------------------------------------------------------------------
# Pre-decode Memory Estimation
# ---------------------------------------------------------------------------


def estimate_core_working_bytes(round_metadata: dict[str, Any]) -> int:
    """Estimate peak memory bytes required for processing a single core from XML bounding boxes/axes."""
    bytes_per_pixel = 4  # float32
    max_pixels = 0
    total_channels = 0

    for r_info in round_metadata.values():
        meta = r_info["metadata"]
        bbox = meta.get("bbox")
        if bbox is not None:
            if hasattr(bbox, "w") and hasattr(bbox, "h"):
                w, h = int(bbox.w), int(bbox.h)
            else:
                _, _, w, h = bbox
            n_px = int(w) * int(h)
        else:
            ax = meta.get("axis_counts", {})
            if "X" not in ax or "Y" not in ax:
                raise ValueError(
                    "Cannot bound image memory without a mosaic bounding box or explicit X/Y dimensions"
                )
            n_px = int(ax["X"]) * int(ax["Y"])
        if n_px <= 0:
            raise ValueError("Image dimensions must be positive")
        max_pixels = max(max_pixels, n_px)

    # Streaming model holds ~3 channel planes simultaneously (reference, moving, warped)
    estimated_peak = max_pixels * bytes_per_pixel * 6
    return estimated_peak


# ---------------------------------------------------------------------------
# Single Core Processing Pipeline
# ---------------------------------------------------------------------------


def process_single_core_cycif(
    assigned_tma: str,
    confirmed_core_id: str,
    round_czi_paths: dict[str, Union[str, Path]],
    output_dir: Union[str, Path],
    receipt_dir: Union[str, Path],
    read_channel_fn: Optional[Callable[[Union[str, Path], int], np.ndarray]] = None,
    extract_metadata_fn: Optional[Callable[[Union[str, Path]], dict[str, Any]]] = None,
    max_working_bytes: int = 32 * 1024**3,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Process all CycIF rounds for one core, registering channels to R1 DAPI.

    Streams channels one by one to respect memory bounds.
    Exports R1 DAPI once, and biological markers from R1-R5 (never R5Q).
    """
    output_path = Path(output_dir)
    receipt_path = Path(receipt_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    receipt_path.mkdir(parents=True, exist_ok=True)

    core_receipt_file = receipt_path / f"{assigned_tma}_{confirmed_core_id}_cycif_receipt.json"

    meta_extractor = extract_metadata_fn or extract_czi_metadata

    if "R1" not in round_czi_paths:
        raise ValueError(
            f"Core {assigned_tma}_{confirmed_core_id}: R1 reference CZI is missing"
        )

    # Step 1: Pre-hash and audit all round CZI files before decoding
    round_metadata: dict[str, Any] = {}
    pre_source_hashes: dict[str, str] = {}
    for r_label, czi_p in sorted(round_czi_paths.items(), key=lambda x: round_sort_key(x[0])):
        czi_path_obj = Path(czi_p)
        source_sha = compute_sha256(czi_path_obj)
        pre_source_hashes[r_label] = source_sha
        meta = meta_extractor(czi_path_obj)
        parsed = parse_cycif_czi_filename(czi_path_obj.name)
        panel_markers = parse_panel_markers(parsed["panel_markers"]) if parsed else []
        channel_audit = audit_czi_panel_channels(czi_path_obj, panel_markers, metadata=meta)

        round_metadata[r_label] = {
            "path": str(czi_path_obj.resolve()),
            "pre_sha256": source_sha,
            "metadata": meta,
            "channel_audit": channel_audit,
            "panel_markers": panel_markers,
        }

    # Step 2: Pre-decode working memory check
    # Reserve additional space for SIFT pyramids and support/QC arrays.
    estimated_mem = 4 * estimate_core_working_bytes(round_metadata)
    if estimated_mem > max_working_bytes:
        raise MemoryError(
            f"{assigned_tma} core {confirmed_core_id}: estimated working memory {estimated_mem / 1024**3:.2f} GB exceeds limit {max_working_bytes / 1024**3:.2f} GB"
        )
    code_bindings = {
        "clinical_roi_cycif.py": compute_sha256(Path(__file__).resolve()),
        "guided_serial_registration.py": compute_sha256(
            Path(__file__).with_name("guided_serial_registration.py")
        ),
        "cycif_zeb1_segment_quantify.py": compute_sha256(
            Path(__file__).with_name("cycif_zeb1_segment_quantify.py")
        ),
        "toolkit_registration": compute_sha256(Path(_toolkit_registration().__file__)),
    }
    import cv2

    versions = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "aicspylibczi": importlib.metadata.version("aicspylibczi"),
        "scikit-image": importlib.metadata.version("scikit-image"),
    }
    parameters = {
        "reference_round": "R1",
        "registration": "toolkit_SIFT_RANSAC_defaults",
        "registration_seed": 42,
        "czi_region": "explicit_native_mosaic_bbox_xywh",
        "tissue_percentile": 60.0,
        "closing_radius": 8,
        "intensity_interpolation": "float32_bilinear",
        "observed_support": "acquired_tiles_and_full_bilinear_coverage",
        "zeb1_correction": "max(0, R5_AF750 - R5_exposure/R5Q_exposure * R5Q_AF750)",
    }

    # Step 3: Check cache if force is False
    if not force and core_receipt_file.exists():
        try:
            with open(core_receipt_file, "r", encoding="utf-8") as f:
                cached_receipt = json.load(f)
            # Verify cached source hashes match current
            cache_valid = (
                cached_receipt.get("code_bindings") == code_bindings
                and cached_receipt.get("versions") == versions
                and cached_receipt.get("parameters") == parameters
                and set(cached_receipt.get("sources", {})) == set(round_metadata)
                and bool(cached_receipt.get("artifacts"))
            )
            for r_label, r_info in round_metadata.items():
                cached_src = cached_receipt.get("sources", {}).get(r_label, {})
                if cached_src.get("sha256") != r_info["pre_sha256"]:
                    cache_valid = False
                    break
            # Check if all output array files exist and have valid hashes
            cached_artifacts = cached_receipt.get("artifacts", {})
            for art_name, art_sha in cached_artifacts.items():
                art_file = output_path / art_name
                if not art_file.exists() or compute_sha256(art_file) != art_sha:
                    cache_valid = False
                    break
            if cache_valid and "table_rows" in cached_receipt:
                print(f"CACHE_HIT cycIF {assigned_tma}_{confirmed_core_id}", flush=True)
                receipt_digest = compute_sha256(core_receipt_file)
                return [
                    {**row, "receipt_sha256": receipt_digest}
                    for row in cached_receipt["table_rows"]
                ]
        except Exception:
            pass  # Recompute on any cache parse/read issue
    print(f"CACHE_MISS_RECOMPUTED cycIF {assigned_tma}_{confirmed_core_id}", flush=True)
    cv2.setRNGSeed(42)

    # Helper for reading channel
    def _read_ch(path: Union[str, Path], ch: int) -> np.ndarray:
        if read_channel_fn is not None:
            image = read_channel_fn(path, ch)
        else:
            import aicspylibczi

            source = str(Path(path).resolve())
            region = next(
                info["metadata"]["bbox"]
                for info in round_metadata.values()
                if info["path"] == source
            )
            if region is None:
                raise ValueError("Native CZI mosaic region is required for explicit geometry")
            czi = aicspylibczi.CziFile(source)
            image = np.squeeze(czi.read_mosaic(C=ch, scale_factor=1.0, region=region))
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError("Each CZI channel must be one finite 2D acquisition plane")
        return image

    # Step 4: Read R1 Reference DAPI
    r1_path = round_metadata["R1"]["path"]
    r1_dapi = _read_ch(r1_path, 0)
    ref_shape = r1_dapi.shape
    ref_pixel_size_x = round_metadata["R1"]["metadata"]["pixel_size_x_um"]
    ref_pixel_size_y = round_metadata["R1"]["metadata"]["pixel_size_y_um"]

    # Verify physical pixel sizes are consistent across all rounds
    for r_label, r_info in round_metadata.items():
        m_x = r_info["metadata"]["pixel_size_x_um"]
        m_y = r_info["metadata"]["pixel_size_y_um"]
        if not (
            np.isclose(m_x, ref_pixel_size_x, rtol=1e-3)
            and np.isclose(m_y, ref_pixel_size_y, rtol=1e-3)
        ):
            raise ValueError(
                f"{assigned_tma} core {confirmed_core_id} round {r_label}: pixel size ({m_x}, {m_y}) conflicts with R1 reference ({ref_pixel_size_x}, {ref_pixel_size_y})"
            )

    # Compute foreground tissue mask on R1 DAPI
    reference_observed = native_observed_support(round_metadata["R1"]["metadata"], ref_shape)
    tissue_mask_arr = compute_tissue_mask(r1_dapi) & reference_observed

    # Step 5: Register all moving rounds to R1 DAPI
    transforms: dict[str, Optional[np.ndarray]] = {"R1": None}
    supports: dict[str, np.ndarray] = {"R1": reference_observed}
    review_stride = max(1, int(np.ceil(max(ref_shape) / 512)))
    registration_review = {"R1": r1_dapi[::review_stride, ::review_stride].astype(np.float32)}
    registration_status: dict[str, Any] = {
        "R1": {"status": "reference", "inliers": 0, "matches": 0, "matrix": None}
    }

    for r_label, r_info in sorted(round_metadata.items(), key=lambda x: round_sort_key(x[0])):
        if r_label == "R1":
            continue
        czi_p = r_info["path"]
        mov_dapi = _read_ch(czi_p, 0)

        M, inliers, matches, status = register_moving_round_to_reference(r1_dapi, mov_dapi)
        if status != "ok" or M is None:
            raise RuntimeError(
                f"Registration failed for {assigned_tma} core {confirmed_core_id} {r_label}: status='{status}'"
            )
        transforms[r_label] = M
        native_support = native_observed_support(r_info["metadata"], mov_dapi.shape)
        supports[r_label] = (
            compute_registered_support(
                ref_shape, mov_dapi.shape, M, native_support=native_support
            )
            & reference_observed
        )
        warped_dapi = warp_image_to_reference(mov_dapi, M, ref_shape)
        registration_review[r_label] = warped_dapi[::review_stride, ::review_stride].copy()
        preview_valid = supports[r_label][::review_stride, ::review_stride]
        reference_preview = registration_review["R1"]
        moving_preview = registration_review[r_label]
        if preview_valid.any() and np.std(reference_preview[preview_valid]) > 0:
            correlation = float(
                np.corrcoef(reference_preview[preview_valid], moving_preview[preview_valid])[
                    0, 1
                ]
            )
        else:
            correlation = None
        registration_status[r_label] = {
            "status": status,
            "inliers": inliers,
            "matches": matches,
            "matrix": M.tolist(),
            "support_fraction": float(supports[r_label].mean()),
            "preview_dapi_pearson": correlation,
            "validation_kind": "same-channel descriptive QC; not independent TRE",
        }
        del warped_dapi, native_support
        del mov_dapi  # Release immediately to keep memory bounded

    # Step 6: Stream and export channels
    # Export R1 DAPI once
    r1_dapi_filename = f"{assigned_tma}_{confirmed_core_id}_R1_DAPI_image.npy"
    r1_dapi_path = output_path / r1_dapi_filename
    r1_support_filename = f"{assigned_tma}_{confirmed_core_id}_R1_DAPI_support.npy"
    r1_support_path = output_path / r1_support_filename
    tissue_mask_filename = f"{assigned_tma}_{confirmed_core_id}_tissue_mask.npy"
    tissue_mask_path = output_path / tissue_mask_filename

    np.save(r1_dapi_path, r1_dapi.astype(np.float32))
    np.save(r1_support_path, supports["R1"])
    np.save(tissue_mask_path, tissue_mask_arr)

    artifacts_sha: dict[str, str] = {
        r1_dapi_filename: compute_sha256(r1_dapi_path),
        r1_support_filename: compute_sha256(r1_support_path),
        tissue_mask_filename: compute_sha256(tissue_mask_path),
    }
    review_name = f"{assigned_tma}_{confirmed_core_id}_round_registration_review.npz"
    np.savez_compressed(output_path / review_name, **registration_review)
    artifacts_sha[review_name] = compute_sha256(output_path / review_name)

    marker_rows: list[dict[str, Any]] = [
        {
            "assigned_tma": assigned_tma,
            "confirmed_core_id": confirmed_core_id,
            "marker": "DAPI",
            "image_path": str(r1_dapi_path.resolve()),
            "support_path": str(r1_support_path.resolve()),
            "tissue_mask_path": str(tissue_mask_path.resolve()),
            "reference_image_path": str(r1_dapi_path.resolve()),
            "pixel_size_x_um": ref_pixel_size_x,
            "pixel_size_y_um": ref_pixel_size_y,
            "source_path": round_metadata["R1"]["path"],
            "source_sha256": round_metadata["R1"]["pre_sha256"],
            "receipt_path": str(core_receipt_file.resolve()),
            "receipt_sha256": "",  # Filled after receipt is written
            "registration_driver": True,
            "section_relationship": "unknown",
            "section_evidence": "",
            "group_id": "",
            "round": "R1",
            "channel_idx": 0,
            "exposure_ms": round_metadata["R1"]["channel_audit"][0]["exposure_ms"],
            "correction_status": "REGISTERED_RAW_COUNTS",
            "background_source_path": "",
            "background_source_sha256": "",
            "background_exposure_ms": None,
            "background_scale": None,
            "raw_registered_image_path": str(r1_dapi_path.resolve()),
            "intensity_units": "camera_counts; not calibrated protein abundance",
            "reference_bbox_xywh": json.dumps(round_metadata["R1"]["metadata"]["bbox"]),
            "saturated_native_pixels": int((r1_dapi >= np.iinfo(np.uint16).max).sum()),
            "native_pixel_count": int(r1_dapi.size),
        }
    ]

    # Pre-read R5Q quench channel 4 if R5 and R5Q are both present and ZEB1 is present
    r5q_ch4_warped: Optional[np.ndarray] = None
    r5q_ch4_support: Optional[np.ndarray] = None
    r5q_exp: Optional[float] = None
    if "R5" in round_metadata and "R5Q" in round_metadata:
        r5q_path = round_metadata["R5Q"]["path"]
        r5q_audit = round_metadata["R5Q"]["channel_audit"]
        if len(r5q_audit) > 4:
            r5q_exp = r5q_audit[4].get("exposure_ms")
            if r5q_exp is not None and r5q_exp > 0:
                raw_r5q = _read_ch(r5q_path, 4)
                r5q_ch4_support = (
                    compute_registered_support(
                        ref_shape,
                        raw_r5q.shape,
                        transforms["R5Q"],
                        native_support=native_observed_support(
                            round_metadata["R5Q"]["metadata"], raw_r5q.shape, channel_idx=4
                        ),
                    )
                    & reference_observed
                )
                r5q_ch4_warped = warp_image_to_reference(raw_r5q, transforms["R5Q"], ref_shape)
                del raw_r5q
            else:
                raise ValueError("R5Q exposure is required for the established ZEB1 correction")

    # Process biological rounds (R1-R5) and their 4 antibody markers (never DAPI again, never R5Q)
    for r_label in BIOLOGICAL_ROUNDS:
        if r_label not in round_metadata:
            continue
        r_info = round_metadata[r_label]
        czi_p = r_info["path"]
        M = transforms[r_label]
        round_sup = supports[r_label]

        for item in r_info["channel_audit"]:
            if item["is_dapi"]:
                continue  # Never export duplicate DAPI from moving rounds
            ch_idx = item["channel_idx"]
            m_name = item["marker"]

            raw_ch = _read_ch(czi_p, ch_idx)
            channel_support = (
                compute_registered_support(
                    ref_shape,
                    raw_ch.shape,
                    M,
                    native_support=native_observed_support(
                        r_info["metadata"], raw_ch.shape, channel_idx=ch_idx
                    ),
                )
                & reference_observed
            )
            saturated_pixels = int((raw_ch >= np.iinfo(np.uint16).max).sum())
            native_pixels = int(raw_ch.size)
            warped_ch = warp_image_to_reference(raw_ch, M, ref_shape)
            del raw_ch
            correction_status = "REGISTERED_RAW_COUNTS"
            correction_scale = None
            raw_registered_path = None

            if (
                m_name == "ZEB1"
                and r5q_ch4_warped is not None
                and item.get("exposure_ms") is not None
                and r5q_exp is not None
            ):
                exp_sig = item["exposure_ms"]
                raw_name = (
                    f"{assigned_tma}_{confirmed_core_id}_{r_label}_{m_name}_registered_raw.npy"
                )
                raw_registered_path = output_path / raw_name
                np.save(raw_registered_path, warped_ch.astype(np.float32))
                artifacts_sha[raw_name] = compute_sha256(raw_registered_path)
                final_ch, correction_scale = backsub_correct(
                    warped_ch, r5q_ch4_warped, exp_sig, r5q_exp
                )
                correction_status = "R5Q_EXPOSURE_SCALED_BACKSUB"
                assert r5q_ch4_support is not None
                final_sup = channel_support & r5q_ch4_support
            else:
                if m_name == "ZEB1":
                    raise ValueError(
                        "ZEB1 requires its recorded R5Q exposure-scaled correction"
                    )
                final_ch = warped_ch.astype(np.float32)
                final_sup = channel_support

            del warped_ch

            img_fname = f"{assigned_tma}_{confirmed_core_id}_{r_label}_{m_name}_image.npy"
            sup_fname = f"{assigned_tma}_{confirmed_core_id}_{r_label}_{m_name}_support.npy"
            img_p = output_path / img_fname
            sup_p = output_path / sup_fname

            np.save(img_p, final_ch)
            np.save(sup_p, final_sup)
            del final_ch

            artifacts_sha[img_fname] = compute_sha256(img_p)
            artifacts_sha[sup_fname] = compute_sha256(sup_p)

            marker_rows.append(
                {
                    "assigned_tma": assigned_tma,
                    "confirmed_core_id": confirmed_core_id,
                    "marker": m_name,
                    "image_path": str(img_p.resolve()),
                    "support_path": str(sup_p.resolve()),
                    "tissue_mask_path": str(tissue_mask_path.resolve()),
                    "reference_image_path": str(r1_dapi_path.resolve()),
                    "pixel_size_x_um": ref_pixel_size_x,
                    "pixel_size_y_um": ref_pixel_size_y,
                    "source_path": r_info["path"],
                    "source_sha256": r_info["pre_sha256"],
                    "receipt_path": str(core_receipt_file.resolve()),
                    "receipt_sha256": "",
                    "registration_driver": False,
                    "section_relationship": "unknown",
                    "section_evidence": "",
                    "group_id": "",
                    "round": r_label,
                    "channel_idx": ch_idx,
                    "exposure_ms": item.get("exposure_ms"),
                    "correction_status": correction_status,
                    "background_source_path": round_metadata["R5Q"]["path"]
                    if correction_scale is not None
                    else "",
                    "background_source_sha256": round_metadata["R5Q"]["pre_sha256"]
                    if correction_scale is not None
                    else "",
                    "background_exposure_ms": r5q_exp if correction_scale is not None else None,
                    "background_scale": correction_scale,
                    "raw_registered_image_path": str((raw_registered_path or img_p).resolve()),
                    "intensity_units": "camera_counts; not calibrated protein abundance",
                    "reference_bbox_xywh": json.dumps(round_metadata["R1"]["metadata"]["bbox"]),
                    "saturated_native_pixels": saturated_pixels,
                    "native_pixel_count": native_pixels,
                }
            )

    if r5q_ch4_warped is not None:
        del r5q_ch4_warped

    # Step 7: Post-decode source hash verification (ensure raw CZIs were not modified)
    for r_label, r_info in round_metadata.items():
        post_sha = compute_sha256(r_info["path"])
        if post_sha != r_info["pre_sha256"]:
            raise RuntimeError(
                f"Source CZI tampering or mutation detected for {r_label}: pre-hash {r_info['pre_sha256']} != post-hash {post_sha}"
            )

    # Step 8: Build and write complete provenance receipt JSON
    receipt_data = {
        "assigned_tma": assigned_tma,
        "confirmed_core_id": confirmed_core_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_round": "R1",
        "reference_shape": list(ref_shape),
        "pixel_size_x_um": ref_pixel_size_x,
        "pixel_size_y_um": ref_pixel_size_y,
        "registration": registration_status,
        "sources": {
            k: {"path": v["path"], "sha256": v["pre_sha256"]} for k, v in round_metadata.items()
        },
        "artifacts": artifacts_sha,
        "table_rows": marker_rows,
        "code_bindings": code_bindings,
        "versions": versions,
        "parameters": parameters,
        "channel_audit": {key: item["channel_audit"] for key, item in round_metadata.items()},
        "reference_bbox_xywh": round_metadata["R1"]["metadata"]["bbox"],
        "registration_review_path": str((output_path / review_name).resolve()),
        "claim_label": "Exploratory",
    }

    with open(core_receipt_file, "w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2)

    receipt_sha = compute_sha256(core_receipt_file)

    # Bind final receipt SHA-256 into all table rows
    for r in marker_rows:
        r["receipt_sha256"] = receipt_sha

    return marker_rows


# ---------------------------------------------------------------------------
# Cohort Runner & Table Export
# ---------------------------------------------------------------------------


def process_clinical_cycif_cohort(
    raw_root: Union[str, Path],
    output_dir: Union[str, Path],
    receipt_dir: Union[str, Path],
    manifest_csv_path: Union[str, Path],
    eligible_cores: Optional[Sequence[tuple[str, str]]] = None,
    read_channel_fn: Optional[Callable[[Union[str, Path], int], np.ndarray]] = None,
    extract_metadata_fn: Optional[Callable[[Union[str, Path]], dict[str, Any]]] = None,
    max_working_bytes: int = 32 * 1024**3,
    force: bool = False,
    continue_on_error: bool = False,
) -> pd.DataFrame:
    """Orchestrate CycIF processing across the entire cohort.

    Verifies input completeness, performs round alignment, extracts continuous markers,
    writes per-core arrays and provenance receipts, and publishes cycif_marker_images.csv.
    """
    inventory_df = inventory_cycif_sources(raw_root, eligible_cores=eligible_cores)
    if inventory_df.empty:
        raise RuntimeError(f"No valid CycIF files found under {raw_root}")

    eligible_df = inventory_df[inventory_df["is_eligible"]].copy()
    if eligible_df.empty:
        raise RuntimeError("No eligible cores found with complete standard rounds (R1-R5, R5Q)")

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    grouped = dict(tuple(eligible_df.groupby(["assigned_tma", "confirmed_core_id"])))
    ordered_keys = (
        list(grouped)
        if eligible_cores is None
        else [key for key in eligible_cores if key in grouped]
    )
    for tma, core in ordered_keys:
        grp = grouped[(tma, core)]
        round_paths = dict(zip(grp["round"], grp["path"]))
        try:
            core_rows = process_single_core_cycif(
                assigned_tma=tma,
                confirmed_core_id=core,
                round_czi_paths=round_paths,
                output_dir=output_dir,
                receipt_dir=receipt_dir,
                read_channel_fn=read_channel_fn,
                extract_metadata_fn=extract_metadata_fn,
                max_working_bytes=max_working_bytes,
                force=force,
            )
        except Exception as error:
            if not continue_on_error:
                raise
            failures.append(
                {
                    "assigned_tma": tma,
                    "confirmed_core_id": core,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            print(f"EXCLUDED cycIF {tma}_{core}: {error}", flush=True)
            core_rows = []
        all_rows.extend(core_rows)
        partial_path = Path(manifest_csv_path).with_suffix(".partial.csv")
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows, columns=CYCIF_MARKER_TABLE_COLUMNS).to_csv(
            partial_path, index=False
        )
        pd.DataFrame(
            failures, columns=["assigned_tma", "confirmed_core_id", "error_type", "reason"]
        ).to_csv(
            Path(manifest_csv_path).with_name("cycif_processing_failures.csv"),
            index=False,
        )

    if not all_rows:
        raise RuntimeError("No core passed cycIF processing; inspect the failure ledger")

    df_out = pd.DataFrame(all_rows, columns=CYCIF_MARKER_TABLE_COLUMNS)

    manifest_file = Path(manifest_csv_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    df_out.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, manifest_file)

    return df_out


def validate_cycif_exports(manifest_path: Union[str, Path]) -> pd.DataFrame:
    """Validate each source-bound receipt and native quantitative image once."""
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    missing = set(CYCIF_MARKER_TABLE_COLUMNS) - set(frame.columns)
    if missing or frame.duplicated(["assigned_tma", "confirmed_core_id", "marker"]).any():
        raise ValueError(f"Invalid cycIF manifest schema or duplicate keys: {missing}")
    checked_receipts: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        receipt_path = Path(record["receipt_path"])
        if record["receipt_path"] not in checked_receipts:
            if compute_sha256(receipt_path) != record["receipt_sha256"]:
                raise ValueError(f"Stale receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            for source in receipt["sources"].values():
                if compute_sha256(source["path"]) != source["sha256"]:
                    raise ValueError(f"Changed raw source: {source['path']}")
            root = Path(record["image_path"]).parent
            for name, digest in receipt["artifacts"].items():
                if compute_sha256(root / name) != digest:
                    raise ValueError(f"Changed cycIF artifact: {root / name}")
            checked_receipts.add(record["receipt_path"])
        image = np.load(record["image_path"], mmap_mode="r")
        support = np.load(record["support_path"], mmap_mode="r")
        tissue = np.load(record["tissue_mask_path"], mmap_mode="r")
        if (
            image.ndim != 2
            or image.dtype != np.float32
            or support.shape != image.shape
            or tissue.shape != image.shape
            or support.dtype != bool
            or tissue.dtype != bool
        ):
            raise ValueError(f"Invalid marker/support geometry: {record['image_path']}")
        if not np.isfinite(image).all() or (image < 0).any():
            raise ValueError(f"Invalid quantitative intensity: {record['image_path']}")
        rows.append(
            {
                "assigned_tma": record["assigned_tma"],
                "confirmed_core_id": record["confirmed_core_id"],
                "marker": record["marker"],
                "shape_y": image.shape[0],
                "shape_x": image.shape[1],
                "observed_pixels": int(support.sum()),
                "shared_tissue_pixels": int((support & tissue).sum()),
                "correction_status": record["correction_status"],
                "verified": True,
            }
        )
    return pd.DataFrame(rows)


def render_cycif_qc(
    manifest_path: Union[str, Path],
    plot_dir: Union[str, Path],
    force: bool = False,
) -> dict[str, Any]:
    """Render native marker patterns and independent round-DAPI alignment reviews."""
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    manifest = Path(manifest_path)
    destination = Path(plot_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest, keep_default_na=False)
    marker_pdf = destination / "cycif_marker_review.pdf"
    round_pdf = destination / "cycif_round_alignment_review.pdf"
    receipt_path = destination / "cycif_plot_receipt.json"
    identity = {
        "manifest_sha256": compute_sha256(manifest),
        "code_sha256": compute_sha256(Path(__file__)),
        "display_max_edge": 512,
        "marker_display_percentile": 99,
    }
    if not force and receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved["identity"] == identity and all(
            Path(item["path"]).exists() and compute_sha256(item["path"]) == item["sha256"]
            for item in saved["artifacts"]
        ):
            print("CACHE_HIT cycIF QC PDFs", flush=True)
            return saved | {"receipt_path": str(receipt_path), "cache_status": "CACHE_HIT"}

    def contrast(image: np.ndarray) -> np.ndarray:
        values = image[np.isfinite(image) & (image > 0)]
        if not values.size:
            return np.zeros_like(image, dtype=float)
        low, high = np.percentile(values, [1, 99])
        return np.clip((image - low) / max(high - low, 1e-12), 0, 1)

    with PdfPages(marker_pdf) as document:
        for marker, group in frame.groupby("marker", sort=True):
            previews = []
            for record in group.to_dict("records"):
                image = np.load(record["image_path"], mmap_mode="r")
                support = np.load(record["support_path"], mmap_mode="r")
                stride = max(1, int(np.ceil(max(image.shape) / 512)))
                previews.append(
                    (
                        record,
                        np.asarray(image[::stride, ::stride]),
                        np.asarray(support[::stride, ::stride]),
                    )
                )
            values = np.concatenate([image[valid] for _, image, valid in previews])
            maximum = max(float(np.percentile(values, 99)), 1.0)
            fig, axes = plt.subplots(3, 8, figsize=(180 / 25.4, 100 / 25.4))
            fig.subplots_adjust(
                left=0.02, right=0.98, bottom=0.14, top=0.9, hspace=0.12, wspace=0.08
            )
            for axis in axes.flat:
                axis.axis("off")
            for record, image, valid in previews:
                core = record["confirmed_core_id"]
                offset = {"TMA1": 0, "TMA2": 4}[record["assigned_tma"]]
                axis = axes["ABC".index(core[0]), int(core[1]) - 1 + offset]
                axis.imshow(
                    np.ma.array(image, mask=~valid),
                    cmap="gray",
                    vmin=0,
                    vmax=maximum,
                    interpolation="nearest",
                )
                axis.text(
                    0.02,
                    0.98,
                    core,
                    transform=axis.transAxes,
                    va="top",
                    color="white",
                    fontsize=5,
                )
            fig.text(0.25, 0.95, "TMA1", ha="center", fontsize=5)
            fig.text(0.75, 0.95, "TMA2", ha="center", fontsize=5)
            correction = "; ".join(sorted(group["correction_status"].unique()))
            fig.text(0.02, 0.075, f"{marker}: {correction}", fontsize=5)
            fig.text(
                0.02,
                0.045,
                "Shared display contrast; camera counts, not protein abundance.",
                fontsize=5,
            )
            fig.text(
                0.02,
                0.02,
                "Generator: clinical_roi_cycif.py; no cell segmentation.",
                fontsize=5,
            )
            document.savefig(fig)
            plt.close(fig)
    with PdfPages(round_pdf) as document:
        for receipt_name in frame["receipt_path"].drop_duplicates():
            receipt = json.loads(Path(receipt_name).read_text())
            with np.load(receipt["registration_review_path"]) as previews:
                reference = contrast(previews["R1"])
                fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 125 / 25.4))
                fig.subplots_adjust(
                    left=0.03, right=0.97, bottom=0.1, top=0.9, hspace=0.15, wspace=0.1
                )
                for axis, label in zip(axes.flat, STANDARD_ROUNDS):
                    moving = contrast(previews[label])
                    overlay = np.stack([reference, moving, reference], axis=-1)
                    axis.imshow(overlay, interpolation="nearest")
                    axis.axis("off")
                    axis.text(
                        0.02,
                        0.98,
                        label,
                        transform=axis.transAxes,
                        color="white",
                        va="top",
                        fontsize=5,
                    )
                    if label == "R1":
                        tissue = np.load(
                            receipt["table_rows"][0]["tissue_mask_path"],
                            mmap_mode="r",
                        )
                        stride = max(1, int(np.ceil(max(tissue.shape) / 512)))
                        mask_preview = tissue[::stride, ::stride]
                        if mask_preview.any() and not mask_preview.all():
                            axis.contour(
                                mask_preview, levels=[0.5], colors=["cyan"], linewidths=0.5
                            )
                fig.text(
                    0.03,
                    0.95,
                    f"{receipt['assigned_tma']} {receipt['confirmed_core_id']}: round DAPI alignment",
                    fontsize=5,
                )
                fig.text(
                    0.03,
                    0.06,
                    "R1 magenta / round green; agreement white. Cyan: heuristic tissue support. Display contrast only.",
                    fontsize=5,
                )
                fig.text(
                    0.03,
                    0.025,
                    "Generator: clinical_roi_cycif.py; descriptive alignment QC, not held-out landmark validation.",
                    fontsize=5,
                )
                document.savefig(fig)
                plt.close(fig)
    receipt = {
        "identity": identity,
        "claim_label": "Exploratory",
        "artifacts": [
            {"path": str(path), "sha256": compute_sha256(path)}
            for path in (marker_pdf, round_pdf)
        ],
        "marker_pdf": str(marker_pdf),
        "round_pdf": str(round_pdf),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("CACHE_MISS_RECOMPUTED cycIF QC PDFs", flush=True)
    return receipt | {
        "receipt_path": str(receipt_path),
        "cache_status": "CACHE_MISS_RECOMPUTED",
    }
