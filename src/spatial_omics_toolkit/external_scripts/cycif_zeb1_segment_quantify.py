"""Generate cohort-wide cycIF nuclear ZEB1 measurements from delivered CZI files.

For each annotated TMA core this script reads the R1 reference, the R5
ZEB1-stained acquisition, and the paired R5Q post-quench blank acquisition.
It registers both later acquisitions to R1 DAPI with OpenCV SIFT/RANSAC,
segments R1 nuclei with scikit-image Otsu/watershed, and applies the Backsub
v0.5.2 exposure-scaled pixel-subtraction method to registered R5 AF750 and
its paired R5Q AF750 blank before nuclear quantification. The adapter works
on registered CZI arrays because Backsub's CLI accepts OME-TIFF input only;
the published Backsub formula is preserved exactly. Raw and blank
measurements remain in the output for audit.

Per-scene checkpoints are reusable only when their receipt matches every
source CZI and this script. Existing CSVs without such a receipt are never
adopted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
TOOLKIT_SRC = PROJECT_ROOT.parent / "02_spatial_toolkit" / "src"
if not (TOOLKIT_SRC / "spatial_omics_toolkit").is_dir():
    raise RuntimeError(f"Spatial toolkit source not found: {TOOLKIT_SRC}")
sys.path.insert(0, str(TOOLKIT_SRC))

from spatial_omics_toolkit.cycif.register import (  # noqa: E402
    get_czi_pixel_size,
    group_czis_by_scene,
    read_czi_channel,
    register_round_pair,
    warp_image,
)
from spatial_omics_toolkit.pla import enhance_dapi, segment_dapi_cells  # noqa: E402

RAW_DIR = (
    PROJECT_ROOT / "02_data_sources" / "04_collaborators" / "from_koei" / "01_mxcycif"
)
PROGRESSION_MAP = PROJECT_ROOT / "02_data_sources" / "01_self" / "progression_map.csv"
TABLES_DIR = PROJECT_ROOT / "03_analysis" / "03_tables"
PLOTS_DIR = PROJECT_ROOT / "03_analysis" / "04_plots"
PER_SCENE_DIR = TABLES_DIR / "cycif_zeb1_by_scene"
QC_DIR = PLOTS_DIR / "00_workspace" / "06_qc_and_diagnostics" / "cycif_zeb1_qc"
FINAL_TABLE = TABLES_DIR / "cycif_zeb1_cell_features.csv"
SUMMARY_TABLE = TABLES_DIR / "cycif_zeb1_scene_summary.csv"
COHORT_RECEIPT = TABLES_DIR / "cycif_zeb1_generation_receipt.json"

REFERENCE_ROUND = "R1"
ZEB1_ROUND = "R5"
QUENCH_ROUND = "R5Q"
DAPI_CHANNEL = 0
ZEB1_CHANNEL = 4
QUENCH_CHANNEL = 4
MIN_REGISTERED_FRACTION = 0.95
MIN_NUCLEUS_AREA_UM2 = 10.0
MAX_NUCLEUS_AREA_UM2 = 500.0
BACKSUB_METHOD = "Backsub exposure-scaled pixel subtraction"
BACKSUB_VERSION = "0.5.2"
BACKSUB_URL = "https://github.com/schapirolabor/background_subtraction"
PREFERRED_METHOD = "CLEAR-AF"
PREFERRED_METHOD_DOI = "10.1101/2025.11.22.689928"
PREFERRED_METHOD_UNAVAILABLE_REASON = (
    "The published preprint provides no released package or code repository."
)
SCRIPT_VERSION = 3


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def channel_exposure_ms(path: Path, channel_name: str) -> float:
    """Read one acquisition exposure from duplicated Zeiss CZI metadata."""
    import aicspylibczi

    metadata = aicspylibczi.CziFile(str(path)).meta
    root = ET.fromstring(metadata) if isinstance(metadata, str) else metadata
    exposures_ms: list[float] = []
    for channel in root.iter():
        if (
            channel.tag.split("}")[-1] != "Channel"
            or channel.get("Name") != channel_name
        ):
            continue
        for node in channel.iter():
            if node.tag.split("}")[-1] != "ExposureTime" or not node.text:
                continue
            value = float(node.text)
            # Zeiss duplicates exposure as activated milliseconds and acquisition
            # nanoseconds (for example, 500 and 500000000 for the same channel).
            exposures_ms.append(value / 1_000_000.0 if value >= 1_000_000 else value)
    if not exposures_ms:
        raise RuntimeError(f"{path.name}: no exposure metadata for {channel_name}")
    if not np.allclose(exposures_ms, exposures_ms[0]):
        raise RuntimeError(
            f"{path.name}: conflicting normalized {channel_name} exposures: {exposures_ms}"
        )
    return exposures_ms[0]


def backsub_correct(
    signal: np.ndarray,
    background: np.ndarray,
    signal_exposure_ms: float,
    background_exposure_ms: float,
) -> tuple[np.ndarray, float]:
    """Apply the published Backsub v0.5.2 formula to registered CZI arrays."""
    if signal_exposure_ms <= 0 or background_exposure_ms <= 0:
        raise ValueError("Backsub exposure values must be positive")
    exposure_scale = signal_exposure_ms / background_exposure_ms
    corrected = np.maximum(
        0.0,
        signal.astype(np.float32)
        - np.float32(exposure_scale) * background.astype(np.float32),
    )
    return corrected, exposure_scale


def supported_label_mean(
    image: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Measure each label only where both registered acquisitions have support."""
    supported_labels = np.where(support, labels, 0)
    sums = ndi.sum(image, labels=supported_labels, index=label_ids)
    counts = ndi.sum(
        np.ones_like(labels, dtype=np.uint8),
        labels=supported_labels,
        index=label_ids,
    )
    return np.divide(
        np.asarray(sums, dtype=float),
        np.asarray(counts, dtype=float),
        out=np.full(len(label_ids), np.nan),
        where=np.asarray(counts) > 0,
    )


def load_scene_manifest() -> pd.DataFrame:
    progression = pd.read_csv(PROGRESSION_MAP)
    progression = progression.loc[
        progression["progression_group"] != "orientation_core"
    ].copy()

    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for slide_dir in sorted(path for path in RAW_DIR.iterdir() if path.is_dir()):
        grouped.update(group_czis_by_scene(str(slide_dir)))

    rows: list[dict[str, object]] = []
    for row in progression.itertuples(index=False):
        scene_label = str(row.slide_scene)
        scene = scene_label.rsplit("_", 1)[-1]
        key = (str(row.tma_id), scene)
        if key not in grouped:
            raise FileNotFoundError(
                f"No cycIF CZI group for {scene_label}; available key count={len(grouped)}"
            )
        rounds = grouped[key]
        missing = {REFERENCE_ROUND, ZEB1_ROUND, QUENCH_ROUND} - set(rounds)
        if missing:
            raise FileNotFoundError(
                f"{scene_label} is missing required cycIF rounds: {sorted(missing)}"
            )
        rows.append(
            {
                "slide_scene": scene_label,
                "tma_id": row.tma_id,
                "core_id": row.core_id,
                "progression_group": row.progression_group,
                "r1_path": Path(rounds[REFERENCE_ROUND]),
                "r5_path": Path(rounds[ZEB1_ROUND]),
                "r5q_path": Path(rounds[QUENCH_ROUND]),
            }
        )
    return pd.DataFrame(rows)


def scene_signature(
    row: dict[str, object], parameters: dict[str, object]
) -> dict[str, object]:
    return {
        "script_version": SCRIPT_VERSION,
        "script": file_signature(Path(__file__)),
        "progression_map": file_signature(PROGRESSION_MAP),
        "r1": file_signature(Path(row["r1_path"])),
        "r5": file_signature(Path(row["r5_path"])),
        "r5q": file_signature(Path(row["r5q_path"])),
        "parameters": parameters,
    }


def cache_valid(
    csv_path: Path, receipt_path: Path, expected: dict[str, object]
) -> bool:
    if not csv_path.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return receipt.get("inputs") == expected and receipt.get(
        "output_sha256"
    ) == sha256_file(csv_path)


def _normalize_for_display(
    image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.7
) -> np.ndarray:
    """Percentile contrast stretch to [0, 1]; display-only, never fed into measurement."""
    positive = image[image > 0]
    if not positive.size:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(positive, [low_pct, high_pct])
    if not hi > lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _merge_rgb(dapi: np.ndarray, zeb1: np.ndarray) -> np.ndarray:
    """DAPI (blue) + corrected ZEB1 (magenta) display composite."""
    dapi_norm = _normalize_for_display(dapi)
    zeb1_norm = _normalize_for_display(zeb1)
    return np.stack([zeb1_norm, np.zeros_like(dapi_norm), dapi_norm], axis=-1)


def _save_qc_pdf(
    scene_label: str,
    ref_dapi: np.ndarray,
    registered_zeb1: np.ndarray,
    registered_quench: np.ndarray,
    corrected_zeb1: np.ndarray,
    nuclei: np.ndarray,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    QC_DIR.mkdir(parents=True, exist_ok=True)
    stride = max(1, math.ceil(max(ref_dapi.shape) / 4000))
    overview_dapi = ref_dapi[::stride, ::stride]
    overview_zeb1 = registered_zeb1[::stride, ::stride]
    overview_quench = registered_quench[::stride, ::stride]
    overview_corrected = corrected_zeb1[::stride, ::stride]

    accepted_ids = np.unique(nuclei)
    accepted_ids = accepted_ids[accepted_ids > 0]
    centroids = ndi.center_of_mass(
        np.ones_like(nuclei, dtype=np.uint8), labels=nuclei, index=accepted_ids
    )
    center_y, center_x = np.median(np.asarray(centroids), axis=0)
    detail_size = 768
    y0 = int(
        np.clip(
            round(center_y) - detail_size // 2,
            0,
            max(0, ref_dapi.shape[0] - detail_size),
        )
    )
    x0 = int(
        np.clip(
            round(center_x) - detail_size // 2,
            0,
            max(0, ref_dapi.shape[1] - detail_size),
        )
    )
    y1, x1 = (
        min(y0 + detail_size, ref_dapi.shape[0]),
        min(x0 + detail_size, ref_dapi.shape[1]),
    )
    detail_dapi = ref_dapi[y0:y1, x0:x1]
    detail_zeb1 = registered_zeb1[y0:y1, x0:x1]
    detail_quench = registered_quench[y0:y1, x0:x1]
    detail_corrected = corrected_zeb1[y0:y1, x0:x1]

    rows = (
        ("overview", overview_dapi, overview_zeb1, overview_quench, overview_corrected),
        (
            "full-resolution detail",
            detail_dapi,
            detail_zeb1,
            detail_quench,
            detail_corrected,
        ),
    )
    fig, axes = plt.subplots(2, 4, figsize=(240 / 25.4, 160 / 25.4))
    for row_index, (
        suffix,
        dapi_image,
        zeb1_image,
        quench_image,
        corrected_image,
    ) in enumerate(rows):
        panels = (
            ("R1 DAPI", _normalize_for_display(dapi_image), "gray"),
            (
                "Registered R5 AF750 (ZEB1 + background)",
                _normalize_for_display(zeb1_image),
                "gray",
            ),
            (
                "Registered R5Q AF750 (post-quench blank)",
                _normalize_for_display(quench_image),
                "gray",
            ),
            ("R5 − exposure-scaled R5Q", _merge_rgb(dapi_image, corrected_image), None),
        )
        for ax, (label, image, cmap) in zip(axes[row_index], panels):
            ax.imshow(
                image,
                cmap=cmap,
                vmin=0 if cmap else None,
                vmax=1 if cmap else None,
                rasterized=True,
            )
            ax.set_xlabel(f"{label}; {suffix}", fontsize=5, fontweight="normal")
            ax.set_title("")
            ax.tick_params(labelsize=5, width=0.5, length=2)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            ax.grid(True, which="major", axis="both", linewidth=0.5, color="#E5E7EB")
            ax.set_axisbelow(True)
    fig.suptitle(scene_label, fontsize=5, fontweight="normal")
    fig.subplots_adjust(
        left=0.045, right=0.995, bottom=0.10, top=0.94, wspace=0.06, hspace=0.16
    )
    out_path = QC_DIR / f"{scene_label}_cycif_zeb1_registration_segmentation_qc.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def process_scene(
    row: dict[str, object], parameters: dict[str, object], force: bool, save_qc: bool
) -> dict[str, object]:
    scene_label = str(row["slide_scene"])
    csv_path = PER_SCENE_DIR / f"{scene_label}.csv"
    receipt_path = PER_SCENE_DIR / f"{scene_label}.receipt.json"
    expected = scene_signature(row, parameters)
    if not force and cache_valid(csv_path, receipt_path, expected):
        cached = pd.read_csv(csv_path, usecols=["cycif_cell_id"])
        return {"slide_scene": scene_label, "n_cells": len(cached), "status": "cached"}

    started = time.time()
    ref_dapi = read_czi_channel(str(row["r1_path"]), DAPI_CHANNEL)
    r5_dapi = read_czi_channel(str(row["r5_path"]), DAPI_CHANNEL)
    r5q_dapi = read_czi_channel(str(row["r5q_path"]), DAPI_CHANNEL)
    r5_transform, r5_inliers, r5_matches, r5_status = register_round_pair(
        ref_dapi, r5_dapi
    )
    if r5_transform is None:
        raise RuntimeError(f"{scene_label}: R5-to-R1 registration failed: {r5_status}")
    r5q_transform, r5q_inliers, r5q_matches, r5q_status = register_round_pair(
        ref_dapi, r5q_dapi
    )
    if r5q_transform is None:
        raise RuntimeError(
            f"{scene_label}: R5Q-to-R1 registration failed: {r5q_status}"
        )

    r5_af750 = read_czi_channel(str(row["r5_path"]), ZEB1_CHANNEL)
    r5q_af750 = read_czi_channel(str(row["r5q_path"]), QUENCH_CHANNEL)
    registered_zeb1 = warp_image(r5_af750, r5_transform, ref_dapi.shape)
    registered_quench = warp_image(r5q_af750, r5q_transform, ref_dapi.shape)
    r5_support = (
        warp_image(np.ones_like(r5_dapi, dtype=np.uint8), r5_transform, ref_dapi.shape)
        > 0
    )
    r5q_support = (
        warp_image(
            np.ones_like(r5q_dapi, dtype=np.uint8), r5q_transform, ref_dapi.shape
        )
        > 0
    )
    joint_support = r5_support & r5q_support

    r5_exposure_ms = channel_exposure_ms(Path(row["r5_path"]), "AF750")
    r5q_exposure_ms = channel_exposure_ms(Path(row["r5q_path"]), "AF750")
    corrected_zeb1, exposure_scale = backsub_correct(
        registered_zeb1,
        registered_quench,
        r5_exposure_ms,
        r5q_exposure_ms,
    )
    exposure_scaled_quench = registered_quench.astype(np.float32) * np.float32(
        exposure_scale
    )

    enhanced = enhance_dapi(ref_dapi)
    nuclei = segment_dapi_cells(
        enhanced,
        threshold_scale=float(parameters["threshold_scale"]),
        min_nucleus_area=int(parameters["min_nucleus_area"]),
        distance_smoothing_sigma=float(parameters["distance_smoothing_sigma"]),
        min_peak_distance=int(parameters["min_peak_distance"]),
        min_peak_height=float(parameters["min_peak_height"]),
    )
    label_ids = np.arange(1, int(nuclei.max()) + 1, dtype=np.int32)
    if not label_ids.size:
        raise RuntimeError(f"{scene_label}: DAPI segmentation produced no nuclei")

    areas = ndi.sum(
        np.ones_like(nuclei, dtype=np.uint8), labels=nuclei, index=label_ids
    )
    centroids = ndi.center_of_mass(
        np.ones_like(nuclei, dtype=np.uint8), labels=nuclei, index=label_ids
    )
    registered_fraction = ndi.mean(joint_support, labels=nuclei, index=label_ids)
    zeb1_raw_mean = supported_label_mean(
        registered_zeb1, nuclei, label_ids, joint_support
    )
    quench_raw_mean = supported_label_mean(
        registered_quench, nuclei, label_ids, joint_support
    )
    quench_scaled_mean = supported_label_mean(
        exposure_scaled_quench, nuclei, label_ids, joint_support
    )
    zeb1_corrected_mean = supported_label_mean(
        corrected_zeb1, nuclei, label_ids, joint_support
    )
    px_um = get_czi_pixel_size(str(row["r1_path"]))
    if px_um is None:
        raise RuntimeError(
            f"{scene_label}: physical pixel size is absent from R1 metadata"
        )
    area_um2 = np.asarray(areas) * px_um**2
    support_keep = registered_fraction >= MIN_REGISTERED_FRACTION
    area_keep = (area_um2 >= MIN_NUCLEUS_AREA_UM2) & (area_um2 <= MAX_NUCLEUS_AREA_UM2)
    keep = support_keep & area_keep
    label_ids = label_ids[keep]
    accepted_lookup = np.zeros(int(nuclei.max()) + 1, dtype=bool)
    accepted_lookup[label_ids] = True
    accepted_nuclei = np.where(accepted_lookup[nuclei], nuclei, 0)

    frame = pd.DataFrame(
        {
            "slide_scene": scene_label,
            "tma_id": row["tma_id"],
            "core_id": row["core_id"],
            "progression_group": row["progression_group"],
            "cycif_cell_id": [f"{scene_label}:{label_id}" for label_id in label_ids],
            "cell_label": label_ids,
            "x_centroid_px": np.asarray([point[1] for point in centroids])[keep],
            "y_centroid_px": np.asarray([point[0] for point in centroids])[keep],
            "nucleus_area_px": np.asarray(areas)[keep],
            "nucleus_area_um2": area_um2[keep],
            "registered_fraction": np.asarray(registered_fraction)[keep],
            "ZEB1_nuclei_raw": np.asarray(zeb1_raw_mean)[keep],
            "ZEB1_nuclei_R5Q_raw": np.asarray(quench_raw_mean)[keep],
            "ZEB1_nuclei_R5Q_scaled": np.asarray(quench_scaled_mean)[keep],
            "ZEB1_nuclei": np.asarray(zeb1_corrected_mean)[keep],
        }
    )
    if frame.empty:
        raise RuntimeError(
            f"{scene_label}: no nuclei retained inside joint R5/R5Q support"
        )
    atomic_csv(csv_path, frame)
    qc_path = (
        _save_qc_pdf(
            scene_label,
            ref_dapi,
            registered_zeb1,
            exposure_scaled_quench,
            corrected_zeb1,
            accepted_nuclei,
        )
        if save_qc
        else None
    )
    receipt = {
        "inputs": expected,
        "output": str(csv_path.resolve()),
        "output_sha256": sha256_file(csv_path),
        "registration": {
            "r5_to_r1": {
                "status": r5_status,
                "n_matches": r5_matches,
                "n_inliers": r5_inliers,
                "inlier_fraction": r5_inliers / r5_matches if r5_matches else 0.0,
                "transform": r5_transform.tolist(),
            },
            "r5q_to_r1": {
                "status": r5q_status,
                "n_matches": r5q_matches,
                "n_inliers": r5q_inliers,
                "inlier_fraction": r5q_inliers / r5q_matches if r5q_matches else 0.0,
                "transform": r5q_transform.tolist(),
            },
        },
        "background_correction": {
            "preferred_method": PREFERRED_METHOD,
            "preferred_method_doi": PREFERRED_METHOD_DOI,
            "preferred_method_unavailable_reason": PREFERRED_METHOD_UNAVAILABLE_REASON,
            "implemented_method": BACKSUB_METHOD,
            "implemented_version": BACKSUB_VERSION,
            "implemented_tool_url": BACKSUB_URL,
            "adapter_reason": (
                "Backsub's CLI accepts OME-TIFF, while correction follows registration "
                "of separate CZI acquisitions; its published formula is preserved."
            ),
            "formula": "max(0, signal - background * signal_exposure/background_exposure)",
            "reference_round": QUENCH_ROUND,
            "reference_channel": QUENCH_CHANNEL,
            "reference_channel_name": "AF750",
            "target_exposure_ms": r5_exposure_ms,
            "reference_exposure_ms": r5q_exposure_ms,
            "reference_exposure_scale": exposure_scale,
        },
        "measurements": {
            "segmented_objects": int(nuclei.max()),
            "retained_nuclei": len(frame),
            "rejected_low_area": int((~(area_um2 >= MIN_NUCLEUS_AREA_UM2)).sum()),
            "rejected_high_area": int((~(area_um2 <= MAX_NUCLEUS_AREA_UM2)).sum()),
            "rejected_registered_support": int((~support_keep).sum()),
            "pixel_size_um": px_um,
        },
        "qc_pdf": str(qc_path.resolve()) if qc_path else None,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(receipt_path, receipt)
    return {"slide_scene": scene_label, "n_cells": len(frame), "status": "generated"}


def finalize_cohort(manifest: pd.DataFrame) -> None:
    frames: list[pd.DataFrame] = []
    scene_receipts: list[dict[str, object]] = []
    for scene_label in manifest["slide_scene"]:
        csv_path = PER_SCENE_DIR / f"{scene_label}.csv"
        receipt_path = PER_SCENE_DIR / f"{scene_label}.receipt.json"
        if not csv_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing generated cycIF checkpoint for {scene_label}"
            )
        frames.append(pd.read_csv(csv_path))
        scene_receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))

    cohort = pd.concat(frames, ignore_index=True)
    atomic_csv(FINAL_TABLE, cohort)

    summary = cohort.groupby(
        ["slide_scene", "tma_id", "core_id", "progression_group"], as_index=False
    ).agg(
        n_cells=("cycif_cell_id", "size"),
        median_ZEB1_nuclei_raw=("ZEB1_nuclei_raw", "median"),
        median_ZEB1_nuclei_R5Q_raw=("ZEB1_nuclei_R5Q_raw", "median"),
        median_ZEB1_nuclei_R5Q_scaled=("ZEB1_nuclei_R5Q_scaled", "median"),
        median_ZEB1_nuclei=("ZEB1_nuclei", "median"),
        zero_fraction_ZEB1_nuclei=(
            "ZEB1_nuclei",
            lambda values: float((values == 0).mean()),
        ),
        median_nucleus_area_um2=("nucleus_area_um2", "median"),
        median_registered_fraction=("registered_fraction", "median"),
    )
    atomic_csv(SUMMARY_TABLE, summary)
    correction_records = [
        receipt["background_correction"] for receipt in scene_receipts
    ]
    if len({json.dumps(record, sort_keys=True) for record in correction_records}) != 1:
        raise RuntimeError(
            "Scene receipts disagree on the ZEB1 background-correction contract"
        )
    atomic_json(
        COHORT_RECEIPT,
        {
            "script_version": SCRIPT_VERSION,
            "output": str(FINAL_TABLE.resolve()),
            "output_sha256": sha256_file(FINAL_TABLE),
            "summary": str(SUMMARY_TABLE.resolve()),
            "summary_sha256": sha256_file(SUMMARY_TABLE),
            "source_scene_receipt_sha256": [
                sha256_file(PER_SCENE_DIR / f"{scene}.receipt.json")
                for scene in manifest["slide_scene"]
            ],
            "background_correction": correction_records[0],
            "n_cells": len(cohort),
            "n_cores": int(cohort["slide_scene"].nunique()),
        },
    )
    print(
        f"Finalized {len(cohort):,} cells across {cohort['slide_scene'].nunique()} cores -> {FINAL_TABLE}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all", action="store_true", help="process all annotated non-orientation cores"
    )
    selection.add_argument("--scenes", nargs="+", help="specific slide_scene values")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="concurrent scene workers; each uses several GB RAM",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate even when a matching receipt exists",
    )
    parser.add_argument(
        "--no-qc", action="store_true", help="do not write per-scene QC PDFs"
    )
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-nucleus-area", type=int, default=40)
    parser.add_argument("--distance-smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--min-peak-distance", type=int, default=5)
    parser.add_argument("--min-peak-height", type=float, default=2.0)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    manifest = load_scene_manifest()
    if args.scenes:
        selected = manifest.loc[manifest["slide_scene"].isin(args.scenes)].copy()
        missing = sorted(set(args.scenes) - set(selected["slide_scene"]))
        if missing:
            raise SystemExit(f"Unknown slide_scene values: {missing}")
    else:
        selected = manifest

    parameters = {
        "threshold_scale": args.threshold_scale,
        "min_nucleus_area": args.min_nucleus_area,
        "distance_smoothing_sigma": args.distance_smoothing_sigma,
        "min_peak_distance": args.min_peak_distance,
        "min_peak_height": args.min_peak_height,
        "minimum_registered_fraction": MIN_REGISTERED_FRACTION,
        "reference_round": REFERENCE_ROUND,
        "zeb1_round": ZEB1_ROUND,
        "quench_round": QUENCH_ROUND,
        "dapi_channel": DAPI_CHANNEL,
        "zeb1_channel": ZEB1_CHANNEL,
        "quench_channel": QUENCH_CHANNEL,
        "background_method": BACKSUB_METHOD,
        "background_method_version": BACKSUB_VERSION,
        "background_method_url": BACKSUB_URL,
        "preferred_background_method": PREFERRED_METHOD,
        "preferred_background_method_doi": PREFERRED_METHOD_DOI,
        "preferred_background_method_unavailable_reason": PREFERRED_METHOD_UNAVAILABLE_REASON,
        "minimum_nucleus_area_um2": MIN_NUCLEUS_AREA_UM2,
        "maximum_nucleus_area_um2": MAX_NUCLEUS_AREA_UM2,
    }
    rows = selected.to_dict("records")
    results: list[dict[str, object]] = []
    if args.jobs == 1:
        for row in rows:
            result = process_scene(
                row, parameters, force=args.force, save_qc=not args.no_qc
            )
            results.append(result)
            print(result)
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(rows))) as pool:
            futures = {
                pool.submit(
                    process_scene, row, parameters, args.force, not args.no_qc
                ): row["slide_scene"]
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(result)

    if args.all:
        finalize_cohort(manifest)
    else:
        print("Selected-scene run complete; cohort table is finalized only by --all.")
    display_summary = pd.DataFrame(results).sort_values("slide_scene")
    print(display_summary.to_string(index=False))


if __name__ == "__main__":
    main()
