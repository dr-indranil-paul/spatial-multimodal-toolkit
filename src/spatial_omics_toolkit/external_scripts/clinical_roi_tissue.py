"""Shared physical-scale tissue-support QC for measured DAPI or HS intensity.

This stage changes support masks only. Native images and round-registration
receipts remain immutable. Masks are image-derived envelopes, not histology.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing

import clinical_roi_cycif as cycif


def derive_tissue_envelope(
    image: np.ndarray,
    observed: np.ndarray,
    *,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    max_edge: int = 1024,
    gaussian_sigma_um: float = 5.0,
    closing_radius_um: float = 15.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Separate a smoothed structural signal from dim acquired background."""
    if image.ndim != 2 or image.shape != observed.shape or observed.dtype != bool:
        raise ValueError("Supply a 2D intensity image and matching boolean observed support")
    settings = (pixel_size_x_um, pixel_size_y_um, gaussian_sigma_um, closing_radius_um)
    if not all(np.isfinite(value) and value > 0 for value in settings) or max_edge < 64:
        raise ValueError("Use positive physical parameters and a meaningful QC grid")
    if not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Structural intensity must be finite and nonnegative")
    height, width = image.shape
    reduction = max(1.0, max(height, width) / max_edge)
    small_width = max(1, int(round(width / reduction)))
    small_height = max(1, int(round(height / reduction)))
    size = (small_width, small_height)
    weights = cv2.resize(observed.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    reduced = cv2.resize(
        np.asarray(image, dtype=np.float32) * observed,
        size,
        interpolation=cv2.INTER_AREA,
    )
    pitch_x = pixel_size_x_um * width / small_width
    pitch_y = pixel_size_y_um * height / small_height
    sigma = (gaussian_sigma_um / pitch_y, gaussian_sigma_um / pitch_x)
    weighted_signal = ndimage.gaussian_filter(reduced, sigma=sigma)
    weighted_coverage = ndimage.gaussian_filter(weights, sigma=sigma)
    smooth = np.divide(
        weighted_signal,
        weighted_coverage,
        out=np.zeros_like(weighted_signal),
        where=weighted_coverage > 0,
    )
    valid = weights >= 0.99
    transformed = np.log1p(np.maximum(smooth, 0))
    values = transformed[valid]
    if not values.size or np.ptp(values) <= 0:
        raise ValueError("Structural intensity has no contrast for a tissue envelope")
    threshold = float(threshold_otsu(values))
    foreground = (transformed > threshold) & valid
    # The anisotropic physical footprint keeps the closing radius in micrometers.
    radius_y = max(1, int(np.ceil(closing_radius_um / pitch_y)))
    radius_x = max(1, int(np.ceil(closing_radius_um / pitch_x)))
    yy, xx = np.ogrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    footprint = ((yy * pitch_y) ** 2 + (xx * pitch_x) ** 2) <= closing_radius_um**2
    closed = binary_closing(foreground, footprint=footprint) & valid
    labels, count = ndimage.label(closed)
    if not count:
        raise ValueError("No tissue component was found")
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    selected = labels == int(np.argmax(areas))
    selected = ndimage.binary_fill_holes(selected) & valid
    native = cv2.resize(
        selected.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    native = native.astype(bool) & observed
    summary = {
        "coarse_shape_y": small_height,
        "coarse_shape_x": small_width,
        "coarse_pitch_x_um": pitch_x,
        "coarse_pitch_y_um": pitch_y,
        "log_otsu_threshold": threshold,
        "smoothed_count_threshold": float(np.expm1(threshold)),
        "candidate_component_count": int(count),
        "selected_component_fraction_of_foreground": float(areas.max() / areas.sum()),
        "refined_tissue_pixels": int(native.sum()),
        "observed_pixels": int(observed.sum()),
    }
    return native, {"intensity": reduced, "valid": valid, "mask": selected}, summary


def refine_cycif_tissue_masks(
    cycif_manifest_path: str | Path,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    plot_dir: str | Path,
    max_edge: int = 1024,
    gaussian_sigma_um: float = 5.0,
    closing_radius_um: float = 15.0,
    force: bool = False,
) -> dict[str, Any]:
    """Publish per-core refined masks and an auditable original/refined comparison."""
    if max_edge < 64 or gaussian_sigma_um <= 0 or closing_radius_um <= 0:
        raise ValueError("Use positive physical smoothing/closing and a meaningful QC grid")
    manifest_path = Path(cycif_manifest_path)
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    dapi_rows = frame.loc[frame["marker"].eq("DAPI")]
    if dapi_rows.empty or dapi_rows.duplicated(["assigned_tma", "confirmed_core_id"]).any():
        raise ValueError("Supply one authenticated R1 DAPI image per core")
    objects, tables, plots = Path(object_dir), Path(table_dir), Path(plot_dir)
    for directory in (objects, tables, plots):
        directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        "max_edge": max_edge,
        "gaussian_sigma_um": gaussian_sigma_um,
        "closing_radius_um": closing_radius_um,
        "reduction": "area_average_with_observed_coverage_weights",
        "threshold": "Otsu_on_log1p_smoothed_DAPI_within_observed_support",
        "selection": "largest_connected_component_then_fill_internal_holes",
        "native_mapping": "nearest_neighbor_then_intersect_native_observed_support",
    }
    rows: list[dict[str, Any]] = []
    for row in dapi_rows.to_dict("records"):
        parent_path = Path(row["receipt_path"])
        parent_sha = cycif.compute_sha256(parent_path)
        if parent_sha != row["receipt_sha256"]:
            raise ValueError("CycIF parent receipt changed")
        parent = json.loads(parent_path.read_text())
        authenticated = next(item for item in parent["table_rows"] if item["marker"] == "DAPI")
        for column in (
            "image_path",
            "support_path",
            "tissue_mask_path",
            "source_path",
            "source_sha256",
        ):
            if row[column] != authenticated[column]:
                raise ValueError(f"DAPI manifest row does not match its receipt: {column}")
        input_hashes = {}
        for column in ("image_path", "support_path", "tissue_mask_path"):
            path = Path(row[column])
            digest = cycif.compute_sha256(path)
            if parent["artifacts"][path.name] != digest:
                raise ValueError(f"Changed parent artifact: {path}")
            input_hashes[column] = {"path": str(path), "sha256": digest}
        if cycif.compute_sha256(row["source_path"]) != row["source_sha256"]:
            raise ValueError("Raw R1 source changed since processing")
        identity = {
            "parent_receipt": str(parent_path),
            "parent_sha256": parent_sha,
            "input_hashes": input_hashes,
            "parameters": parameters,
            "code_sha256": cycif.compute_sha256(Path(__file__)),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
            "skimage_version": importlib.metadata.version("scikit-image"),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = objects / f"{row['assigned_tma']}_{row['confirmed_core_id']}" / key
        directory.mkdir(parents=True, exist_ok=True)
        receipt_path = directory / "receipt.json"
        cached = None
        if not force and receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if saved["identity"] == identity and all(
                Path(item["path"]).exists()
                and cycif.compute_sha256(item["path"]) == item["sha256"]
                for item in saved["artifacts"]
            ):
                cached = saved
        if cached is None:
            image = np.load(row["image_path"], mmap_mode="r")
            observed = np.load(row["support_path"], mmap_mode="r")
            original = np.load(row["tissue_mask_path"], mmap_mode="r")
            native, previews, summary = derive_tissue_envelope(
                image,
                observed,
                pixel_size_x_um=float(row["pixel_size_x_um"]),
                pixel_size_y_um=float(row["pixel_size_y_um"]),
                max_edge=max_edge,
                gaussian_sigma_um=gaussian_sigma_um,
                closing_radius_um=closing_radius_um,
            )
            native_path = directory / "tissue_mask.npy"
            preview_path = directory / "review.npz"
            np.save(native_path, native)
            previews["original_mask"] = cv2.resize(
                original.astype(np.uint8),
                (summary["coarse_shape_x"], summary["coarse_shape_y"]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            np.savez_compressed(preview_path, **previews)
            summary.update(
                original_tissue_pixels=int(original.sum()),
                mask_jaccard_with_original=float(
                    (native & original).sum() / (native | original).sum()
                ),
            )
            cached = {
                "identity": identity,
                "summary": summary,
                "mask_path": str(native_path),
                "preview_path": str(preview_path),
                "claim_label": "Exploratory",
                "artifacts": [
                    {"path": str(path), "sha256": cycif.compute_sha256(path)}
                    for path in (native_path, preview_path)
                ],
            }
            receipt_path.write_text(json.dumps(cached, indent=2))
            print(
                f"CACHE_MISS_RECOMPUTED tissue support {row['assigned_tma']}_{row['confirmed_core_id']}",
                flush=True,
            )
        else:
            print(
                f"CACHE_HIT tissue support {row['assigned_tma']}_{row['confirmed_core_id']}",
                flush=True,
            )
        rows.append(
            {
                "assigned_tma": row["assigned_tma"],
                "confirmed_core_id": row["confirmed_core_id"],
                "reference_image_path": row["image_path"],
                "reference_image_sha256": input_hashes["image_path"]["sha256"],
                "reference_support_path": row["support_path"],
                "original_tissue_mask_path": row["tissue_mask_path"],
                "tissue_mask_path": cached["mask_path"],
                "tissue_mask_sha256": cycif.compute_sha256(cached["mask_path"]),
                "receipt_path": str(receipt_path),
                "receipt_sha256": cycif.compute_sha256(receipt_path),
                "preview_path": cached["preview_path"],
                "pixel_size_x_um": row["pixel_size_x_um"],
                "pixel_size_y_um": row["pixel_size_y_um"],
                "support_kind": "image-derived main-core envelope; not histological segmentation",
                **cached["summary"],
            }
        )
    result = pd.DataFrame(rows)
    output_manifest = tables / "cycif_refined_tissue_masks.csv"
    result.to_csv(output_manifest, index=False)
    pdf_path = plots / "cycif_refined_tissue_mask_review.pdf"
    global_receipt = tables / "cycif_refined_tissue_masks.receipt.json"
    plot_identity = {
        "manifest_sha256": cycif.compute_sha256(output_manifest),
        "code_sha256": cycif.compute_sha256(Path(__file__)),
        "matplotlib_version": importlib.metadata.version("matplotlib"),
    }
    saved_plot = json.loads(global_receipt.read_text()) if global_receipt.exists() else {}
    if (
        not force
        and saved_plot.get("identity") == plot_identity
        and pdf_path.exists()
        and saved_plot.get("pdf_sha256") == cycif.compute_sha256(pdf_path)
    ):
        print("CACHE_HIT tissue-support QC PDF", flush=True)
    else:
        render_tissue_review(result, pdf_path)
        global_receipt.write_text(
            json.dumps(
                {
                    "identity": plot_identity,
                    "manifest_path": str(output_manifest),
                    "pdf_path": str(pdf_path),
                    "pdf_sha256": cycif.compute_sha256(pdf_path),
                    "source_cycif_manifest_path": str(manifest_path),
                    "source_cycif_manifest_sha256": cycif.compute_sha256(manifest_path),
                    "claim_label": "Exploratory",
                    "parameters": parameters,
                },
                indent=2,
            )
        )
    return {
        "manifest": result,
        "manifest_path": output_manifest,
        "review_pdf": pdf_path,
        "receipt_path": global_receipt,
    }


def render_tissue_review(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    modality_label: str = "DAPI",
) -> None:
    """Compare original and refined support on one measured structural image.

    Each row supplies assigned_tma, confirmed_core_id, and a preview_path NPZ
    containing intensity, valid, mask, and original_mask on the same YX grid.
    """
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import plotnine_style_helper as style

    style.apply_matplotlib_style()
    with PdfPages(path) as document:
        for row in frame.to_dict("records"):
            with np.load(row["preview_path"]) as arrays:
                image = arrays["intensity"]
                maximum = max(float(np.percentile(image[arrays["valid"]], 99)), 1.0)
                figure, axes = plt.subplots(1, 2, figsize=(140 / 25.4, 85 / 25.4))
                figure.subplots_adjust(
                    left=0.03, right=0.97, top=0.88, bottom=0.16, wspace=0.08
                )
                for axis, name in zip(axes, ("original_mask", "mask")):
                    axis.imshow(image, cmap="gray", vmin=0, vmax=maximum)
                    axis.contour(arrays[name], levels=[0.5], colors=["cyan"], linewidths=0.5)
                    axis.axis("off")
                figure.text(
                    0.03,
                    0.94,
                    f"{row['assigned_tma']} {row['confirmed_core_id']}: {modality_label}-derived tissue support",
                    fontsize=5,
                )
                figure.text(
                    0.25, 0.9, "Original native-percentile mask", ha="center", fontsize=5
                )
                figure.text(
                    0.75,
                    0.9,
                    "Area average / Gaussian / log-Otsu envelope",
                    ha="center",
                    fontsize=5,
                )
                figure.text(
                    0.03,
                    0.09,
                    "Cyan = image-derived support; inspect tissue retention and exclusion of background / neighboring fragments.",
                    fontsize=5,
                )
                figure.text(
                    0.03,
                    0.04,
                    "Generator: clinical_roi_tissue.py; native intensities remain unchanged.",
                    fontsize=5,
                )
                document.savefig(figure)
                plt.close(figure)
