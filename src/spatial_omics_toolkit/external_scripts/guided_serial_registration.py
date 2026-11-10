"""Guided channel-aware serial-section registration for PREVAIL Figure 6.

Structural registration images are kept separate from quantitative channels. Every
reported metric is calculated from loaded images or user-supplied landmarks.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, registration, transform


def enable_deeperhistreg_native_deps(vips_bin: str | Path | None = None) -> None:
    """Registers the libvips DLL directory before importing deeperhistreg/pyvips.

    On this host, conda-forge's libvips build causes a GObject double-type-registration
    conflict with pyvips (traced during a 2026-08-21 VALIS installation investigation).
    The officially-distributed libvips-build-win64-mxe release (correctly-named
    libvips-42.dll) works cleanly instead. Call this before ``import deeperhistreg`` or
    ``import pyvips``; it is a no-op if the directory does not exist (e.g. on a host where
    libvips is already on PATH some other way).
    """
    import os

    default_bin = Path(
        r"/home/groups/EmiliLab/pauli/00_environments/vips-dev-x64/vips-dev-8.18/bin"
    )
    bin_dir = Path(vips_bin) if vips_bin is not None else default_bin
    if not bin_dir.is_dir():
        return
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(str(bin_dir))


@dataclass(frozen=True)
class RegistrationConfig:
    dapi_weight: float = 0.45
    autofluorescence_weight: float = 0.35
    edge_weight: float = 0.20
    tissue_percentile: float = 60.0
    closing_radius: int = 8
    orb_features: int = 12000
    ratio_test: float = 0.78
    ransac_residual_px: float = 4.0
    ransac_trials: int = 20000
    flow_attachment: float = 20.0
    flow_tightness: float = 0.5
    flow_num_warp: int = 10


@dataclass
class RegistrationCandidate:
    name: str
    matrix: np.ndarray
    fixed: np.ndarray
    moving: np.ndarray
    affine_warped: np.ndarray
    deformable_warped: np.ndarray | None = None
    flow_v: np.ndarray | None = None
    flow_u: np.ndarray | None = None
    match_fixed_xy: np.ndarray | None = None
    match_moving_xy: np.ndarray | None = None
    metrics: dict | None = None


def robust01(image: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Image has no finite pixels")
    lo, hi = np.percentile(finite, (low, high))
    return (
        np.zeros_like(values) if hi <= lo else np.clip((values - lo) / (hi - lo), 0, 1)
    )


def project_stack(stack: np.ndarray, method: str = "max") -> np.ndarray:
    values = np.asarray(stack)
    if values.ndim == 2:
        return values.astype(np.float32, copy=False)
    if values.ndim != 3:
        raise ValueError(f"Expected 2D or ZYX data, received {values.shape}")
    if method == "max":
        return np.max(values, axis=0).astype(np.float32)
    if method == "median":
        return np.median(values, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported projection: {method}")


def tissue_mask(
    image: np.ndarray, percentile: float = 60.0, closing_radius: int = 8
) -> np.ndarray:
    normalized = robust01(image)
    nonzero = normalized[normalized > 0]
    threshold = np.percentile(nonzero, percentile) if nonzero.size else 0.0
    mask = morphology.closing(normalized >= threshold, morphology.disk(closing_radius))
    mask = ndi.binary_fill_holes(mask)
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask.astype(bool)
    largest = max(measure.regionprops(labels), key=lambda region: region.area)
    return labels == largest.label


def tissue_centroid(image: np.ndarray) -> tuple[float, float]:
    """Returns the (row, col) centroid of the largest tissue component in `image`."""
    mask = tissue_mask(image)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return image.shape[0] / 2.0, image.shape[1] / 2.0
    return float(ys.mean()), float(xs.mean())


def center_crop_or_pad(
    image: np.ndarray, size: int, centroid: tuple[float, float] | None = None
) -> np.ndarray:
    """Returns `image` cropped/zero-padded to a (size, size) canvas centered on `centroid`
    (row, col), or the array's own tissue centroid via :func:`tissue_centroid` if omitted.
    Used to reconcile independently-detected core crops (different tissue extents per
    modality) onto one common physical canvas before shape-equality checks."""
    h, w = image.shape[:2]
    cy, cx = centroid if centroid is not None else tissue_centroid(image)
    y0 = int(round(cy - size / 2.0))
    x0 = int(round(cx - size / 2.0))
    out = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    src_y0, src_y1 = max(y0, 0), min(y0 + size, h)
    src_x0, src_x1 = max(x0, 0), min(x0 + size, w)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    dst_y1, dst_x1 = dst_y0 + (src_y1 - src_y0), dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def nuclear_density(dapi: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    normalized = robust01(dapi)
    positive = normalized[normalized > 0]
    threshold = filters.threshold_otsu(positive) if positive.size else 0.0
    nuclei = morphology.remove_small_objects(normalized >= threshold, min_size=5)
    return robust01(ndi.gaussian_filter(nuclei.astype(np.float32), sigma=sigma))


def build_structural_composite(channels, roles, config=RegistrationConfig()):
    """Create a morphology-only composite from explicit channel roles."""
    if set(channels) != set(roles):
        raise ValueError("Every channel requires exactly one role")
    allowed = {"dapi", "autofluorescence", "structural", "quantitative_only", "exclude"}
    if set(roles.values()) - allowed:
        raise ValueError(f"Unsupported roles: {sorted(set(roles.values()) - allowed)}")
    projected = {name: project_stack(image) for name, image in channels.items()}
    usable = [
        name
        for name, role in roles.items()
        if role in {"dapi", "autofluorescence", "structural"}
    ]
    if not usable:
        raise ValueError("At least one registration-capable channel is required")
    components = {}
    dapi = [name for name, role in roles.items() if role == "dapi"]
    af = [name for name, role in roles.items() if role == "autofluorescence"]
    structural = [name for name, role in roles.items() if role == "structural"]
    if dapi:
        components["dapi"] = robust01(
            np.mean([robust01(projected[n]) for n in dapi], axis=0)
        )
        components["nuclear_density"] = nuclear_density(components["dapi"])
    if af:
        components["autofluorescence"] = robust01(
            np.mean([robust01(projected[n]) for n in af], axis=0)
        )
    if structural:
        components["structural"] = robust01(
            np.mean([robust01(projected[n]) for n in structural], axis=0)
        )
    base = np.mean([robust01(projected[name]) for name in usable], axis=0)
    components["edges"] = robust01(filters.sobel(base))
    mask_source = components.get("autofluorescence", components.get("structural", base))
    components["tissue_mask"] = tissue_mask(
        mask_source, config.tissue_percentile, config.closing_radius
    ).astype(np.float32)
    weighted = []
    if "dapi" in components:
        weighted.extend(
            [
                (config.dapi_weight, components["dapi"]),
                (config.dapi_weight * 0.5, components["nuclear_density"]),
            ]
        )
    if "autofluorescence" in components:
        weighted.append(
            (config.autofluorescence_weight, components["autofluorescence"])
        )
    if "structural" in components:
        weighted.append((config.autofluorescence_weight, components["structural"]))
    weighted.append((config.edge_weight, components["edges"]))
    total = sum(weight for weight, _ in weighted)
    components["composite"] = (
        robust01(sum(weight * image for weight, image in weighted) / total)
        * components["tissue_mask"]
    )
    return components


def plot_channel_audit(channels, roles, components):
    names = list(channels) + list(components)
    cols, rows = 4, int(np.ceil(len(names) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        image = project_stack(channels[name]) if name in channels else components[name]
        ax.imshow(robust01(image), cmap="gray")
        ax.set_title(f"{name}\nrole={roles[name]}" if name in channels else name)
        ax.axis("off")
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    fig.suptitle("Channel audit: registration signal retained versus excluded")
    fig.tight_layout()
    return fig


def estimate_affine_orb(fixed, moving, config=RegistrationConfig()):
    fixed8 = np.round(robust01(fixed) * 255).astype(np.uint8)
    moving8 = np.round(robust01(moving) * 255).astype(np.uint8)
    detector = cv2.ORB_create(nfeatures=config.orb_features)
    key_fixed, desc_fixed = detector.detectAndCompute(fixed8, None)
    key_moving, desc_moving = detector.detectAndCompute(moving8, None)
    if desc_fixed is None or desc_moving is None:
        raise RuntimeError("ORB found insufficient descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_moving, desc_fixed, k=2)
    good = [
        first
        for first, second in pairs
        if first.distance < config.ratio_test * second.distance
    ]
    if len(good) < 6:
        raise RuntimeError(f"Only {len(good)} robust matches; at least 6 required")
    moving_xy = np.float32([key_moving[m.queryIdx].pt for m in good])
    fixed_xy = np.float32([key_fixed[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffine2D(
        moving_xy,
        fixed_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_residual_px,
        maxIters=config.ransac_trials,
        confidence=0.995,
        refineIters=25,
    )
    if matrix is None or inliers is None:
        raise RuntimeError("RANSAC affine estimation failed")
    keep = inliers.ravel().astype(bool)
    warped = cv2.warpAffine(
        np.asarray(moving, np.float32),
        matrix,
        (fixed.shape[1], fixed.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return RegistrationCandidate(
        "ORB + RANSAC affine",
        matrix,
        np.asarray(fixed, np.float32),
        np.asarray(moving, np.float32),
        warped,
        match_fixed_xy=fixed_xy[keep],
        match_moving_xy=moving_xy[keep],
    )


def refine_deformable(candidate, config=RegistrationConfig()):
    fixed, moving = robust01(candidate.fixed), robust01(candidate.affine_warped)
    flow_v, flow_u = registration.optical_flow_tvl1(
        fixed,
        moving,
        attachment=config.flow_attachment,
        tightness=config.flow_tightness,
        num_warp=config.flow_num_warp,
        prefilter=True,
    )
    yy, xx = np.meshgrid(
        np.arange(fixed.shape[0]), np.arange(fixed.shape[1]), indexing="ij"
    )
    candidate.deformable_warped = transform.warp(
        moving,
        np.array([yy + flow_v, xx + flow_u]),
        mode="constant",
        preserve_range=True,
    ).astype(np.float32)
    candidate.flow_v, candidate.flow_u = (
        flow_v.astype(np.float32),
        flow_u.astype(np.float32),
    )
    return candidate


def _ncc(a, b, mask):
    x, y = robust01(a)[mask], robust01(b)[mask]
    return (
        float(np.corrcoef(x, y)[0, 1])
        if x.size > 1 and np.std(x) and np.std(y)
        else float("nan")
    )


def _mi(a, b, mask, bins=64):
    x, y = robust01(a)[mask], robust01(b)[mask]
    if x.size < 2:
        return float("nan")
    hist = np.histogram2d(x, y, bins=bins, range=((0, 1), (0, 1)))[0]
    p = hist / hist.sum()
    expected = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
    valid = p > 0
    return float(np.sum(p[valid] * np.log(p[valid] / expected[valid])))


def jacobian_determinant(flow_v, flow_u):
    dv_dy, dv_dx = np.gradient(flow_v)
    du_dy, du_dx = np.gradient(flow_u)
    return (1 + dv_dy) * (1 + du_dx) - dv_dx * du_dy


def evaluate_candidate(candidate, fixed_mask=None):
    fixed = robust01(candidate.fixed)
    fixed_mask = (
        tissue_mask(fixed) if fixed_mask is None else np.asarray(fixed_mask, bool)
    )
    stages = {"unregistered": candidate.moving, "affine": candidate.affine_warped}
    if candidate.deformable_warped is not None:
        stages["deformable"] = candidate.deformable_warped
    metrics = {}
    for stage, image in stages.items():
        moving_mask = tissue_mask(image)
        intersection = np.logical_and(fixed_mask, moving_mask)
        union = np.logical_or(fixed_mask, moving_mask)
        metrics[f"{stage}_dice"] = float(
            2 * intersection.sum() / (fixed_mask.sum() + moving_mask.sum())
        )
        metrics[f"{stage}_iou"] = (
            float(intersection.sum() / union.sum()) if union.any() else float("nan")
        )
        metrics[f"{stage}_ncc"] = _ncc(fixed, image, intersection)
        metrics[f"{stage}_mutual_information"] = _mi(fixed, image, intersection)
    if candidate.match_fixed_xy is not None:
        prediction = (
            np.column_stack(
                [candidate.match_moving_xy, np.ones(len(candidate.match_moving_xy))]
            )
            @ candidate.matrix.T
        )
        residual = np.linalg.norm(prediction - candidate.match_fixed_xy, axis=1)
        metrics.update(
            inlier_count=int(len(residual)),
            inlier_tre_median_px=float(np.median(residual)),
            inlier_tre_p95_px=float(np.percentile(residual, 95)),
        )
    if candidate.flow_v is not None:
        jac = jacobian_determinant(candidate.flow_v, candidate.flow_u)
        mag = np.hypot(candidate.flow_v, candidate.flow_u)
        metrics.update(
            folded_fraction=float(np.mean(jac <= 0)),
            jacobian_p01=float(np.percentile(jac, 1)),
            jacobian_p99=float(np.percentile(jac, 99)),
            displacement_median_px=float(np.median(mag[fixed_mask])),
            displacement_p95_px=float(np.percentile(mag[fixed_mask], 95)),
        )
    candidate.metrics = metrics
    return metrics


def load_landmarks(path):
    table = pd.read_csv(path)
    required = {"landmark_id", "fixed_x", "fixed_y", "moving_x", "moving_y", "role"}
    if required - set(table):
        raise ValueError(f"Missing columns: {sorted(required - set(table))}")
    if not set(table.role).issubset({"fit", "validation"}):
        raise ValueError("role must be fit or validation")
    return table


def transform_points_affine(points_xy, matrix):
    values = np.asarray(points_xy, float)
    return np.column_stack([values, np.ones(len(values))]) @ matrix.T


def landmark_metrics(table, candidate, pixel_size_um):
    validation = table.loc[table.role == "validation"].copy()
    if validation.empty:
        raise ValueError("At least one held-out validation landmark is required")
    predicted = transform_points_affine(
        validation[["moving_x", "moving_y"]], candidate.matrix
    )
    fixed = validation[["fixed_x", "fixed_y"]].to_numpy(float)
    errors = np.linalg.norm(predicted - fixed, axis=1)
    validation[["predicted_x", "predicted_y"]] = predicted
    validation["error_px"] = errors
    validation["error_um"] = errors * float(pixel_size_um)
    summary = {
        "validation_n": int(len(validation)),
        "tre_median_um": float(np.median(validation.error_um)),
        "tre_p95_um": float(np.percentile(validation.error_um, 95)),
        "tre_max_um": float(validation.error_um.max()),
    }
    return validation, summary


def plot_registration_diagnostics(candidate, fixed_mask=None):
    if candidate.metrics is None:
        evaluate_candidate(candidate, fixed_mask)
    fixed = robust01(candidate.fixed)
    final = robust01(
        candidate.deformable_warped
        if candidate.deformable_warped is not None
        else candidate.affine_warped
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    images = [
        ("Fixed structural image", fixed),
        ("Moving before registration", robust01(candidate.moving)),
        ("Affine result", robust01(candidate.affine_warped)),
        ("Final result", final),
    ]
    for ax, (title, image) in zip(axes[0], images):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    overlay = np.zeros((*fixed.shape, 3), np.float32)
    overlay[..., 0] = fixed
    overlay[..., 1] = final
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Red=fixed, green=moving, yellow=agreement")
    axes[1, 0].axis("off")
    checker = np.indices(fixed.shape).sum(0) // 64 % 2
    axes[1, 1].imshow(np.where(checker, fixed, final), cmap="gray")
    axes[1, 1].set_title("Checkerboard: boundaries and tears")
    axes[1, 1].axis("off")
    if candidate.flow_v is not None:
        jac = jacobian_determinant(candidate.flow_v, candidate.flow_u)
        mag = np.hypot(candidate.flow_v, candidate.flow_u)
        im = axes[1, 2].imshow(jac, cmap="coolwarm", vmin=0.5, vmax=1.5)
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
        axes[1, 2].set_title("Jacobian: ≤0 means folding (FAIL)")
        im = axes[1, 3].imshow(mag, cmap="magma")
        fig.colorbar(im, ax=axes[1, 3], fraction=0.046)
        axes[1, 3].set_title("Local displacement magnitude (px)")
    else:
        axes[1, 2].axis("off")
        axes[1, 3].axis("off")
    fig.suptitle("Registration diagnostics: statistics must agree with anatomy")
    fig.tight_layout()
    return fig


def plot_metric_defense(metrics):
    rows = [
        {"stage": stage, "metric": metric, "value": metrics[f"{stage}_{metric}"]}
        for stage in ("unregistered", "affine", "deformable")
        for metric in ("dice", "iou", "ncc", "mutual_information")
        if f"{stage}_{metric}" in metrics
    ]
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric in ("dice", "iou", "ncc"):
        subset = frame[frame.metric == metric]
        axes[0].plot(subset.stage, subset.value, marker="o", label=metric.upper())
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Higher is better")
    axes[0].set_title("Overlap and structural agreement")
    axes[0].legend()
    mi = frame[frame.metric == "mutual_information"]
    axes[1].bar(mi.stage, mi.value)
    axes[1].set_title("Mutual information")
    axes[1].set_ylabel("Higher is better")
    fig.tight_layout()
    return fig, frame


def decision_table(
    candidate, landmark_summary=None, max_folded_fraction=0.0, min_dice_gain=0.02
):
    metrics = candidate.metrics or evaluate_candidate(candidate)
    final = "deformable" if candidate.deformable_warped is not None else "affine"
    checks = [
        (
            "Tissue overlap improves",
            metrics[f"{final}_dice"] - metrics["unregistered_dice"] >= min_dice_gain,
            metrics[f"{final}_dice"] - metrics["unregistered_dice"],
            f">= {min_dice_gain}",
        ),
        (
            "No deformation folding",
            metrics.get("folded_fraction", 0.0) <= max_folded_fraction,
            metrics.get("folded_fraction", 0.0),
            f"<= {max_folded_fraction}",
        ),
        (
            "Mutual information improves",
            metrics[f"{final}_mutual_information"]
            > metrics["unregistered_mutual_information"],
            metrics[f"{final}_mutual_information"]
            - metrics["unregistered_mutual_information"],
            "> 0",
        ),
    ]
    if landmark_summary is not None:
        checks.append(
            (
                "Held-out landmarks supplied",
                landmark_summary.get("validation_n", 0) >= 6,
                landmark_summary.get("validation_n", 0),
                ">= 6",
            )
        )
    table = pd.DataFrame(checks, columns=["criterion", "pass", "observed", "required"])
    table["status"] = np.where(table["pass"], "PASS", "FAIL")
    return table


def save_registration_bundle(
    output_dir, candidate, config, decision, landmark_rows=None
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "affine_matrix.npy", candidate.matrix)
    if candidate.flow_v is not None:
        np.savez_compressed(
            output / "deformable_field.npz",
            flow_v=candidate.flow_v,
            flow_u=candidate.flow_u,
        )
    pd.DataFrame([candidate.metrics or {}]).to_csv(
        output / "registration_metrics.csv", index=False
    )
    decision.to_csv(output / "registration_decision.csv", index=False)
    if landmark_rows is not None:
        landmark_rows.to_csv(output / "held_out_landmark_errors.csv", index=False)
    provenance = {
        "candidate": candidate.name,
        "config": asdict(config),
        "decision": "PASS" if bool(decision["pass"].all()) else "FAIL",
    }
    (output / "registration_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance
