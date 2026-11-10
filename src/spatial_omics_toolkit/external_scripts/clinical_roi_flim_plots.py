"""Receipt-bound, descriptive plots for the reusable raw-FLIM workflow.

This module consumes saved stage arrays from ``clinical_roi_flim``. It never
reads raw decay cubes, estimates phasors, calibrates, filters, or fits models.
Histograms use all saved supported pixels; image-only stride is documented.
No pixel is treated as an independent biological replicate.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle
from matplotlib.ticker import NullFormatter, StrMethodFormatter

STYLE_PATH = Path(__file__).resolve().parent.parent / "01_utils/plotnine_style_helper.py"
if str(STYLE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(STYLE_PATH.parent))
from plotnine_style_helper import apply_matplotlib_style  # noqa: E402

Stage = Mapping[str, Any]
Labels = Mapping[str, str] | None
Renderer = Callable[[Sequence[str], Path, dict[str, Any]], list[Path]]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(stage: Stage, key: str) -> dict[str, np.ndarray]:
    from clinical_roi_flim import load_flim_detector

    return load_flim_detector(stage, key)


def _keyset(stages: Sequence[Stage]) -> list[str]:
    keys = list(stages[0]["detectors"])
    if not keys:
        raise ValueError("Plot requires at least one selected data block")
    for stage in stages[1:]:
        if set(stage["detectors"]) != set(keys):
            raise ValueError("Before/after plots require exactly matching acquisition/block keys")
    return keys


def _label(stage: Stage, key: str, labels: Labels) -> str:
    entry = stage["detectors"][key]
    if labels:
        for lookup in (key, entry.get("acquisition_id"), str(entry.get("path", ""))):
            if lookup in labels:
                return str(labels[lookup]) + f" / block {entry.get('detector_index', '?')}"
    source = Path(entry.get("path", entry.get("source_path", key)))
    name = f"{source.parent.name}/{source.name}"
    if len(name) > 65:
        name = name[:40] + "..." + name[-22:]
    return f"{name} / block {entry.get('detector_index', '?')}"


def _state_label(stage: Stage, key: str) -> str:
    entry = stage["detectors"][key]
    status = entry.get("calibration_status", stage.get("calibration_status", "UNCALIBRATED"))
    frequency = entry.get("frequency_mhz", stage.get("frequency_mhz"))
    return str(status).replace("_", " ") + (f"; {frequency:g} MHz" if isinstance(frequency, (int, float)) else "")


def _source_identity(stage: Stage, keys: Sequence[str]) -> dict[str, Any]:
    receipt = Path(stage["receipt_path"]).resolve(strict=True)
    recorded = json.loads(receipt.read_text())
    expected = {str(Path(item["path"]).resolve()): item["sha256"] for item in recorded["artifacts"]}
    objects: dict[str, Any] = {}
    for key in keys:
        path = Path(stage["detectors"][key]["object_path"]).resolve(strict=True)
        digest = _sha256(path)
        if expected.get(str(path)) != digest:
            raise ValueError(f"Plotted object differs from its completed stage receipt: {path}")
        objects[key] = {"path": str(path), "sha256": digest, "metadata": stage["detectors"][key]}
    return {
        "receipt_path": str(receipt),
        "receipt_sha256": _sha256(receipt),
        "objects": objects,
    }


def _render_cached(
    name: str,
    stages: Sequence[Stage],
    plot_dir: Path,
    parameters: dict[str, Any],
    renderer: Renderer,
    force: bool,
) -> dict[str, Any]:
    """Bind output reuse to code, style, stage receipts, arrays and display choices."""
    keys = _keyset(stages)
    if int(parameters["detectors_per_page"]) < 1 or int(parameters["max_image_edge"]) < 1:
        raise ValueError("detectors_per_page and max_image_edge must be positive")
    plot_dir = Path(plot_dir).resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = plot_dir / f"{name}_plot_receipt.json"
    identity = {
        "schema": "flim_plots_v1",
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "style_path": str(STYLE_PATH),
        "style_sha256": _sha256(STYLE_PATH),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "parameters": parameters,
        "stages": [_source_identity(stage, keys) for stage in stages],
    }
    # Normalize Path/NumPy metadata consistently for in-memory and JSON receipts.
    identity = json.loads(json.dumps(identity, sort_keys=True, default=str))
    if receipt_path.is_file() and not force:
        previous = json.loads(receipt_path.read_text())
        outputs = previous.get("outputs", {})
        if previous.get("identity") == identity and outputs and all(
            Path(path).is_file() and _sha256(Path(path)) == digest
            for path, digest in outputs.items()
        ):
            return {"pdf_paths": [Path(path) for path in outputs], "receipt_path": receipt_path,
                    "cache_status": "CACHE_HIT"}
    apply_matplotlib_style()
    diagnostics: dict[str, Any] = {}
    paths = renderer(keys, plot_dir, diagnostics)
    receipt = {
        "identity": identity,
        "claim_label": "Exploratory",
        "replication_note": "Pixel histograms describe these acquisitions; no biological inference or P values.",
        "image_display": "Native array order; nearest-neighbor display stride only. All histogram/support counts use native arrays.",
        "diagnostics": diagnostics,
        "outputs": {str(path): _sha256(path) for path in paths},
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(receipt_path)
    return {"pdf_paths": paths, "receipt_path": receipt_path,
            "cache_status": "FORCE_RECOMPUTE" if force else "CACHE_MISS_RECOMPUTED"}


def _page(rows: int, heading: str, columns: Sequence[str]) -> tuple[Figure, np.ndarray]:
    """Fixed 180-mm canvas with fixed row height and external panel labels."""
    height_mm = 22.0 + rows * 53.0
    fig, axes = plt.subplots(rows, 3, squeeze=False, figsize=(180 / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.065, right=0.925, bottom=16 / height_mm,
                        top=1 - 17 / height_mm, wspace=0.60, hspace=0.79)
    fig.text(0.065, 1 - 3.0 / height_mm, heading, va="top", fontsize=5)
    for index, label in enumerate(columns):
        pos = axes[0, index].get_position()
        fig.text(pos.x0, 1 - 8.0 / height_mm, label, va="top", fontsize=5)
    fig.text(0.065, 2.0 / height_mm,
             "Exploratory acquisition QC; pixels are descriptive. Generator: clinical_roi_flim_plots.py",
             fontsize=5, va="bottom")
    return fig, axes


def _row_label(fig: Figure, axes: np.ndarray, label: str) -> None:
    pos = axes[0].get_position()
    fig.text(pos.x0, pos.y1 + 0.026 * (75 / (fig.get_figheight() * 25.4)), label,
             fontsize=5, va="bottom")


def _format_axis(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(which="both", width=0.5, labelsize=5, pad=1)
    ax.set_axisbelow(True)
    # Avoid MathText log labels: their superscripts break exact 5-pt Arial.
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_major_formatter(StrMethodFormatter("{x:g}"))
            axis.set_minor_formatter(NullFormatter())


def _map(
    ax: Axes,
    array: np.ndarray,
    *,
    max_image_edge: int,
    cmap: str | ListedColormap,
    limits: tuple[float, float],
    colorbar_label: str | None = None,
) -> dict[str, Any]:
    data = np.asarray(array)
    original_shape = list(data.shape)
    leading_plane: list[int] = []
    if data.ndim > 2:
        leading_plane = [0] * (data.ndim - 2)
        data = data[tuple(leading_plane)]
    if data.ndim < 2:
        data = np.atleast_2d(data)
    stride = max(1, math.ceil(max(data.shape) / max_image_edge))
    handle = ax.imshow(data[::stride, ::stride], cmap=cmap, vmin=limits[0], vmax=limits[1],
                       origin="upper", interpolation="none",
                       extent=(-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    _format_axis(ax)
    annotation = f"display stride {stride}"
    if leading_plane:
        annotation = f"leading plane {leading_plane}; " + annotation
    ax.set_xlabel(annotation)
    if colorbar_label is not None:
        colorbar = ax.figure.colorbar(handle, ax=ax, fraction=0.05, pad=0.035)
        colorbar.set_label(colorbar_label, fontsize=5)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(labelsize=5, width=0.5, pad=1)
    return {"native_shape": original_shape, "leading_plane": leading_plane,
            "display_stride": stride, "display_limits": list(limits)}


def _finite_limits(arrays: Sequence[np.ndarray], quantiles: tuple[float, float] = (1, 99)) -> tuple[float, float]:
    values = [np.asarray(array)[np.isfinite(array)] for array in arrays]
    values = [array for array in values if array.size]
    if not values:
        return (0.0, 1.0)
    lo, hi = np.percentile(np.concatenate(values), quantiles)
    if hi <= lo:
        delta = max(abs(float(lo)) * 0.01, 0.5)
        lo, hi = lo - delta, hi + delta
    return (float(lo), float(hi))


def _support(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(arrays.get("valid", np.asarray(arrays["intensity"]) > 0), dtype=bool)
    if "real" in arrays and "imag" in arrays:
        valid = valid & np.isfinite(arrays["real"]) & np.isfinite(arrays["imag"])
    return valid


def _density(ax: Axes, arrays: Mapping[str, np.ndarray], bins: int = 160) -> dict[str, int | float]:
    support = _support(arrays)
    real, imag = arrays["real"][support], arrays["imag"][support]
    histogram, xedges, yedges = np.histogram2d(real, imag, bins=bins, range=((-1.05, 1.05), (-1.05, 1.05)))
    positive = np.ma.masked_less(histogram.T, 1)
    handle = ax.imshow(positive, origin="lower", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                       cmap="viridis", norm=LogNorm(vmin=1, vmax=max(2, int(histogram.max()))),
                       interpolation="none")
    # Geometric references only, not simulated decay data or fitted trajectories.
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="#777777", linewidth=0.5))
    ax.add_patch(Arc((0.5, 0), 1, 1, theta1=0, theta2=180, edgecolor="#D55E00", linewidth=0.5))
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel="G", ylabel="S")
    ax.set_aspect("equal")
    colorbar = ax.figure.colorbar(handle, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label("pixels / bin", fontsize=5)
    colorbar.formatter = StrMethodFormatter("{x:g}")
    colorbar.update_ticks()
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(labelsize=5, width=0.5, pad=1)
    _format_axis(ax)
    # A one-count pixel lies on the unit circle; floating-point roundoff must
    # not turn its last-bit error into an apparent out-of-disk QC finding.
    radius_squared_tolerance = 64 * max(np.finfo(real.dtype).eps, np.finfo(imag.dtype).eps)
    result = {"supported_pixels": int(support.sum()),
              "outside_unit_disk": int(np.count_nonzero(real * real + imag * imag > 1 + radius_squared_tolerance)),
              "unit_disk_radius_squared_tolerance": float(radius_squared_tolerance),
              "outside_display_window": int(support.sum() - histogram.sum())}
    ax.text(0, -0.19, f"{result['supported_pixels']:,} pixels; outside disk {result['outside_unit_disk']:,}\n"
            f"outside displayed G/S limits {result['outside_display_window']:,}", transform=ax.transAxes,
            fontsize=5, va="top")
    return result


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, format="pdf", dpi=300, metadata={"Creator": "clinical_roi_flim_plots.py"})
    plt.close(fig)


def plot_flim_qc(
    qc: Stage,
    plot_dir: Path,
    *,
    labels: Labels = None,
    detectors_per_page: int = 3,
    max_image_edge: int = 1024,
    force: bool = False,
) -> dict[str, Any]:
    """Plot native total counts, global decay at recorded times, and count histogram."""
    def render(keys: Sequence[str], output: Path, diagnostics: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for page, start in enumerate(range(0, len(keys), detectors_per_page), 1):
            subset = keys[start:start + detectors_per_page]
            fig, axes = _page(len(subset), "Raw FLIM QC", ("Total counts (display p1-p99)", "Recorded global decay", "Native pixel count distribution"))
            for row, key in enumerate(subset):
                arrays = _load(qc, key)
                intensity = arrays["intensity"]
                _row_label(fig, axes[row], _label(qc, key, labels))
                limits = _finite_limits([intensity])
                view = _map(axes[row, 0], intensity, max_image_edge=max_image_edge, cmap="gray",
                            limits=limits, colorbar_label="total counts")
                decay = arrays["global_decay"]
                times = arrays["time_seconds"] * 1e9
                positive = np.isfinite(decay) & (decay > 0)
                axes[row, 1].plot(times[positive], decay[positive], color="#0072B2", linewidth=0.5)
                axes[row, 1].set(xlabel="Recorded time (ns)", ylabel="Counts / time bin", yscale="log")
                values = intensity[np.isfinite(intensity) & (intensity >= 0)]
                axes[row, 2].hist(np.log1p(values), bins=80, histtype="step", color="#0072B2", linewidth=0.5)
                axes[row, 2].set(xlabel="log(1 + total counts / pixel)", ylabel="Pixels", yscale="log")
                for ax in axes[row, 1:]:
                    _format_axis(ax)
                diagnostics[key] = {"image": view, "native_pixel_count": int(intensity.size),
                                    "zero_count_pixels": int(np.count_nonzero(intensity == 0)),
                                    "time_bins": int(times.size)}
            path = output / f"flim_raw_qc_page{page:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        return paths

    return _render_cached("flim_raw_qc", [qc], plot_dir,
                          {"labels": labels, "detectors_per_page": detectors_per_page, "max_image_edge": max_image_edge}, render, force)


def plot_flim_phasors(
    phasor: Stage,
    plot_dir: Path,
    *,
    labels: Labels = None,
    detectors_per_page: int = 3,
    max_image_edge: int = 1024,
    force: bool = False,
) -> dict[str, Any]:
    """Plot saved G/S maps and full-pixel log density on fixed G/S axes.

    The unit circle and single-exponential semicircle are references, never
    exclusion masks. Each density colorbar reports its own absolute bin counts.
    """
    def render(keys: Sequence[str], output: Path, diagnostics: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for page, start in enumerate(range(0, len(keys), detectors_per_page), 1):
            subset = keys[start:start + detectors_per_page]
            fig, axes = _page(len(subset), "FLIM phasor coordinates; gray unit circle / orange single-exponential semicircle",
                              ("Saved real component", "Saved imaginary component", "All supported pixels"))
            for row, key in enumerate(subset):
                arrays = _load(phasor, key)
                _row_label(fig, axes[row], _label(phasor, key, labels) + " | " + _state_label(phasor, key))
                support = _support(arrays)
                views = {}
                for col, component in enumerate(("real", "imag")):
                    values = np.where(support, arrays[component], np.nan)
                    views[component] = _map(axes[row, col], values, max_image_edge=max_image_edge,
                                             cmap="coolwarm", limits=(-1, 1), colorbar_label="G" if col == 0 else "S")
                diagnostics[key] = {"maps": views, "density": _density(axes[row, 2], arrays)}
            path = output / f"flim_phasor_coordinates_page{page:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        return paths

    return _render_cached("flim_phasor_coordinates", [phasor], plot_dir,
                          {"labels": labels, "detectors_per_page": detectors_per_page, "max_image_edge": max_image_edge}, render, force)


def plot_flim_mask(
    raw: Stage,
    masked: Stage,
    plot_dir: Path,
    *,
    labels: Labels = None,
    detectors_per_page: int = 3,
    max_image_edge: int = 1024,
    force: bool = False,
) -> dict[str, Any]:
    """Display saved support before/after thresholding and native pixel coverage."""
    def render(keys: Sequence[str], output: Path, diagnostics: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for page, start in enumerate(range(0, len(keys), detectors_per_page), 1):
            subset = keys[start:start + detectors_per_page]
            fig, axes = _page(len(subset), "FLIM support review; blue = supported; white = excluded",
                              ("Before count threshold", "After count threshold", "Supported pixels / all native pixels"))
            for row, key in enumerate(subset):
                before, after = _load(raw, key), _load(masked, key)
                valid_before, valid_after = _support(before), _support(after)
                if valid_before.shape != valid_after.shape:
                    raise ValueError(f"Mask comparison changed native geometry for {key}")
                cutoff = masked["detectors"][key].get("minimum_total_counts")
                cutoff_label = f" | minimum total counts {cutoff:g}" if isinstance(cutoff, (int, float)) else ""
                _row_label(fig, axes[row], _label(masked, key, labels) + cutoff_label)
                views = [_map(axes[row, col], mask, max_image_edge=max_image_edge,
                              cmap=ListedColormap(["white", "#0072B2"]), limits=(0, 1))
                         for col, mask in enumerate((valid_before, valid_after))]
                totals = [int(valid_before.sum()), int(valid_after.sum())]
                percentages = [100 * count / valid_before.size for count in totals]
                axes[row, 2].bar(("Before", "After"), percentages, color=("#56B4E9", "#0072B2"),
                                 edgecolor="black", linewidth=0.5)
                axes[row, 2].set(ylabel="Supported native pixels (%)", ylim=(0, 110))
                for x, (count, percent) in enumerate(zip(totals, percentages, strict=True)):
                    axes[row, 2].text(x, percent + 3, f"{percent:.2f}%\n{count:,} pixels", ha="center", fontsize=5)
                _format_axis(axes[row, 2])
                diagnostics[key] = {"views": views, "before_supported": totals[0], "after_supported": totals[1],
                                    "all_native_pixels": int(valid_before.size),
                                    "newly_included_pixels": int(np.count_nonzero(valid_after & ~valid_before))}
            path = output / f"flim_count_mask_review_page{page:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        return paths

    return _render_cached("flim_count_mask_review", [raw, masked], plot_dir,
                          {"labels": labels, "detectors_per_page": detectors_per_page, "max_image_edge": max_image_edge}, render, force)


def plot_flim_filter(
    before: Stage,
    after: Stage,
    plot_dir: Path,
    *,
    labels: Labels = None,
    detectors_per_page: int = 3,
    max_image_edge: int = 1024,
    force: bool = False,
) -> dict[str, Any]:
    """Compare saved coordinate maps and density before/after optional smoothing.

    Each map PDF contains one acquisition/block in two rows (G and S). Separate
    density pages use the requested detector count per page. No filter is run.
    """
    def render(keys: Sequence[str], output: Path, diagnostics: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for index, key in enumerate(keys, 1):
            raw, smooth = _load(before, key), _load(after, key)
            valid_before, valid_after = _support(raw), _support(smooth)
            if valid_before.shape != valid_after.shape:
                raise ValueError(f"Filter comparison changed native geometry for {key}")
            joint = valid_before & valid_after
            fig, axes = _page(2, _label(after, key, labels) + " | optional phasor smoothing; " + _state_label(after, key),
                              ("Before filtering", "After filtering", "Difference on joint support (p99 |difference| limits)"))
            maps: dict[str, Any] = {}
            for row, component in enumerate(("real", "imag")):
                symbol = "G" if row == 0 else "S"
                _row_label(fig, axes[row], symbol + " coordinate")
                first = np.where(valid_before, raw[component], np.nan)
                second = np.where(valid_after, smooth[component], np.nan)
                delta = np.where(joint, second - first, np.nan)
                finite = np.abs(delta[np.isfinite(delta)])
                limit = max(float(np.percentile(finite, 99)) if finite.size else 0.0, np.finfo(np.float32).eps)
                maps[component] = [
                    _map(axes[row, 0], first, max_image_edge=max_image_edge, cmap="coolwarm", limits=(-1, 1), colorbar_label=symbol),
                    _map(axes[row, 1], second, max_image_edge=max_image_edge, cmap="coolwarm", limits=(-1, 1), colorbar_label=symbol),
                    _map(axes[row, 2], delta, max_image_edge=max_image_edge, cmap="coolwarm", limits=(-limit, limit), colorbar_label="change in " + symbol),
                ]
            diagnostics[key] = {"maps": maps, "joint_supported_pixels": int(joint.sum()),
                                "support_lost": int(np.count_nonzero(valid_before & ~valid_after)),
                                "support_gained": int(np.count_nonzero(valid_after & ~valid_before)),
                                "intensity_arrays_equal": bool(np.array_equal(raw["intensity"], smooth["intensity"], equal_nan=True))}
            path = output / f"flim_filter_coordinate_maps_block{index:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        for page, start in enumerate(range(0, len(keys), detectors_per_page), 1):
            subset = keys[start:start + detectors_per_page]
            fig, axes = _page(len(subset), "Optional phasor smoothing: coordinate density and displacement",
                              ("Before filtering", "After filtering", "Coordinate displacement on joint support"))
            for row, key in enumerate(subset):
                raw, smooth = _load(before, key), _load(after, key)
                _row_label(fig, axes[row], _label(after, key, labels))
                diagnostics[key]["density_before"] = _density(axes[row, 0], raw)
                diagnostics[key]["density_after"] = _density(axes[row, 1], smooth)
                joint = _support(raw) & _support(smooth)
                distance = np.hypot(smooth["real"][joint] - raw["real"][joint], smooth["imag"][joint] - raw["imag"][joint])
                axes[row, 2].hist(distance, bins=80, histtype="step", color="#0072B2", linewidth=0.5)
                axes[row, 2].set(xlabel="Euclidean change in (G, S)", ylabel="Joint-support pixels", yscale="log")
                _format_axis(axes[row, 2])
            path = output / f"flim_filter_density_review_page{page:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        return paths

    return _render_cached("flim_filter_review", [before, after], plot_dir,
                          {"labels": labels, "detectors_per_page": detectors_per_page, "max_image_edge": max_image_edge}, render, force)


def plot_flim_lifetimes(
    lifetime: Stage,
    plot_dir: Path,
    *,
    labels: Labels = None,
    detectors_per_page: int = 3,
    max_image_edge: int = 1024,
    limits_ns: tuple[float, float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plot saved apparent lifetimes without re-estimating or imposing chemistry.

    Defaults use the pooled phase/modulation p1-p99 per block for display only;
    finite counts and tails outside that display range are reported. Explicit
    ``limits_ns`` applies shared bounds to every block. Undefined values remain
    missing. The calibration status is visible on every row.
    """
    if limits_ns is not None and (not all(np.isfinite(limits_ns)) or limits_ns[1] <= limits_ns[0]):
        raise ValueError("limits_ns must contain two finite increasing values")

    def render(keys: Sequence[str], output: Path, diagnostics: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for page, start in enumerate(range(0, len(keys), detectors_per_page), 1):
            subset = keys[start:start + detectors_per_page]
            heading = "Apparent FLIM lifetimes; display limits " + ("specified in ns" if limits_ns else "pooled phase/modulation p1-p99 per block")
            fig, axes = _page(len(subset), heading, ("Apparent phase lifetime", "Apparent modulation lifetime", "Distributions; full finite counts and clipped tails"))
            for row, key in enumerate(subset):
                arrays = _load(lifetime, key)
                valid = _support(arrays)
                phase = np.where(valid, arrays["phase_lifetime_ns"], np.nan)
                modulation = np.where(valid, arrays["modulation_lifetime_ns"], np.nan)
                limits = limits_ns or _finite_limits([phase, modulation])
                _row_label(fig, axes[row], _label(lifetime, key, labels) + " | " + _state_label(lifetime, key))
                views: dict[str, Any] = {}
                for col, (name, values, color) in enumerate((("phase", phase, "#0072B2"), ("modulation", modulation, "#D55E00"))):
                    views[name] = _map(axes[row, col], values, max_image_edge=max_image_edge, cmap="viridis", limits=limits, colorbar_label="ns")
                    finite = values[np.isfinite(values)]
                    outside = int(np.count_nonzero((finite < limits[0]) | (finite > limits[1])))
                    axes[row, 2].hist(finite, bins=80, range=limits, histtype="step", linewidth=0.5, color=color,
                                     label=f"{name}: {finite.size:,} finite; {outside:,} outside")
                    views[name].update({"finite_pixels": int(finite.size), "outside_display_limits": outside,
                                        "supported_but_undefined": int(valid.sum() - finite.size)})
                axes[row, 2].set(xlabel="Apparent lifetime (ns)", ylabel="Pixels", xlim=limits)
                axes[row, 2].legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=5)
                _format_axis(axes[row, 2])
                status = lifetime["detectors"][key].get("lifetime_status", "")
                if status:
                    axes[row, 2].text(0, -0.3, str(status).replace("_", " "), transform=axes[row, 2].transAxes, fontsize=5, va="top")
                diagnostics[key] = {"maps": views, "calibration": _state_label(lifetime, key), "lifetime_status": status}
            path = output / f"flim_apparent_lifetimes_page{page:02d}.pdf"
            _save(fig, path)
            paths.append(path)
        return paths

    return _render_cached("flim_apparent_lifetimes", [lifetime], plot_dir,
                          {"labels": labels, "detectors_per_page": detectors_per_page, "max_image_edge": max_image_edge,
                           "limits_ns": limits_ns}, render, force)
