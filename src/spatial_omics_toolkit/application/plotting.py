from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_arial() -> str:
    """Register the shared Arial files and reject silent font substitution."""
    from matplotlib import font_manager

    family = font_manager.FontProperties(family="Arial", style="normal", weight="normal")
    try:
        return str(font_manager.findfont(family, fallback_to_default=False))
    except ValueError:
        software_root = Path(os.environ.get("PAULI_FAST_SOFTWARES", "/home/groups/EmiliLab/pauli/00_environments"))
        font_dir = software_root / "share" / "fonts" / "arial"
        if font_dir.is_dir():
            for font in sorted(font_dir.iterdir()):
                if font.is_file() and font.suffix.lower() == ".ttf":
                    font_manager.fontManager.addfont(str(font))
        try:
            return str(font_manager.findfont(family, fallback_to_default=False))
        except ValueError as error:
            raise RuntimeError(f"Arial is required for scientific plots; install/register the approved font under {font_dir}.") from error


# Descriptive views of saved artifacts only: category colors do not aggregate
# observations. Units and replication remain source-defined.
import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_ROWS = 500_000
MAX_PIXELS = 16_000_000
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PlotSettings:
    kind: str = "scatter"
    x: str = ""
    y: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xscale: str = "linear"
    yscale: str = "linear"
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    color: str = "#2B6CB0"
    cmap: str = "viridis"
    marker: str = "o"
    line_style: str = "-"
    category: str = ""
    colorbar: bool = False
    marker_size: float = 3.14
    line_width: float = 0.5
    opacity: float = 1.0
    font: str = "Arial"
    font_size: float = 5.0
    grid: bool = True
    legend: bool = False
    width_mm: float = 85.0
    height_mm: float = 65.0
    dpi: int = 300
    vmin: float | None = None
    vmax: float | None = None


def source_path(path: Path, allowed_root: Path) -> Path:
    """Resolve an existing bounded file within the approved workspace."""
    resolved = path.expanduser().resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Source must be a file within the approved workspace.")
    if resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"Source exceeds {MAX_SOURCE_BYTES // (1024 * 1024)} MiB.")
    if resolved.suffix.lower() not in TABLE_SUFFIXES | IMAGE_SUFFIXES:
        raise ValueError("Unsupported source. Existing PDFs cannot be replotted.")
    return resolved


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: Path, allowed_root: Path) -> tuple[pd.DataFrame | np.ndarray, str]:
    source = source_path(path, allowed_root)
    digest = source_hash(source)
    if source.suffix.lower() in TABLE_SUFFIXES:
        if source.suffix.lower() == ".parquet":
            # Inspect metadata before materializing potentially compressed tables.
            import pyarrow.parquet as pq

            metadata = pq.ParquetFile(source).metadata
            if metadata.num_rows > MAX_ROWS or metadata.num_columns > 1000:
                raise ValueError("Table exceeds the row or column limit.")
            if sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) > MAX_SOURCE_BYTES * 4:
                raise ValueError("Uncompressed table exceeds the memory limit.")
            data = pd.read_parquet(source)
        else:
            data = pd.read_csv(source, sep="\t" if source.suffix.lower() == ".tsv" else ",", nrows=MAX_ROWS + 1)
            if len(data) > MAX_ROWS or len(data.columns) > 1000:
                raise ValueError("Table exceeds the row or column limit.")
        if data.empty:
            raise ValueError("Table contains no rows.")
    else:
        from PIL import Image

        with Image.open(source) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"Image exceeds {MAX_PIXELS:,} pixels; no downsampled export is substituted.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported; select a single-plane artifact.")
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            data = np.asarray(image).copy()
        if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[2] not in (3, 4)):
            raise ValueError("Only single-plane numeric or RGB/RGBA images are supported.")
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Image pixels must be numeric.")
    if source_hash(source) != digest:
        raise ValueError("Source changed while loading; select it again.")
    return data, digest


def validate_settings(settings: PlotSettings) -> None:
    from matplotlib import colors, colormaps

    for name in ("kind", "x", "y", "xlabel", "ylabel", "xscale", "yscale", "color", "cmap", "marker", "line_style", "font", "category"):
        if not isinstance(getattr(settings, name), str):
            raise ValueError(f"{name} must be a string.")
    for name in ("grid", "legend", "colorbar"):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError(f"{name} must be a boolean.")
    for name in ("xmin", "xmax", "ymin", "ymax", "vmin", "vmax", "marker_size", "line_width", "opacity", "font_size", "width_mm", "height_mm", "dpi"):
        value = getattr(settings, name)
        if value is None and name in {"xmin", "xmax", "ymin", "ymax", "vmin", "vmax"}:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not isinstance(settings.dpi, int):
        raise ValueError("dpi must be an integer.")

    if settings.kind not in {"scatter", "line", "image"}:
        raise ValueError("Unsupported plot type.")
    if settings.xscale not in {"linear", "log"} or settings.yscale not in {"linear", "log"}:
        raise ValueError("Axis scales must be linear or log.")
    for lo, hi, label in ((settings.xmin, settings.xmax, "x"), (settings.ymin, settings.ymax, "y"), (settings.vmin, settings.vmax, "color")):
        if any(v is not None and not math.isfinite(v) for v in (lo, hi)):
            raise ValueError(f"{label} limits must be finite.")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"{label} minimum must be less than maximum.")
    for scale, lo, hi in ((settings.xscale, settings.xmin, settings.xmax), (settings.yscale, settings.ymin, settings.ymax)):
        if scale == "log" and any(v is not None and v <= 0 for v in (lo, hi)):
            raise ValueError("Log-scale limits must be positive.")
    if not (20 <= settings.width_mm <= 300 and 20 <= settings.height_mm <= 300 and 72 <= settings.dpi <= 1200):
        raise ValueError("Canvas must be 20–300 mm per side and DPI 72–1200.")
    if not (1 <= settings.font_size <= 36 and 0 <= settings.line_width <= 10 and 0 < settings.marker_size <= 500 and 0 <= settings.opacity <= 1):
        raise ValueError("Font, line, marker or opacity setting is outside its allowed range.")
    if not colors.is_color_like(settings.color) or settings.cmap not in colormaps:
        raise ValueError("Invalid color or colormap.")
    if settings.marker not in {"o", ".", "s", "^", "x", "+"} or settings.line_style not in {"-", "--", ":", "-."}:
        raise ValueError("Invalid marker or line style.")
    if not settings.font.strip():
        raise ValueError("A font family is required.")


def validate_table_columns(data: pd.DataFrame, settings: PlotSettings) -> None:
    """Reject settings that would silently relabel different source measurements."""
    for axis, column in (("X", settings.x), ("Y", settings.y)):
        if not column or column not in data.columns:
            raise ValueError(f"Saved {axis} column {column!r} is not in this table. Remove the uploaded settings and select columns for this source.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Saved {axis} column {column!r} is not numeric in this table. Remove the uploaded settings and select numeric columns.")
    if settings.category and settings.category not in data.columns:
        raise ValueError(f"Saved category column {settings.category!r} is not in this table. Remove the uploaded settings and select a category for this source.")


def make_figure(data: pd.DataFrame | np.ndarray, settings: PlotSettings) -> Figure:
    """Render unaggregated source values; caller owns and closes the figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    validate_settings(settings)
    if settings.font.lower() == "arial":
        ensure_arial()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=settings.font), fallback_to_default=False)
        except ValueError as error:
            raise ValueError(f"Font {settings.font!r} is unavailable; no substitute was used.") from error
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": settings.font, "font.size": settings.font_size,
          "font.weight": "normal", "axes.linewidth": settings.line_width, "grid.linewidth": settings.line_width,
          "lines.linewidth": settings.line_width, "axes.labelsize": settings.font_size,
          "axes.titlesize": settings.font_size, "xtick.labelsize": settings.font_size,
          "ytick.labelsize": settings.font_size, "legend.fontsize": settings.font_size,
          "legend.title_fontsize": settings.font_size, "figure.labelsize": settings.font_size,
          "axes.labelweight": "normal", "axes.titleweight": "normal"}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(settings.width_mm / 25.4, settings.height_mm / 25.4), dpi=settings.dpi)
        try:
            if isinstance(data, pd.DataFrame):
                if settings.kind not in {"scatter", "line"} or settings.x not in data or settings.y not in data:
                    raise ValueError("Choose two numeric table columns and a table plot type.")
                if not pd.api.types.is_numeric_dtype(data[settings.x]) or not pd.api.types.is_numeric_dtype(data[settings.y]):
                    raise ValueError("Selected columns must be numeric.")
                x = data[settings.x].to_numpy(dtype=float, na_value=np.nan)
                y = data[settings.y].to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.all():
                    raise ValueError(f"{int((~valid).sum())} rows have missing or nonfinite selected values; no rows were silently omitted.")
                if (settings.xscale == "log" and np.any(x <= 0)) or (settings.yscale == "log" and np.any(y <= 0)):
                    raise ValueError("Log axes require all selected source values to be positive.")
                if settings.category:
                    if settings.category not in data:
                        raise ValueError("The selected category column is unavailable.")
                    categories = data[settings.category]
                    if categories.isna().any():
                        raise ValueError("Category values include missing entries; no rows were silently omitted.")
                    codes, values = pd.factorize(categories, sort=False)
                    if len(values) > 20:
                        raise ValueError("Category coloring supports at most 20 distinct values.")
                    if len({str(value) for value in values}) != len(values):
                        raise ValueError("Category values have ambiguous display labels.")
                    palette = mpl.colormaps[settings.cmap].resampled(max(len(values), 2))
                    series = [(codes == index, palette(index), str(value)) for index, value in enumerate(values)]
                else:
                    series = [(np.ones(len(data), dtype=bool), settings.color, settings.y)]
                for mask, color, label in series:
                    if settings.kind == "scatter":
                        ax.scatter(x[mask], y[mask], color=color, marker=settings.marker, s=settings.marker_size, alpha=settings.opacity, label=label)
                    else:
                        ax.plot(x[mask], y[mask], color=color, marker=settings.marker, markersize=math.sqrt(settings.marker_size), linestyle=settings.line_style, linewidth=settings.line_width, alpha=settings.opacity, label=label)
            else:
                if settings.kind != "image":
                    raise ValueError("Image sources support the image view only.")
                if data.ndim == 3 and (settings.vmin is not None or settings.vmax is not None or settings.colorbar):
                    raise ValueError("Color limits and colorbars apply only to numeric grayscale images.")
                image = ax.imshow(data, cmap=settings.cmap, vmin=settings.vmin, vmax=settings.vmax, alpha=settings.opacity, interpolation="nearest")
                if settings.colorbar:
                    colorbar = fig.colorbar(image, ax=ax)
                    colorbar.ax.tick_params(labelsize=settings.font_size, width=settings.line_width)
            ax.set_xlabel(settings.xlabel or (settings.x if isinstance(data, pd.DataFrame) else ""))
            ax.set_ylabel(settings.ylabel or (settings.y if isinstance(data, pd.DataFrame) else ""))
            ax.set_xscale(settings.xscale)
            ax.set_yscale(settings.yscale)
            if settings.xmin is not None or settings.xmax is not None:
                ax.set_xlim(settings.xmin, settings.xmax)
            if settings.ymin is not None or settings.ymax is not None:
                ax.set_ylim(settings.ymin, settings.ymax)
            ax.set_axisbelow(True)
            ax.grid(settings.grid, which="major", color="#E5E7EB", linewidth=settings.line_width)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(settings.line_width)
            ax.tick_params(width=settings.line_width, labelsize=settings.font_size)
            if settings.legend and isinstance(data, pd.DataFrame):
                ax.legend(title=settings.category or None, fontsize=settings.font_size, title_fontsize=settings.font_size)
            fig.subplots_adjust(left=0.18, bottom=0.18, right=0.96, top=0.96)
            return fig
        except Exception:
            plt.close(fig)
            raise


def export_plot(source: Path, allowed_root: Path, run_dir: Path, settings: PlotSettings) -> tuple[Path, Path]:
    """Write a unique PDF and JSON receipt, never modifying the source."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    resolved = source_path(source, allowed_root)
    run = run_dir.resolve(strict=True)
    if not run.is_relative_to(allowed_root.resolve(strict=True)) or not run.is_dir():
        raise ValueError("Run directory is outside the approved workspace.")
    data, digest = load_source(resolved, allowed_root)
    fig = make_figure(data, settings)
    try:
        dpi = settings.dpi
        if isinstance(data, np.ndarray):
            # PDF rasterization must preserve at least the source pixel dimensions.
            fig.canvas.draw()
            bounds = fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            dpi = max(dpi, math.ceil(data.shape[1] / bounds.width), math.ceil(data.shape[0] / bounds.height))
            if dpi > 2400 or (settings.width_mm / 25.4 * dpi) * (settings.height_mm / 25.4 * dpi) > 100_000_000:
                raise ValueError("Source-resolution export exceeds the 2400 DPI or 100 megapixel safety limit.")
        if source_hash(resolved) != digest:
            raise ValueError("Source changed before export.")
        directory = run / "plot_exports"
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.resolve().is_relative_to(run):
            raise ValueError("Unsafe plot export directory.")
        stem = f"{resolved.stem[:80]}_{settings.kind}_{uuid.uuid4().hex}"
        pdf, receipt = directory / f"{stem}.pdf", directory / f"{stem}.json"
        stream = pdf.open("xb")
        try:
            with stream, mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                fig.savefig(stream, format="pdf", dpi=dpi)
        except Exception:
            pdf.unlink(missing_ok=True)
            raise
        try:
            provenance = {"source": str(resolved), "source_sha256": digest, "settings": asdict(settings),
                          "effective_export_dpi": dpi, "view": "descriptive saved-source view; no aggregation or inference",
                          "source_shape": list(data.shape)}
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump(provenance, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception:
            pdf.unlink()
            raise
        return pdf, receipt
    finally:
        plt.close(fig)
