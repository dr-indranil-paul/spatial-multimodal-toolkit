from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from spatial_omics_toolkit.application.plotting import (
    IMAGE_SUFFIXES, TABLE_SUFFIXES, PlotSettings, export_plot, load_source, make_figure,
    validate_settings, validate_table_columns,
)


def _optional_number(label: str, value: float | None, key: str) -> float | None:
    text = st.text_input(label, value="" if value is None else str(value), key=key).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a number or blank.") from error


def render_plot_controls(source: Path, run_dir: Path, allowed_root: Path) -> None:
    """Customize a new descriptive view within the selected result pane."""
    suffix = source.suffix.lower()
    if suffix not in TABLE_SUFFIXES | IMAGE_SUFFIXES:
        st.caption("Customization requires a saved CSV, TSV, Parquet, or single-plane image. Existing PDFs remain unchanged and cannot be re-plotted.")
        return
    key = hashlib.sha256(f"{run_dir.resolve()}:{source}".encode()).hexdigest()[:20]
    with st.expander("Customize and export a descriptive PDF", expanded=False):
        st.caption("New view of the selected saved artifact, not a recreation of its canonical figure. No aggregation or inference is performed; units and observational units remain those of the source.")
        if not st.checkbox("Load source for customization", value=False, key=f"plot_enable_{key}"):
            return
        try:
            data, digest = load_source(source, allowed_root)
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            st.warning(f"Cannot load this source: {error}")
            return
        st.caption(f"Source SHA-256: {digest} · shape: {data.shape}")
        defaults = PlotSettings(kind="image" if isinstance(data, np.ndarray) else "scatter")
        uploaded = st.file_uploader("Reuse plot settings JSON", type=["json"], key=f"plot_upload_{key}")
        if uploaded is not None:
            try:
                if uploaded.size > 64 * 1024:
                    raise ValueError("Settings file exceeds 64 KiB.")
                value: Any = json.loads(uploaded.getvalue())
                if isinstance(value, dict) and "settings" in value:
                    value = value["settings"]
                if not isinstance(value, dict) or set(value) - {field.name for field in fields(PlotSettings)}:
                    raise ValueError("Expected a plot settings object or export provenance receipt.")
                defaults = PlotSettings(**value)
                validate_settings(defaults)
                if (isinstance(data, np.ndarray)) != (defaults.kind == "image"):
                    raise ValueError("Settings do not match this source type.")
                if isinstance(data, pd.DataFrame):
                    validate_table_columns(data, defaults)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                st.error(f"Invalid settings: {error}")
                return
        # Settings-file hash isolates widget state when a different profile is uploaded.
        profile = hashlib.sha256(json.dumps(asdict(defaults), sort_keys=True).encode()).hexdigest()[:10]
        prefix = f"plot_{key}_{profile}"
        def select(label: str, options: list[str], current: str, name: str) -> str:
            return st.selectbox(label, options, index=options.index(current) if current in options else 0, key=f"{prefix}_{name}")
        try:
            if isinstance(data, pd.DataFrame):
                numeric = [str(column) for column in data.select_dtypes(include="number").columns]
                if not numeric:
                    st.info("This table has no numeric columns to plot.")
                    return
                kind = select("View", ["scatter", "line"], defaults.kind, "kind")
                x = select("X column", numeric, defaults.x, "x")
                y = select("Y column", numeric, defaults.y if defaults.y else numeric[min(1, len(numeric) - 1)], "y")
                category = select("Color by category (optional)", ["", *[str(column) for column in data.columns if str(column)]], defaults.category, "category")
                st.caption("Category colors distinguish saved rows without aggregation. Line plots connect rows within each category in saved order; missing or nonfinite values are reported, not silently removed.")
            else:
                kind, x, y, category = "image", "", "", ""
                st.caption("Image view uses the complete saved image; export is refused rather than downsampled when safety bounds are exceeded. The source may itself be a preview.")
            a, b = st.columns(2)
            with a:
                xlabel = st.text_input("X label", value=defaults.xlabel, key=f"{prefix}_xlabel")
                xscale = select("X scale", ["linear", "log"], defaults.xscale, "xscale")
                xmin = _optional_number("X minimum (blank = automatic)", defaults.xmin, f"{prefix}_xmin")
                xmax = _optional_number("X maximum (blank = automatic)", defaults.xmax, f"{prefix}_xmax")
            with b:
                ylabel = st.text_input("Y label", value=defaults.ylabel, key=f"{prefix}_ylabel")
                yscale = select("Y scale", ["linear", "log"], defaults.yscale, "yscale")
                ymin = _optional_number("Y minimum (blank = automatic)", defaults.ymin, f"{prefix}_ymin")
                ymax = _optional_number("Y maximum (blank = automatic)", defaults.ymax, f"{prefix}_ymax")
            color, cmap, marker, line_style = defaults.color, defaults.cmap, defaults.marker, defaults.line_style
            vmin, vmax, colorbar = defaults.vmin, defaults.vmax, False
            if kind == "image":
                cmap = select("Grayscale colormap", ["viridis", "gray", "magma", "cividis", "inferno", "plasma"], defaults.cmap, "cmap")
                if data.ndim == 2:
                    vmin = _optional_number("Color minimum (blank = automatic)", defaults.vmin, f"{prefix}_vmin")
                    vmax = _optional_number("Color maximum (blank = automatic)", defaults.vmax, f"{prefix}_vmax")
                    colorbar = st.checkbox("Show colorbar", value=defaults.colorbar, key=f"{prefix}_colorbar")
                else:
                    st.caption("RGB/RGBA pixels retain their source colors; colormap and color limits do not apply.")
                    vmin, vmax = None, None
            else:
                if category:
                    cmap = select("Category palette", ["tab10", "tab20", "viridis", "cividis", "plasma"], defaults.cmap, "category_cmap")
                else:
                    color = st.color_picker("Color", value=defaults.color, key=f"{prefix}_color")
                marker = select("Marker", ["o", ".", "s", "^", "x", "+"], defaults.marker, "marker")
                if kind == "line":
                    line_style = select("Line style", ["-", "--", ":", "-."], defaults.line_style, "line_style")
            a, b = st.columns(2)
            with a:
                opacity = st.slider("Opacity", 0.0, 1.0, float(defaults.opacity), 0.05, key=f"{prefix}_opacity")
                line_width = st.number_input("Line/spine/grid width (pt)", min_value=0.0, max_value=10.0, value=float(defaults.line_width), key=f"{prefix}_line_width")
                marker_size = st.number_input("Marker area (pt²)", min_value=0.01, max_value=500.0, value=float(defaults.marker_size), key=f"{prefix}_marker_size")
                grid = st.checkbox("Major grid", value=defaults.grid, key=f"{prefix}_grid")
                legend = st.checkbox("Legend", value=defaults.legend, key=f"{prefix}_legend")
            with b:
                font = st.text_input("Font family (must be installed)", value=defaults.font, key=f"{prefix}_font")
                font_size = st.number_input("Font size (pt)", min_value=1.0, max_value=36.0, value=float(defaults.font_size), key=f"{prefix}_font_size")
                width_mm = st.number_input("Canvas width (mm)", min_value=20.0, max_value=300.0, value=float(defaults.width_mm), key=f"{prefix}_width")
                height_mm = st.number_input("Canvas height (mm)", min_value=20.0, max_value=300.0, value=float(defaults.height_mm), key=f"{prefix}_height")
                dpi = st.number_input("Export DPI (minimum for raster sources)", min_value=72, max_value=1200, value=int(defaults.dpi), key=f"{prefix}_dpi")
            settings = PlotSettings(kind=kind, x=x, y=y, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale,
                                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, color=color, cmap=cmap, marker=marker,
                                    line_style=line_style, category=category, colorbar=colorbar, marker_size=marker_size, line_width=line_width, opacity=opacity,
                                    font=font, font_size=font_size, grid=grid, legend=legend, width_mm=width_mm,
                                    height_mm=height_mm, dpi=int(dpi), vmin=vmin, vmax=vmax)
            validate_settings(settings)
            st.download_button("Download reusable settings", json.dumps(asdict(settings), indent=2), file_name="plot_settings.json", mime="application/json", key=f"{prefix}_settings")
            if st.button("Preview selected settings", key=f"{prefix}_preview"):
                import matplotlib.pyplot as plt
                fig = make_figure(data, settings)
                try:
                    st.pyplot(fig)
                finally:
                    plt.close(fig)
            identity = hashlib.sha256((digest + json.dumps(asdict(settings), sort_keys=True)).encode()).hexdigest()
            state_key = f"plot_generated_{key}"
            saved = st.session_state.get(state_key)
            if saved and saved["identity"] != identity:
                del st.session_state[state_key]
                saved = None
            if st.button("Generate PDF and provenance", key=f"{prefix}_export"):
                pdf, receipt = export_plot(source, allowed_root, run_dir, settings)
                receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
                if receipt_data["source_sha256"] != digest:
                    raise ValueError("Source changed since the controls were loaded; generate again.")
                saved = {"identity": identity, "pdf": str(pdf), "receipt": str(receipt)}
                st.session_state[state_key] = saved
            if saved:
                pdf, receipt = Path(saved["pdf"]), Path(saved["receipt"])
                st.success(f"Saved {pdf.name} and {receipt.name} in {pdf.parent}.")
                st.download_button("Download generated PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf", key=f"{prefix}_pdf")
                st.download_button("Download provenance JSON", receipt.read_bytes(), file_name=receipt.name, mime="application/json", key=f"{prefix}_receipt")
        except (OSError, ValueError, RuntimeError, ImportError, TypeError) as error:
            st.error(f"Plot unavailable: {error}")
