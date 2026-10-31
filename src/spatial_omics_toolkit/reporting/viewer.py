"""Self-contained interactive HTML5 spatial viewer generator.

Generates standalone, zero-dependency HTML/JS/Canvas spatial viewers for single-cell
and multiplexed spatial omics data with high-performance pan/zoom, dynamic color-by
phenotype or continuous marker, tooltips, and ROI inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { display: flex; height: 100vh; background: #1a1a1a; color: #e0e0e0; overflow: hidden; }
        #sidebar { width: 340px; background: #242424; border-right: 1px solid #333; display: flex; flex-direction: column; padding: 16px; gap: 16px; overflow-y: auto; }
        #main-view { flex: 1; position: relative; display: flex; flex-direction: column; }
        #canvas-container { flex: 1; position: relative; background: #121212; cursor: grab; }
        #canvas-container:active { cursor: grabbing; }
        canvas { display: block; width: 100%; height: 100%; }
        h1 { font-size: 1.1rem; font-weight: 600; color: #fff; border-bottom: 1px solid #333; padding-bottom: 8px; }
        .control-group { display: flex; flex-direction: column; gap: 6px; }
        label { font-size: 0.8rem; font-weight: 500; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; }
        select, input[type="range"] { background: #333; border: 1px solid #444; color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 0.9rem; width: 100%; }
        button { background: #0066cc; color: #fff; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; font-weight: 500; }
        button:hover { background: #0052a3; }
        .btn-secondary { background: #3a3a3a; }
        .btn-secondary:hover { background: #4a4a4a; }
        #legend { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; max-height: 240px; overflow-y: auto; background: #1c1c1c; padding: 8px; border-radius: 4px; border: 1px solid #333; }
        .legend-item { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 2px 4px; border-radius: 2px; }
        .legend-item:hover { background: #2a2a2a; }
        .legend-color { width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }
        #tooltip { position: absolute; display: none; background: rgba(20, 20, 20, 0.95); border: 1px solid #555; padding: 8px 12px; border-radius: 4px; font-size: 0.8rem; pointer-events: none; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        .stats-badge { background: #333; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; }
        .btn-row { display: flex; gap: 8px; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>{{TITLE}}</h1>
        <div class="control-group">
            <span class="stats-badge">Total Cells: <strong id="total-cells-count">0</strong></span>
        </div>
        <div class="control-group">
            <label for="color-mode">Color Mode</label>
            <select id="color-mode">
                <option value="phenotype">Cell Phenotype</option>
                <option value="continuous">Continuous Marker</option>
            </select>
        </div>
        <div id="continuous-controls" class="control-group" style="display: none;">
            <label for="marker-select">Select Marker</label>
            <select id="marker-select"></select>
        </div>
        <div class="control-group">
            <label>Legend / Phenotypes</label>
            <div id="legend"></div>
        </div>
        <div class="control-group">
            <label for="point-size">Point Radius: <span id="point-size-val">2.5</span> px</label>
            <input type="range" id="point-size" min="0.5" max="10" step="0.5" value="2.5">
        </div>
        <div class="control-group btn-row">
            <button id="btn-reset-view" class="btn-secondary">Reset View</button>
            <button id="btn-export-png">Export PNG</button>
        </div>
    </div>
    <div id="main-view">
        <div id="canvas-container">
            <canvas id="spatial-canvas"></canvas>
            <div id="tooltip"></div>
        </div>
    </div>

    <script>
        const DATA = {{CELL_DATA_JSON}};
        const PALETTE = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"
        ];

        const container = document.getElementById("canvas-container");
        const canvas = document.getElementById("spatial-canvas");
        const ctx = canvas.getContext("2d");
        const tooltip = document.getElementById("tooltip");
        const legendDiv = document.getElementById("legend");
        const markerSelect = document.getElementById("marker-select");
        const colorModeSelect = document.getElementById("color-mode");
        const continuousControls = document.getElementById("continuous-controls");
        const pointSizeInput = document.getElementById("point-size");
        const pointSizeVal = document.getElementById("point-size-val");

        document.getElementById("total-cells-count").innerText = DATA.x.length.toLocaleString();

        let width = container.clientWidth;
        let height = container.clientHeight;
        let scale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let isDragging = false;
        let startX, startY;
        let pointRadius = 2.5;

        // Assign colors to categorical phenotypes
        const uniqueTypes = [...new Set(DATA.type)];
        const typeColors = {};
        const activeTypes = new Set(uniqueTypes);
        uniqueTypes.forEach((t, i) => {
            typeColors[t] = PALETTE[i % PALETTE.length];
        });

        // Populate markers dropdown
        if (DATA.markers && Object.keys(DATA.markers).length > 0) {
            Object.keys(DATA.markers).forEach(m => {
                const opt = document.createElement("option");
                opt.value = m;
                opt.innerText = m;
                markerSelect.appendChild(opt);
            });
        }

        function initLegend() {
            legendDiv.innerHTML = "";
            uniqueTypes.forEach(t => {
                const row = document.createElement("div");
                row.className = "legend-item";
                row.innerHTML = `<div class="legend-color" style="background:${typeColors[t]};"></div><span>${t}</span>`;
                row.onclick = () => {
                    if (activeTypes.has(t)) {
                        activeTypes.delete(t);
                        row.style.opacity = "0.3";
                    } else {
                        activeTypes.add(t);
                        row.style.opacity = "1.0";
                    }
                    draw();
                };
                legendDiv.appendChild(row);
            });
        }

        function resizeCanvas() {
            width = container.clientWidth;
            height = container.clientHeight;
            canvas.width = width * window.devicePixelRatio;
            canvas.height = height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            draw();
        }

        function resetView() {
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let i = 0; i < DATA.x.length; i++) {
                if (DATA.x[i] < minX) minX = DATA.x[i];
                if (DATA.x[i] > maxX) maxX = DATA.x[i];
                if (DATA.y[i] < minY) minY = DATA.y[i];
                if (DATA.y[i] > maxY) maxY = DATA.y[i];
            }
            const dataW = maxX - minX || 1;
            const dataH = maxY - minY || 1;
            const scaleX = (width - 40) / dataW;
            const scaleY = (height - 40) / dataH;
            scale = Math.min(scaleX, scaleY);
            offsetX = 20 - minX * scale + (width - 40 - dataW * scale) / 2;
            offsetY = 20 - minY * scale + (height - 40 - dataH * scale) / 2;
            draw();
        }

        function draw() {
            ctx.fillStyle = "#121212";
            ctx.fillRect(0, 0, width, height);

            const isContinuous = colorModeSelect.value === "continuous";
            const markerName = markerSelect.value;
            const markerVals = isContinuous && DATA.markers ? DATA.markers[markerName] : null;
            let minM = 0, maxM = 1;
            if (markerVals) {
                minM = Math.min(...markerVals);
                maxM = Math.max(...markerVals) || 1;
            }

            for (let i = 0; i < DATA.x.length; i++) {
                const cType = DATA.type[i];
                if (!isContinuous && !activeTypes.has(cType)) continue;

                const sx = DATA.x[i] * scale + offsetX;
                const sy = DATA.y[i] * scale + offsetY;

                if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue;

                if (isContinuous && markerVals) {
                    const norm = Math.max(0, Math.min(1, (markerVals[i] - minM) / (maxM - minM + 1e-6)));
                    const r = Math.floor(255 * norm);
                    const b = Math.floor(255 * (1 - norm));
                    ctx.fillStyle = `rgb(${r}, 40, ${b})`;
                } else {
                    ctx.fillStyle = typeColors[cType] || "#ffffff";
                }

                ctx.beginPath();
                ctx.arc(sx, sy, pointRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        // Interactive Event Handlers
        window.addEventListener("resize", resizeCanvas);
        container.addEventListener("mousedown", (e) => {
            isDragging = true;
            startX = e.clientX - offsetX;
            startY = e.clientY - offsetY;
        });
        window.addEventListener("mousemove", (e) => {
            if (isDragging) {
                offsetX = e.clientX - startX;
                offsetY = e.clientY - startY;
                draw();
            } else {
                // Tooltip query
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                let found = -1;
                for (let i = 0; i < DATA.x.length; i++) {
                    const sx = DATA.x[i] * scale + offsetX;
                    const sy = DATA.y[i] * scale + offsetY;
                    const distSq = (mx - sx) ** 2 + (my - sy) ** 2;
                    if (distSq < (pointRadius + 4) ** 2) {
                        found = i;
                        break;
                    }
                }
                if (found !== -1) {
                    tooltip.style.display = "block";
                    tooltip.style.left = (e.clientX + 12) + "px";
                    tooltip.style.top = (e.clientY + 12) + "px";
                    let tipHtml = `<strong>Cell #${found}</strong><br>Type: ${DATA.type[found]}<br>X: ${DATA.x[found].toFixed(1)}, Y: ${DATA.y[found].toFixed(1)}`;
                    if (DATA.markers) {
                        Object.keys(DATA.markers).forEach(m => {
                            tipHtml += `<br>${m}: ${DATA.markers[m][found].toFixed(2)}`;
                        });
                    }
                    tooltip.innerHTML = tipHtml;
                } else {
                    tooltip.style.display = "none";
                }
            }
        });
        window.addEventListener("mouseup", () => isDragging = false);
        container.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const zoomFactor = e.deltaY < 0 ? 1.2 : 0.8;
            scale *= zoomFactor;
            offsetX = mouseX - (mouseX - offsetX) * zoomFactor;
            offsetY = mouseY - (mouseY - offsetY) * zoomFactor;
            draw();
        });

        colorModeSelect.onchange = () => {
            continuousControls.style.display = colorModeSelect.value === "continuous" ? "flex" : "none";
            draw();
        };
        markerSelect.onchange = draw;
        pointSizeInput.oninput = () => {
            pointRadius = parseFloat(pointSizeInput.value);
            pointSizeVal.innerText = pointRadius;
            draw();
        };
        document.getElementById("btn-reset-view").onclick = resetView;
        document.getElementById("btn-export-png").onclick = () => {
            const link = document.createElement("a");
            link.download = "spatial_view.png";
            link.href = canvas.toDataURL("image/png");
            link.click();
        };

        // Init
        initLegend();
        resizeCanvas();
        resetView();
    </script>
</body>
</html>
"""


def export_interactive_html_viewer(
    df: pd.DataFrame,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    type_col: Optional[str] = "cell_type",
    marker_cols: Optional[Sequence[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Spatial Omics Interactive Viewer",
    max_cells: int = 100000,
) -> str:
    """Generate a self-contained, standalone interactive HTML spatial viewer.

    Args:
        df: Cells DataFrame with coordinates, phenotype labels, and marker intensities.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        type_col: Column for categorical phenotype labeling and filtering.
        marker_cols: Optional list of numeric marker columns for continuous expression mapping.
        output_path: Optional file path to save the HTML document.
        title: Page and header title.
        max_cells: Maximum cell subsample ceiling for browser memory safety.

    Returns:
        Full HTML string of the self-contained interactive viewer.
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame.")

    work = df.copy()
    if len(work) > max_cells:
        logger.warning(f"Subsampling {len(work)} cells to {max_cells} for HTML viewer.")
        work = work.sample(n=max_cells, random_state=42)

    cell_data: Dict[str, Any] = {
        "x": np.round(work[x_col].values.astype(float), 2).tolist(),
        "y": np.round(work[y_col].values.astype(float), 2).tolist(),
        "type": (
            work[type_col].astype(str).tolist()
            if type_col and type_col in work.columns
            else ["Cell"] * len(work)
        ),
        "markers": {},
    }

    if marker_cols:
        for m in marker_cols:
            if m in work.columns and pd.api.types.is_numeric_dtype(work[m]):
                cell_data["markers"][m] = np.round(
                    work[m].fillna(0.0).values.astype(float), 2
                ).tolist()

    json_payload = json.dumps(cell_data)
    html = HTML_TEMPLATE.replace("{{TITLE}}", title).replace(
        "{{CELL_DATA_JSON}}", json_payload
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(html, encoding="utf-8")
        logger.info(f"Saved interactive HTML viewer: {out_p}")

    return html
