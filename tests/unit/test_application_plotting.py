"""Synthetic fixtures test rendering and safety, not scientific results."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spatial_omics_toolkit.application.plotting import (
    MAX_PIXELS, PlotSettings, export_plot, load_source, make_figure, source_path,
    validate_settings, validate_table_columns,
)


def test_table_view_preserves_values_and_export_provenance(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    source = tmp_path / "measurements.csv"
    source.write_text("position,intensity\n1,4\n2,7\n3,2\n", encoding="utf-8")
    original = source.read_bytes()
    settings = PlotSettings(x="position", y="intensity", font="DejaVu Sans", width_mm=80, height_mm=60)
    data, digest = load_source(source, tmp_path)
    fig = make_figure(data, settings)
    try:
        np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets(), [[1, 4], [2, 7], [3, 2]])
    finally:
        plt.close(fig)
    pdf, sidecar = export_plot(source, tmp_path, tmp_path, settings)
    pdf2, _ = export_plot(source, tmp_path, tmp_path, settings)
    assert pdf != pdf2 and pdf.read_bytes().startswith(b"%PDF")
    assert source.read_bytes() == original
    receipt = json.loads(sidecar.read_text())
    assert receipt["source_sha256"] == digest
    assert receipt["settings"]["width_mm"] == 80
    assert receipt["source_shape"] == [3, 2]
    fitz = pytest.importorskip("fitz")
    with fitz.open(pdf) as document:
        assert len(document) == 1
        assert document[0].rect.width == pytest.approx(80 / 25.4 * 72, abs=0.02)
        assert document[0].rect.height == pytest.approx(60 / 25.4 * 72, abs=0.02)
        spans = [span for block in document[0].get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
        assert spans and all(span["size"] == pytest.approx(5, abs=0.1) for span in spans)
        assert all("DejaVuSans" in span["font"] for span in spans)


def test_image_export_preserves_source_and_reports_resolution(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12)).save(source)
    original = source.read_bytes()
    settings = PlotSettings(kind="image", font="DejaVu Sans")
    data, _ = load_source(source, tmp_path)
    assert data.shape == (10, 12)
    pdf, receipt = export_plot(source, tmp_path, tmp_path, settings)
    assert pdf.exists() and source.read_bytes() == original
    assert json.loads(receipt.read_text())["effective_export_dpi"] >= settings.dpi


def test_rejects_unsafe_paths_and_unsupported_pdf(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside.csv"
    outside.write_text("x,y\n1,2\n")
    link = tmp_path / "link.csv"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="approved workspace"):
        source_path(link, tmp_path)
    pdf = tmp_path / "existing.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="PDF"):
        load_source(pdf, tmp_path)


def test_rejects_invalid_or_missing_values_and_log_scale() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, np.nan]})
    with pytest.raises(ValueError, match="nonfinite"):
        make_figure(data, PlotSettings(x="x", y="y", font="DejaVu Sans"))
    with pytest.raises(ValueError, match="positive"):
        make_figure(data.fillna(2), PlotSettings(x="x", y="y", xscale="log", font="DejaVu Sans"))
    with pytest.raises(ValueError, match="minimum"):
        validate_settings(PlotSettings(xmin=5, xmax=1))
    with pytest.raises(ValueError, match="unavailable"):
        make_figure(data.fillna(2), PlotSettings(x="x", y="y", font="font-that-does-not-exist-1937"))


def test_rejects_oversized_and_multiframe_images(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("L", (MAX_PIXELS // 1000 + 1, 1000)).save(source)
    with pytest.raises(ValueError, match="pixels"):
        load_source(source, tmp_path)
    stack = tmp_path / "stack.tif"
    Image.new("L", (4, 4)).save(stack, save_all=True, append_images=[Image.new("L", (4, 4))])
    with pytest.raises(ValueError, match="Multi-frame"):
        load_source(stack, tmp_path)


def test_pdf_name_collision_never_removes_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "values.tsv"
    source.write_text("x\ty\n1\t2\n")
    exports = tmp_path / "plot_exports"
    exports.mkdir()
    existing = exports / "values_scatter_fixed.pdf"
    existing.write_bytes(b"previous output")
    with patch("spatial_omics_toolkit.application.plotting.uuid.uuid4") as uid:
        uid.return_value.hex = "fixed"
        with pytest.raises(FileExistsError):
            export_plot(source, tmp_path, tmp_path, PlotSettings(x="x", y="y", font="DejaVu Sans"))
    assert existing.read_bytes() == b"previous output"


def test_parquet_source_values(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "values.parquet"
    expected = pd.DataFrame({"x": [1.0, 2.0], "y": [4.0, 8.0]})
    expected.to_parquet(source)
    actual, _ = load_source(source, tmp_path)
    pd.testing.assert_frame_equal(actual, expected)


def test_export_failure_does_not_leave_partial_pdf(tmp_path: Path) -> None:
    source = tmp_path / "values.csv"
    source.write_text("x,y\n1,2\n")
    with patch("matplotlib.figure.Figure.savefig", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            export_plot(source, tmp_path, tmp_path, PlotSettings(x="x", y="y", font="DejaVu Sans"))
    assert not list((tmp_path / "plot_exports").glob("*.pdf"))


def test_imported_settings_reject_incompatible_measurements() -> None:
    table = pd.DataFrame({"position": [1, 2], "intensity": [3, 4], "group": ["A", "B"]})
    valid = PlotSettings(x="position", y="intensity", category="group", ylabel="Intensity")
    validate_table_columns(table, valid)
    for settings, expected in (
        (PlotSettings(x="old_position", y="intensity", xlabel="Old position"), "X column"),
        (PlotSettings(x="position", y="old_intensity", ylabel="Old intensity"), "Y column"),
        (PlotSettings(x="position", y="intensity", category="old_group"), "category column"),
        (PlotSettings(x="position", y="group", ylabel="Old numeric value"), "not numeric"),
        (PlotSettings(x="", y="intensity", xlabel="Old position"), "X column"),
    ):
        with pytest.raises(ValueError, match=expected):
            validate_table_columns(table, settings)
