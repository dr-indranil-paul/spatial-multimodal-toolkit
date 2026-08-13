"""step01_scan_intake — pre-registration intake QC for raw CZI microscopy files.

Ported and generalized from
``02_spatial_proteomics/03_analysis/02_scripts/cycif_qc.py`` (935 lines) —
the pre-registration raw-image intake QC pipeline: CZI inventory, filename
parsing, round x scene coverage checking, OME-XML channel/pixel-size/
dimension validation, DAPI preview, and HTML report generation. This is
**not** the same module as the already-shipped
``spatial_omics_toolkit.analysis.cycif_qc`` (step06), which is
post-segmentation marker-intensity QC on an already-extracted cell x
marker matrix — a much smaller, different-purpose file that happens to
share a name with the source script this module ports.

What changed to generalize
---------------------------
- Filename-parsing regex, expected round vocabulary, expected channel
  count, and per-slide expected file counts are now parameters
  (:class:`~spatial_omics_toolkit.cycif.config.RawImageIntakeConfig`)
  instead of hardcoded constants tied to the PREVAIL hBE-TMA cohort. The
  values observed in that cohort ship as ``PREVAIL_BARRETTS_INTAKE_CONFIG``,
  the worked default — a different acquisition batch needs a different
  config object, not a source edit.
- The source's ``QC_COLORS`` dict keyed plot colors by literal slide
  identifiers (``'hBE-TMA1-1': '#2563eb'``), which silently hardcodes one
  manuscript's slide names into a shared constant. Replaced here with
  ``_slide_color``/``_round_color``, which assign a color deterministically
  from a fixed palette based on each dataset's *observed* slides/rounds.
- ``find_sample_czi``/``find_all_round_paths`` in the source reconstruct
  the raw filename from its parsed components using the same hardcoded
  ``{round}_{markers}_{slide}_{date}__{acqid}-{scene}.czi`` format string
  used to *parse* filenames — a second, silent copy of the naming
  convention that would silently drift from ``filename_pattern``. Here the
  scanned file path is carried straight through ``parse_filenames``
  instead of being rebuilt, so there is exactly one place
  (``config.filename_pattern``) that encodes the naming convention.
- The source hardcoded a per-slide expected CZI count of 60 as the default
  when none was supplied. That is a per-acquisition-batch fact, not a
  panel constant, so ``RawImageIntakeConfig.expected_files_per_slide``
  defaults to ``None`` (check skipped, reported as "not specified") rather
  than inventing a number that would not hold for a different batch.
- Channel-count checking, per-slide pixel-size consistency checking, and
  round x scene coverage checking are otherwise unchanged algorithms.

Every check function returns a structured pass/fail record rather than
raising by default, so ``step01_scan_intake`` always returns a usable
report even when the raw data has real gaps — this satisfies this lab's
"fail loudly, never silently" rule via its "return a structured problem"
branch. Pass ``strict=True`` to raise a ``RuntimeError`` instead once
you've reviewed the report and want the notebook to stop on any failure.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from spatial_omics_toolkit.cycif.config import (
    DEFAULT_EXPECTED_ROUNDS,
    DEFAULT_RAW_FILENAME_PATTERN,
    PREVAIL_BARRETTS_INTAKE_CONFIG,
    RawImageIntakeConfig,
)

logger = logging.getLogger("spatial_omics_toolkit.cycif.intake")

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
"""Standard OME-XML namespace URI — not a naming convention, kept as a constant."""

_PALETTE = ("#2563eb", "#7c3aed", "#0891b2", "#059669", "#d97706",
            "#dc2626", "#64748b", "#db2777", "#65a30d", "#0d9488")
"""Fixed color palette cycled deterministically per *observed* slide/round,
replacing the source's dict keyed by literal hBE-TMA slide identifiers."""

_FILENAME_GROUPS = ("round", "markers", "slide", "date", "acq_id", "scene")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Filename parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_intake_filename(
    filename: str,
    pattern: str = DEFAULT_RAW_FILENAME_PATTERN,
) -> Optional[Dict[str, str]]:
    """Parse a raw image filename with a configurable named-group regex.

    Ported from ``registration.py::parse_czi_filename``. The matching
    algorithm (a single ``re.match``) is unchanged; only the pattern
    itself is now a parameter instead of a hardcoded module constant.

    Args:
        filename: raw image filename (e.g. a ``.czi`` basename).
        pattern: regex with named groups ``round``, ``markers``, ``slide``,
            ``date``, ``acq_id``, ``scene``.

    Returns:
        A dict with those six keys (``date`` normalized to ``YYYY-MM-DD``),
        or ``None`` if the filename does not match ``pattern``.

    Raises:
        ValueError: if ``pattern`` matches but is missing one of the
            required named groups — a silent typo in a custom pattern
            must not be mistaken for "this file didn't match".
    """
    m = re.match(pattern, filename)
    if not m:
        return None
    gd = m.groupdict()
    missing = set(_FILENAME_GROUPS) - set(gd)
    if missing:
        raise ValueError(
            f"filename_pattern matched {filename!r} but is missing required "
            f"named group(s) {sorted(missing)}; pattern must define named "
            f"groups {_FILENAME_GROUPS}."
        )
    gd["date"] = gd["date"].replace("_", "-")
    return gd


def _round_sort_key(label: str) -> Tuple[int, int]:
    """Sort round labels in biological order: R1 < R2 < ... < R5 < R5Q.

    Ported from ``mcmicro.py::_round_sort_key_local``.
    """
    m = re.match(r"^R(\d+)(\D*)$", str(label))
    if m:
        return (int(m.group(1)), 1 if m.group(2) else 0)
    return (10**6, 0)


def _slide_color(slide: str, all_slides: Sequence[str]) -> str:
    ordered = sorted(set(all_slides))
    idx = ordered.index(slide) if slide in ordered else 0
    return _PALETTE[idx % len(_PALETTE)]


def _round_color(round_label: str, all_rounds: Sequence[str]) -> str:
    ordered = sorted(set(all_rounds), key=_round_sort_key)
    idx = ordered.index(round_label) if round_label in ordered else 0
    return _PALETTE[idx % len(_PALETTE)]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Raw file inventory
# ═══════════════════════════════════════════════════════════════════════════

def scan_raw_images(raw_dir: str, extension: str = ".czi") -> pd.DataFrame:
    """Scan a directory of per-slide subdirectories for raw image files.

    Ported from ``cycif_qc.py::scan_czis``, generalized: the file
    extension is now a parameter instead of a hardcoded ``*.czi`` glob.

    Args:
        raw_dir: directory containing one subdirectory per slide/sample.
        extension: file extension to scan for (default ``.czi``).

    Returns:
        DataFrame with columns ``slide``, ``fname``, ``path``, ``size_gb``.

    Raises:
        FileNotFoundError: if ``raw_dir`` does not exist.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_path}")

    slide_dirs = sorted(d for d in raw_path.iterdir() if d.is_dir())
    logger.info(f"Slide directories found: {len(slide_dirs)}")

    records = []
    for sd in slide_dirs:
        for f in sorted(sd.glob(f"*{extension}")):
            records.append({
                "slide": sd.name,
                "fname": f.name,
                "path": str(f),
                "size_gb": round(f.stat().st_size / 1024**3, 4),
            })

    df = pd.DataFrame(records, columns=["slide", "fname", "path", "size_gb"])
    total_gb = df["size_gb"].sum() if len(df) else 0.0
    logger.info(f"Total {extension} files: {len(df)}  |  Total size: {total_gb:.1f} GB")
    return df


def parse_filenames(
    df_files: pd.DataFrame,
    pattern: str = DEFAULT_RAW_FILENAME_PATTERN,
) -> Tuple[pd.DataFrame, List[str]]:
    """Parse every scanned filename, carrying the original path through.

    Ported from ``cycif_qc.py::parse_filenames``. Unlike the source, the
    original ``path`` (and directory-derived ``slide``) are carried into
    the parsed table directly rather than only ``slide``/``size_gb`` — see
    the module docstring for why (avoids ever rebuilding a filename from
    its parsed pieces).

    Args:
        df_files: output of :func:`scan_raw_images`.
        pattern: passed to :func:`parse_intake_filename`.

    Returns:
        ``(df_parsed, unparsed_fnames)``.
    """
    parsed = []
    unparsed: List[str] = []

    for _, row in df_files.iterrows():
        p = parse_intake_filename(row["fname"], pattern=pattern)
        if p:
            parsed.append({**p, "slide": row["slide"], "path": row["path"], "size_gb": row["size_gb"]})
        else:
            unparsed.append(row["fname"])

    columns = ["round", "markers", "slide", "date", "acq_id", "scene", "path", "size_gb"]
    df_parsed = pd.DataFrame(parsed, columns=columns)
    logger.info(f"Parsed OK: {len(df_parsed)} / {len(df_files)}")
    if unparsed:
        logger.warning(f"Failed to parse {len(unparsed)} filename(s)")
    return df_parsed, unparsed


def check_round_scene_coverage(
    df_parsed: pd.DataFrame,
    expected_rounds: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Build a round x scene coverage matrix and list missing combinations.

    Ported from ``cycif_qc.py::check_round_scene_coverage``, generalized:
    ``expected_rounds`` defaults to :data:`DEFAULT_EXPECTED_ROUNDS`
    (the hBE-TMA round vocabulary) instead of being hardcoded inline.

    Args:
        df_parsed: output of :func:`parse_filenames`.
        expected_rounds: round labels every ``(slide, scene)`` must have.
            Defaults to :data:`DEFAULT_EXPECTED_ROUNDS`.

    Returns:
        ``(pivot, missing)`` — ``pivot`` is a round x scene file-count
        table; ``missing`` is a list of ``{slide, scene, round}`` records
        for every combination absent from the data. An empty ``missing``
        list means full coverage.
    """
    if expected_rounds is None:
        expected_rounds = DEFAULT_EXPECTED_ROUNDS
    expected_set = set(expected_rounds)

    if len(df_parsed) == 0:
        return pd.DataFrame(), []

    pivot = df_parsed.pivot_table(
        index="round", columns="scene", values="size_gb",
        aggfunc="count", fill_value=0,
    )

    missing: List[Dict[str, str]] = []
    for (slide, sc), grp in df_parsed.groupby(["slide", "scene"]):
        rounds_present = set(grp["round"])
        for r in expected_set:
            if r not in rounds_present:
                missing.append({"slide": slide, "scene": sc, "round": r})

    if missing:
        logger.warning(f"{len(missing)} missing round/scene combination(s)")
    else:
        logger.info("All expected rounds present for every scene")
    return pivot, missing


# ═══════════════════════════════════════════════════════════════════════════
# 3. OME-XML sidecar metadata
# ═══════════════════════════════════════════════════════════════════════════

def _parse_ome_xml(xml_path: str):
    """Parse channel names, pixel size, and dimensions from an OME-XML sidecar.

    Ported unchanged from ``cycif_qc.py::_parse_ome_xml``.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image = root.find(".//ome:Image", OME_NS)
        if image is None:
            return None, None, None
        pixels = image.find("ome:Pixels", OME_NS)
        if pixels is None:
            return None, None, None

        px = pixels.get("PhysicalSizeX")
        if px:
            px = float(px)

        channels = [ch.get("Name", "") for ch in pixels.findall("ome:Channel", OME_NS)]
        dims = {
            "SizeX": pixels.get("SizeX"),
            "SizeY": pixels.get("SizeY"),
            "SizeC": pixels.get("SizeC"),
            "SizeZ": pixels.get("SizeZ"),
            "SizeT": pixels.get("SizeT"),
        }
        return channels, px, dims
    except Exception as e:
        return None, None, str(e)


def parse_ome_sidecars(slide_dirs: Sequence[Path], xml_suffix: str = ".ome.xml") -> List[Dict[str, Any]]:
    """Parse every OME-XML sidecar across a set of slide directories.

    Ported from ``cycif_qc.py::parse_ome_sidecars``, generalized:
    ``xml_suffix`` is now a parameter instead of a hardcoded ``*.ome.xml`` glob.

    Returns:
        List of ``{slide, xml_file, channels, n_channels, pixel_size_um, dims}``.
    """
    records: List[Dict[str, Any]] = []
    for sd in slide_dirs:
        sd = Path(sd)
        for xml_path in sorted(sd.glob(f"*{xml_suffix}")):
            ch_names, px_size, dims = _parse_ome_xml(str(xml_path))
            records.append({
                "slide": sd.name,
                "xml_file": xml_path.name,
                "channels": ch_names,
                "n_channels": len(ch_names) if ch_names else 0,
                "pixel_size_um": px_size,
                "dims": dims,
            })
    logger.info(f"OME-XML sidecars parsed: {len(records)}")
    return records


def check_channel_counts(
    xml_records: List[Dict[str, Any]],
    expected: int = 5,
) -> Tuple[Dict[int, List[str]], int]:
    """Tally channel counts across sidecars and count mismatches.

    Ported unchanged from ``cycif_qc.py::check_channel_counts``.

    Returns:
        ``(counts_per_channel, n_mismatches)``.
    """
    ch_counts: Dict[int, List[str]] = defaultdict(list)
    for rec in xml_records:
        ch_counts[rec["n_channels"]].append(rec["xml_file"])
    bad = sum(1 for r in xml_records if r["n_channels"] != expected)
    return dict(ch_counts), bad


def check_pixel_sizes(
    xml_records: List[Dict[str, Any]],
    bounds_um: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Return a per-file pixel-size table, optionally flagged against sanity bounds.

    The per-file table itself is ported unchanged from
    ``cycif_qc.py::check_pixel_sizes``. ``bounds_um`` is a new, opt-in
    addition not present in the source (which only ever checked per-slide
    *consistency*, via :func:`build_intake_report` — that check still runs
    regardless of ``bounds_um``); when supplied, an additional
    ``within_bounds`` column is added.

    Args:
        xml_records: output of :func:`parse_ome_sidecars`.
        bounds_um: optional ``(min, max)`` plausible pixel size in microns.

    Returns:
        DataFrame with columns ``slide``, ``xml``, ``pixel_size_um``
        (plus ``within_bounds`` if ``bounds_um`` is given).
    """
    records = [{
        "slide": r["slide"], "xml": r["xml_file"],
        "pixel_size_um": r["pixel_size_um"],
    } for r in xml_records if r["pixel_size_um"] is not None]
    df = pd.DataFrame(records, columns=["slide", "xml", "pixel_size_um"])
    if bounds_um is not None and len(df):
        lo, hi = bounds_um
        df = df.copy()
        df["within_bounds"] = df["pixel_size_um"].between(lo, hi)
    return df


def check_dimensions(xml_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a per-file image-dimension table.

    Ported unchanged from ``cycif_qc.py::check_dimensions``.
    """
    records = []
    for r in xml_records:
        d = r["dims"]
        if d and isinstance(d, dict):
            records.append({
                "slide": r["slide"],
                "file": r["xml_file"],
                "Width": int(d.get("SizeX") or 0),
                "Height": int(d.get("SizeY") or 0),
                "Channels": int(d.get("SizeC") or 0),
                "Z-slices": int(d.get("SizeZ") or 0),
            })
    return pd.DataFrame(records, columns=["slide", "file", "Width", "Height", "Channels", "Z-slices"])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Structured checks report
# ═══════════════════════════════════════════════════════════════════════════

def build_intake_report(
    df_files: pd.DataFrame,
    df_parsed: pd.DataFrame,
    unparsed: List[str],
    xml_records: List[Dict[str, Any]],
    px_df: pd.DataFrame,
    missing: List[Dict[str, str]],
    config: RawImageIntakeConfig = PREVAIL_BARRETTS_INTAKE_CONFIG,
) -> pd.DataFrame:
    """Build the pass/fail check table.

    Ported from ``cycif_qc.py::build_qc_report``, generalized: the
    per-slide expected-count check uses ``config.expected_files_per_slide``
    and is *skipped* (status ``—``, not a fabricated pass or fail) when
    that mapping is not supplied, instead of silently assuming 60 files
    per slide as the source did. Channel-count, pixel-size-consistency,
    and round-coverage checks are otherwise unchanged.

    Returns:
        DataFrame with columns ``Check``, ``Status`` (``✓``/``⚠``/``—``), ``Detail``.
    """
    checks = []
    actual_counts = df_files["slide"].value_counts().to_dict() if len(df_files) else {}
    slides = sorted(df_files["slide"].unique()) if len(df_files) else []

    if config.expected_files_per_slide is None:
        checks.append({
            "Check": "Per-slide file count", "Status": "—",
            "Detail": "expected_files_per_slide not specified — check skipped",
        })
    else:
        for slide, exp in config.expected_files_per_slide.items():
            act = actual_counts.get(slide, 0)
            status = "✓" if act == exp else "⚠"
            checks.append({"Check": f"{slide} file count", "Status": status, "Detail": f"{act} / {exp}"})

    status = "✓" if len(unparsed) == 0 else "⚠"
    checks.append({"Check": "Filename parsing", "Status": status,
                    "Detail": f"{len(df_parsed)}/{len(df_files)} OK"})

    bad_ch = sum(1 for r in xml_records if r["n_channels"] != config.expected_channels)
    status = "✓" if bad_ch == 0 else "⚠"
    checks.append({"Check": f"Channel count ({config.expected_channels}/{config.expected_channels})",
                    "Status": status, "Detail": f"{bad_ch} mismatch(es)"})

    bad_px = 0
    if len(px_df) > 0:
        unique_px = px_df.groupby("slide")["pixel_size_um"].apply(lambda x: x.unique())
        bad_px = sum(1 for v in unique_px if len(v) > 1)
    status = "✓" if bad_px == 0 else "⚠"
    checks.append({"Check": "Pixel size consistent per slide", "Status": status,
                    "Detail": f"{bad_px} slide(s) with multiple values"})

    if config.pixel_size_bounds_um is not None and "within_bounds" in px_df.columns:
        bad_bounds = int((~px_df["within_bounds"]).sum())
        status = "✓" if bad_bounds == 0 else "⚠"
        checks.append({
            "Check": f"Pixel size within {config.pixel_size_bounds_um} µm",
            "Status": status, "Detail": f"{bad_bounds} file(s) out of bounds",
        })

    status = "✓" if len(missing) == 0 else "⚠"
    checks.append({"Check": f"All {len(config.expected_rounds)} expected rounds present",
                    "Status": status, "Detail": f"{len(missing)} missing"})

    return pd.DataFrame(checks, columns=["Check", "Status", "Detail"])


# ═══════════════════════════════════════════════════════════════════════════
# 5. Locating sample files (for preview)
# ═══════════════════════════════════════════════════════════════════════════

def find_sample_raw_file(
    df_parsed: pd.DataFrame,
    slide: Optional[str] = None,
    scene: str = "sceneA1",
    round_label: str = "R1",
) -> Optional[str]:
    """Look up the on-disk path for a specific (slide, scene, round).

    Ported from ``cycif_qc.py::find_sample_czi``, generalized: reads the
    ``path`` column already carried through by :func:`parse_filenames`
    instead of reconstructing the filename from its parsed pieces (see
    module docstring).
    """
    if len(df_parsed) == 0:
        return None
    if slide is None:
        slides = sorted(df_parsed["slide"].unique())
        slide = slides[0] if slides else None
    if slide is None:
        return None

    match = df_parsed[
        (df_parsed["slide"] == slide) & (df_parsed["scene"] == scene) & (df_parsed["round"] == round_label)
    ]
    if len(match) == 0:
        return None
    path = match.iloc[0]["path"]
    return path if Path(path).exists() else None


def find_all_round_paths(
    df_parsed: pd.DataFrame,
    slide: Optional[str] = None,
    scene: str = "sceneA1",
) -> Dict[str, str]:
    """Look up on-disk paths for every round of a given (slide, scene).

    Ported from ``cycif_qc.py::find_all_round_paths``, generalized the
    same way as :func:`find_sample_raw_file`.
    """
    if len(df_parsed) == 0:
        return {}
    if slide is None:
        slides = sorted(df_parsed["slide"].unique())
        slide = slides[0] if slides else None
    if slide is None:
        return {}

    match = df_parsed[(df_parsed["slide"] == slide) & (df_parsed["scene"] == scene)]
    result = {}
    for _, row in match.iterrows():
        if Path(row["path"]).exists():
            result[row["round"]] = row["path"]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. QC plots
# ═══════════════════════════════════════════════════════════════════════════
# Uses this lab's plot defaults (5pt font, 0.5pt lines, s=3.14 scatter,
# pdf.fonttype=42, exact mm canvas, no bbox_inches='tight') applied locally
# via plt.rc_context so importing this module never mutates global
# matplotlib state. Per-panel in-axis titles are an explicit, documented
# override of the lab default: these are multi-panel diagnostic QC report
# images (not manuscript figures) where each panel is a different
# slide/round and is unreadable without a label identifying which.

_LAB_PLOT_RC = {
    "font.size": 5,
    "font.weight": "normal",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,
}


def _mm_figsize(width_mm: float, height_mm: float) -> Tuple[float, float]:
    return (width_mm / 25.4, height_mm / 25.4)


def _fig_to_png(fig, out_path: Path) -> str:
    fig.savefig(out_path, format="png", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out_path)


def plot_size_distribution(df_files: pd.DataFrame, out_path: Path) -> str:
    """File size histogram per slide. Ported from ``plot_size_distribution``."""
    import matplotlib.pyplot as plt

    if len(df_files) == 0:
        return ""
    slides = sorted(df_files["slide"].unique())
    n = max(len(slides), 1)
    with plt.rc_context(_LAB_PLOT_RC):
        fig, axes = plt.subplots(1, n, figsize=_mm_figsize(55 * n, 45), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, slide in zip(axes, slides):
            grp = df_files[df_files["slide"] == slide]
            ax.hist(grp["size_gb"], bins=20, color=_slide_color(slide, slides), edgecolor="white")
            ax.set_title(f"{slide} (n={len(grp)}, {grp['size_gb'].sum():.1f} GB)")
            ax.set_xlabel("File size (GB)")
            ax.set_ylabel("Files")
            med = grp["size_gb"].median()
            ax.axvline(med, color="red", ls="--", label=f"median={med:.2f}")
            ax.legend()
        return _fig_to_png(fig, out_path)


def plot_slide_bar(df_files: pd.DataFrame, out_path: Path) -> str:
    """Bar chart: file count and total GB per slide. Ported from ``plot_slide_bar``."""
    import matplotlib.pyplot as plt

    if len(df_files) == 0:
        return ""
    summary = df_files.groupby("slide").agg(Files=("fname", "count"), Total_GB=("size_gb", "sum")).reset_index()
    colors = [_slide_color(s, summary["slide"]) for s in summary["slide"]]

    with plt.rc_context(_LAB_PLOT_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_mm_figsize(90, 45), constrained_layout=True)
        ax1.bar(summary["slide"], summary["Files"], color=colors, edgecolor="white", width=0.5)
        ax1.set_title("File Count by Slide")
        ax1.set_ylabel("Files")
        for i, v in enumerate(summary["Files"]):
            ax1.text(i, v, str(v), ha="center", va="bottom")

        ax2.bar(summary["slide"], summary["Total_GB"], color=colors, edgecolor="white", width=0.5)
        ax2.set_title("Total Volume by Slide")
        ax2.set_ylabel("GB")
        for i, v in enumerate(summary["Total_GB"]):
            ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom")
        return _fig_to_png(fig, out_path)


def plot_channel_counts(xml_records: List[Dict[str, Any]], out_path: Path) -> str:
    """Bar chart: files per observed channel count. Ported from ``plot_channel_counts``."""
    import matplotlib.pyplot as plt

    if not xml_records:
        return ""
    ch_counts: Dict[int, int] = defaultdict(int)
    for r in xml_records:
        ch_counts[r["n_channels"]] += 1

    with plt.rc_context(_LAB_PLOT_RC):
        fig, ax = plt.subplots(figsize=_mm_figsize(50, 35), constrained_layout=True)
        channels = sorted(ch_counts.keys())
        counts = [ch_counts[c] for c in channels]
        bars = ax.bar([str(c) for c in channels], counts, color="#0891b2", edgecolor="white", width=0.5)
        ax.set_title("Channel Count Distribution")
        ax.set_xlabel("Channels per file")
        ax.set_ylabel("Files")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(cnt), ha="center", va="bottom")
        return _fig_to_png(fig, out_path)


def plot_pixel_sizes(px_df: pd.DataFrame, out_path: Path) -> str:
    """Jittered scatter + boxplot of pixel sizes per slide. Ported from ``plot_pixel_sizes``
    (dropping the seaborn dependency; matplotlib only)."""
    import matplotlib.pyplot as plt

    if len(px_df) == 0:
        return ""
    slides = sorted(px_df["slide"].unique())
    with plt.rc_context(_LAB_PLOT_RC):
        fig, ax = plt.subplots(figsize=_mm_figsize(55, 40), constrained_layout=True)
        data = [px_df.loc[px_df["slide"] == s, "pixel_size_um"].to_numpy() for s in slides]
        ax.boxplot(data, tick_labels=slides, widths=0.3, showfliers=False)
        rng = np.random.default_rng(0)
        for i, (s, vals) in enumerate(zip(slides, data), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=3.14, color=_slide_color(s, slides), alpha=0.7)
        ax.set_title("Pixel Size by Slide")
        ax.set_ylabel("Pixel size (µm)")
        return _fig_to_png(fig, out_path)


def plot_dimensions(dim_df: pd.DataFrame, out_path: Path) -> str:
    """Scatter of image width vs height. Ported from ``plot_dimensions``."""
    import matplotlib.pyplot as plt

    if len(dim_df) == 0:
        return ""
    slides = sorted(dim_df["slide"].unique())
    with plt.rc_context(_LAB_PLOT_RC):
        fig, ax = plt.subplots(figsize=_mm_figsize(55, 55), constrained_layout=True)
        for slide, grp in dim_df.groupby("slide"):
            ax.scatter(grp["Width"], grp["Height"], label=slide, s=3.14, color=_slide_color(slide, slides))
        ax.set_title("Image Dimensions")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.legend()
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, "k--", alpha=0.3)
        return _fig_to_png(fig, out_path)


def _read_raw_channel(path: str, channel: int = 0, scale_factor: float = 1.0) -> np.ndarray:
    """Read one stitched channel from a per-scene mosaic CZI (lazy import).

    Ported from ``registration.py::read_czi_channel``. Requires the
    optional ``aicspylibczi`` dependency (``pip install
    spatial-omics-toolkit[imaging]``); raises ``ImportError`` with an
    install hint if it is missing, rather than failing this module's
    import for callers who never need a raw pixel preview.
    """
    try:
        import aicspylibczi
    except ImportError as e:
        raise ImportError(
            "Reading raw CZI pixel data requires the optional 'aicspylibczi' "
            "dependency. Install it with: pip install spatial-omics-toolkit[imaging]"
        ) from e
    czi = aicspylibczi.CziFile(path)
    img = czi.read_mosaic(C=channel, scale_factor=scale_factor)
    return np.squeeze(img)


def plot_dapi_preview(r1_path: str, scene_label: str, out_path: Path, dapi_channel: int = 0) -> str:
    """Low-resolution DAPI preview of one raw file. Ported from ``plot_dapi_preview``.

    Requires ``aicspylibczi`` (see :func:`_read_raw_channel`); returns
    ``""`` (does not raise) if the file is missing, matching the source's
    "preview is best-effort" behavior — the intake report itself is not
    gated on this succeeding.
    """
    import matplotlib.pyplot as plt

    if not r1_path or not Path(r1_path).exists():
        return ""
    try:
        dapi = _read_raw_channel(r1_path, channel=dapi_channel, scale_factor=0.1)
        with plt.rc_context(_LAB_PLOT_RC):
            fig, ax = plt.subplots(figsize=_mm_figsize(80, 80), constrained_layout=True)
            ax.imshow(dapi, cmap="gray", vmin=0, vmax=np.percentile(dapi, 99.5))
            ax.set_title(f"{scene_label} — R1 DAPI (10% scale)")
            ax.axis("off")
            h, w = dapi.shape
            ax.text(5, 15, f"{w}x{h} px | 10% scale", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
            return _fig_to_png(fig, out_path)
    except Exception as e:
        logger.warning(f"DAPI preview failed: {e}")
        return ""


def plot_round_dapi_grid(czi_paths: Dict[str, str], scene_label: str, out_path: Path, dapi_channel: int = 0) -> str:
    """Grid of DAPI previews across all rounds for one scene. Ported from ``plot_round_dapi_grid``."""
    import matplotlib.pyplot as plt

    valid = {rnd: p for rnd, p in czi_paths.items() if p and Path(p).exists()}
    if not valid:
        return ""

    rounds = sorted(valid.keys(), key=_round_sort_key)
    n = len(rounds)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    with plt.rc_context(_LAB_PLOT_RC):
        fig, axes = plt.subplots(rows, cols, figsize=_mm_figsize(cols * 40, rows * 40), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, rnd in zip(axes, rounds):
            try:
                dapi = _read_raw_channel(valid[rnd], channel=dapi_channel, scale_factor=0.05)
                ax.imshow(dapi, cmap="gray", vmin=0, vmax=np.percentile(dapi, 99.5))
                ax.set_title(rnd)
            except Exception:
                ax.text(0.5, 0.5, f"{rnd}\n(no data)", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")

        for ax in axes[len(rounds):]:
            ax.axis("off")
            ax.set_visible(False)

        fig.suptitle(f"{scene_label} — DAPI across rounds (5% scale)")
        return _fig_to_png(fig, out_path)


def generate_all_qc_plots(
    df_files: pd.DataFrame,
    xml_records: List[Dict[str, Any]],
    px_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    reports_dir: str,
    sample_r1_path: Optional[str] = None,
    sample_scene_label: Optional[str] = None,
    all_round_paths: Optional[Dict[str, str]] = None,
    dapi_channel: int = 0,
) -> Dict[str, str]:
    """Generate every QC plot PNG and return relative paths for the HTML report.

    Ported from ``cycif_qc.py::generate_all_qc_plots``.
    """
    img_dir = Path(reports_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def rel(name: str) -> str:
        return f"images/{name}.png"

    plots: Dict[str, str] = {}
    plots["size_dist"] = rel("size_dist") if plot_size_distribution(df_files, img_dir / "size_dist.png") else ""
    plots["slide_bar"] = rel("slide_bar") if plot_slide_bar(df_files, img_dir / "slide_bar.png") else ""
    if xml_records:
        plots["channels"] = rel("channels") if plot_channel_counts(xml_records, img_dir / "channels.png") else ""
    if len(px_df) > 0:
        plots["pixel_sizes"] = rel("pixel_sizes") if plot_pixel_sizes(px_df, img_dir / "pixel_sizes.png") else ""
    if len(dim_df) > 0:
        plots["dimensions"] = rel("dimensions") if plot_dimensions(dim_df, img_dir / "dimensions.png") else ""
    if sample_r1_path:
        label = sample_scene_label or "Sample"
        plots["dapi"] = rel("dapi") if plot_dapi_preview(sample_r1_path, label, img_dir / "dapi.png", dapi_channel) else ""
    if all_round_paths and len(all_round_paths) > 1:
        label = sample_scene_label or "Sample"
        plots["dapi_rounds"] = rel("dapi_rounds") if plot_round_dapi_grid(all_round_paths, label, img_dir / "dapi_rounds.png", dapi_channel) else ""
    return plots


# ═══════════════════════════════════════════════════════════════════════════
# 7. HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _status_badge(status: str) -> str:
    if status == "✓":
        return '<span class="badge pass">✓ PASS</span>'
    if status == "⚠":
        return '<span class="badge warn">⚠ FAIL</span>'
    return f'<span class="badge">{status}</span>'


def _html_escape(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pct(a: float, b: float) -> str:
    return "—" if b == 0 else f"{100.0 * a / b:.1f}%"


def _build_css() -> str:
    return """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.6;padding:0}
.container{max-width:1300px;margin:0 auto;padding:20px}
.header{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);color:#fff;padding:32px 40px;border-radius:14px;margin-bottom:28px}
.header h1{font-size:26px;font-weight:700;margin-bottom:6px}
.header .sub{font-size:14px;opacity:.85;margin-bottom:12px}
.header .meta{font-size:13px;opacity:.7;display:flex;gap:20px;flex-wrap:wrap}
.ribbon{display:flex;gap:14px;margin-bottom:28px;flex-wrap:wrap}
.ribbon-card{flex:1;min-width:140px;background:#fff;border-radius:10px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}
.ribbon-card .num{font-size:30px;font-weight:700;line-height:1.2}
.ribbon-card .lbl{font-size:12px;color:#64748b;margin-top:2px}
.ribbon-card.pass .num{color:#16a34a}
.ribbon-card.info .num{color:#0f3460}
.tabs{display:flex;gap:2px;margin-bottom:0;overflow-x:auto;flex-wrap:wrap}
.tab-btn{padding:10px 18px;background:#e2e8f0;border:none;border-radius:8px 8px 0 0;font-size:13px;font-weight:600;color:#475569;cursor:pointer;white-space:nowrap}
.tab-btn.active{background:#fff;color:#0f172a}
.tab-panel{display:none;background:#fff;border-radius:0 10px 10px 10px;padding:28px;box-shadow:0 1px 4px rgba(0,0,0,.07)}
.tab-panel.active{display:block}
h2{font-size:20px;color:#0f172a;margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #e2e8f0}
h3{font-size:16px;color:#334155;margin:24px 0 12px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
th{background:#f1f5f9;color:#475569;font-weight:600;padding:9px 11px;text-align:left;border-bottom:2px solid #e2e8f0}
td{padding:8px 11px;border-bottom:1px solid #f1f5f9}
.badge{display:inline-block;padding:2px 11px;border-radius:12px;font-size:11px;font-weight:600}
.badge.pass{background:#dcfce7;color:#166534}
.badge.warn{background:#fef2f2;color:#991b1b}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}
.g-item{display:flex;align-items:center;gap:10px;background:#f8fafc;padding:11px 15px;border-radius:8px;border-left:4px solid #e2e8f0}
.g-item.pass{border-left-color:#16a34a}
.g-item.fail{border-left-color:#dc2626}
.g-item .gl{flex:1;font-size:13px}
.g-item .gd{font-size:12px;color:#64748b}
.plot-box{text-align:center;margin:18px 0;padding:12px;background:#fafafa;border-radius:10px;border:1px solid #e2e8f0}
.plot-box img{max-width:100%;height:auto;border-radius:6px}
.footer{text-align:center;font-size:12px;color:#94a3b8;margin-top:36px;padding:18px 0}
code{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12px}
</style>"""


def build_html_report(
    df_files: pd.DataFrame,
    df_parsed: pd.DataFrame,
    unparsed: List[str],
    xml_records: List[Dict[str, Any]],
    px_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    report: pd.DataFrame,
    missing: List[Dict[str, str]],
    qc_plots: Dict[str, str],
    raw_dir: str,
    reports_dir: str,
    run_label: str,
    config: RawImageIntakeConfig = PREVAIL_BARRETTS_INTAKE_CONFIG,
    sample_info: Optional[Dict[str, str]] = None,
    pipeline_name: str = "spatial_omics_toolkit",
) -> str:
    """Generate the self-contained intake QC HTML report.

    Ported from ``cycif_qc.py::build_html_report``, generalized: the
    hardcoded "PREVAIL Spatial Proteomics Pipeline" footer and the
    hardcoded "6 rounds (R1-R5, R5Q)" coverage sentence are now derived
    from ``pipeline_name``/``config.expected_rounds``.
    """
    parts = []
    total_files = len(df_files)
    total_gb = f"{df_files['size_gb'].sum():.1f}" if total_files else "0.0"
    slide_list = sorted(df_files["slide"].unique()) if total_files else []
    n_pass = int((report["Status"] == "✓").sum()) if len(report) else 0
    n_checked = int((report["Status"] != "—").sum()) if len(report) else 0
    n_scenes = len(df_parsed[["slide", "scene"]].drop_duplicates()) if len(df_parsed) else 0
    overall = "PASS" if n_checked and n_pass == n_checked else "ISSUES DETECTED" if n_checked else "NO CHECKS RUN"
    overall_color = "#16a34a" if overall == "PASS" else "#dc2626"

    parts.append(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Raw Image Intake QC — {_html_escape(run_label)}</title>{_build_css()}</head><body>
<div class="container">
<div class="header"><h1>Raw Image Intake QC Report</h1>
<div class="sub">Pre-registration integrity check for raw microscopy files ({_html_escape(pipeline_name)})</div>
<div class="meta">
<span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
<span>{_html_escape(Path(raw_dir).name)}</span>
<span>{total_files} files ({total_gb} GB)</span>
<span>{n_scenes} scenes</span>
<span><strong style="color:{overall_color}">{overall}</strong></span>
</div></div>""")

    parts.append(f"""<div class="ribbon">
<div class="ribbon-card pass"><div class="num">{n_pass}/{n_checked}</div><div class="lbl">Checks Passed</div></div>
<div class="ribbon-card info"><div class="num">{total_files}</div><div class="lbl">Files Scanned</div></div>
<div class="ribbon-card info"><div class="num">{total_gb} GB</div><div class="lbl">Total Data</div></div>
<div class="ribbon-card info"><div class="num">{len(slide_list)}</div><div class="lbl">Slides</div></div>
<div class="ribbon-card info"><div class="num">{n_scenes}</div><div class="lbl">Scenes</div></div>
</div>""")

    tabs = [("exec", "Executive Summary"), ("inventory", "File Inventory"), ("qcplots", "QC Plots"),
            ("filenames", "Filename Validation"), ("metadata", "OME Metadata"), ("preview", "Sample Preview")]
    parts.append('<div class="tabs">')
    for i, (tid, tlabel) in enumerate(tabs):
        parts.append(f'<button class="tab-btn{" active" if i == 0 else ""}" onclick="switchTab(\'{tid}\')">{_html_escape(tlabel)}</button>')
    parts.append("</div>")

    # Tab 1: Executive Summary
    parts.append('<div id="tab-exec" class="tab-panel active">')
    parts.append("<h2>Executive Summary</h2>")
    if len(report) > 0:
        parts.append('<div class="grid">')
        for _, r in report.iterrows():
            cls = "pass" if r["Status"] == "✓" else "fail" if r["Status"] == "⚠" else ""
            parts.append(f'<div class="g-item {cls}"><span class="badge {cls}">{_html_escape(r["Status"])}</span>'
                         f'<span class="gl">{_html_escape(r["Check"])}</span>'
                         f'<span class="gd">{_html_escape(r["Detail"])}</span></div>')
        parts.append("</div>")
    parts.append("<h3>Scene Coverage</h3>")
    parts.append(f'<p>Expecting <strong>{len(config.expected_rounds)}</strong> round(s) per scene: '
                 f'<code>{_html_escape(", ".join(config.expected_rounds))}</code>. '
                 f'{"All present." if not missing else f"{len(missing)} missing combination(s) — see Filename Validation tab."}</p>')
    if qc_plots.get("slide_bar"):
        parts.append(f'<div class="plot-box"><img src="{qc_plots["slide_bar"]}"></div>')
    parts.append("</div>")

    # Tab 2: Inventory
    parts.append('<div id="tab-inventory" class="tab-panel">')
    parts.append("<h2>File Inventory</h2>")
    parts.append(f'<p>Total <strong>{total_files}</strong> files across <strong>{len(slide_list)}</strong> slide(s), '
                 f'<strong>{total_gb} GB</strong>.</p>')
    if total_files:
        grp = df_files.groupby("slide")["size_gb"]
        parts.append("<h3>Per-Slide Details</h3>")
        parts.append("<table><tr><th>Slide</th><th>Files</th><th>Total (GB)</th><th>Min (GB)</th><th>Max (GB)</th><th>Median (GB)</th></tr>")
        for s in slide_list:
            g = grp.get_group(s)
            parts.append(f"<tr><td>{_html_escape(s)}</td><td>{len(g)}</td><td>{g.sum():.2f}</td>"
                         f"<td>{g.min():.2f}</td><td>{g.max():.2f}</td><td>{g.median():.2f}</td></tr>")
        parts.append("</table>")
    if qc_plots.get("size_dist"):
        parts.append(f'<h3>File Size Distribution</h3><div class="plot-box"><img src="{qc_plots["size_dist"]}"></div>')
    parts.append("</div>")

    # Tab 3: QC Plots
    parts.append('<div id="tab-qcplots" class="tab-panel">')
    parts.append("<h2>Quality Control Plots</h2>")
    for pid, pname in [("channels", "Channel Counts"), ("pixel_sizes", "Pixel Sizes"), ("dimensions", "Dimensions")]:
        if qc_plots.get(pid):
            parts.append(f'<h3>{pname}</h3><div class="plot-box"><img src="{qc_plots[pid]}"></div>')
    parts.append("</div>")

    # Tab 4: Filename Validation
    parts.append('<div id="tab-filenames" class="tab-panel">')
    parts.append("<h2>Filename Parsing Validation</h2>")
    if len(df_parsed) > 0:
        parts.append(f'<p>Parsed <strong>{len(df_parsed)}/{total_files}</strong> filenames successfully '
                     f'({_pct(len(df_parsed), total_files)}).'
                     f'{"" if not unparsed else f" <span style=color:#dc2626>{len(unparsed)} failure(s)</span>."}</p>')
        rounds_ordered = sorted(df_parsed["round"].unique(), key=_round_sort_key)
        scenes_ordered = sorted(df_parsed["scene"].unique())
        pivot_tmp = df_parsed.pivot_table(index="round", columns="scene", values="size_gb", aggfunc="count", fill_value=0)
        parts.append("<h3>Round x Scene Matrix (file count)</h3>")
        parts.append("<table><tr><th>Round</th>" + "".join(f"<th>{_html_escape(s)}</th>" for s in scenes_ordered) + "</tr>")
        for rnd in rounds_ordered:
            parts.append(f"<tr><td><strong>{rnd}</strong></td>")
            for sc in scenes_ordered:
                v = pivot_tmp.loc[rnd, sc] if rnd in pivot_tmp.index and sc in pivot_tmp.columns else 0
                bg = "background:#dcfce7" if v >= 1 else "background:#fef2f2"
                parts.append(f'<td style="{bg};text-align:center">{int(v)}</td>')
            parts.append("</tr>")
        parts.append("</table>")
    if unparsed:
        parts.append("<h3>Unparsed Filenames</h3><table><tr><th>Filename</th></tr>")
        for fn in unparsed:
            parts.append(f"<tr><td><code>{_html_escape(fn)}</code></td></tr>")
        parts.append("</table>")
    parts.append('<div style="margin-top:14px">')
    if not missing:
        parts.append('<span style="color:#16a34a;font-weight:600">All expected rounds present for every scene.</span>')
    else:
        parts.append(f'<span style="color:#dc2626;font-weight:600">{len(missing)} missing round/scene combination(s).</span>')
        parts.append("<table><tr><th>Slide</th><th>Scene</th><th>Missing round</th></tr>")
        for m in missing:
            parts.append(f"<tr><td>{_html_escape(m['slide'])}</td><td>{_html_escape(m['scene'])}</td><td>{_html_escape(m['round'])}</td></tr>")
        parts.append("</table>")
    parts.append("</div></div>")

    # Tab 5: OME Metadata
    parts.append('<div id="tab-metadata" class="tab-panel">')
    parts.append("<h2>OME Metadata &amp; Channel Analysis</h2>")
    if xml_records:
        parts.append(f"<p>Parsed <strong>{len(xml_records)}</strong> OME-XML sidecar(s) "
                     f"({_pct(len(xml_records), total_files)} of files).</p>")
        ch_counts: Dict[int, int] = defaultdict(int)
        for r in xml_records:
            ch_counts[r["n_channels"]] += 1
        parts.append("<h3>Channel Count Distribution</h3><table><tr><th>Channels</th><th>Files</th><th>%</th></tr>")
        for n in sorted(ch_counts):
            parts.append(f"<tr><td>{n}</td><td>{ch_counts[n]}</td><td>{_pct(ch_counts[n], len(xml_records))}</td></tr>")
        parts.append("</table>")
        bad_ch = sum(1 for r in xml_records if r["n_channels"] != config.expected_channels)
        color = "#16a34a" if bad_ch == 0 else "#dc2626"
        msg = (f"All {len(xml_records)} files have {config.expected_channels} channels." if bad_ch == 0
               else f"{bad_ch} file(s) have non-standard channel counts.")
        parts.append(f'<div style="color:{color}">{msg}</div>')
        if len(px_df) > 0:
            parts.append("<h3>Pixel Size (µm/px)</h3>")
            parts.append("<table><tr><th>Slide</th><th>Count</th><th>Mean</th><th>Std</th><th>Unique</th><th>Status</th></tr>")
            for s in slide_list:
                sd = px_df[px_df["slide"] == s]
                if len(sd) == 0:
                    continue
                uniq = sd["pixel_size_um"].unique()
                ok = len(uniq) == 1
                parts.append(f"<tr><td>{_html_escape(s)}</td><td>{len(sd)}</td><td>{sd['pixel_size_um'].mean():.4f}</td>"
                             f"<td>{sd['pixel_size_um'].std():.4f}</td>"
                             f"<td>{', '.join(f'{v:.4f}' for v in uniq)}</td>"
                             f"<td>{_status_badge('✓' if ok else '⚠')}</td></tr>")
            parts.append("</table>")
        if qc_plots.get("channels"):
            parts.append(f'<div class="plot-box"><img src="{qc_plots["channels"]}"></div>')
        if qc_plots.get("pixel_sizes"):
            parts.append(f'<div class="plot-box"><img src="{qc_plots["pixel_sizes"]}"></div>')
        if len(dim_df) > 0:
            parts.append("<h3>Image Dimensions</h3>")
            parts.append("<table><tr><th>Dimension</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr>")
            for col in ["Width", "Height", "Channels", "Z-slices"]:
                if col in dim_df.columns:
                    parts.append(f"<tr><td><strong>{col}</strong></td><td>{dim_df[col].mean():.0f}</td>"
                                 f"<td>{dim_df[col].median():.0f}</td><td>{dim_df[col].min():.0f}</td>"
                                 f"<td>{dim_df[col].max():.0f}</td></tr>")
            parts.append("</table>")
        if qc_plots.get("dimensions"):
            parts.append(f'<div class="plot-box"><img src="{qc_plots["dimensions"]}"></div>')
    parts.append("</div>")

    # Tab 6: Sample Preview
    parts.append('<div id="tab-preview" class="tab-panel">')
    parts.append("<h2>Sample DAPI Preview</h2>")
    if sample_info:
        parts.append(f'<p>Slide: <strong>{_html_escape(sample_info.get("slide", ""))}</strong> | '
                     f'Scene: <strong>{_html_escape(sample_info.get("scene", ""))}</strong></p>')
    if qc_plots.get("dapi"):
        parts.append(f'<div class="plot-box"><img src="{qc_plots["dapi"]}"></div>')
    if qc_plots.get("dapi_rounds"):
        parts.append(f'<h3>All Rounds — DAPI Comparison</h3><div class="plot-box"><img src="{qc_plots["dapi_rounds"]}"></div>')
    if not qc_plots.get("dapi"):
        parts.append('<p style="color:#94a3b8;font-style:italic">No DAPI preview available (pass generate_dapi_preview=True and ensure aicspylibczi is installed).</p>')
    parts.append("</div>")

    parts.append("""<script>
function switchTab(id){document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active')});
document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active')});
document.getElementById('tab-'+id).classList.add('active');
[].forEach.call(document.querySelectorAll('.tab-btn'),function(b){if(b.getAttribute('onclick')&&b.getAttribute('onclick').includes(id))b.classList.add('active')});}
</script>""")

    html_path = str(Path(reports_dir) / "intake_qc_report.html")
    parts.append(f'<div class="footer">Generated by <strong>{_html_escape(pipeline_name)}.step01_scan_intake</strong><br>'
                 f'<code>{_html_escape(html_path)}</code></div>')
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 8. step01_scan_intake — the public step function
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IntakeScanResult:
    """Return value of :func:`step01_scan_intake`."""

    inventory: pd.DataFrame
    parsed: pd.DataFrame
    unparsed: List[str]
    coverage_pivot: pd.DataFrame
    coverage_missing: List[Dict[str, str]]
    xml_records: List[Dict[str, Any]]
    pixel_sizes: pd.DataFrame
    dimensions: pd.DataFrame
    checks: pd.DataFrame
    html_report_path: Optional[str] = None

    @property
    def all_checks_passed(self) -> bool:
        """True iff every *run* check (Status != '—') passed. False if none ran."""
        checked = self.checks[self.checks["Status"] != "—"] if len(self.checks) else self.checks
        return len(checked) > 0 and bool((checked["Status"] == "✓").all())


def step01_scan_intake(
    raw_dir: str,
    reports_dir: str,
    config: RawImageIntakeConfig = PREVAIL_BARRETTS_INTAKE_CONFIG,
    sample_slide: Optional[str] = None,
    sample_scene: str = "sceneA1",
    sample_round: str = "R1",
    run_label: Optional[str] = None,
    generate_html_report: bool = True,
    generate_dapi_preview: bool = False,
    pipeline_name: str = "spatial_omics_toolkit",
    strict: bool = False,
    verbose: bool = True,
) -> IntakeScanResult:
    """Scan, parse, and validate a raw CZI intake directory before registration.

    Ported and generalized from ``cycif_qc.py::run_full_qc`` — the
    end-to-end orchestration of inventory scan, filename parsing, round x
    scene coverage checking, OME-XML metadata validation, and HTML report
    generation. See the module docstring for what was changed to
    generalize this away from the PREVAIL hBE-TMA cohort's specific
    naming/counts.

    Every individual check function used here (:func:`parse_filenames`,
    :func:`check_round_scene_coverage`, :func:`check_channel_counts`,
    :func:`check_pixel_sizes`, :func:`build_intake_report`) returns a
    structured result rather than raising, so a genuinely incomplete or
    malformed intake directory does not silently look fine — it shows up
    as ``⚠`` rows in ``result.checks`` and non-empty
    ``result.coverage_missing`` / ``result.unparsed``. Set ``strict=True``
    to turn any check failure into a raised ``RuntimeError`` once you've
    reviewed the report once and want subsequent runs to stop hard.

    Args:
        raw_dir: directory containing one subdirectory per slide/sample,
            each holding raw ``config.raw_extension`` files and (optionally)
            ``*{config.ome_xml_suffix}`` sidecars.
        reports_dir: directory to write ``intake_qc_report.html``,
            ``intake_inventory.csv``, ``intake_checks.csv``, and
            ``images/*.png`` into (created if missing).
        config: naming/count assumptions; defaults to the PREVAIL hBE-TMA
            worked example. Pass a different :class:`RawImageIntakeConfig`
            for a different acquisition batch.
        sample_slide / sample_scene / sample_round: which (slide, scene,
            round) to use for the DAPI preview tab. ``sample_slide=None``
            (default) picks the first slide alphabetically.
        run_label: label embedded in the report; defaults to a timestamp.
        generate_html_report: if False, skip HTML/plot generation and
            only return the structured tables (fast path for scripted use).
        generate_dapi_preview: if True, attempt a raw-pixel DAPI preview
            (requires the optional ``aicspylibczi`` dependency). Off by
            default since it needs an extra dependency and reads actual
            image data.
        pipeline_name: embedded in the report header/footer.
        strict: if True, raise ``RuntimeError`` if any check in the
            resulting report has status ``⚠`` (fail) — including unparsed
            filenames or missing round/scene coverage.
        verbose: log progress via the module logger.

    Returns:
        :class:`IntakeScanResult`.

    Raises:
        FileNotFoundError: if ``raw_dir`` does not exist.
        RuntimeError: if ``strict=True`` and any check failed.
    """
    if run_label is None:
        run_label = datetime.now().strftime("%Y%m%d_%H%M")

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_path}")
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # 1. Inventory
    df_files = scan_raw_images(raw_dir, extension=config.raw_extension)
    slide_dirs = sorted(d for d in raw_path.iterdir() if d.is_dir())

    # 2. Filename parsing + coverage
    df_parsed, unparsed = parse_filenames(df_files, pattern=config.filename_pattern)
    pivot, missing = check_round_scene_coverage(df_parsed, expected_rounds=config.expected_rounds)

    # 3. OME-XML metadata
    xml_records = parse_ome_sidecars(slide_dirs, xml_suffix=config.ome_xml_suffix)
    px_df = check_pixel_sizes(xml_records, bounds_um=config.pixel_size_bounds_um)
    dim_df = check_dimensions(xml_records)

    # 4. Structured report
    checks = build_intake_report(df_files, df_parsed, unparsed, xml_records, px_df, missing, config=config)

    if verbose:
        n_pass = int((checks["Status"] == "✓").sum())
        n_checked = int((checks["Status"] != "—").sum())
        logger.info(f"step01_scan_intake: {n_pass}/{n_checked} checks passed "
                    f"({len(df_files)} files, {len(missing)} missing round/scene combos, "
                    f"{len(unparsed)} unparsed filenames)")

    html_path: Optional[str] = None
    if generate_html_report:
        if sample_slide is None:
            slides = sorted(df_files["slide"].unique())
            sample_slide = slides[0] if slides else None
        sample_r1 = find_sample_raw_file(df_parsed, sample_slide, sample_scene, sample_round) if generate_dapi_preview else None
        sample_rounds = find_all_round_paths(df_parsed, sample_slide, sample_scene) if generate_dapi_preview else {}
        sample_label = f"{sample_slide} / {sample_scene}" if sample_slide else None

        qc_plots = generate_all_qc_plots(
            df_files, xml_records, px_df, dim_df, str(reports_path),
            sample_r1_path=sample_r1, sample_scene_label=sample_label,
            all_round_paths=sample_rounds,
        )

        sample_info = {"slide": sample_slide, "scene": sample_scene} if sample_slide else None
        html = build_html_report(
            df_files, df_parsed, unparsed, xml_records, px_df, dim_df,
            checks, missing, qc_plots, raw_dir, str(reports_path), run_label,
            config=config, sample_info=sample_info, pipeline_name=pipeline_name,
        )
        html_path = str(reports_path / "intake_qc_report.html")
        Path(html_path).write_text(html, encoding="utf-8")

        df_files.to_csv(reports_path / "intake_inventory.csv", index=False)
        checks.to_csv(reports_path / "intake_checks.csv", index=False)
        if verbose:
            logger.info(f"  HTML report: {html_path}")

    result = IntakeScanResult(
        inventory=df_files, parsed=df_parsed, unparsed=unparsed,
        coverage_pivot=pivot, coverage_missing=missing, xml_records=xml_records,
        pixel_sizes=px_df, dimensions=dim_df, checks=checks, html_report_path=html_path,
    )

    if strict:
        failed = checks[checks["Status"] == "⚠"]
        if len(failed):
            raise RuntimeError(
                f"step01_scan_intake: {len(failed)} check(s) failed under strict=True:\n"
                + failed.to_string(index=False)
            )

    return result
