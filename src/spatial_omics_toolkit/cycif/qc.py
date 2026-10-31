"""step06_qc_filter_intensities — post-segmentation marker-intensity QC.

Thin wrapper around the existing
``spatial_omics_toolkit.analysis.cycif_qc`` functions (per-marker
signal-to-noise ratio, background subtraction). This is post-segmentation QC
on an already-extracted cell x marker intensity matrix — distinct from
``step01_scan_intake``, which is pre-registration QC on raw CZI/ND2 files
(see ``02_spatial_proteomics/03_analysis/02_scripts/cycif_qc.py``, a much
larger, different-purpose file despite the similar name).
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from spatial_omics_toolkit.analysis.cycif_qc import (
    calculate_marker_snr,
    filter_background_signal,
)
from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig

PathLike = Union[str, Path]


def step06_qc_filter_intensities(
    intensity_df: pd.DataFrame,
    background_percentile: float = 10.0,
    signal_percentile: float = 90.0,
    background_thresholds: Optional[Dict[str, float]] = None,
    output_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-marker SNR and subtract background from a cells x markers matrix.

    Args:
        intensity_df: cells x markers mean-intensity table.
        background_percentile / signal_percentile: percentiles defining the
            background/signal baseline used for the reported SNR (does not
            affect the subtraction itself unless ``background_thresholds`` is
            omitted, in which case the background percentile value per marker
            is used as the subtraction threshold).
        background_thresholds: explicit per-marker background cutoff to
            subtract. If omitted, uses each marker's ``background_percentile``
            value from ``intensity_df`` itself.
        output_dir: if given, writes ``step06_marker_snr.png`` (a bar chart of
            per-marker SNR, sorted, so a low-SNR marker is immediately visible
            before you trust it downstream) — the check-before-proceeding
            artifact for this step.

    Returns:
        A copy of ``intensity_df`` with background subtracted (floored at 0).
        The per-marker SNR report is attached as ``.attrs["marker_snr"]`` on
        the returned frame.
    """
    snr = calculate_marker_snr(
        intensity_df,
        background_percentile=background_percentile,
        signal_percentile=signal_percentile,
    )
    if background_thresholds is None:
        background_thresholds = {
            col: float(
                intensity_df[col].dropna().quantile(background_percentile / 100.0)
            )
            for col in intensity_df.select_dtypes(include="number").columns
        }
    filtered = filter_background_signal(intensity_df, background_thresholds)
    filtered.attrs["marker_snr"] = snr

    if output_dir is not None and snr:
        import matplotlib.pyplot as plt

        out = ensure_dir(output_dir)
        markers = sorted(snr, key=snr.get)
        fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(markers)), 4))
        ax.bar(markers, [snr[m] for m in markers], color="steelblue")
        ax.axhline(
            1.0,
            color="red",
            linestyle="--",
            linewidth=1,
            label="SNR = 1 (no signal above background)",
        )
        ax.set_ylabel("Signal-to-noise ratio")
        ax.set_title("step06: per-marker SNR (post-segmentation intensity QC)")
        ax.tick_params(axis="x", rotation=90)
        ax.legend()
        fig.tight_layout()
        savefig(fig, out / "step06_marker_snr.png", verbose=verbose)

    return filtered
