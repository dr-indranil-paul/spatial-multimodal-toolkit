"""step12_compare_groups — per-core two-group comparison with multiplicity correction.

Ported and generalized from
02_spatial_proteomics/03_analysis/02_scripts/tma_compare_groups.py.

Improvements over the source script: uses Welch's t-test (``equal_var=False``)
rather than Student's, since per-core sample sizes here are small (the
PREVAIL cohort has 9 vs. 9 cores) and unequal-variance is the safer default;
BH-FDR is applied to every metric scanned, always, not only to the ones that
happened to be nominally significant.

**This is a simple two-group per-core test, not a mixed-effects model.** If
your study design has multiple cores per patient/specimen, "core" is not the
same thing as "biological unit" — aggregate to one row per biological unit
(or pass a biological-unit id and average across a patient's cores) before
calling this, or every p-value here is inflated. See DESIGN.md non-negotiable
#2 and the pre-specified mixed-effects design in this project's own
``figure7_preregistration.md`` if that is the model your manuscript claim
needs — this function does not implement it.
"""

import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig

PathLike = Union[str, Path]

try:
    from scipy.stats import false_discovery_control as _bh_fdr
except ImportError:  # scipy < 1.11
    def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
        pvals = np.asarray(pvals, dtype=float)
        order = np.argsort(pvals)
        ranked = pvals[order] * len(pvals) / (np.arange(len(pvals)) + 1)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(adjusted)
        out[order] = np.clip(adjusted, 0, 1)
        return out


def step12_compare_groups(
    core_df: pd.DataFrame,
    group_col: str,
    metrics: Sequence[str],
    biological_unit_col: Optional[str] = None,
    output_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare every metric in ``metrics`` between exactly two groups, BH-corrected.

    Args:
        core_df: one row per core (e.g. from ``step09_aggregate_per_core`` or
            ``step10_spatial_features``), with a grouping column.
        group_col: column with exactly two non-null category values to compare.
        metrics: numeric columns to test.
        biological_unit_col: optional column identifying the true independent
            biological unit (e.g. patient id). If given and it has fewer
            unique values than rows of ``core_df``, a warning is raised — the
            n reported by this function is rows of ``core_df``, which may
            overstate independent replication.
        output_dir: if given, writes one violin+strip plot per metric
            (``step12_<metric>.png``, restored from the source project's own
            ``tma_compare_groups.py::plot_group_comp``) plus
            ``step12_comparison_table.csv`` — the visual form of every row in
            the returned table, so a nominally-significant raw p-value that
            disappears after FDR correction is visible as "two overlapping
            clouds of points," not just a number.

    Returns:
        One row per metric: ``n_group_a``, ``n_group_b``, mean per group,
        ``diff``, ``cohens_d``, ``welch_t_p``, ``mannwhitney_p``, ``fdr_bh_p``.
    """
    if group_col not in core_df.columns:
        raise ValueError(f"group_col {group_col!r} not found in core_df.columns")

    groups = sorted(core_df[group_col].dropna().unique().tolist())
    if len(groups) != 2:
        raise ValueError(
            f"step12_compare_groups requires exactly 2 groups in {group_col!r}, found {groups}"
        )
    group_a, group_b = groups

    if biological_unit_col is not None and biological_unit_col in core_df.columns:
        n_units = core_df[biological_unit_col].nunique()
        if n_units < len(core_df):
            warnings.warn(
                f"{len(core_df)} rows map to only {n_units} distinct {biological_unit_col!r} values — "
                f"rows of core_df are not independent biological replicates. Aggregate to one row per "
                f"{biological_unit_col} before trusting these p-values as n={len(core_df)}.",
                stacklevel=2,
            )

    rows = []
    p_raw = []
    for metric in metrics:
        if metric not in core_df.columns:
            continue
        vals_a = core_df.loc[core_df[group_col] == group_a, metric].dropna().to_numpy()
        vals_b = core_df.loc[core_df[group_col] == group_b, metric].dropna().to_numpy()

        row = {
            "metric": metric,
            "n_group_a": len(vals_a),
            "n_group_b": len(vals_b),
            "group_a": group_a,
            "group_b": group_b,
            "mean_a": np.nan,
            "mean_b": np.nan,
            "diff": np.nan,
            "cohens_d": np.nan,
            "welch_t_p": np.nan,
            "mannwhitney_p": np.nan,
        }
        if len(vals_a) >= 2 and len(vals_b) >= 2:
            row["mean_a"] = float(vals_a.mean())
            row["mean_b"] = float(vals_b.mean())
            row["diff"] = row["mean_a"] - row["mean_b"]
            pooled_var = (vals_a.var(ddof=1) + vals_b.var(ddof=1)) / 2
            row["cohens_d"] = float(row["diff"] / np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
            _, row["welch_t_p"] = ttest_ind(vals_a, vals_b, equal_var=False)
            _, row["mannwhitney_p"] = mannwhitneyu(vals_a, vals_b)
        rows.append(row)
        p_raw.append(row["welch_t_p"])

    out = pd.DataFrame(rows)
    finite_mask = out["welch_t_p"].notna()
    out["fdr_bh_p"] = np.nan
    if finite_mask.any():
        out.loc[finite_mask, "fdr_bh_p"] = _bh_fdr(out.loc[finite_mask, "welch_t_p"].to_numpy())

    n_a = int(core_df[group_col].eq(group_a).sum())
    n_b = int(core_df[group_col].eq(group_b).sum())
    print(f"Compared {len(out)} metrics: {group_a} (n={n_a}) vs {group_b} (n={n_b}); "
          f"n is rows of core_df — verify these are independent biological units before citing.")

    if output_dir is not None:
        import matplotlib.pyplot as plt

        out_dir = ensure_dir(output_dir)
        csv_path = out_dir / "step12_comparison_table.csv"
        out.to_csv(csv_path, index=False)
        if verbose:
            print(f"  Comparison table -> {csv_path}")

        for metric in metrics:
            if metric not in core_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(4, 4))
            data = [
                core_df.loc[core_df[group_col] == group_a, metric].dropna().to_numpy(),
                core_df.loc[core_df[group_col] == group_b, metric].dropna().to_numpy(),
            ]
            parts = ax.violinplot(data, showmedians=True)
            for i, vals in enumerate(data, start=1):
                jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.08
                ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", alpha=0.6, s=12)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([group_a, group_b])
            row = out.loc[out["metric"] == metric]
            fdr = row["fdr_bh_p"].iloc[0] if len(row) else np.nan
            raw = row["welch_t_p"].iloc[0] if len(row) else np.nan
            ax.set_title(f"{metric}\nraw p={raw:.3g}  FDR p={fdr:.3g}", fontsize=9)
            fig.tight_layout()
            savefig(fig, out_dir / f"step12_{metric}.png", verbose=verbose)

    return out
