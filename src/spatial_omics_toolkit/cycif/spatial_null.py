"""step11_spatial_null — spatial null permutation control for neighbor-enrichment claims.

DESIGN.md non-negotiable #5: "Spatial feature testing enforces explicit
spatial null permutation models (distance-decay permutations or label
swapping within tissue boundaries) before claiming spatial niche
enrichment." No script anywhere in 02_spatial_proteomics implements this —
the one permutation routine found there
(``run_exhaustive_spatial_ml.py::empirical_permutations``) permutes
*outcome labels* to null a classifier AUROC, which is a valid predictive
null but not a spatial one. This module is new code, not a port.

The control implemented here is **label swapping within tissue boundaries**:
cell coordinates are held fixed (they are the observed tissue geometry) and
cell-type labels are randomly reassigned among the eligible cell population
(the whole core, or — when ``tissue_mask_col`` is supplied — only among cells
inside the same tissue-mask region), rebuilding the same
``spatial.<source>.to.<target>.knn<k>.delta`` neighbor-enrichment statistic
computed by ``step10_spatial_features`` under each permutation. This is the
standard neighborhood-enrichment permutation test used in the spatial
proteomics literature (e.g. histoCAT/squidpy-style ``nhood_enrichment``).

A full continuous distance-decay/CSR null (randomizing point *positions*
rather than labels) is a natural extension but is not implemented here —
label swapping is the more directly applicable control for a categorical
cell-type neighbor-enrichment claim and is what this function provides.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig


def _knn_delta(
    tm: np.ndarray,
    sm: np.ndarray,
    indices: np.ndarray,
    k: int,
    n: int,
) -> float:
    src = np.flatnonzero(sm)
    if len(src) == 0 or not tm.any():
        return np.nan
    kk = min(k, n - 1)
    neigh = tm[indices[src, 1:kk + 1]]
    observed = neigh.mean(axis=1)
    expected = (tm.sum() - tm[src].astype(float)) / max(n - 1, 1)
    return float(np.nanmean(observed - expected))


def _core_spatial_null(
    grp: pd.DataFrame,
    x_col: str,
    y_col: str,
    type_col: str,
    source_type: str,
    target_type: str,
    k: int,
    n_permutations: int,
    rng: np.random.Generator,
    tissue_mask_col: Optional[str],
    alternative: str,
) -> dict:
    x = pd.to_numeric(grp.get(x_col), errors="coerce").to_numpy(float)
    y = pd.to_numeric(grp.get(y_col), errors="coerce").to_numpy(float)
    types = grp.get(type_col, pd.Series("Other", index=grp.index)).fillna("Other").astype(str).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, types = x[ok], y[ok], types[ok]

    if tissue_mask_col is not None and tissue_mask_col in grp.columns:
        eligible = grp[tissue_mask_col].fillna(False).to_numpy(bool)[ok]
    else:
        eligible = np.ones(len(types), dtype=bool)

    n = len(x)
    result = {
        "n_cells": n,
        "n_source": int((types == source_type).sum()),
        "n_target": int((types == target_type).sum()),
        "n_permutations": n_permutations,
        "observed_delta": np.nan,
        "null_mean": np.nan,
        "null_std": np.nan,
        "p_value": np.nan,
    }
    if n < 3 or result["n_source"] == 0 or result["n_target"] == 0:
        return result

    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    kmax = min(k, n - 1)
    _, indices = tree.query(pts, k=kmax + 1)

    sm = types == source_type
    tm = types == target_type
    observed = _knn_delta(tm, sm, indices, k, n)
    result["observed_delta"] = observed

    if not np.isfinite(observed):
        return result

    null_deltas = np.empty(n_permutations)
    permutable_idx = np.flatnonzero(eligible)
    fixed_labels = types.copy()
    for i in range(n_permutations):
        shuffled = fixed_labels.copy()
        shuffled[permutable_idx] = rng.permutation(fixed_labels[permutable_idx])
        sm_p = shuffled == source_type
        tm_p = shuffled == target_type
        null_deltas[i] = _knn_delta(tm_p, sm_p, indices, k, n)

    null_deltas = null_deltas[np.isfinite(null_deltas)]
    result["null_mean"] = float(np.mean(null_deltas)) if len(null_deltas) else np.nan
    result["null_std"] = float(np.std(null_deltas)) if len(null_deltas) else np.nan
    result["_null_deltas"] = null_deltas

    if len(null_deltas) == 0:
        return result
    if alternative == "greater":
        p = (1 + np.sum(null_deltas >= observed)) / (len(null_deltas) + 1)
    elif alternative == "less":
        p = (1 + np.sum(null_deltas <= observed)) / (len(null_deltas) + 1)
    else:
        p = (1 + np.sum(np.abs(null_deltas) >= abs(observed))) / (len(null_deltas) + 1)
    result["p_value"] = float(p)
    return result


def step11_spatial_null(
    df: pd.DataFrame,
    source_type: str,
    target_type: str,
    core_col: str = "slide_scene",
    x_col: str = "DAPI_X",
    y_col: str = "DAPI_Y",
    type_col: str = "cell_type",
    k: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
    tissue_mask_col: Optional[str] = None,
    alternative: str = "two-sided",
    output_dir: Optional[str] = None,
    max_histograms: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Label-swap spatial null for one ``source_type -> target_type`` neighbor-enrichment pair.

    Tests the same statistic as ``step10_spatial_features``'s
    ``spatial.<source_type>.to.<target_type>.knn<k>.delta`` column: the mean
    fraction of a source-type cell's k nearest neighbors that are
    target-type, minus the fraction expected under random label assignment.
    Cell coordinates (the observed tissue geometry) are held fixed; cell-type
    labels are permuted ``n_permutations`` times among eligible cells to
    build an empirical null distribution, per core.

    Args:
        source_type / target_type: the two cell types being tested for
            spatial association.
        k: neighborhood size (must match the ``step10_spatial_features`` knn
            value you are validating).
        tissue_mask_col: optional boolean per-cell column. When given, label
            permutation is restricted to cells within the same tissue-mask
            region rather than the whole core ("label swapping within tissue
            boundaries" per DESIGN.md, rather than across the full field).
        alternative: ``"two-sided"`` (default), ``"greater"`` (test for
            enrichment only), or ``"less"`` (test for depletion only).

    Returns:
        One row per core: ``observed_delta``, ``null_mean``, ``null_std``,
        ``p_value``, ``n_permutations``, and cell counts. A core with fewer
        than 3 source or target cells returns NaNs rather than a fabricated
        p-value.

    This validates one pair/k at a time by design — running it over every
    pair in a large ``spatial_pairs`` grid means every resulting p-value is a
    multiple-comparison and must be BH-corrected before any claim, exactly
    like ``step12_compare_groups``.

    Args:
        output_dir: if given, writes ``step11_null_<source>_to_<target>_<core>.png``
            for up to ``max_histograms`` cores — the null distribution
            histogram with the observed statistic marked as a vertical line,
            so you can see *why* a p-value came out the way it did rather
            than trusting a single number.
        max_histograms: cap on how many per-core null histograms to render.
    """
    if core_col not in df.columns:
        raise ValueError(f"core_col {core_col!r} not found in df.columns")

    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(output_dir) if output_dir is not None else None
    plotted = 0
    rows = []
    for core, grp in df.groupby(core_col):
        result = _core_spatial_null(
            grp, x_col, y_col, type_col, source_type, target_type, k,
            n_permutations, rng, tissue_mask_col, alternative,
        )
        null_deltas = result.pop("_null_deltas", None)
        result[core_col] = core
        rows.append(result)

        if out_dir is not None and null_deltas is not None and len(null_deltas) and plotted < max_histograms:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(null_deltas, bins=40, color="lightgray", label=f"null (n={len(null_deltas)} permutations)")
            ax.axvline(result["observed_delta"], color="red", linewidth=2,
                       label=f"observed = {result['observed_delta']:.4f}")
            ax.set_xlabel(f"{source_type} -> {target_type} knn{k} enrichment delta")
            ax.set_title(f"step11: spatial null — {core} (p={result['p_value']:.4f})")
            ax.legend(fontsize=7)
            fig.tight_layout()
            safe_core = str(core).replace("/", "_")
            savefig(fig, out_dir / f"step11_null_{source_type}_to_{target_type}_{safe_core}.png", verbose=verbose)
            plotted += 1

    out = pd.DataFrame(rows)
    cols = [core_col] + [c for c in out.columns if c != core_col]
    result_df = out[cols].sort_values(core_col).reset_index(drop=True)

    if out_dir is not None:
        csv_path = out_dir / f"step11_spatial_null_{source_type}_to_{target_type}.csv"
        result_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"  Spatial null results -> {csv_path}")

    return result_df
