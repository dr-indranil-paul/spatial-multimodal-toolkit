"""step10_spatial_features — per-core spatial neighborhood and density features.

Ported and generalized from
02_spatial_proteomics/03_analysis/02_scripts/run_exhaustive_spatial_ml.py
(``spatial_core_features``, ``add_spatial_grid_features``). Column names and
cell-type pairs are now parameters instead of module-level constants tied to
one manuscript's panel.

Any neighbor-enrichment feature produced here (the ``spatial.<source>.to.<target>``
columns) is a *descriptive* statistic. Before treating one of these as evidence
of spatial niche organization rather than incidental composition, run
``step11_spatial_null`` on the same source/target/k and report the permutation
p-value alongside it (DESIGN.md non-negotiable #5).
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig

PathLike = Union[str, Path]


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _safe_mean(values) -> float:
    x = np.asarray(
        pd.to_numeric(np.asarray(values).reshape(-1), errors="coerce"), dtype=float
    )
    return _finite(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _cv(values) -> float:
    x = np.asarray(
        pd.to_numeric(np.asarray(values).reshape(-1), errors="coerce"), dtype=float
    )
    x = x[np.isfinite(x)]
    return _finite(np.std(x) / np.mean(x)) if len(x) and np.mean(x) != 0 else np.nan


def _gini(values) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x) or np.sum(np.maximum(x, 0)) <= 0:
        return np.nan
    x = np.sort(np.maximum(x, 0))
    return _finite(
        (2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum())
    )


def _entropy(probs) -> float:
    p = np.asarray(list(probs), dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    return _finite(-np.sum(p * np.log(p))) if len(p) else np.nan


def _grid_features(
    out: Dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
    types: np.ndarray,
    grid: int,
    broad_types: Sequence[str],
) -> None:
    """ROI-grid composition heterogeneity for one grid resolution."""
    if len(x) < 3:
        return
    xlo, xhi = np.nanpercentile(x, [1, 99])
    ylo, yhi = np.nanpercentile(y, [1, 99])
    width, height = max(xhi - xlo, 1.0), max(yhi - ylo, 1.0)
    xb = np.clip(((x - xlo) / width * grid).astype(int), 0, grid - 1)
    yb = np.clip(((y - ylo) / height * grid).astype(int), 0, grid - 1)
    bid = yb * grid + xb
    counts = np.bincount(bid, minlength=grid * grid).astype(float)
    dens = counts / (width * height / (grid * grid))
    prefix = f"roi{grid}"
    out[f"{prefix}.density_mean"] = _finite(np.mean(dens))
    out[f"{prefix}.density_cv"] = _cv(dens)
    out[f"{prefix}.density_gini"] = _gini(dens)
    out[f"{prefix}.empty_fraction"] = _finite(np.mean(counts == 0))
    out[f"{prefix}.density_max_fraction"] = _finite(
        np.max(counts) / max(np.sum(counts), 1.0)
    )
    rows, weights = [], []
    for b in range(grid * grid):
        m = bid == b
        if m.any():
            rows.append([np.mean(types[m] == t) for t in broad_types])
            weights.append(m.sum())
    if rows:
        comp = np.asarray(rows, float)
        w = np.asarray(weights, float)
        glob = np.asarray([np.mean(types == t) for t in broad_types])
        out[f"{prefix}.composition_heterogeneity"] = _finite(
            np.average(np.sum((comp - glob) ** 2, axis=1), weights=w)
        )
        out[f"{prefix}.composition_entropy_mean"] = _finite(
            np.average([_entropy(r) for r in comp], weights=w)
        )
        for j, t in enumerate(broad_types):
            out[f"{prefix}.{t}.bin_fraction_cv"] = _cv(comp[:, j])
            out[f"{prefix}.{t}.max_bin_fraction"] = _finite(np.max(comp[:, j]))


def _core_features(
    grp: pd.DataFrame,
    x_col: str,
    y_col: str,
    type_col: str,
    broad_types: Sequence[str],
    spatial_pairs: Sequence[Tuple[str, str]],
    program_cols: Sequence[str],
    morphology_cols: Sequence[str],
    grid_sizes: Sequence[int],
    knn_k_values: Sequence[int],
    radii: Sequence[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {"cell.n_cells": float(len(grp))}
    x = pd.to_numeric(grp.get(x_col), errors="coerce").to_numpy(float)
    y = pd.to_numeric(grp.get(y_col), errors="coerce").to_numpy(float)
    types = (
        grp.get(type_col, pd.Series("Other", index=grp.index))
        .fillna("Other")
        .astype(str)
        .to_numpy()
    )
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, types = x[ok], y[ok], types[ok]
    n = len(x)
    out["cell.n_cells_with_coordinates"] = float(n)

    for t in broad_types:
        mask = types == t
        out[f"state.count.{t}"] = float(mask.sum())
        out[f"state.fraction.{t}"] = _finite(mask.mean()) if n else np.nan

    for col in program_cols:
        if col in grp:
            out[f"program.cell.{col}.mean"] = _safe_mean(grp[col])
            out[f"program.cell.{col}.cv"] = _cv(grp[col])

    for col in morphology_cols:
        if col in grp:
            out[f"morphology.{col}.median"] = _finite(
                np.nanmedian(pd.to_numeric(grp[col], errors="coerce"))
            )
            out[f"morphology.{col}.cv"] = _cv(grp[col])

    if n < 3:
        return out

    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    kmax = min(max(knn_k_values, default=10), n - 1)
    distances, indices = tree.query(pts, k=kmax + 1)
    nn = distances[:, 1]
    out["density.nn_distance.mean"] = _safe_mean(nn)
    out["density.nn_distance.median"] = _finite(np.nanmedian(nn))

    for k in sorted(set(knn_k_values)):
        if k >= n:
            continue
        kth = distances[:, k]
        local = k / (np.pi * np.maximum(kth, 1e-9) ** 2)
        out[f"density.knn{k}.distance.mean"] = _safe_mean(kth)
        out[f"density.knn{k}.local_density.mean"] = _safe_mean(local)

    xlo, xhi = np.nanpercentile(x, [1, 99])
    ylo, yhi = np.nanpercentile(y, [1, 99])
    width, height = max(xhi - xlo, 1.0), max(yhi - ylo, 1.0)
    out["density.bbox"] = float(n / (width * height))

    for grid in grid_sizes:
        _grid_features(out, x, y, types, grid, broad_types)

    for source, target in spatial_pairs:
        sm = types == source
        tm = types == target
        key = f"spatial.{source}.to.{target}"
        if not sm.any() or not tm.any():
            for k in knn_k_values:
                out[f"{key}.knn{k}.mean"] = np.nan
                out[f"{key}.knn{k}.delta"] = np.nan
            for r in radii:
                out[f"{key}.radius{r:g}.mean"] = np.nan
            continue
        src = np.flatnonzero(sm)
        target_pts = pts[tm]
        target_tree = cKDTree(target_pts)
        for k in knn_k_values:
            kk = min(k, n - 1)
            neigh = tm[indices[src, 1 : kk + 1]]
            observed = neigh.mean(axis=1)
            expected = (tm.sum() - tm[src].astype(float)) / max(n - 1, 1)
            out[f"{key}.knn{k}.mean"] = _safe_mean(observed)
            out[f"{key}.knn{k}.delta"] = _safe_mean(observed - expected)
        for r in radii:
            counts = np.asarray(
                target_tree.query_ball_point(pts[sm], r=r, return_length=True), float
            )
            out[f"{key}.radius{r:g}.mean"] = _safe_mean(counts)
            out[f"{key}.radius{r:g}.delta"] = _safe_mean(
                counts / max(np.sum(tm), 1) - (np.pi * r * r) / (width * height)
            )

    return out


def step10_spatial_features(
    df: pd.DataFrame,
    core_col: str = "slide_scene",
    x_col: str = "DAPI_X",
    y_col: str = "DAPI_Y",
    type_col: str = "cell_type",
    broad_types: Optional[Sequence[str]] = None,
    spatial_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    program_cols: Sequence[str] = (
        "EL_score",
        "IH_score",
        "ME_score",
        "hybrid_balance",
        "program_entropy",
    ),
    morphology_cols: Sequence[str] = ("nuclei_area", "cell_area"),
    grid_sizes: Sequence[int] = (4, 8),
    knn_k_values: Sequence[int] = (5, 10),
    radii: Sequence[float] = (50.0, 100.0),
    output_dir: Optional[PathLike] = None,
    max_scatter_cores: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per core of density, neighborhood, and ROI-grid heterogeneity features.

    Args:
        df: per-cell table with coordinate and cell-type columns (typically
            the output of ``step07_classify_cells`` / ``step08_score_programs``).
        core_col: imaging/biological unit column to group by.
        x_col / y_col: cell centroid coordinate columns.
        type_col: categorical cell-type column.
        broad_types: cell types to report per-core count/fraction for;
            defaults to the categories observed in ``df[type_col]``.
        spatial_pairs: ``(source_type, target_type)`` pairs to compute
            neighbor-enrichment features for; defaults to every ordered pair
            of ``broad_types``. Each pair produces a
            ``spatial.<source>.to.<target>.knn<k>.delta`` feature: the mean
            fraction of a source cell's k nearest neighbors that are the
            target type, minus the fraction expected under random mixing.
            **This is a descriptive statistic, not a significance claim** —
            pair it with ``step11_spatial_null`` before making an enrichment
            claim.
        program_cols / morphology_cols: per-cell numeric columns to
            mean/CV-summarize per core, if present.
        grid_sizes: ROI grid resolutions (N x N bins) for composition
            heterogeneity features.
        knn_k_values / radii: neighborhood sizes for density and
            neighbor-enrichment features.
        output_dir: if given, writes one ``step10_spatial_map_<core>.png`` per
            core (up to ``max_scatter_cores``) — the actual tissue map, cells
            plotted at their real coordinates colored by type. This is the
            single most direct check that the spatial data loaded correctly
            (tissue shape, cell density, obvious artifacts) before trusting
            any downstream neighbor-enrichment number.
        max_scatter_cores: cap on how many per-core spatial maps to render,
            so calling this on a large cohort doesn't silently write hundreds
            of large PNGs.
    """
    if core_col not in df.columns:
        raise ValueError(f"core_col {core_col!r} not found in df.columns")

    if broad_types is None:
        broad_types = (
            sorted(df[type_col].dropna().unique().tolist())
            if type_col in df.columns
            else []
        )
    if spatial_pairs is None:
        spatial_pairs = [(a, b) for a in broad_types for b in broad_types if a != b]

    rows = []
    out_dir = ensure_dir(output_dir) if output_dir is not None else None
    plotted = 0
    for core, grp in df.groupby(core_col):
        feats = _core_features(
            grp,
            x_col,
            y_col,
            type_col,
            broad_types,
            spatial_pairs,
            program_cols,
            morphology_cols,
            grid_sizes,
            knn_k_values,
            radii,
        )
        feats[core_col] = core
        rows.append(feats)

        if out_dir is not None and plotted < max_scatter_cores:
            import matplotlib.pyplot as plt

            x = pd.to_numeric(grp.get(x_col), errors="coerce")
            y = pd.to_numeric(grp.get(y_col), errors="coerce")
            types = (
                grp.get(type_col, pd.Series("Other", index=grp.index))
                .fillna("Other")
                .astype(str)
            )
            fig, ax = plt.subplots(figsize=(6, 6))
            for t in sorted(types.unique()):
                m = types == t
                ax.scatter(x[m], y[m], s=2, alpha=0.5, label=t)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(f"step10: spatial map — {core}")
            ax.legend(markerscale=4, fontsize=6, loc="upper right", ncol=2)
            fig.tight_layout()
            savefig(fig, out_dir / f"step10_spatial_map_{core}.png", verbose=verbose)
            plotted += 1

    out = pd.DataFrame(rows)
    cols = [core_col] + [c for c in out.columns if c != core_col]
    result = out[cols].sort_values(core_col).reset_index(drop=True)

    if out_dir is not None:
        csv_path = out_dir / "step10_spatial_features.csv"
        result.to_csv(csv_path, index=False)
        if verbose:
            print(f"  Spatial feature table -> {csv_path}")

    return result
