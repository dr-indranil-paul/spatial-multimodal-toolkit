"""step09_aggregate_per_core — cell-level table -> one row per biological/imaging unit.

Ported and generalized from
02_spatial_proteomics/03_analysis/02_scripts/tma_cell_classify.py::aggregate_per_core.

This produces *mean*-based per-core summary metrics (cell-type fractions,
marker intensity means, program-score means) for exploratory QC and group
comparison (``step12_compare_groups``). It is a different aggregation from
``spatial_omics_toolkit.domain.pseudobulk.preaggregate_pseudobulk``, which
*sums* raw count/expression matrices per biological unit as the mandatory
pre-aggregation step before containerized R differential testing (DESIGN.md
non-negotiable #4). Use this function for per-core summary statistics; use
``preaggregate_pseudobulk`` when the destination is a differential-expression
container.

A "core" here is an imaging field, not automatically a biological replicate.
If multiple cores come from the same patient/specimen, aggregate again (or
pass a patient-level ``core_col``) before treating rows of the output as
independent units in any statistical test.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig

PathLike = Union[str, Path]


def step09_aggregate_per_core(
    df: pd.DataFrame,
    core_col: str = "slide_scene",
    type_col: str = "cell_type",
    type_categories: Optional[Sequence[str]] = None,
    marker_cols: Optional[Sequence[str]] = None,
    program_score_cols: Optional[Sequence[str]] = None,
    extra_group_col: Optional[str] = None,
    epithelial_flag_col: str = "is_epithelial",
    hybrid_flag_col: str = "is_hybrid",
    immune_flag_col: str = "is_immune",
    myeloid_flag_col: str = "is_myeloid",
    proliferating_flag_col: str = "is_proliferating",
    output_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate a per-cell table to one summary row per ``core_col`` value.

    Args:
        df: per-cell table, typically the output of ``step08_score_programs``.
        core_col: column identifying the imaging/biological unit to aggregate to.
        type_col: categorical cell-type column (e.g. from ``step07_classify_cells``).
        type_categories: cell-type fraction columns to emit; defaults to the
            categories observed in ``df[type_col]``.
        marker_cols: marker intensity columns to mean-aggregate; defaults to
            all numeric columns not already covered by a flag/score/id column.
        program_score_cols: program score mean columns to emit (e.g.
            ``["EL_score", "IH_score", "ME_score"]``); auto-detected from any
            column ending in ``_score`` if not given.
        extra_group_col: optional additional identifier column to carry
            through unchanged (one value per core; e.g. a TMA/batch id).
        epithelial_flag_col / hybrid_flag_col / immune_flag_col /
            myeloid_flag_col / proliferating_flag_col: boolean per-cell flag
            columns used to compute composite fractions; skipped if absent.

        output_dir: if given, writes ``step09_core_table.csv`` (the aggregated
            table itself — the primary artifact to inspect row by row) and
            ``step09_core_composition_heatmap.png`` (core x cell-type
            fraction heatmap, so an outlier core jumps out visually).

    Returns:
        One row per core, sorted by ``core_col``.
    """
    if core_col not in df.columns:
        raise ValueError(f"core_col {core_col!r} not found in df.columns")
    if type_col not in df.columns:
        raise ValueError(f"type_col {type_col!r} not found in df.columns")

    if type_categories is None:
        type_categories = sorted(df[type_col].dropna().unique().tolist())
    if program_score_cols is None:
        program_score_cols = [c for c in df.columns if c.endswith("_score") and c != "hybrid_balance"]
    if marker_cols is None:
        reserved = {core_col, type_col, extra_group_col} | set(program_score_cols)
        reserved |= {c for c in df.columns if c.startswith("is_") or c.endswith("_z")
                     or c in ("dominant_program", "hybrid_balance", "program_entropy")}
        marker_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in reserved]

    rows = []
    for core, grp in df.groupby(core_col):
        n = len(grp)
        counts = grp[type_col].value_counts()
        row = {core_col: core, "total_cells": n}
        if extra_group_col is not None and extra_group_col in grp.columns:
            row[extra_group_col] = grp[extra_group_col].iloc[0]
        for cat in type_categories:
            row[cat] = counts.get(cat, 0) / n if n else np.nan
        for col in marker_cols:
            if col in grp.columns:
                row[col] = grp[col].mean()
        for col in program_score_cols:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
        if "dominant_program" in grp.columns:
            for name in grp["dominant_program"].dropna().unique():
                row[f"dominant_{name}_frac"] = (grp["dominant_program"] == name).mean()
        if "hybrid_balance" in grp.columns:
            row["hybrid_balance_mean"] = grp["hybrid_balance"].mean()
        if "program_entropy" in grp.columns:
            row["program_entropy_mean"] = grp["program_entropy"].mean()
        if immune_flag_col in grp.columns:
            row["immune_fraction"] = grp[immune_flag_col].mean()
        if myeloid_flag_col in grp.columns:
            row["myeloid_fraction"] = grp[myeloid_flag_col].mean()
        if proliferating_flag_col in grp.columns:
            row["proliferating_fraction"] = grp[proliferating_flag_col].mean()
        if hybrid_flag_col in grp.columns:
            row["hybrid_fraction"] = grp[hybrid_flag_col].mean()
            if epithelial_flag_col in grp.columns:
                epi = grp[grp[epithelial_flag_col]]
                row["epithelial_hybrid_fraction"] = epi[hybrid_flag_col].mean() if len(epi) else 0.0
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(core_col).reset_index(drop=True)

    if output_dir is not None:
        import matplotlib.pyplot as plt

        out_dir = ensure_dir(output_dir)
        csv_path = out_dir / "step09_core_table.csv"
        result.to_csv(csv_path, index=False)
        if verbose:
            print(f"  Core table -> {csv_path}")

        present_types = [t for t in type_categories if t in result.columns]
        if present_types:
            heat = result.set_index(core_col)[present_types]
            fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(present_types) + 2), max(4, 0.35 * len(heat))))
            im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(present_types)))
            ax.set_xticklabels(present_types, rotation=90, fontsize=7)
            ax.set_yticks(range(len(heat)))
            ax.set_yticklabels(heat.index, fontsize=7)
            ax.set_title("step09: cell-type fraction per core")
            fig.colorbar(im, ax=ax, label="fraction")
            fig.tight_layout()
            savefig(fig, out_dir / "step09_core_composition_heatmap.png", verbose=verbose)

    return result
