"""step07_classify_cells / step08_score_programs — per-cell marker-threshold typing.

Ported and generalized from
02_spatial_proteomics/03_analysis/02_scripts/tma_cell_classify.py::classify_all
and ::compute_emax_programs (validated on the PREVAIL hBE-TMA CyCIF cohort).
The gating algorithm (per-core z-score normalization, dual-gate epithelial vs.
mesenchymal assignment, mutually-exclusive priority typing) is preserved
exactly; marker column names are now supplied via
:class:`~spatial_omics_toolkit.cycif.config.CellTypeMarkerConfig` instead of
being hardcoded to one panel.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy as _scipy_entropy

from spatial_omics_toolkit.cycif.config import CellTypeMarkerConfig, ProgramDefinition
from spatial_omics_toolkit.cycif.qc_plots import ensure_dir, savefig

PathLike = Union[str, Path]


def zscore_per_core(
    df: pd.DataFrame,
    columns: Sequence[str],
    core_col: str = "slide_scene",
) -> pd.DataFrame:
    """Add ``<column>_z`` columns: each cell's value z-scored against its own core.

    A core with zero variance for a marker gets z = 0.0 for every cell in that
    core (matches the original script's behavior; avoids inventing signal from
    noise-free channels).
    """
    out = df.copy()
    for col in columns:
        grouped = out.groupby(core_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        z = (out[col] - mean) / std
        out[f"{col}_z"] = z.where(std > 0, 0.0).fillna(0.0)
    return out


def step07_classify_cells(
    df: pd.DataFrame,
    config: CellTypeMarkerConfig,
    core_col: str = "slide_scene",
    verbose: bool = True,
    output_dir: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Classify every cell into a mutually-exclusive type using per-core z-score gating.

    Algorithm (unchanged from the validated PREVAIL hBE-TMA pipeline):

    1. Z-score every marker in ``config.z_score_markers`` within each core.
    2. Epithelial gate: max z-score across ``config.epithelial_markers`` > z_threshold.
       Non-epithelial gate: that same max z-score < ``config.non_epithelial_gate``.
    3. Mesenchymal signal: any of ``config.mesenchymal_markers`` > z_threshold.
       Strong mesenchymal: **all** of ``config.mesenchymal_markers`` > z_threshold.
    4. Priority-ordered mutually-exclusive type assignment: Hybrid_epithelial >
       Epithelial > Mesenchymal_like > Myeloid_immune > Lymphoid_immune >
       Endothelial > Stromal > Proliferating > Other.

    Returns a copy of ``df`` with added boolean gate columns, ``cell_type``, and
    the per-core z-score columns.

    n.b. This assigns a type per *cell*. Cells are never independent
    biological replicates — aggregate to one row per biological unit (see
    ``step09_aggregate_per_core``) before any group-level statistical claim.

    Args:
        output_dir: if given, writes two check-before-proceeding PNGs:
            ``step07_cell_type_composition.png`` (stacked bar of cell-type
            fraction per core — spot an obviously broken core immediately)
            and ``step07_gate_diagnostic.png`` (epithelial vs. mesenchymal
            max-marker-z scatter, colored by assigned type — the direct
            visual check on whether the z-threshold gates look right on
            *this* dataset before trusting the labels downstream).
    """
    if core_col not in df.columns:
        raise ValueError(f"core_col {core_col!r} not found in df.columns")

    z = zscore_per_core(df, config.z_score_markers, core_col=core_col)

    def zcol(name: str) -> np.ndarray:
        col = f"{name}_z"
        return z[col].to_numpy() if col in z.columns else np.zeros(len(z))

    epi_max = np.maximum.reduce([zcol(m) for m in config.epithelial_markers])
    non_epi = epi_max < config.non_epithelial_gate

    is_epithelial = epi_max > config.z_threshold
    mes_scores = [zcol(m) for m in config.mesenchymal_markers]
    has_mes = np.logical_or.reduce([s > config.z_threshold for s in mes_scores])
    has_mes_strong = np.logical_and.reduce([s > config.z_threshold for s in mes_scores])

    is_hybrid = is_epithelial & has_mes
    is_epi_only = is_epithelial & ~has_mes
    is_meso = has_mes_strong & ~is_epithelial

    z_immune = z[f"{config.immune_marker}_z"].to_numpy()
    z_apc = z[f"{config.antigen_presentation_marker}_z"].to_numpy()
    z_myeloid_markers = [zcol(m) for m in config.myeloid_markers]
    z_stromal = z[f"{config.stromal_marker}_z"].to_numpy()
    z_endothelial = z[f"{config.endothelial_marker}_z"].to_numpy()
    z_prolif = z[f"{config.proliferation_marker}_z"].to_numpy()

    is_immune = (z_immune > config.z_high_threshold) & non_epi
    is_myeloid = (
        np.logical_or.reduce([m > config.z_threshold for m in z_myeloid_markers])
        | ((z_immune > config.z_threshold) & (z_apc > config.z_threshold))
    ) & non_epi
    is_apc = (z_apc > config.z_threshold) & (z_immune > config.z_threshold) & non_epi
    is_stromal = (z_stromal > config.z_threshold) & non_epi
    is_endothelial = (z_endothelial > config.z_threshold) & non_epi
    is_prolif = z_prolif > config.z_threshold

    n = len(df)
    cell_type = np.full(n, "Other", dtype=object)
    cell_type[is_hybrid] = "Hybrid_epithelial"
    cell_type[is_epi_only] = "Epithelial"
    cell_type[is_meso] = "Mesenchymal_like"
    imm_ne = is_immune & ~is_epithelial
    cell_type[imm_ne & is_myeloid] = "Myeloid_immune"
    cell_type[imm_ne & ~is_myeloid] = "Lymphoid_immune"
    cell_type[is_endothelial & ~is_epithelial & ~is_immune] = "Endothelial"
    cell_type[is_stromal & ~is_epithelial & ~is_immune & ~is_endothelial] = "Stromal"
    cell_type[
        is_prolif & ~is_epithelial & ~is_immune & ~is_endothelial & ~is_stromal & ~is_meso
    ] = "Proliferating"

    labels = pd.DataFrame(
        {
            "is_epithelial": is_epithelial,
            "is_hybrid": is_hybrid,
            "is_mesenchymal": is_meso,
            "is_epithelial_only": is_epi_only,
            "is_stromal": is_stromal,
            "is_endothelial": is_endothelial,
            "is_immune": is_immune,
            "is_myeloid": is_myeloid,
            "is_apc": is_apc,
            "is_proliferating": is_prolif,
            "cell_type": cell_type,
        },
        index=df.index,
    )
    out = pd.concat([z, labels], axis=1)

    if verbose:
        print(f"Classified {n:,} cells across {df[core_col].nunique()} cores "
              f"(z_threshold={config.z_threshold}, z_high_threshold={config.z_high_threshold})")
        for name, arr in [
            ("is_epithelial", is_epithelial), ("is_hybrid", is_hybrid),
            ("is_mesenchymal", is_meso), ("is_immune", is_immune),
            ("is_myeloid", is_myeloid), ("is_stromal", is_stromal),
            ("is_endothelial", is_endothelial), ("is_proliferating", is_prolif),
        ]:
            print(f"  {name:22s}: {arr.mean() * 100:5.1f}%")
        print("  cell_type distribution:")
        for u, c in pd.Series(cell_type).value_counts().items():
            print(f"    {u:20s}: {c:>7,} ({c / n * 100:.1f}%)")

    if output_dir is not None:
        import matplotlib.pyplot as plt

        out_dir = ensure_dir(output_dir)

        composition = (
            out.groupby(core_col)["cell_type"].value_counts(normalize=True).unstack(fill_value=0.0)
        )
        fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(composition)), 5))
        composition.plot(kind="bar", stacked=True, ax=ax, width=0.85)
        ax.set_ylabel("Fraction of cells")
        ax.set_title("step07: cell-type composition per core")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        fig.tight_layout()
        savefig(fig, out_dir / "step07_cell_type_composition.png", verbose=verbose)

        fig, ax = plt.subplots(figsize=(6, 6))
        for t in sorted(out["cell_type"].unique()):
            m = out["cell_type"] == t
            ax.scatter(epi_max[m.to_numpy()], np.maximum.reduce([zcol(x) for x in config.mesenchymal_markers])[m.to_numpy()],
                       s=3, alpha=0.4, label=t)
        ax.axvline(config.z_threshold, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(config.z_threshold, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("max epithelial-marker z-score")
        ax.set_ylabel("max mesenchymal-marker z-score")
        ax.set_title("step07: gate diagnostic (does the z-threshold split look right?)")
        ax.legend(markerscale=3, fontsize=7, loc="upper right")
        fig.tight_layout()
        savefig(fig, out_dir / "step07_gate_diagnostic.png", verbose=verbose)

    return out


def step08_score_programs(
    df: pd.DataFrame,
    programs: Sequence[ProgramDefinition],
    z_suffix: str = "_z",
    verbose: bool = True,
    output_dir: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Compute per-cell composite program scores and derived balance/entropy metrics.

    For each :class:`ProgramDefinition`, the program score is the sum of its
    member markers' per-core z-scores (requires ``step07_classify_cells`` — or
    :func:`zscore_per_core` directly — to have already added ``<marker>_z``
    columns). Also computes:

    - ``dominant_program``: the program with the highest score for that cell.
    - ``hybrid_balance``: sum of all program scores / (1 + |max score - min score|),
      clipped to [-10, 10]. High when a cell scores comparably across programs
      rather than dominated by one.
    - ``program_entropy``: Shannon entropy of the absolute, L1-normalized
      program-score vector. High when program engagement is spread out.

    Returns a copy of ``df`` with one ``<program.name>_score`` column per
    program plus ``dominant_program``, ``hybrid_balance``, ``program_entropy``.

    Args:
        output_dir: if given, writes ``step08_program_scores.png``
            (per-program score histograms + dominant-program bar chart).
    """
    out = df.copy()
    prog_cols = []
    for prog in programs:
        missing = [f"{m}{z_suffix}" for m in prog.marker_columns if f"{m}{z_suffix}" not in out.columns]
        if missing:
            raise ValueError(
                f"Program {prog.name!r} references z-score columns not present in df: {missing}. "
                f"Run step07_classify_cells (or zscore_per_core) first."
            )
        col = f"{prog.name}_score"
        out[col] = sum(out[f"{m}{z_suffix}"] for m in prog.marker_columns)
        prog_cols.append(col)

    scores = out[prog_cols].to_numpy()
    prog_names = [p.name for p in programs]
    out["dominant_program"] = [prog_names[i] for i in np.argmax(scores, axis=1)]

    row_max = scores.max(axis=1)
    row_min = scores.min(axis=1)
    balance = scores.sum(axis=1) / (1 + np.abs(row_max - row_min))
    out["hybrid_balance"] = np.clip(balance, -10, 10)

    def _row_entropy(row: np.ndarray) -> float:
        v = np.abs(row)
        s = v.sum()
        return 0.0 if s == 0 else float(_scipy_entropy(v / s))

    out["program_entropy"] = np.apply_along_axis(_row_entropy, 1, scores)

    if verbose:
        for col in prog_cols:
            print(f"  {col}: {out[col].mean():.2f} +/- {out[col].std():.2f}")
        for name in prog_names:
            c = (out["dominant_program"] == name).sum()
            print(f"  Dominant {name}: {c:>6,} ({c / len(out) * 100:.1f}%)")

    if output_dir is not None:
        import matplotlib.pyplot as plt

        out_dir = ensure_dir(output_dir)
        fig, axes = plt.subplots(1, len(prog_cols) + 1, figsize=(4 * (len(prog_cols) + 1), 4))
        for ax, col in zip(axes[:-1], prog_cols):
            ax.hist(out[col].to_numpy(), bins=60, color="steelblue")
            ax.set_title(col)
            ax.set_xlabel("score")
        out["dominant_program"].value_counts().reindex(prog_names).plot(kind="bar", ax=axes[-1], color="darkorange")
        axes[-1].set_title("dominant_program")
        fig.suptitle("step08: per-cell program score distributions")
        fig.tight_layout()
        savefig(fig, out_dir / "step08_program_scores.png", verbose=verbose)

    return out
