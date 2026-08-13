"""step13_pseudobulk — sum-aggregate a per-cell count/expression matrix to sample level.

Thin wrapper around ``spatial_omics_toolkit.domain.pseudobulk.preaggregate_pseudobulk``.
Distinct from ``step09_aggregate_per_core`` (mean-based per-core summary
metrics for exploratory QC/comparison): this is the *sum*-based
pre-aggregation DESIGN.md mandates before handing any matrix to a
containerized R differential-expression stage (edgeR/limma-voom/muscat).
"""

from typing import Dict

import pandas as pd

from spatial_omics_toolkit.domain.pseudobulk import preaggregate_pseudobulk


def step13_pseudobulk(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sample_col: str = "biological_unit_id",
    group_col: str = "cell_type",
) -> Dict[str, pd.DataFrame]:
    """Sum-aggregate cells/spots to sample-level pseudobulk counts, per cell-type group.

    Args:
        expression_df: cells x features raw count/intensity matrix.
        metadata_df: per-cell metadata containing ``sample_col`` (the
            independent biological unit — a patient/specimen id, not a core
            id, unless one core truly is one biological unit in your design)
            and ``group_col``.
        sample_col: column identifying the independent biological unit.
        group_col: column identifying the cell-type/niche group to
            aggregate within.

    Returns:
        ``{group_name: samples x features summed count matrix}``.
    """
    return preaggregate_pseudobulk(expression_df, metadata_df, sample_col=sample_col, group_col=group_col)
