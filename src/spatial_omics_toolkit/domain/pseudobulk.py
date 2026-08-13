"""Domain logic for Python-side pseudobulk matrix pre-aggregation."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def preaggregate_pseudobulk(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sample_col: str = "biological_unit_id",
    group_col: str = "cell_type"
) -> Dict[str, pd.DataFrame]:
    """Pre-aggregate single-cell / spatial spot count matrices to sample-level pseudobulk counts.
    
    Args:
        expression_df: Cells/Spots x Features raw count DataFrame
        metadata_df: Cells/Spots metadata containing sample_col and group_col
        sample_col: Column defining independent biological unit ID
        group_col: Column defining cell type or spatial niche grouping
        
    Returns:
        Dict mapping group_name -> DataFrame (Samples x Features count matrix)
    """
    if not metadata_df.index.equals(expression_df.index):
        raise ValueError("Metadata index does not match expression matrix index.")

    merged_df = expression_df.copy()
    merged_df[sample_col] = metadata_df[sample_col]
    merged_df[group_col] = metadata_df[group_col]

    pseudobulk_matrices: Dict[str, pd.DataFrame] = {}
    groups = merged_df[group_col].unique()

    for grp in groups:
        subset = merged_df[merged_df[group_col] == grp]
        # Sum counts across cells/spots per biological unit (sample)
        aggregated = subset.groupby(sample_col).sum(numeric_only=True)
        pseudobulk_matrices[str(grp)] = aggregated

    return pseudobulk_matrices
