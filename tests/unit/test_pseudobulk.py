"""Unit tests for pseudobulk pre-aggregation logic."""

import pandas as pd
import numpy as np
from spatial_omics_toolkit.domain.pseudobulk import preaggregate_pseudobulk


def test_pseudobulk_aggregation():
    """Verify that cell-level count matrices are correctly summed per biological unit."""
    # 4 cells, 2 genes
    expression_df = pd.DataFrame(
        {
            "GeneA": [1, 2, 10, 20],
            "GeneB": [5, 5, 50, 50],
        },
        index=["Cell1", "Cell2", "Cell3", "Cell4"],
    )

    metadata_df = pd.DataFrame(
        {
            "biological_unit_id": ["Patient_1", "Patient_1", "Patient_2", "Patient_2"],
            "cell_type": ["T_cell", "T_cell", "T_cell", "T_cell"],
        },
        index=["Cell1", "Cell2", "Cell3", "Cell4"],
    )

    res = preaggregate_pseudobulk(
        expression_df,
        metadata_df,
        sample_col="biological_unit_id",
        group_col="cell_type",
    )
    assert "T_cell" in res
    t_cell_matrix = res["T_cell"]

    # Check summed counts
    assert t_cell_matrix.loc["Patient_1", "GeneA"] == 3
    assert t_cell_matrix.loc["Patient_1", "GeneB"] == 10
    assert t_cell_matrix.loc["Patient_2", "GeneA"] == 30
    assert t_cell_matrix.loc["Patient_2", "GeneB"] == 100
