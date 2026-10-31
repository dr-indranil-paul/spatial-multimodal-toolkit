"""Tissue-mask-aware spatial analysis, Gini spatial nulls, and neighborhood graphs."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_tissue_masked_gini(
    densities: np.ndarray, tissue_mask: Optional[np.ndarray] = None
) -> float:
    """Compute Gini inequality coefficient over valid tissue-masked spatial grids.

    Prevents core-geometry artifacts by restricting evaluation to valid tissue mask regions.

    Args:
        densities: Array of cell densities or intensity values
        tissue_mask: Optional boolean array matching densities where True = valid tissue

    Returns:
        Float Gini coefficient in range [0, 1]
    """
    if tissue_mask is not None:
        values = densities[tissue_mask]
    else:
        values = densities

    values = np.asarray(values, dtype=np.float64)
    values = values[values >= 0]

    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def compute_spatial_neighborhood_graph(
    coordinates_df: pd.DataFrame,
    radius_pixels: float = 50.0,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
) -> Dict[int, List[int]]:
    """Construct k-d tree spatial radius neighborhood graph for cells.

    Args:
        coordinates_df: Cells DataFrame containing X and Y centroid coordinates
        radius_pixels: Distance threshold in pixels for neighbor connectivity

    Returns:
        Dict mapping cell_id -> list of neighbor cell_ids within radius
    """
    from scipy.spatial import KDTree

    coords = coordinates_df[[x_col, y_col]].values
    cell_ids = list(coordinates_df.index)

    tree = KDTree(coords)
    pairs = tree.query_pairs(r=radius_pixels)

    graph: Dict[int, List[int]] = {cid: [] for cid in cell_ids}
    for i, j in pairs:
        cid_i = cell_ids[i]
        cid_j = cell_ids[j]
        graph[cid_i].append(cid_j)
        graph[cid_j].append(cid_i)

    return graph
