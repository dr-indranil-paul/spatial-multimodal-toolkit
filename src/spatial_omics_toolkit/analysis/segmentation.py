"""Cell segmentation helper utilities and centroid extraction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_cell_centroids(
    segmentation_mask: np.ndarray
) -> pd.DataFrame:
    """Compute X, Y centroids and pixel area for labeled segmentation masks.
    
    Args:
        segmentation_mask: 2D integer array where 0 is background and N > 0 are cell IDs
        
    Returns:
        DataFrame with columns [cell_id, centroid_x, centroid_y, area_pixels]
    """
    if segmentation_mask.ndim != 2:
        raise ValueError("Segmentation mask must be a 2D integer numpy array.")

    cell_ids = np.unique(segmentation_mask)
    cell_ids = cell_ids[cell_ids > 0]  # Exclude background 0

    records: List[Dict[str, float]] = []
    for cid in cell_ids:
        ys, xs = np.where(segmentation_mask == cid)
        records.append({
            "cell_id": int(cid),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "area_pixels": float(len(xs))
        })

    return pd.DataFrame(records).set_index("cell_id")
