"""Unit tests for spatial analysis engines (CyCIF QC, Segmentation, Spatial ML)."""

import numpy as np
import pandas as pd
from spatial_omics_toolkit.analysis.cycif_qc import (
    calculate_marker_snr,
    filter_background_signal,
)
from spatial_omics_toolkit.analysis.segmentation import compute_cell_centroids
from spatial_omics_toolkit.analysis.spatial_ml import (
    compute_tissue_masked_gini,
    compute_spatial_neighborhood_graph,
)


def test_cycif_marker_snr_and_filter():
    """Test CyCIF marker signal-to-noise calculation and background subtraction."""
    df = pd.DataFrame(
        {
            "CK8_18": [10.0, 12.0, 100.0, 120.0, 150.0],
            "CD45": [1.0, 2.0, 2.0, 50.0, 60.0],
        }
    )

    snr = calculate_marker_snr(df)
    assert "CK8_18" in snr
    assert "CD45" in snr
    assert snr["CK8_18"] > 1.0

    filtered = filter_background_signal(df, {"CK8_18": 10.0, "CD45": 2.0})
    assert filtered["CK8_18"].min() == 0.0
    assert filtered["CD45"].min() == 0.0


def test_compute_cell_centroids():
    """Test cell centroid extraction from 2D segmentation masks."""
    mask = np.zeros((10, 10), dtype=int)
    mask[2:5, 2:5] = 1  # Cell 1
    mask[7:9, 7:9] = 2  # Cell 2

    centroids = compute_cell_centroids(mask)
    assert len(centroids) == 2
    assert 1 in centroids.index
    assert 2 in centroids.index
    assert centroids.loc[1, "area_pixels"] == 9
    assert centroids.loc[2, "area_pixels"] == 4


def test_compute_tissue_masked_gini():
    """Test tissue-masked Gini coefficient computation."""
    densities = np.array([10.0, 10.0, 10.0, 100.0, 0.0])
    mask = np.array([True, True, True, True, False])  # Exclude background zero

    gini = compute_tissue_masked_gini(densities, mask)
    assert 0.0 <= gini <= 1.0


def test_compute_spatial_neighborhood_graph():
    """Test KD-Tree spatial neighborhood graph construction."""
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0, 100.0],
            "centroid_y": [0.0, 10.0, 100.0],
        },
        index=[1, 2, 3],
    )

    graph = compute_spatial_neighborhood_graph(df, radius_pixels=20.0)
    assert 2 in graph[1]  # Cell 1 connects to Cell 2
    assert 1 in graph[2]  # Cell 2 connects to Cell 1
    assert 3 not in graph[1]  # Cell 3 is > 20px away
