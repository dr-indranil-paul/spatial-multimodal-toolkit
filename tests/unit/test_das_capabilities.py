"""Comprehensive unit tests for all newly implemented DAS_v2 capabilities.

Covers:
- Vector loss-weight cell typing and built-in Maxey matrices.
- SVM active learning classifier and uncertainty sampling.
- Empirical Bayes ComBat batch effect correction and confounding checks.
- Spatial biomarker lag smoothing and pairwise cell contact matrices.
- Virtual therapy / cell knockout simulation.
- Spatial polygon ROI filtering and boundary distance calculation.
- Ruifrok-Johnston optical density color deconvolution.
- Flat-field polynomial shading and tile-edge gain illumination correction.
- Multi-scale pyramid H&E-to-fluorescence texture registration.
- Interactive standalone HTML spatial viewer generator.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.analysis.batch_correction import correct_batch_combat
from spatial_omics_toolkit.analysis.cell_typing import (
    MAXEY_IMMUNE_MATRIX,
    MAXEY_PRIMARY_MATRIX,
    MAXEY_TUMOR_MATRIX,
    align_features_to_weights,
    loss_weights_to_dataframe,
    predict_celltype_svm,
    scale_features,
    train_celltype_svm,
    uncertainty_sampling,
    vector_classify_cells,
)
from spatial_omics_toolkit.analysis.deconvolution import (
    STAIN_VECTORS,
    build_stain_matrix,
    deconvolve_color_stains,
)
from spatial_omics_toolkit.analysis.illumination import (
    correct_flatfield_shading,
    correct_tile_edge_gain,
)
from spatial_omics_toolkit.analysis.spatial_interactions import (
    compute_cell_interaction_matrix,
    compute_distance_to_boundary,
    compute_spatial_lag_features,
    filter_cells_by_polygon,
    simulate_cell_knockout,
)
from spatial_omics_toolkit.cycif.register import register_he_to_fluorescence
from spatial_omics_toolkit.reporting.viewer import export_interactive_html_viewer


# =====================================================================
# 1. Vector Cell Typing Tests
# =====================================================================


def test_loss_weights_to_dataframe():
    df_w = loss_weights_to_dataframe(MAXEY_PRIMARY_MATRIX)
    assert isinstance(df_w, pd.DataFrame)
    assert "1: endothelial" in df_w.columns
    assert "CD31" in df_w.index
    assert df_w.loc["CD31", "1: endothelial"] == 2.0


def test_align_features_to_weights_exact_and_prefix():
    weights = pd.DataFrame(
        {
            "Epithelial": [5.0, 0.0],
            "Immune": [0.0, 3.0],
        },
        index=["Ecad", "CD45"],
    )

    # Test exact matching
    df_exact = pd.DataFrame({"Ecad": [10.0, 2.0], "CD45": [1.0, 20.0]})
    sub_df, sub_w = align_features_to_weights(df_exact, weights, match_mode="exact")
    assert list(sub_df.columns) == ["Ecad", "CD45"]

    # Test prefix matching (e.g. Ecad_mean, CD45_nuclei)
    df_prefix = pd.DataFrame({"Ecad_mean": [10.0, 2.0], "CD45_nuclei": [1.0, 20.0]})
    sub_df2, sub_w2 = align_features_to_weights(df_prefix, weights, match_mode="prefix")
    assert list(sub_df2.columns) == ["Ecad", "CD45"]


def test_vector_classify_cells_basic_and_margin():
    df = pd.DataFrame(
        {
            "CD31_mean": [100.0, 2.0, 1.0, 5.0],
            "CD45_mean": [1.0, 80.0, 2.0, 4.0],
            "Ecad_mean": [2.0, 1.0, 90.0, 5.0],
            "aSMA_mean": [1.0, 1.0, 1.0, 5.0],
            "Vim_mean": [1.0, 2.0, 1.0, 5.0],
        }
    )

    result = vector_classify_cells(
        df,
        loss_weights=MAXEY_PRIMARY_MATRIX,
        scaling="zscore",
        confidence_threshold=0.3,
    )

    assert "vector_celltype" in result.columns
    assert "vector_confidence" in result.columns
    assert "vector_margin" in result.columns
    assert result.loc[0, "vector_celltype"] == "1: endothelial"
    assert result.loc[1, "vector_celltype"] == "2: immune"
    assert result.loc[2, "vector_celltype"] == "3: epithelial"


# =====================================================================
# 2. SVM Cell Classifier & Active Learning Tests
# =====================================================================


def test_train_and_predict_celltype_svm():
    # Synthetic annotated training set
    np.random.seed(42)
    n = 60
    df_train = pd.DataFrame(
        {
            "Ecad": np.concatenate(
                [np.random.normal(10, 1, n // 2), np.random.normal(1, 0.5, n // 2)]
            ),
            "CD45": np.concatenate(
                [np.random.normal(1, 0.5, n // 2), np.random.normal(10, 1, n // 2)]
            ),
            "cell_type": ["Epithelial"] * (n // 2) + ["Immune"] * (n // 2),
        }
    )

    svm_res = train_celltype_svm(
        df_train, target_col="cell_type", feature_cols=["Ecad", "CD45"]
    )
    assert svm_res.training_accuracy > 0.90
    assert set(svm_res.classes) == {"Epithelial", "Immune"}

    # Predict on test data
    df_test = pd.DataFrame(
        {
            "Ecad": [11.0, 1.2, 5.5],
            "CD45": [0.8, 9.5, 5.5],
        }
    )
    preds = predict_celltype_svm(svm_res, df_test, confidence_threshold=0.5)
    assert preds.loc[0, "svm_celltype"] == "Epithelial"
    assert preds.loc[1, "svm_celltype"] == "Immune"
    assert "svm_confidence" in preds.columns

    # Active learning / uncertainty sampling
    uncertain = uncertainty_sampling(svm_res, df_test, n_samples=1)
    assert len(uncertain) == 1
    assert uncertain.index[0] == 2  # ambiguous midpoint cell has highest uncertainty


# =====================================================================
# 3. ComBat Batch Effect Correction Tests
# =====================================================================


def test_correct_batch_combat_parametric():
    np.random.seed(42)
    n_per_batch = 20
    # Batch 1 has an additive shift of +5.0
    b1 = np.random.normal(loc=10.0, scale=1.0, size=(n_per_batch, 3))
    b2 = np.random.normal(loc=5.0, scale=1.0, size=(n_per_batch, 3))

    df = pd.DataFrame(np.vstack([b1, b2]), columns=["Marker1", "Marker2", "Marker3"])
    batch = ["BatchA"] * n_per_batch + ["BatchB"] * n_per_batch

    # Before correction, batch means differ by ~5.0
    assert (
        abs(
            df.iloc[:n_per_batch]["Marker1"].mean()
            - df.iloc[n_per_batch:]["Marker1"].mean()
        )
        > 4.0
    )

    corrected = correct_batch_combat(df, batch=batch, parametric=True)

    # After correction, batch means are aligned within 0.5
    mean_a = corrected.iloc[:n_per_batch]["Marker1"].mean()
    mean_b = corrected.iloc[n_per_batch:]["Marker1"].mean()
    assert abs(mean_a - mean_b) < 0.5


def test_correct_batch_combat_confounding_raises_stop():
    df = pd.DataFrame(
        {
            "Marker1": [1.0, 2.0, 3.0, 4.0],
            "Condition": ["Progressor", "Progressor", "NonProgressor", "NonProgressor"],
        }
    )
    # Batch is 100% collinear with Condition -> Non-identifiable design
    batch = ["Batch1", "Batch1", "Batch2", "Batch2"]

    with pytest.raises(ValueError, match="STOP: NON-IDENTIFIABLE DESIGN"):
        correct_batch_combat(df[["Marker1"]], batch=batch, covariates=df[["Condition"]])


# =====================================================================
# 4. Spatial Biomarker Smoothing & Interaction Matrices Tests
# =====================================================================


def test_compute_spatial_lag_features():
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0, 100.0],
            "centroid_y": [0.0, 0.0, 100.0],
            "MarkerA": [10.0, 20.0, 100.0],
        }
    )
    lag_df = compute_spatial_lag_features(
        df, feature_cols=["MarkerA"], radius=20.0, include_self=True
    )
    assert "MarkerA_neighbor_mean" in lag_df.columns
    # Cells (0,0) and (10,0) are within radius 20 -> mean is (10+20)/2 = 15.0
    assert abs(lag_df.loc[0, "MarkerA_neighbor_mean"] - 15.0) < 1e-3


def test_compute_cell_interaction_matrix():
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0, 20.0, 100.0],
            "centroid_y": [0.0, 0.0, 0.0, 100.0],
            "cell_type": ["Epithelial", "Immune", "Immune", "Stromal"],
        }
    )
    res = compute_cell_interaction_matrix(
        df, type_col="cell_type", radius=15.0, n_permutations=20
    )
    assert res.contact_counts.loc["Epithelial", "Immune"] >= 1.0
    assert res.enrichment_zscores is not None
    assert res.p_values is not None


# =====================================================================
# 5. Virtual Therapy / Cell Knockout Simulation Tests
# =====================================================================


def test_simulate_cell_knockout():
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 10.0, 20.0, 30.0],
            "centroid_y": [0.0, 0.0, 0.0, 0.0],
            "cell_type": ["Epithelial", "Immune", "Immune", "Stromal"],
        }
    )
    res = simulate_cell_knockout(df, target_types="Immune", radius=15.0)
    assert res.depleted_count == 2
    assert len(res.retained_cells) == 2
    assert "Immune" not in res.retained_cells["cell_type"].values
    assert res.disruption_score >= 0.0


# =====================================================================
# 6. Polygon ROI Filtering & Distance to Boundary Tests
# =====================================================================


def test_filter_cells_by_polygon():
    df = pd.DataFrame(
        {
            "centroid_x": [5.0, 50.0, 95.0],
            "centroid_y": [5.0, 50.0, 95.0],
        }
    )
    # Square ROI from (0,0) to (20,20)
    square_poly = [(0, 0), (20, 0), (20, 20), (0, 20)]
    inside_df = filter_cells_by_polygon(df, square_poly, inside=True)
    assert len(inside_df) == 1
    assert inside_df.index[0] == 0


def test_compute_distance_to_boundary():
    df = pd.DataFrame(
        {
            "centroid_x": [0.0, 3.0, 10.0],
            "centroid_y": [0.0, 4.0, 0.0],
        }
    )
    # Point at origin (0, 0)
    boundary = [(0.0, 0.0)]
    dist_df = compute_distance_to_boundary(df, boundary)
    assert dist_df.loc[0, "distance_to_boundary"] == 0.0
    assert dist_df.loc[1, "distance_to_boundary"] == 5.0  # sqrt(3^2 + 4^2) = 5.0


# =====================================================================
# 7. Ruifrok-Johnston Color Deconvolution Tests
# =====================================================================


def test_deconvolve_color_stains_aec_hem():
    # Synthetic RGB image with simulated Hematoxylin (blue/purple) and AEC (red/brown)
    rgb = np.ones((20, 20, 3), dtype=np.uint8) * 240
    # Simulate Hematoxylin absorbance on top half
    rgb[:10, :, 0] = 100  # reduce red
    rgb[:10, :, 1] = 100  # reduce green -> bluish
    # Simulate AEC on bottom half
    rgb[10:, :, 2] = 80  # reduce blue -> reddish

    res = deconvolve_color_stains(rgb, stains="AEC_HEM", output_dtype="float32")
    assert "Hematoxylin" in res.channels
    assert "AEC" in res.channels
    assert res.channels["Hematoxylin"].shape == (20, 20)
    assert (
        res.channels["Hematoxylin"][:10, :].mean()
        > res.channels["Hematoxylin"][10:, :].mean()
    )


# =====================================================================
# 8. Illumination & Flat-field Shading Correction Tests
# =====================================================================


def test_correct_flatfield_shading_and_edge_gain():
    # Parabolic vignette pattern
    y, x = np.mgrid[-10:10, -10:10]
    shading_groundtruth = 1.0 - 0.002 * (x**2 + y**2)
    clean_signal = np.ones((20, 20), dtype=float) * 1000.0
    vignetted = (clean_signal * shading_groundtruth).astype(np.uint16)

    corrected, surface = correct_flatfield_shading(vignetted, degree=2, downsample=1)
    assert corrected.shape == (20, 20)
    # Corrected image should have reduced variance across the field
    assert corrected.std() < vignetted.std()

    # Edge gain compensation
    edge_corr = correct_tile_edge_gain(
        corrected, edge_width_pixels=5, edge_gain_factor=1.2
    )
    assert edge_corr.shape == (20, 20)


# =====================================================================
# 9. H&E to Fluorescence Texture Registration Tests
# =====================================================================


def test_register_he_to_fluorescence():
    # Create a synthetic pattern of nuclei
    fl = np.zeros((64, 64), dtype=np.uint16)
    fl[20:25, 20:25] = 2000
    fl[40:45, 40:45] = 3000

    # Create H&E with dark nuclei shifted by tx=+4, ty=+3
    he = np.ones((64, 64, 3), dtype=np.uint8) * 220
    he[23:28, 24:29, :] = 40  # dark nucleus at (20+3, 20+4)
    he[43:48, 44:49, :] = 40

    res = register_he_to_fluorescence(
        he,
        fl,
        search_radius=10,
        pyramid_scales=(2, 1),
    )
    assert res["status"] == "ok"
    assert "warped_he" in res
    assert abs(res["tx"] - (-4.0)) <= 2.0
    assert abs(res["ty"] - (-3.0)) <= 2.0


# =====================================================================
# 10. Interactive HTML Spatial Viewer Tests
# =====================================================================


def test_export_interactive_html_viewer(tmp_path):
    df = pd.DataFrame(
        {
            "centroid_x": [10.0, 20.0, 30.0],
            "centroid_y": [15.0, 25.0, 35.0],
            "cell_type": ["Epithelial", "Immune", "Stromal"],
            "CD45": [1.2, 55.4, 3.1],
            "Ecad": [88.2, 2.1, 4.0],
        }
    )
    out_html = tmp_path / "spatial_viewer.html"
    html_str = export_interactive_html_viewer(
        df,
        type_col="cell_type",
        marker_cols=["CD45", "Ecad"],
        output_path=out_html,
        title="Test Viewer",
    )

    assert out_html.exists()
    assert '<canvas id="spatial-canvas">' in html_str
    assert "Test Viewer" in html_str
    assert "CD45" in html_str
