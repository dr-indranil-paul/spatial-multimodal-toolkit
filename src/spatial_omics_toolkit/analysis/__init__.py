"""Analysis engines ported and standardized from spatial proteomics workflows."""

from importlib import import_module


_EXPORT_MODULES = {
    "correct_batch_combat": "spatial_omics_toolkit.analysis.batch_correction",
    "MAXEY_IMMUNE_MATRIX": "spatial_omics_toolkit.analysis.cell_typing",
    "MAXEY_PRIMARY_MATRIX": "spatial_omics_toolkit.analysis.cell_typing",
    "MAXEY_TUMOR_MATRIX": "spatial_omics_toolkit.analysis.cell_typing",
    "SVMClassificationResult": "spatial_omics_toolkit.analysis.cell_typing",
    "loss_weights_to_dataframe": "spatial_omics_toolkit.analysis.cell_typing",
    "predict_celltype_svm": "spatial_omics_toolkit.analysis.cell_typing",
    "scale_features": "spatial_omics_toolkit.analysis.cell_typing",
    "train_celltype_svm": "spatial_omics_toolkit.analysis.cell_typing",
    "uncertainty_sampling": "spatial_omics_toolkit.analysis.cell_typing",
    "vector_classify_cells": "spatial_omics_toolkit.analysis.cell_typing",
    "calculate_marker_snr": "spatial_omics_toolkit.analysis.cycif_qc",
    "filter_background_signal": "spatial_omics_toolkit.analysis.cycif_qc",
    "PRESET_STAIN_MATRICES": "spatial_omics_toolkit.analysis.deconvolution",
    "STAIN_VECTORS": "spatial_omics_toolkit.analysis.deconvolution",
    "DeconvolutionResult": "spatial_omics_toolkit.analysis.deconvolution",
    "build_stain_matrix": "spatial_omics_toolkit.analysis.deconvolution",
    "deconvolve_color_stains": "spatial_omics_toolkit.analysis.deconvolution",
    "correct_flatfield_shading": "spatial_omics_toolkit.analysis.illumination",
    "correct_tile_edge_gain": "spatial_omics_toolkit.analysis.illumination",
    "compute_cell_centroids": "spatial_omics_toolkit.analysis.segmentation",
    "CellInteractionResult": "spatial_omics_toolkit.analysis.spatial_interactions",
    "VirtualTherapyResult": "spatial_omics_toolkit.analysis.spatial_interactions",
    "compute_cell_interaction_matrix": "spatial_omics_toolkit.analysis.spatial_interactions",
    "compute_distance_to_boundary": "spatial_omics_toolkit.analysis.spatial_interactions",
    "compute_spatial_lag_features": "spatial_omics_toolkit.analysis.spatial_interactions",
    "filter_cells_by_polygon": "spatial_omics_toolkit.analysis.spatial_interactions",
    "simulate_cell_knockout": "spatial_omics_toolkit.analysis.spatial_interactions",
    "compute_spatial_neighborhood_graph": "spatial_omics_toolkit.analysis.spatial_ml",
    "compute_tissue_masked_gini": "spatial_omics_toolkit.analysis.spatial_ml",
}

__all__ = [*sorted(_EXPORT_MODULES)]


def __getattr__(name: str):
    """Load an analysis engine only when its public export is requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public exports to notebook completion without importing them."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))
