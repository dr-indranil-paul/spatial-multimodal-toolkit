"""SpatialOmics Toolkit: Institute-grade spatial and single-cell multiomics analytics platform.

Notebook analysts drive the CyCIF / spatial-proteomics pipeline one numbered
step at a time from this top-level namespace::

    import spatial_omics_toolkit as sot
    cells = sot.step07_classify_cells(cells, sot.PREVAIL_BARRETTS_MARKER_CONFIG)
    cells = sot.step08_score_programs(cells, sot.PREVAIL_BARRETTS_EMT_PROGRAMS)
    cores = sot.step09_aggregate_per_core(cells)

Tab-complete ``sot.step<TAB>`` in Jupyter to see every step in pipeline
order. See ``spatial_omics_toolkit.cycif`` for the full step table and
``README.md`` for a worked quickstart.
"""

__version__ = "0.1.0"
__author__ = "Dr. Indranil Paul"

from importlib import import_module


_EXPORT_MODULES = {
    "CellTypeMarkerConfig": "spatial_omics_toolkit.cycif",
    "ExportNamingConfig": "spatial_omics_toolkit.cycif",
    "ImageConversionConfig": "spatial_omics_toolkit.cycif",
    "MCMICROConfig": "spatial_omics_toolkit.cycif",
    "PIPEXConfig": "spatial_omics_toolkit.cycif",
    "ProgramDefinition": "spatial_omics_toolkit.cycif",
    "RawImageIntakeConfig": "spatial_omics_toolkit.cycif",
    "RegistrationConfig": "spatial_omics_toolkit.cycif",
    "SlurmDispatchConfig": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_CELL_TYPES": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_CONVERSION_CONFIG": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_EMT_PROGRAMS": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_INTAKE_CONFIG": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_MARKER_CONFIG": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_MARKER_MAP": "spatial_omics_toolkit.cycif",
    "PREVAIL_BARRETTS_SLURM_CONFIG": "spatial_omics_toolkit.cycif",
    "bridge_mcmicro_to_pipex": "spatial_omics_toolkit.cycif",
    "step01_scan_intake": "spatial_omics_toolkit.cycif",
    "step02_convert_raw_images": "spatial_omics_toolkit.cycif",
    "step03_register_rounds": "spatial_omics_toolkit.cycif",
    "step04_export_registered": "spatial_omics_toolkit.cycif",
    "step05_segment_cells": "spatial_omics_toolkit.cycif",
    "step06_qc_filter_intensities": "spatial_omics_toolkit.cycif",
    "step07_classify_cells": "spatial_omics_toolkit.cycif",
    "step08_score_programs": "spatial_omics_toolkit.cycif",
    "step09_aggregate_per_core": "spatial_omics_toolkit.cycif",
    "step10_spatial_features": "spatial_omics_toolkit.cycif",
    "step11_spatial_null": "spatial_omics_toolkit.cycif",
    "step12_compare_groups": "spatial_omics_toolkit.cycif",
    "step13_pseudobulk": "spatial_omics_toolkit.cycif",
    "PLAProcessingConfig": "spatial_omics_toolkit.pla",
    "detect_pla_spots": "spatial_omics_toolkit.pla",
    "enhance_dapi": "spatial_omics_toolkit.pla",
    "enhance_dot_display": "spatial_omics_toolkit.pla",
    "estimate_af_scale": "spatial_omics_toolkit.pla",
    "inspect_pla_czi": "spatial_omics_toolkit.pla",
    "load_pla_scene": "spatial_omics_toolkit.pla",
    "make_tissue_mask": "spatial_omics_toolkit.pla",
    "process_pla_batch": "spatial_omics_toolkit.pla",
    "process_pla_czi": "spatial_omics_toolkit.pla",
    "segment_dapi_cells": "spatial_omics_toolkit.pla",
    "subtract_autofluorescence": "spatial_omics_toolkit.pla",
    "MAXEY_IMMUNE_MATRIX": "spatial_omics_toolkit.analysis",
    "MAXEY_PRIMARY_MATRIX": "spatial_omics_toolkit.analysis",
    "MAXEY_TUMOR_MATRIX": "spatial_omics_toolkit.analysis",
    "CellInteractionResult": "spatial_omics_toolkit.analysis",
    "DeconvolutionResult": "spatial_omics_toolkit.analysis",
    "PRESET_STAIN_MATRICES": "spatial_omics_toolkit.analysis",
    "STAIN_VECTORS": "spatial_omics_toolkit.analysis",
    "SVMClassificationResult": "spatial_omics_toolkit.analysis",
    "VirtualTherapyResult": "spatial_omics_toolkit.analysis",
    "build_stain_matrix": "spatial_omics_toolkit.analysis",
    "calculate_marker_snr": "spatial_omics_toolkit.analysis",
    "compute_cell_centroids": "spatial_omics_toolkit.analysis",
    "compute_cell_interaction_matrix": "spatial_omics_toolkit.analysis",
    "compute_distance_to_boundary": "spatial_omics_toolkit.analysis",
    "compute_spatial_lag_features": "spatial_omics_toolkit.analysis",
    "compute_spatial_neighborhood_graph": "spatial_omics_toolkit.analysis",
    "compute_tissue_masked_gini": "spatial_omics_toolkit.analysis",
    "correct_batch_combat": "spatial_omics_toolkit.analysis",
    "correct_flatfield_shading": "spatial_omics_toolkit.analysis",
    "correct_tile_edge_gain": "spatial_omics_toolkit.analysis",
    "deconvolve_color_stains": "spatial_omics_toolkit.analysis",
    "filter_background_signal": "spatial_omics_toolkit.analysis",
    "filter_cells_by_polygon": "spatial_omics_toolkit.analysis",
    "loss_weights_to_dataframe": "spatial_omics_toolkit.analysis",
    "predict_celltype_svm": "spatial_omics_toolkit.analysis",
    "scale_features": "spatial_omics_toolkit.analysis",
    "simulate_cell_knockout": "spatial_omics_toolkit.analysis",
    "train_celltype_svm": "spatial_omics_toolkit.analysis",
    "uncertainty_sampling": "spatial_omics_toolkit.analysis",
    "vector_classify_cells": "spatial_omics_toolkit.analysis",
    "register_he_to_fluorescence": "spatial_omics_toolkit.cycif.register",
    "export_interactive_html_viewer": "spatial_omics_toolkit.reporting.viewer",
}

__all__ = [*sorted(_EXPORT_MODULES), "__author__", "__version__"]


def __getattr__(name: str):
    """Load public capability exports only when callers request them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public exports to notebook completion without importing them."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))
