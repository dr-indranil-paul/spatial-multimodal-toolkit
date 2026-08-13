"""CyCIF / spatial-proteomics notebook step API.

This subpackage holds the implementation; the public entry point is the
numbered ``stepNN_*`` functions re-exported from
``spatial_omics_toolkit.__init__`` so a notebook analyst only ever needs::

    import spatial_omics_toolkit as sot
    sot.step07_classify_cells(...)

Full pipeline, in order (see DESIGN.md and this project's wiki
``requirements/notebook_first_step_api`` page for the source-mapping and
integrity notes behind each step):

============  ===============================  ==========================================
step          function                         status
============  ===============================  ==========================================
step01        step01_scan_intake               raw CZI/ND2 intake QC (imaging)
step02        step02_convert_raw_images         CZI/ND2 -> OME-TIFF, SLURM-dispatched (imaging)
step03        step03_register_rounds            SIFT+RANSAC multi-round registration (imaging)
step04        step04_export_registered          composite/per-channel export + provenance (imaging)
step05        step05_segment_cells              MCMICRO or PIPEX segmentation (imaging)
step06        step06_qc_filter_intensities      post-segmentation marker SNR/background QC
step07        step07_classify_cells             per-core z-score marker-threshold typing
step08        step08_score_programs             composite per-cell program scores
step09        step09_aggregate_per_core         mean-based per-core summary table
step10        step10_spatial_features           per-core density/neighborhood features
step11        step11_spatial_null               spatial null permutation control (NEW)
step12        step12_compare_groups             BH-corrected two-group comparison
step13        step13_pseudobulk                 sum-aggregation before R differential stats
============  ===============================  ==========================================

Steps 01-05 need the ``imaging`` extra (``pip install -e ".[imaging]"``) and,
for anything beyond small/synthetic inputs, an active SLURM allocation — see
each function's docstring. Steps 06-13 only need the core dependencies and
run interactively in a notebook.
"""

from spatial_omics_toolkit.cycif.aggregate import step09_aggregate_per_core
from spatial_omics_toolkit.cycif.classify import step07_classify_cells, step08_score_programs, zscore_per_core
from spatial_omics_toolkit.cycif.compare import step12_compare_groups
from spatial_omics_toolkit.cycif.config import (
    CellTypeMarkerConfig,
    ExportNamingConfig,
    ImageConversionConfig,
    MCMICROConfig,
    PIPEXConfig,
    ProgramDefinition,
    PREVAIL_BARRETTS_CELL_TYPES,
    PREVAIL_BARRETTS_CONVERSION_CONFIG,
    PREVAIL_BARRETTS_EMT_PROGRAMS,
    PREVAIL_BARRETTS_INTAKE_CONFIG,
    PREVAIL_BARRETTS_MARKER_CONFIG,
    PREVAIL_BARRETTS_MARKER_MAP,
    PREVAIL_BARRETTS_SLURM_CONFIG,
    RawImageIntakeConfig,
    RegistrationConfig,
    SlurmDispatchConfig,
)
from spatial_omics_toolkit.cycif.convert import step02_convert_raw_images
from spatial_omics_toolkit.cycif.export_images import step04_export_registered
from spatial_omics_toolkit.cycif.intake import step01_scan_intake
from spatial_omics_toolkit.cycif.pseudobulk import step13_pseudobulk
from spatial_omics_toolkit.cycif.qc import step06_qc_filter_intensities
from spatial_omics_toolkit.cycif.register import step03_register_rounds
from spatial_omics_toolkit.cycif.segment import bridge_mcmicro_to_pipex, step05_segment_cells
from spatial_omics_toolkit.cycif.spatial_features import step10_spatial_features
from spatial_omics_toolkit.cycif.spatial_null import step11_spatial_null

__all__ = [
    "step01_scan_intake",
    "step02_convert_raw_images",
    "step03_register_rounds",
    "step04_export_registered",
    "step05_segment_cells",
    "step06_qc_filter_intensities",
    "step07_classify_cells",
    "step08_score_programs",
    "step09_aggregate_per_core",
    "step10_spatial_features",
    "step11_spatial_null",
    "step12_compare_groups",
    "step13_pseudobulk",
    "bridge_mcmicro_to_pipex",
    "zscore_per_core",
    "CellTypeMarkerConfig",
    "ExportNamingConfig",
    "ImageConversionConfig",
    "MCMICROConfig",
    "PIPEXConfig",
    "ProgramDefinition",
    "RawImageIntakeConfig",
    "RegistrationConfig",
    "SlurmDispatchConfig",
    "PREVAIL_BARRETTS_CELL_TYPES",
    "PREVAIL_BARRETTS_CONVERSION_CONFIG",
    "PREVAIL_BARRETTS_EMT_PROGRAMS",
    "PREVAIL_BARRETTS_INTAKE_CONFIG",
    "PREVAIL_BARRETTS_MARKER_CONFIG",
    "PREVAIL_BARRETTS_MARKER_MAP",
    "PREVAIL_BARRETTS_SLURM_CONFIG",
]
