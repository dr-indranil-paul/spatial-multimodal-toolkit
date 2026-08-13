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

from spatial_omics_toolkit.cycif import (  # noqa: F401
    CellTypeMarkerConfig,
    ExportNamingConfig,
    ImageConversionConfig,
    MCMICROConfig,
    PIPEXConfig,
    ProgramDefinition,
    RawImageIntakeConfig,
    RegistrationConfig,
    SlurmDispatchConfig,
    PREVAIL_BARRETTS_CELL_TYPES,
    PREVAIL_BARRETTS_CONVERSION_CONFIG,
    PREVAIL_BARRETTS_EMT_PROGRAMS,
    PREVAIL_BARRETTS_INTAKE_CONFIG,
    PREVAIL_BARRETTS_MARKER_CONFIG,
    PREVAIL_BARRETTS_MARKER_MAP,
    PREVAIL_BARRETTS_SLURM_CONFIG,
    bridge_mcmicro_to_pipex,
    step01_scan_intake,
    step02_convert_raw_images,
    step03_register_rounds,
    step04_export_registered,
    step05_segment_cells,
    step06_qc_filter_intensities,
    step07_classify_cells,
    step08_score_programs,
    step09_aggregate_per_core,
    step10_spatial_features,
    step11_spatial_null,
    step12_compare_groups,
    step13_pseudobulk,
)
