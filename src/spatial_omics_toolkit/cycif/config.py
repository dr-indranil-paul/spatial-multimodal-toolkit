"""Panel-specific configuration objects for the CyCIF notebook step API.

Every function in :mod:`spatial_omics_toolkit.cycif` that needs to know which
marker column means "epithelial" or "immune" takes one of these config objects
as an explicit argument rather than hardcoding marker names. The
``PREVAIL_BARRETTS_*`` constants below are the worked example this package's
own case study uses (25-marker hBE-TMA CyCIF panel) — they are a *default you
can start from*, not an institute-wide truth. A different panel needs a
different config, not a code change.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CellTypeMarkerConfig:
    """Marker-column configuration for :func:`spatial_omics_toolkit.step07_classify_cells`.

    Preserves the dual-gate, per-core z-score classification algorithm
    validated on the PREVAIL hBE-TMA cohort (epithelial gate on max marker
    z-score, non-epithelial gate excludes cells with any epithelial signal,
    single-marker gates for immune/stromal/endothelial/proliferation,
    mutually-exclusive priority assignment). Only the marker *names* are
    configurable — the gating logic itself is a domain decision, not a
    free-form rule engine.
    """

    epithelial_markers: Tuple[str, ...]
    """Columns whose max z-score defines the epithelial gate (e.g. Ecad, CK5, CK19)."""

    mesenchymal_markers: Tuple[str, ...]
    """Columns whose z-scores define mesenchymal signal (e.g. Vim, ZEB1)."""

    immune_marker: str
    """Single pan-immune marker column (e.g. CD45)."""

    myeloid_markers: Tuple[str, ...]
    """Columns marking myeloid lineage (e.g. IBA1, CD11c)."""

    antigen_presentation_marker: str
    """Marker used jointly with the immune marker to flag antigen-presenting cells (e.g. HLA-DR)."""

    stromal_marker: str
    """Single stromal marker column (e.g. aSMA)."""

    endothelial_marker: str
    """Single endothelial marker column (e.g. CD31)."""

    proliferation_marker: str
    """Single proliferation marker column (e.g. Ki67)."""

    z_score_markers: Tuple[str, ...]
    """All columns that get per-core z-scored; must be a superset of every marker referenced above."""

    z_threshold: float = 1.5
    """Positive-gate z-score threshold (top ~7% per core at the default)."""

    z_high_threshold: float = 2.0
    """Stricter z-score threshold used for the immune gate (top ~2.5% per core at the default)."""

    non_epithelial_gate: float = 0.5
    """A cell is treated as "not epithelial" when its max epithelial-marker z-score is below this."""

    def __post_init__(self) -> None:
        referenced = set(self.epithelial_markers) | set(self.mesenchymal_markers) | {
            self.immune_marker,
            self.antigen_presentation_marker,
            self.stromal_marker,
            self.endothelial_marker,
            self.proliferation_marker,
        } | set(self.myeloid_markers)
        missing = referenced - set(self.z_score_markers)
        if missing:
            raise ValueError(
                f"z_score_markers must include every marker referenced elsewhere in the config; "
                f"missing: {sorted(missing)}"
            )


# Marker short-name -> (intensity column, subcellular compartment, biological role).
# This is the panel actually used for the PREVAIL hBE-TMA cohort (see
# 02_spatial_proteomics/03_analysis/02_scripts/tma_cell_classify.py::MARKER_MAP).
# Ships as the worked example for this package's case study.
PREVAIL_BARRETTS_MARKER_MAP: Dict[str, Tuple[str, str, str]] = {
    "Ecad": ("Ecad_cytoplasm", "cytoplasm", "epithelial"),
    "Vim": ("Vim_cytoplasm", "cytoplasm", "mesenchymal"),
    "ZEB1": ("ZEB1_nuclei", "nuclei", "EMT_TF"),
    "CK5": ("CK5_cytoplasm", "cytoplasm", "epithelial"),
    "CK19": ("CK19_cytoplasm", "cytoplasm", "epithelial"),
    "aSMA": ("aSMA_cytoplasm", "cytoplasm", "stromal"),
    "CD31": ("CD31_perinuc5", "perinuc5", "endothelial"),
    "CAV1": ("CAV1_cytoplasm", "cytoplasm", "stromal"),
    "CD45": ("CD45_perinuc5", "perinuc5", "pan_immune"),
    "CD11c": ("CD11c_perinuc5", "perinuc5", "myeloid"),
    "IBA1": ("IBA1_perinuc5", "perinuc5", "myeloid"),
    "HLA_DR": ("HLA-DR_cytoplasm", "cytoplasm", "antigen_presentation"),
    "Ki67": ("Ki67_nuclei", "nuclei", "proliferation"),
    "PCNA": ("PCNA_nuclei", "nuclei", "proliferation"),
    "p53_nuc": ("p53_nuclei", "nuclei", "tumor_suppressor"),
    "p53_cyto": ("p53_cytoplasm", "cytoplasm", "tumor_suppressor"),
    "MUC1": ("MUC1_cytoplasm", "cytoplasm", "mucin"),
    "MUC2": ("MUC2_cytoplasm", "cytoplasm", "mucin"),
    "MUC5AC": ("MUC5AC_cytoplasm", "cytoplasm", "mucin"),
    "MUC6": ("MUC6_cytoplasm", "cytoplasm", "mucin"),
    "CD44": ("CD44_cytoplasm", "cytoplasm", "stemness"),
    "Ecad_mem": ("Ecad_cellmem2p25", "cellmem2p25", "epithelial_membrane"),
    "DAPI3": ("DAPI3_nuclei", "nuclei", "dapi"),
    "DAPI4": ("DAPI4_nuclei", "nuclei", "dapi"),
}

PREVAIL_BARRETTS_MARKER_CONFIG = CellTypeMarkerConfig(
    epithelial_markers=("Ecad_cytoplasm", "CK5_cytoplasm", "CK19_cytoplasm"),
    mesenchymal_markers=("Vim_cytoplasm", "ZEB1_nuclei"),
    immune_marker="CD45_perinuc5",
    myeloid_markers=("IBA1_perinuc5", "CD11c_perinuc5"),
    antigen_presentation_marker="HLA-DR_cytoplasm",
    stromal_marker="aSMA_cytoplasm",
    endothelial_marker="CD31_perinuc5",
    proliferation_marker="Ki67_nuclei",
    z_score_markers=(
        "Ecad_cytoplasm", "Vim_cytoplasm", "ZEB1_nuclei",
        "CK5_cytoplasm", "CK19_cytoplasm",
        "aSMA_cytoplasm", "CD31_perinuc5", "CD45_perinuc5",
        "CD11c_perinuc5", "IBA1_perinuc5", "HLA-DR_cytoplasm",
        "Ki67_nuclei",
        "MUC1_cytoplasm", "MUC2_cytoplasm", "MUC5AC_cytoplasm", "MUC6_cytoplasm",
    ),
)


@dataclass(frozen=True)
class ProgramDefinition:
    """One composite per-cell program score: the sum of member markers' per-core z-scores."""

    name: str
    marker_columns: Tuple[str, ...]


# The EMT program triads (EL/IH/ME) from tma_cell_classify.py::compute_emax_programs.
# Manuscript-specific program definition, shipped as the case-study default.
PREVAIL_BARRETTS_EMT_PROGRAMS: Tuple[ProgramDefinition, ...] = (
    ProgramDefinition("EL", ("Ecad_cytoplasm", "CK5_cytoplasm", "CK19_cytoplasm")),
    ProgramDefinition("IH", ("Ecad_cytoplasm", "Vim_cytoplasm", "ZEB1_nuclei")),
    ProgramDefinition("ME", ("Vim_cytoplasm", "ZEB1_nuclei")),
)

# Broad cell-type categories produced by step07_classify_cells, in priority order.
PREVAIL_BARRETTS_CELL_TYPES: Tuple[str, ...] = (
    "Hybrid_epithelial", "Epithelial", "Mesenchymal_like",
    "Myeloid_immune", "Lymphoid_immune", "Endothelial", "Stromal",
    "Proliferating", "Other",
)


# ═══════════════════════════════════════════════════════════════════════════
# step01_scan_intake configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_RAW_FILENAME_PATTERN = (
    r"^(?P<round>R\d+Q?)_"
    r"(?P<markers>[A-Za-z0-9.\-]+)_"
    r"(?P<slide>hBE-TMA\d+-\d+)_"
    r"(?P<date>\d{4}_\d{2}_\d{2})__"
    r"(?P<acq_id>\d+)-"
    r"(?P<scene>scene[A-Z]\d+)\.czi$"
)
"""Observed hBE-TMA CZI naming convention from the PREVAIL cohort intake
(``02_spatial_proteomics/03_analysis/02_scripts/cycif_qc.py::parse_czi_filename``,
via ``registration.py``). Ships as the *default* value of
``RawImageIntakeConfig.filename_pattern`` — a different acquisition naming
scheme is a different pattern string, not a code change. The pattern must
define the six named groups ``round``, ``markers``, ``slide``, ``date``,
``acq_id``, ``scene`` (:func:`spatial_omics_toolkit.cycif.intake.parse_intake_filename`
validates this and raises if a custom pattern is missing one)."""

DEFAULT_EXPECTED_ROUNDS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R5Q")
"""Round vocabulary observed in the PREVAIL hBE-TMA cohort (5 antibody rounds
plus one blank/quench-check round, R5Q). Ships as the default value of
``RawImageIntakeConfig.expected_rounds``."""


@dataclass(frozen=True)
class RawImageIntakeConfig:
    """Config for :func:`spatial_omics_toolkit.cycif.intake.step01_scan_intake`.

    Bundles every naming/count assumption the source ``cycif_qc.py`` intake
    script hardcoded (filename regex, round vocabulary, expected channel
    count, per-slide expected file counts), so a different acquisition batch
    needs a different config object, not a source edit. The channel-count
    check, per-slide pixel-size consistency check, and round x scene
    coverage check themselves are unchanged from the source script.
    """

    filename_pattern: str = DEFAULT_RAW_FILENAME_PATTERN
    """Regex with named groups round/markers/slide/date/acq_id/scene."""

    expected_rounds: Tuple[str, ...] = DEFAULT_EXPECTED_ROUNDS
    """Round labels every (slide, scene) is expected to have one file for."""

    expected_channels: int = 5
    """Channel count every OME-XML sidecar is expected to report."""

    raw_extension: str = ".czi"
    """Raw image file extension to scan for."""

    ome_xml_suffix: str = ".ome.xml"
    """Sidecar metadata file suffix, matched as ``*{suffix}`` per slide dir."""

    expected_files_per_slide: Optional[Mapping[str, int]] = None
    """Optional ``{slide_name: expected_file_count}``. ``None`` (default)
    skips the per-slide file-count check rather than assuming a
    batch-specific number — the source script's hardcoded default of 60
    was one manuscript's acquisition-batch fact, not a panel constant, so
    it is not carried over as a silent default here."""

    pixel_size_bounds_um: Optional[Tuple[float, float]] = None
    """Optional ``(min, max)`` plausible pixel size in microns. Not present
    in the source script (which only checked per-slide consistency, which
    is still always run); off by default so nothing is flagged that the
    original pipeline never flagged. Documented as a new, opt-in addition,
    not a ported behavior."""


PREVAIL_BARRETTS_INTAKE_CONFIG = RawImageIntakeConfig()
"""Worked default for the PREVAIL hBE-TMA CyCIF cohort intake scan — every
field is the value observed in that cohort's raw CZI acquisitions."""


# ═══════════════════════════════════════════════════════════════════════════
# step02_convert_raw_images configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImageConversionConfig:
    """Config for :func:`spatial_omics_toolkit.cycif.convert.step02_convert_raw_images`.

    Bundles the contrast-stretch/MIP/composite behavior ported from
    ``process_czi.py`` / ``process_nd2.py`` so panel-specific fluorophore
    naming and file extensions can be supplied without editing the
    conversion code. The contrast-stretch/MIP/composite algorithm itself
    (2nd/98th percentile stretch, max-intensity projection over Z,
    additive RGB pseudocolor composite) is unchanged from the source.
    """

    generate_mip: bool = True
    generate_composite: bool = True

    contrast_percentiles: Tuple[float, float] = (2.0, 98.0)
    """Percentile bounds for the contrast stretch — unchanged from the
    source scripts' hardcoded ``(2, 98)``, now a parameter."""

    custom_color_map: Optional[Mapping[str, Tuple[float, float, float]]] = None
    """Optional ``{channel_name_substring: (r, g, b)}`` in ``[0, 1]``,
    checked before the built-in DAPI/GFP/RFP/Cy5 keyword map in
    ``get_color_map``. Not present in the source script — a documented
    addition for panels whose fluorophore names don't match the built-in
    keyword set (e.g. a non-Zeiss/Nikon naming scheme)."""

    czi_extension: str = ".czi"
    nd2_extension: str = ".nd2"


PREVAIL_BARRETTS_CONVERSION_CONFIG = ImageConversionConfig()


@dataclass(frozen=True)
class SlurmDispatchConfig:
    """SLURM ``sbatch`` options for dispatching step02 conversion as a batch job.

    This lab's HPC policy (``AGENTS.md`` #2) requires anything compute-heavy
    or >~1 minute to run via SLURM, never the login node or an interactive
    notebook kernel. Gigapixel CZI/ND2 contrast-stretch + MIP + composite
    generation is squarely in that category, so
    ``step02_convert_raw_images`` builds an ``sbatch`` script by default
    instead of processing pixels in-process (see ``convert.py`` module
    docstring). Job scripts, logs, and any staged files must be written to
    paths on shared/cluster-visible storage — session-local scratch
    directories are node-local and invisible to compute nodes; this class
    does not hardcode a lab-specific root, so the caller is responsible for
    passing ``jobs_dir``/``output_folder`` paths that satisfy that.
    """

    job_name: str = "sot_step02_convert"
    cpus_per_task: int = 8
    mem_gb: int = 64
    time_limit: str = "04:00:00"
    partition: Optional[str] = None
    conda_env: Optional[str] = None
    extra_sbatch_lines: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cpus_per_task > 32:
            raise ValueError(
                f"cpus_per_task={self.cpus_per_task} exceeds this cluster's "
                f"per-node batch-partition limit (<=32; AGENTS.md #2 sizing rule). "
                f"Requesting 48, for example, is rejected by SLURM outright."
            )


PREVAIL_BARRETTS_SLURM_CONFIG = SlurmDispatchConfig()


@dataclass(frozen=True)
class RegistrationConfig:
    """Algorithm and sanity-bound parameters for
    :func:`spatial_omics_toolkit.cycif.register.step03_register_rounds`.

    Defaults are copied unchanged from the validated SIFT+RANSAC multi-round
    CyCIF registration pipeline
    (``02_spatial_proteomics/03_analysis/02_scripts/registration.py``). These
    are not free-form tuning knobs: ``max_angle_deg``, ``scale_range``, and
    ``max_shift_fraction`` are physically-motivated bounds for same-slide TMA
    re-staining cycles on an AxioScan (rotation/scale should be ~0deg/~1.0;
    only translational tissue slippage between rounds is expected). A fitted
    transform outside these bounds indicates a registration failure, not a
    legitimate tissue deformation, and is rejected rather than silently
    applied — do not widen them just to make a failure "pass". Change them
    only with a physically-motivated reason for a different imaging setup.
    """

    n_features: int = 10000
    """Max SIFT features per image."""

    flann_checks: int = 100
    """FLANN KD-tree leaf checks — higher = better recall, slower."""

    lowe_ratio: float = 0.75
    """Lowe's ratio test threshold for filtering ambiguous keypoint matches."""

    ransac_threshold: float = 3.0
    """RANSAC reprojection error threshold, in pixels."""

    ransac_confidence: float = 0.95
    """RANSAC confidence level."""

    ransac_max_iters: int = 20000
    """RANSAC iteration cap."""

    max_angle_deg: float = 5.0
    """Sanity bound: reject a fitted transform with rotation beyond +/- this angle."""

    scale_range: Tuple[float, float] = (0.95, 1.05)
    """Sanity bound: reject a fitted transform with scale outside this range."""

    max_shift_fraction: float = 0.5
    """Sanity bound: reject a fitted transform whose translation exceeds this
    fraction of the reference image diagonal."""


# Koei's CZI naming convention for the PREVAIL hBE-TMA cohort:
#   R{round}_{markers}_{slide}_{YYYY_MM_DD}__{acqid}-{scene}.czi
# Example: R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_2026_05_16__17930-sceneA1.czi
# Ships as the worked-example default for
# spatial_omics_toolkit.cycif.register.parse_czi_filename — a different
# acquisition naming scheme is a parameter to that function, not a code change.
# Must have exactly 6 capture groups, in order: round, markers, slide, date, acq_id, scene.
PREVAIL_BARRETTS_CZI_FILENAME_PATTERN: str = (
    r'^(R\d+Q?)_'
    r'([A-Za-z0-9.\-]+)_'
    r'(hBE-TMA\d+-\d+)_'
    r'(\d{4}_\d{2}_\d{2})__'
    r'(\d+)-'
    r'(scene[A-Z]\d+)\.czi$'
)

# Round-label sort convention: R1 < R2 < ... < R5 < R5Q. Plain alphabetical
# sort breaks for R10+ (R10 sorts before R2) and puts R5Q before R5 in some
# locales; spatial_omics_toolkit.cycif.register._round_sort_key parses this
# pattern into a numeric-aware (round_number, is_Q_suffix) key instead. A
# cohort using a different round-label scheme (e.g. "Cycle3") passes its own
# pattern with the same two capture groups (numeric part, optional suffix flag).
PREVAIL_BARRETTS_ROUND_LABEL_PATTERN: str = r'^R(\d+)(Q?)$'


# ---------------------------------------------------------------------------
# step04_export_registered / step05_segment_cells configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportNamingConfig:
    """Output filename templates for
    :func:`spatial_omics_toolkit.cycif.export_images.step04_export_registered`.

    Ported from the fixed string literals scattered through
    ``02_spatial_proteomics/03_analysis/02_scripts/export.py`` (e.g.
    ``f'{rnd}_valid_mask.tif'``, ``f'{scene_label}_composite.ome.tif'``).
    None of these are panel- or manuscript-specific — they are this lab's
    filename convention — so they are now templates with the original
    strings as defaults rather than literals baked into the function bodies.
    Every template is formatted with a subset of ``{scene_label}``,
    ``{round}``, and ``{marker}`` depending on which artefact it names.
    """

    channel_filename_template: str = "{round}_{marker}.tif"
    """PIPEX single-channel TIFF filename, formatted with round + marker."""

    composite_filename_template: str = "{scene_label}_composite.ome.tif"
    """Composite OME-TIFF filename, formatted with scene_label."""

    metadata_filename_template: str = "{scene_label}_registration_metadata.json"
    """Provenance JSON filename, formatted with scene_label."""

    valid_mask_filename_template: str = "{round}_valid_mask.tif"
    """Per-round valid-pixel mask filename, formatted with round."""

    combined_mask_filename: str = "combined_valid_mask.tif"
    """Intersection valid-pixel mask filename (no formatting fields)."""


@dataclass(frozen=True)
class MCMICROConfig:
    """Container/pipeline configuration for the MCMICRO backend of
    :func:`spatial_omics_toolkit.cycif.segment.step05_segment_cells`.

    Ported from ``02_spatial_proteomics/03_analysis/02_scripts/mcmicro.py``.
    The original script hardcoded the wrapper/install paths as bare local
    variables (``SOFTWARES/bin/mcmicro``, a local ``mcmicro/main.nf``
    checkout) and the Nextflow profile name (``'exacloud'``, this HPC's
    cluster profile). Here ``mcmicro_bin`` and ``mcmicro_nf`` are required
    constructor arguments with no default — there is no institute-wide
    correct value for another lab's MCMICRO install location, so forcing an
    explicit value is preferred over a silently-wrong default. ``profile``
    keeps the original default because it is a legitimate generic default
    for *this* HPC's Nextflow profile catalogue, and remains fully
    overridable for any other cluster/profile name.
    """

    mcmicro_bin: str
    """Path to the ``mcmicro`` wrapper script (handles Java/Nextflow env setup)."""

    mcmicro_nf: str
    """Path to the local MCMICRO ``main.nf`` checkout."""

    profile: str = "exacloud"
    """Nextflow ``-profile`` value. Default is this HPC's cluster profile name."""

    segmenter: str = "mccellpose"
    """MCMICRO segmentation module: 'mccellpose' (recommended), 'cellpose', or 'unmicst'."""

    dapi_channel: int = 1
    """1-based DAPI channel index in the composite OME-TIFF."""

    cellpose_model: str = "cyto3"
    """CellPose pretrained model name."""

    cellpose_diameter: int = 30
    """Expected nucleus diameter in pixels (informational; recorded in provenance)."""

    cellpose_expansion: int = 10
    """Cytoplasm-mask expansion around the nucleus, in pixels."""

    use_gpu: bool = False
    """Append ',gpu' to the Nextflow profile and pass --use-gpu to the segmenter."""

    extra_options: str = ""
    """Additional flags appended to the segmenter option string in params.yml."""

    extra_args: str = ""
    """Additional raw arguments appended to the ``nextflow run`` command line."""

    timeout_seconds: int = 14400
    """Seconds before the MCMICRO subprocess is killed (default 4 h)."""


@dataclass(frozen=True)
class PIPEXConfig:
    """Pipeline configuration for the PIPEX backend of
    :func:`spatial_omics_toolkit.cycif.segment.step05_segment_cells`.

    Ported from ``02_spatial_proteomics/03_analysis/02_scripts/pipex.py``.
    ``pipex_home`` (the PIPEX install directory containing
    ``segmentation.py``/``analysis.py``) and ``nuclei_marker`` (the DAPI
    channel's TIFF stem, e.g. ``'R1_DAPI'``) are panel/install-specific and
    have no institute-wide default, so both are required constructor
    arguments rather than baked-in literals.
    """

    pipex_home: str
    """Directory containing PIPEX's segmentation.py / analysis.py."""

    nuclei_marker: str
    """TIFF stem for the nuclear channel (e.g. 'R1_DAPI')."""

    nuclei_diameter: int = 30
    """Expected nucleus diameter in pixels, passed to StarDist."""

    nuclei_expansion: int = 10
    """Cytoplasm expansion around the nucleus in pixels."""

    membrane_markers: Tuple[str, ...] = ()
    """Optional membrane-channel TIFF stems used for watershed refinement."""

    measure_markers: Optional[Tuple[str, ...]] = None
    """All channels to quantify. If None, computed from a marker_map at call
    time via :func:`spatial_omics_toolkit.cycif.segment.get_measure_markers`
    (every non-DAPI channel)."""

    exclude_rounds: Tuple[str, ...] = ()
    """Round labels to omit entirely when auto-deriving measure_markers
    (e.g. a blank/no-antibody QC round)."""

    extra_flags: str = ""
    """Additional raw flags appended to the PIPEX segmentation.py command line."""

    timeout_seconds: int = 7200
    """Seconds before the PIPEX subprocess is killed (default 2 h)."""
