"""step05_segment_cells — cell segmentation via MCMICRO or PIPEX.

Ported and generalized from
``02_spatial_proteomics/03_analysis/02_scripts/mcmicro.py`` (MCMICRO backend:
containerized, CellPose-based, SLURM/Nextflow-native) and
``02_spatial_proteomics/03_analysis/02_scripts/pipex.py`` (PIPEX backend:
StarDist-based, runs in-process). The two source scripts implemented
interchangeable segmentation backends producing structurally comparable
per-cell tables; this module exposes both through one step function,
:func:`step05_segment_cells`, selected via ``backend="mcmicro"`` or
``backend="pipex"``.

What changed to generalize it
------------------------------
- Container/install paths, image tags, and the Nextflow profile name were
  local variables or bare defaults in the source scripts, tied to this
  lab's install (``SOFTWARES/bin/mcmicro``, a local MCMICRO checkout, the
  ``'exacloud'`` cluster profile, ``$SOFTWARES/envs/spatial-proteomics/pipex/``).
  They are now fields on
  :class:`~spatial_omics_toolkit.cycif.config.MCMICROConfig` /
  :class:`~spatial_omics_toolkit.cycif.config.PIPEXConfig`, with the
  install-location fields required (no institute-wide correct default
  exists) and the profile/algorithm-parameter fields keeping their
  original values as defaults.
- Marker-panel-specific column names never appear in this module. The
  MCMICRO->PIPEX column bridge (:func:`bridge_mcmicro_to_pipex`) only
  renames the fixed, tool-defined columns (``CellID``, ``X_centroid``, ...)
  that MCQuant/PIPEX themselves define; the marker columns it passes
  through are supplied by the caller via ``analysis_markers``.
- The **fail-loudly, no-silent-fallback command-builder pattern is
  preserved exactly**: ``build_*_cmd`` constructs a command string,
  ``subprocess.run`` executes it, and a non-zero exit code raises
  ``RuntimeError`` immediately. If MCMICRO fails, this module does not
  retry with PIPEX and report the same result — the caller gets a raised
  exception naming the failed backend and exact command, and any
  substitution has to be an explicit, separate, visibly-labeled call
  (mirrors DESIGN.md §7's fallback policy: a substitution is never silently
  treated as an equivalent result).

HPC / SLURM requirement
------------------------
Both backends can process a gigapixel composite image, and MCMICRO drives a
containerized Nextflow pipeline. Per this lab's HPC policy, that work is
compute-heavy and must run under a SLURM allocation — **never on a login
node**. This module's job is to build the correct command and manage I/O; it
does not enforce SLURM itself (it cannot detect a login node from inside a
Python process reliably). A notebook user must call
``step05_segment_cells(..., run=True)`` from *within* an interactive
``srun`` session or an ``sbatch`` batch job, not from a login-node Jupyter
kernel. Use ``run=False`` to build and inspect the command without
executing it (e.g. to hand off to a separate ``sbatch`` script).
"""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from spatial_omics_toolkit.cycif.config import MCMICROConfig, PIPEXConfig

logger = logging.getLogger("spatial_omics_toolkit.cycif.segment")

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_sort_key(label: str):
    """Sort round labels in biological order: R1 < R2 < ... < R5 < R5Q.

    Local port of ``mcmicro.py::_round_sort_key_local``, kept independent of
    ``spatial_omics_toolkit.cycif.register`` on purpose (same reason the
    source script kept its own copy rather than importing the registration
    module's sort key: this module must be importable/usable standalone,
    without pulling in the registration stage).
    """
    m = re.match(r"^R(\d+)(Q?)$", label)
    if m:
        return (int(m.group(1)), 1 if m.group(2) else 0)
    return (999, 0)


def _run_subprocess(cmd: str, backend_name: str, cwd: Optional[PathLike] = None, timeout: int = 14400) -> int:
    """Execute a command via subprocess; raise RuntimeError on non-zero exit.

    Shared fail-loudly runner for both backends — preserves
    ``run_mcmicro`` / ``run_pipex_cmd``'s behavior exactly: log the command,
    run it, and raise immediately on failure so a notebook stops rather than
    silently proceeding to load missing/stale segmentation output.
    """
    logger.info(f"  [{backend_name}] CMD: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, timeout=timeout)
    if result.returncode != 0:
        msg = f"{backend_name} exited with code {result.returncode}.\nCMD: {cmd}"
        if cwd:
            msg += f"\nWorking dir: {cwd}"
        raise RuntimeError(msg)
    return result.returncode


# ---------------------------------------------------------------------------
# MCMICRO backend
# ---------------------------------------------------------------------------

def build_markers_csv(marker_map: Dict[str, List[str]]) -> str:
    """Generate MCMICRO ``markers.csv`` content from a project marker map.

    Unchanged from the source script. Channel order must exactly match the
    channel order in the composite OME-TIFF produced by
    ``step04_export_registered`` / ``build_composite_ometiff``.

    Naming convention: ``{round}_{marker}`` (e.g. ``R1_DAPI``, ``R1_MUC1``).
    This is the same convention step04 uses for composite channel names, so
    MCMICRO's quantification columns line up with PIPEX channel names
    without any translation beyond :func:`bridge_mcmicro_to_pipex`'s
    structural renames.

    Args:
        marker_map: ``{round_label: [ch0_name, ch1_name, ...]}``.

    Returns:
        CSV string with header ``cycle,marker_name``.
    """
    rounds_sorted = sorted(marker_map.keys(), key=_round_sort_key)
    lines = ["cycle,marker_name"]
    for cycle_num, rnd in enumerate(rounds_sorted, start=1):
        for marker in marker_map[rnd]:
            lines.append(f"{cycle_num},{rnd}_{marker}")
    return "\n".join(lines)


def build_params_yml(config: MCMICROConfig) -> str:
    """Generate MCMICRO ``params.yml`` content from an :class:`MCMICROConfig`.

    Unchanged algorithm from the source script's ``build_params_yml``, now
    reading every value from ``config`` instead of individual keyword
    arguments with lab-specific defaults baked into the function signature.

    Configures ``start-at: segmentation`` (a pre-registered OME-TIFF is
    supplied in ``registration/`` — Ashlar is skipped since step03 already
    performed SIFT+RANSAC registration) and ``stop-at: quantification``.
    """
    gpu_flag = "--use-gpu " if config.use_gpu else ""

    if config.segmenter in ("mccellpose", "cellpose"):
        seg_opts = (
            f"{gpu_flag}"
            f"--channel {config.dapi_channel} "
            f"--pretrained_model {config.cellpose_model} "
            f"--expand-size {config.cellpose_expansion}"
        )
        if config.extra_options:
            seg_opts += f" {config.extra_options}"
        return (
            f"workflow:\n"
            f"  start-at: segmentation\n"
            f"  stop-at: quantification\n"
            f"  probability-maps: {config.segmenter}\n"
            f"\n"
            f"options:\n"
            f"  {config.segmenter}: >-\n"
            f"    {seg_opts}\n"
        )
    return (
        f"workflow:\n"
        f"  start-at: segmentation\n"
        f"  stop-at: quantification\n"
        f"  probability-maps: {config.segmenter}\n"
    )


def prepare_mcmicro_project(
    scene_label: str,
    composite_ometiff_path: PathLike,
    marker_map: Dict[str, List[str]],
    project_dir: PathLike,
    config: MCMICROConfig,
    overwrite: bool = False,
) -> Path:
    """Create an MCMICRO project directory for one registered scene.

    Unchanged from the source script's ``prepare_mcmicro_project``, now
    parameterized by an :class:`MCMICROConfig` instead of individual
    keyword arguments.

    Directory layout after this call::

        {project_dir}/
          markers.csv
          params.yml
          registration/{scene_label}.ome.tif   (symlink, no file copy)

    Args:
        scene_label: e.g. ``'hBE-TMA1-1_sceneA1'``.
        composite_ometiff_path: path to the composite OME-TIFF (from
            ``step04_export_registered``).
        marker_map: ``{round_label: [marker_names]}``.
        project_dir: root directory for the MCMICRO project.
        config: :class:`MCMICROConfig` with segmenter/CellPose parameters.
        overwrite: if True, remove and recreate an existing project dir.

    Returns:
        Path to the created project directory.
    """
    project_dir = Path(project_dir)

    if project_dir.exists() and overwrite:
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    markers_content = build_markers_csv(marker_map)
    n_channels = len(markers_content.splitlines()) - 1
    (project_dir / "markers.csv").write_text(markers_content)
    logger.info(f"  markers.csv: {n_channels} channels")

    (project_dir / "params.yml").write_text(build_params_yml(config))

    reg_dir = project_dir / "registration"
    reg_dir.mkdir(exist_ok=True)
    link_target = reg_dir / f"{scene_label}.ome.tif"
    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()
    link_target.symlink_to(Path(composite_ometiff_path).resolve())

    logger.info(f"  registration/{scene_label}.ome.tif -> {composite_ometiff_path}")
    logger.info(f"  MCMICRO project ready: {project_dir}")
    return project_dir


def build_mcmicro_cmd(
    project_dir: PathLike,
    config: MCMICROConfig,
) -> str:
    """Build the MCMICRO ``nextflow run`` command string.

    Unchanged from the source script's ``build_mcmicro_cmd``, now reading
    ``mcmicro_bin`` / ``mcmicro_nf`` / ``profile`` / ``use_gpu`` /
    ``extra_args`` from ``config`` instead of individual arguments whose
    defaults hardcoded this lab's install paths.

    Args:
        project_dir: MCMICRO project directory (from
            :func:`prepare_mcmicro_project`).
        config: :class:`MCMICROConfig` supplying the wrapper/install paths.

    Returns:
        Command string suitable for ``subprocess.run(cmd, shell=True)``.
    """
    full_profile = f"{config.profile},gpu" if config.use_gpu else config.profile
    parts = [
        str(config.mcmicro_bin),
        "run",
        str(config.mcmicro_nf),
        f"--in {project_dir}",
        f"-profile {full_profile}",
    ]
    if config.extra_args:
        parts.append(config.extra_args)
    return " ".join(parts)


def run_mcmicro(cmd: str, timeout: int = 14400) -> int:
    """Execute an MCMICRO command via subprocess.

    Raises ``RuntimeError`` on non-zero exit so a notebook stops rather than
    silently proceeding with missing segmentation output. See the module
    docstring's HPC/SLURM note — this call is only safe from inside a SLURM
    allocation, never a login node.

    Args:
        cmd: command string from :func:`build_mcmicro_cmd`.
        timeout: seconds before killing the process (default 4 h).

    Returns:
        Return code (0 = success).
    """
    return _run_subprocess(cmd, "MCMICRO", cwd=None, timeout=timeout)


def load_mcmicro_results(project_dir: PathLike, prefer: str = "mccellpose") -> Optional[pd.DataFrame]:
    """Load the MCMICRO per-cell Spatial Feature Table (quantification CSV).

    Unchanged from the source script. MCMICRO writes CSVs to
    ``{project_dir}/quantification/{segmenter}--{scene_label}_cell.csv``.

    Args:
        project_dir: MCMICRO project directory.
        prefer: segmenter name substring to prefer when multiple CSVs exist.

    Returns:
        DataFrame with ``CellID``, ``X_centroid``, ``Y_centroid``, ``Area``,
        ``Eccentricity``, ``Solidity``, and one column per marker in
        ``markers.csv``. Returns None and logs a warning if no CSV is found
        (segmentation has not completed, or MCMICRO failed upstream of
        quantification without this call being reached — the caller should
        already have seen a raised ``RuntimeError`` from
        :func:`run_mcmicro` in that case).
    """
    project_dir = Path(project_dir)
    quant_dir = project_dir / "quantification"
    if not quant_dir.exists():
        logger.warning(f"  quantification/ not found in {project_dir} — run MCMICRO first")
        return None

    csvs = sorted(quant_dir.glob("*_cell.csv"))
    if not csvs:
        logger.warning(f"  No *_cell.csv found in {quant_dir}")
        return None

    preferred = [p for p in csvs if prefer in p.name]
    chosen = preferred[0] if preferred else csvs[0]

    df = pd.read_csv(chosen)
    logger.info(f"  Loaded {len(df):,} cells from {chosen.name}")
    return df


def bridge_mcmicro_to_pipex(
    mcmicro_df: pd.DataFrame,
    pipex_data_dir: PathLike,
    analysis_markers: List[str],
) -> str:
    """Convert an MCMICRO Spatial Feature Table to PIPEX ``cell_data.csv`` format.

    Ported unchanged from ``mcmicro.py::bridge_to_pipex``. This is a
    genuinely reusable structural bridge (fixed MCQuant column names ->
    fixed PIPEX column names), not manuscript-specific — the only
    caller-supplied content is ``analysis_markers``, the ordered marker
    column list to carry through.

    Column mapping:

    ======================  ================
    MCMICRO                 PIPEX
    ======================  ================
    CellID                  cell_id
    X_centroid              x
    Y_centroid              y
    Area                    size
    Eccentricity            eccentricity
    Solidity                solidity
    (not present)           memref = 0
    {R1_MUC1, R2_CK5, ...}  (unchanged)
    ======================  ================

    Written to ``{pipex_data_dir}/analysis/cell_data.csv`` — the exact path
    PIPEX's ``analysis.py`` reads from, so MCMICRO segmentation output can
    feed PIPEX's downstream clustering/neighborhood/export tooling.

    Args:
        mcmicro_df: DataFrame from :func:`load_mcmicro_results`.
        pipex_data_dir: PIPEX data directory (contains single-channel TIFFs).
        analysis_markers: ordered marker column names to include (e.g. from
            :func:`get_measure_markers` — excludes DAPI and blank rounds).

    Returns:
        Absolute path string to the written ``cell_data.csv``.

    Raises:
        ValueError: if neither an ``X_centroid``/``column_centroid``/``x``
            nor a ``Y_centroid``/``row_centroid``/``y`` column is present
            (MCQuant version mismatch — fail loudly rather than write a
            table with silently-wrong or missing coordinates).
    """
    df = mcmicro_df.copy()

    x_col = next((c for c in ["X_centroid", "column_centroid", "x"] if c in df.columns), None)
    y_col = next((c for c in ["Y_centroid", "row_centroid", "y"] if c in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError(
            f"Cannot find centroid columns in MCMICRO output. "
            f"Available columns: {list(df.columns[:20])}"
        )

    rename = {"CellID": "cell_id", x_col: "x", y_col: "y"}
    if "Area" in df.columns:
        rename["Area"] = "size"
    if "Eccentricity" in df.columns:
        rename["Eccentricity"] = "eccentricity"
    if "Solidity" in df.columns:
        rename["Solidity"] = "solidity"
    df = df.rename(columns=rename)

    if "memref" not in df.columns:
        df["memref"] = 0

    missing = [m for m in analysis_markers if m not in df.columns]
    if missing:
        logger.warning(
            f"  {len(missing)} markers missing from MCMICRO output: "
            + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        )

    core_cols = [
        c for c in ["cell_id", "size", "x", "y", "solidity", "eccentricity", "memref"]
        if c in df.columns
    ]
    marker_cols = [m for m in analysis_markers if m in df.columns]
    out_df = df[core_cols + marker_cols]

    out_dir = Path(pipex_data_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_data.csv"
    out_df.to_csv(out_path, index=False)

    logger.info(f"  Bridge -> {out_path}  ({len(out_df):,} cells x {len(marker_cols)} markers)")
    return str(out_path)


# ---------------------------------------------------------------------------
# PIPEX backend
# ---------------------------------------------------------------------------

def get_measure_markers(
    marker_map: Dict[str, List[str]],
    exclude_rounds: Optional[List[str]] = None,
) -> List[str]:
    """Return all non-DAPI marker names in ``{round}_{marker}`` format.

    Unchanged from the source script. These are the names PIPEX's
    ``-measure_markers`` / ``-analysis_markers`` flags expect, matching the
    filenames :func:`spatial_omics_toolkit.cycif.export_images.extract_pipex_channels`
    writes.

    Args:
        marker_map: ``{round_label: [ch0_name, ch1_name, ...]}``.
        exclude_rounds: rounds to skip entirely (e.g. a blank round).

    Returns:
        List of strings like ``['R1_MUC1', 'R1_MUC5AC', 'R2_CK5', ...]``.
    """
    exclude = set(exclude_rounds or [])
    markers = []
    for rnd, names in marker_map.items():
        if rnd in exclude:
            continue
        for name in names:
            if "dapi" not in name.lower():
                markers.append(f"{rnd}_{name}")
    return markers


def build_pipex_segmentation_cmd(
    data_dir: PathLike,
    config: PIPEXConfig,
    measure_markers: Optional[List[str]] = None,
) -> str:
    """Build the PIPEX ``segmentation.py`` command string.

    Unchanged from the source script's ``build_segmentation_cmd``, now
    reading ``nuclei_marker`` / ``nuclei_diameter`` / ``nuclei_expansion`` /
    ``membrane_markers`` / ``extra_flags`` from ``config``.

    Args:
        data_dir: path to the flat single-channel TIFF folder (from
            ``step04_export_registered``'s PIPEX output).
        config: :class:`PIPEXConfig` with segmentation parameters.
        measure_markers: channels to quantify; if None, uses
            ``config.measure_markers`` (and if that is also None, PIPEX's
            own default of all non-DAPI channels).

    Returns:
        Full command string (uses ``sys.executable`` for the correct Python
        environment).
    """
    parts = [
        sys.executable,
        "segmentation.py",
        f"-data={data_dir}",
        f"-nuclei_marker={config.nuclei_marker}",
        f"-nuclei_diameter={config.nuclei_diameter}",
        f"-nuclei_expansion={config.nuclei_expansion}",
    ]
    if config.membrane_markers:
        parts.append(f"-membrane_markers={','.join(config.membrane_markers)}")
    markers = measure_markers if measure_markers is not None else config.measure_markers
    if markers:
        parts.append(f"-measure_markers={','.join(markers)}")
    if config.extra_flags:
        parts.append(config.extra_flags)
    return " ".join(str(p) for p in parts)


def build_pipex_analysis_cmd(
    data_dir: PathLike,
    analysis_markers: Optional[List[str]] = None,
    log_norm: bool = True,
    leiden: bool = True,
    leiden_resolution: float = 0.5,
    kmeans: int = 0,
    extra_flags: str = "",
) -> str:
    """Build the PIPEX ``analysis.py`` command string.

    Unchanged from the source script's ``build_analysis_cmd``. PIPEX's
    ``analysis.py`` performs normalization/Leiden-or-K-means clustering,
    which is downstream of segmentation proper (this package's
    ``step07_classify_cells`` performs the toolkit's own per-core z-score
    typing instead of PIPEX's clustering) — this builder is provided as a
    reusable building block for a notebook that wants PIPEX's native
    clustering, but :func:`step05_segment_cells` does not call it by
    default.

    Args:
        data_dir: same path passed to segmentation (analysis reads from
            there).
        analysis_markers: markers to use for clustering; default = all
            measure_markers.
        log_norm: apply log1p normalization before clustering.
        leiden: run Leiden community detection.
        leiden_resolution: Leiden resolution parameter.
        kmeans: number of K-means clusters (0 = skip).
        extra_flags: any additional PIPEX flags.

    Returns:
        Full command string.
    """
    parts = [
        sys.executable,
        "analysis.py",
        f"-data={data_dir}",
        f'-log_norm={"yes" if log_norm else "no"}',
        f'-leiden={"yes" if leiden else "no"}',
    ]
    if leiden:
        parts.append(f"-leiden_resolution={leiden_resolution}")
    if kmeans:
        parts.append(f"-kmeans={kmeans}")
    if analysis_markers:
        parts.append(f"-analysis_markers={','.join(analysis_markers)}")
    if extra_flags:
        parts.append(extra_flags)
    return " ".join(str(p) for p in parts)


def run_pipex_cmd(cmd: str, pipex_home: PathLike, timeout: int = 7200) -> int:
    """Run a PIPEX command from the ``pipex_home`` directory.

    Raises ``RuntimeError`` on non-zero return code so a notebook stops
    rather than silently proceeding to the next step with missing
    segmentation output. See the module docstring's HPC/SLURM note.

    Args:
        cmd: command string from :func:`build_pipex_segmentation_cmd` /
            :func:`build_pipex_analysis_cmd`.
        pipex_home: directory containing ``segmentation.py`` /
            ``analysis.py``.
        timeout: seconds before killing the process (default 2 h).

    Returns:
        Return code (0 = success).
    """
    return _run_subprocess(cmd, "PIPEX", cwd=pipex_home, timeout=timeout)


def load_pipex_results(data_dir: PathLike) -> Optional[pd.DataFrame]:
    """Load the PIPEX per-cell quantification CSV.

    Unchanged from the source script. PIPEX writes quantification to
    ``{data_dir}/analysis/*_cell.csv``.

    Returns:
        DataFrame, or None (with a logged warning) if no result file is
        found — segmentation has not been run yet, or PIPEX failed (in
        which case the caller should already have seen a raised
        ``RuntimeError`` from :func:`run_pipex_cmd`).
    """
    data_dir = Path(data_dir)
    analysis_dir = data_dir / "analysis"

    candidates = sorted(analysis_dir.glob("*_cell.csv")) + sorted(analysis_dir.glob("*.csv"))
    candidates = [p for p in candidates if "QC" not in p.name]

    if not candidates:
        logger.warning(f"  No PIPEX result CSV in {analysis_dir}")
        return None

    df = pd.read_csv(candidates[0])
    logger.info(f"  Loaded {len(df):,} cells from {candidates[0].name}")
    return df


def load_all_pipex_results(pipex_root: PathLike, scene_labels: List[str]) -> Dict[str, pd.DataFrame]:
    """Load PIPEX results for multiple scenes into a dict keyed by scene label.

    Unchanged from the source script. Scenes with no result file are simply
    omitted from the returned dict (each individual load already logs a
    warning via :func:`load_pipex_results`).
    """
    results = {}
    for label in scene_labels:
        df = load_pipex_results(Path(pipex_root) / label)
        if df is not None:
            results[label] = df
    return results


# ---------------------------------------------------------------------------
# step05 orchestrator
# ---------------------------------------------------------------------------

def step05_segment_cells(
    backend: str,
    *,
    # --- MCMICRO backend ---
    scene_label: Optional[str] = None,
    composite_ometiff_path: Optional[PathLike] = None,
    marker_map: Optional[Dict[str, List[str]]] = None,
    project_dir: Optional[PathLike] = None,
    mcmicro_config: Optional[MCMICROConfig] = None,
    overwrite_project: bool = False,
    # --- PIPEX backend ---
    data_dir: Optional[PathLike] = None,
    pipex_config: Optional[PIPEXConfig] = None,
    measure_markers: Optional[List[str]] = None,
    # --- shared ---
    run: bool = True,
    load_results: bool = True,
    bridge_to_pipex_format: bool = False,
    pipex_bridge_dir: Optional[PathLike] = None,
    analysis_markers: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Segment cells from a registered/exported scene via MCMICRO or PIPEX.

    Ported and generalized from
    ``02_spatial_proteomics/03_analysis/02_scripts/mcmicro.py`` and
    ``pipex.py``, which drove these two backends as separate, mutually
    exclusive scripts. This is that orchestration promoted to the
    numbered-step calling convention, with the backend selected explicitly
    via ``backend`` rather than by which script the analyst happened to run.

    Command-builder pattern (unchanged, preserved intentionally — see module
    docstring): ``build_*_cmd`` -> ``subprocess.run`` -> ``RuntimeError`` on
    non-zero exit. A failed backend raises immediately; this function never
    catches that and silently retries with the other backend.

    **HPC/SLURM requirement**: if ``run=True`` (the default — matches both
    source scripts' always-run behavior), this function executes an actual
    container/CellPose or StarDist job via ``subprocess.run`` in this
    process. That is compute-heavy per this lab's HPC policy and must only
    be called from inside a SLURM allocation (``srun``/``sbatch``), never
    from a login-node Jupyter kernel. Pass ``run=False`` to build (and
    inspect / hand off to a separate ``sbatch`` script) the command without
    executing it — this function's job is to build the correct command and
    manage I/O, not to enforce SLURM itself.

    Args:
        backend: ``"mcmicro"`` or ``"pipex"`` (case-insensitive).

        scene_label, composite_ometiff_path, marker_map, project_dir,
        mcmicro_config: required when ``backend="mcmicro"``.
            ``composite_ometiff_path`` is the OME-TIFF from
            ``step04_export_registered``; ``project_dir`` is created/reused
            for the MCMICRO project (markers.csv, params.yml,
            registration/ symlink).
        overwrite_project: if True, remove and recreate an existing MCMICRO
            ``project_dir`` before preparing it.

        data_dir, pipex_config: required when ``backend="pipex"``.
            ``data_dir`` is the flat single-channel-TIFF folder from
            ``step04_export_registered``'s PIPEX output
            (``pipex_dir/{scene_label}``).
        measure_markers: overrides ``pipex_config.measure_markers`` for this
            call; if both are None and ``marker_map`` is supplied, computed
            via :func:`get_measure_markers`.

        run: execute the built command via subprocess (default True). If
            False, the command is built and returned but not run, and
            ``returncode``/``cells`` in the result are None.
        load_results: after a successful run, load the per-cell table via
            :func:`load_mcmicro_results` / :func:`load_pipex_results`.
        bridge_to_pipex_format: after a successful MCMICRO run (and
            ``load_results=True``), also call
            :func:`bridge_mcmicro_to_pipex` to write a PIPEX-format
            ``cell_data.csv``. Only valid with ``backend="mcmicro"`` — PIPEX
            output is already in its own native format.
        pipex_bridge_dir: PIPEX data directory to bridge into; required if
            ``bridge_to_pipex_format=True``.
        analysis_markers: ordered marker column list for the bridge step;
            required if ``bridge_to_pipex_format=True``. Defaults to
            ``measure_markers`` if not given and ``measure_markers`` was
            resolved.
        verbose: print a one-line summary of what ran.

    Returns:
        Dict shaped consistently across backends::

            {
                'backend': 'mcmicro' | 'pipex',
                'cmd': str,
                'returncode': int | None,
                'cells': pd.DataFrame | None,
                'n_cells': int | None,
                'project_dir' | 'data_dir': str,
                'bridge_path': str | None,
            }

    Raises:
        ValueError: unknown backend, missing required arguments for the
            selected backend, or ``bridge_to_pipex_format=True`` used with
            ``backend="pipex"``.
        RuntimeError: the underlying MCMICRO/PIPEX subprocess exited
            non-zero (fail loudly — never silently swallowed).
    """
    backend_norm = backend.lower().strip()

    if backend_norm == "mcmicro":
        missing = [
            name
            for name, val in [
                ("scene_label", scene_label),
                ("composite_ometiff_path", composite_ometiff_path),
                ("marker_map", marker_map),
                ("project_dir", project_dir),
                ("mcmicro_config", mcmicro_config),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(f"backend='mcmicro' requires: {', '.join(missing)}")
        if bridge_to_pipex_format and pipex_bridge_dir is None:
            raise ValueError("pipex_bridge_dir is required when bridge_to_pipex_format=True")

        prepare_mcmicro_project(
            scene_label,
            composite_ometiff_path,
            marker_map,
            project_dir,
            mcmicro_config,
            overwrite=overwrite_project,
        )
        cmd = build_mcmicro_cmd(project_dir, mcmicro_config)

        returncode: Optional[int] = None
        cells: Optional[pd.DataFrame] = None
        bridge_path: Optional[str] = None

        if run:
            returncode = run_mcmicro(cmd, timeout=mcmicro_config.timeout_seconds)
            if load_results:
                cells = load_mcmicro_results(project_dir, prefer=mcmicro_config.segmenter)
                if bridge_to_pipex_format and cells is not None:
                    markers_for_bridge = analysis_markers if analysis_markers is not None else (
                        measure_markers if measure_markers is not None else get_measure_markers(marker_map)
                    )
                    bridge_path = bridge_mcmicro_to_pipex(cells, pipex_bridge_dir, markers_for_bridge)

        if verbose:
            status = "not run (run=False)" if not run else f"returncode={returncode}"
            n = f"{len(cells):,} cells" if cells is not None else "no cells loaded"
            print(f"step05_segment_cells(backend='mcmicro', scene={scene_label}): {status}, {n}")

        return {
            "backend": "mcmicro",
            "cmd": cmd,
            "returncode": returncode,
            "cells": cells,
            "n_cells": None if cells is None else len(cells),
            "project_dir": str(project_dir),
            "bridge_path": bridge_path,
        }

    elif backend_norm == "pipex":
        missing = [
            name
            for name, val in [("data_dir", data_dir), ("pipex_config", pipex_config)]
            if val is None
        ]
        if missing:
            raise ValueError(f"backend='pipex' requires: {', '.join(missing)}")
        if bridge_to_pipex_format:
            raise ValueError(
                "bridge_to_pipex_format is only meaningful for backend='mcmicro' "
                "(PIPEX output is already in its own native format)."
            )

        markers = measure_markers
        if markers is None and pipex_config.measure_markers is None and marker_map is not None:
            markers = get_measure_markers(marker_map, exclude_rounds=list(pipex_config.exclude_rounds))

        cmd = build_pipex_segmentation_cmd(data_dir, pipex_config, measure_markers=markers)

        returncode = None
        cells = None
        if run:
            returncode = run_pipex_cmd(cmd, pipex_config.pipex_home, timeout=pipex_config.timeout_seconds)
            if load_results:
                cells = load_pipex_results(data_dir)

        if verbose:
            status = "not run (run=False)" if not run else f"returncode={returncode}"
            n = f"{len(cells):,} cells" if cells is not None else "no cells loaded"
            print(f"step05_segment_cells(backend='pipex', data_dir={data_dir}): {status}, {n}")

        return {
            "backend": "pipex",
            "cmd": cmd,
            "returncode": returncode,
            "cells": cells,
            "n_cells": None if cells is None else len(cells),
            "data_dir": str(data_dir),
            "bridge_path": None,
        }

    else:
        raise ValueError(f"Unknown backend {backend!r}; must be 'mcmicro' or 'pipex'.")
