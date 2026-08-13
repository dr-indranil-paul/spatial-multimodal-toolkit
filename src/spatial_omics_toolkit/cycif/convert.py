"""step02_convert_raw_images — CZI/ND2 -> OME-TIFF, MIP, pseudocolor composite.

Ported and generalized from
``02_spatial_proteomics/03_analysis/02_scripts/process_czi.py``,
``process_nd2.py``, and ``batch.py`` — fluorophore-aware colour mapping,
16-bit contrast stretching, OME-TIFF export, maximum-intensity projection
(MIP), and pseudocolored RGB composite generation for raw CZI/ND2
microscopy files.

What changed to generalize
---------------------------
- Input/output file extensions are now parameters
  (:class:`~spatial_omics_toolkit.cycif.config.ImageConversionConfig`
  ``.czi_extension`` / ``.nd2_extension``), not hardcoded ``*.czi`` /
  ``*.nd2`` globs.
- ``get_color_map``'s fluorophore-name keyword matching (DAPI/GFP/RFP/Cy5
  substrings -> RGB) is preserved exactly and is genuinely
  panel-agnostic (it is keyed on generic fluorophore/laser-line names,
  not this manuscript's marker names), but now accepts an optional
  ``custom_color_map`` checked first, for panels whose channel names
  don't hit the built-in keywords.
- The contrast-stretch percentiles (source hardcoded ``(2, 98)``) are now
  a parameter (``ImageConversionConfig.contrast_percentiles``); the
  stretch algorithm itself is unchanged.
- **Fail-loud change**: the source's per-file loop caught every exception
  and only ``print()``-ed a warning before silently continuing to the
  next file (``process_czis``/``process_nd2s``). That violates this lab's
  "never silently continue" policy — a corrupt or unreadable raw file
  must not just vanish from a multi-hundred-file batch run. Here,
  ``convert_czi_file``/``convert_nd2_file`` each return a structured
  ``{status, outputs, error}`` record per file, and
  ``convert_raw_images_inprocess`` raises ``RuntimeError`` summarizing
  every failure at the end unless ``strict=False`` is passed.
- **Dispatch, not in-process execution, by default**: contrast-stretching
  + MIP + composite generation over gigapixel CZI/ND2 stacks is
  compute-heavy, multi-minute work. Per this lab's HPC policy
  (``AGENTS.md`` #2) and this package's own ``DESIGN.md`` non-negotiable
  #10, that must never run on the login node or block a notebook kernel.
  ``step02_convert_raw_images`` therefore *builds* (and optionally
  submits) a SLURM ``sbatch`` script by default — mirroring how
  ``mcmicro.py::build_mcmicro_cmd``/``run_mcmicro`` build and dispatch a
  command rather than doing segmentation in-process — instead of calling
  the pixel-processing functions directly in the calling process.
  ``convert_raw_images_inprocess`` remains directly callable (and is what
  the dispatched SLURM job actually runs, via this module's ``__main__``
  CLI) for small test/dev inputs where blocking briefly is acceptable.

Raw CZI reading needs the optional ``aicspylibczi`` dependency; raw ND2
reading needs the optional ``nd2`` dependency. Both are imported lazily
inside the functions that need them (not at module import time) so this
module — including ``get_color_map``/``create_rgb_composite``/
``enhance_image``, and the SLURM job-building functions, none of which
touch raw pixels — imports cleanly without either installed. Install both
with ``pip install spatial-omics-toolkit[imaging]``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from skimage import exposure, img_as_ubyte

from spatial_omics_toolkit.cycif.config import (
    ImageConversionConfig,
    PREVAIL_BARRETTS_CONVERSION_CONFIG,
    PREVAIL_BARRETTS_SLURM_CONFIG,
    SlurmDispatchConfig,
)

logger = logging.getLogger("spatial_omics_toolkit.cycif.convert")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Color mapping — ported unchanged (generalized with an optional override)
# ═══════════════════════════════════════════════════════════════════════════

def get_color_map(
    channel_name: str,
    custom_map: Optional[Mapping[str, Tuple[float, float, float]]] = None,
) -> List[float]:
    """Map a fluorophore/channel name to an RGB triple in ``[0, 1]``.

    Ported unchanged from ``process_czi.py::get_color_map`` (keyword
    matching on generic fluorophore/laser-line substrings — DAPI, GFP/
    FITC/488, RFP/Cy3, Cy5/647 — not this manuscript's marker names, so
    this is genuinely panel-agnostic already). ``custom_map`` is a new,
    opt-in addition: entries are matched (case-insensitively, as a
    substring) before the built-in keyword table.

    Args:
        channel_name: channel name from CZI/ND2 metadata (e.g. ``"DAPI"``).
        custom_map: optional ``{name_substring: (r, g, b)}`` override.

    Returns:
        Three-element RGB list in ``[0, 1]``.
    """
    name = str(channel_name).upper()
    if custom_map:
        for key, rgb in custom_map.items():
            if str(key).upper() in name:
                return list(rgb)
    if any(x in name for x in ["DAPI", "HOECHST", "405"]):
        return [0.1, 0.4, 1.0]  # Blue
    if any(x in name for x in ["GFP", "FITC", "488", "GREEN"]):
        return [0.0, 1.0, 0.0]  # Green
    if any(x in name for x in ["RFP", "CY3", "561", "555", "RED"]):
        return [1.0, 0.5, 0.0]  # Orange / Red
    if any(x in name for x in ["CY5", "647", "FAR RED"]):
        return [1.0, 0.0, 0.0]  # Deep Red
    return [1.0, 1.0, 1.0]      # Default White


def create_rgb_composite(
    stack_2d: np.ndarray,
    channel_names: Optional[List[str]],
    custom_color_map: Optional[Mapping[str, Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Create a pseudocolored RGB uint8 composite from a 2D multi-channel stack.

    Ported unchanged from ``process_czi.py::create_rgb_composite``.

    Args:
        stack_2d: multi-channel image, shape ``(C, Y, X)``.
        channel_names: per-channel names for colour mapping; ``None``
            yields white for all channels.
        custom_color_map: forwarded to :func:`get_color_map`.

    Returns:
        RGB image of shape ``(Y, X, 3)``, dtype ``uint8``.
    """
    num_channels = stack_2d.shape[0]
    h, w = stack_2d.shape[1], stack_2d.shape[2]
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(num_channels):
        name = channel_names[i] if channel_names else f"Ch{i}"
        color = np.array(get_color_map(name, custom_color_map))
        channel_norm = stack_2d[i].astype(np.float32) / 65535.0
        for c in range(3):
            composite[:, :, c] += channel_norm * color[c]

    return img_as_ubyte(np.clip(composite, 0, 1))


def enhance_image(image_data: np.ndarray, percentiles: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """16-bit contrast stretching by percentile.

    Ported from ``process_czi.py::enhance_image``; the stretch algorithm
    is unchanged, only ``percentiles`` (source hardcoded ``(2, 98)``) is
    now a parameter.

    Args:
        image_data: input image (any dtype; ``uint16`` preferred).
        percentiles: ``(low, high)`` percentile bounds for the stretch.

    Returns:
        Contrast-stretched ``uint16`` image.
    """
    p_lo, p_hi = np.percentile(image_data, percentiles)
    enhanced = exposure.rescale_intensity(image_data, in_range=(p_lo, p_hi), out_range=(0, 65535))
    return enhanced.astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-file conversion — heavy pixel processing, lazy raw-format imports
# ═══════════════════════════════════════════════════════════════════════════

def convert_czi_file(
    czi_path,
    output_dir,
    config: ImageConversionConfig = PREVAIL_BARRETTS_CONVERSION_CONFIG,
) -> Dict:
    """Convert one CZI file to OME-TIFF (+ optional MIP + pseudocolor composite).

    Ported from the per-file body of ``process_czi.py::process_czis``.
    Requires the optional ``aicspylibczi`` dependency (lazy-imported);
    if missing, returns a structured "failed" record with an install hint
    rather than raising at import time or silently skipping the file.

    Args:
        czi_path: path to the ``.czi`` file.
        output_dir: directory to write outputs into (created by the caller).
        config: contrast/MIP/composite behavior.

    Returns:
        ``{"path", "format", "status" ("ok"|"failed"), "outputs", "error"}``.
        Never raises — failures are reported in the record so a caller
        processing many files can decide how to react (see
        :func:`convert_raw_images_inprocess`).
    """
    czi_path = Path(czi_path)
    output_dir = Path(output_dir)
    try:
        import aicspylibczi
    except ImportError as e:
        return {
            "path": str(czi_path), "format": "czi", "status": "failed", "outputs": [],
            "error": f"aicspylibczi not installed ({e}). Install with: "
                     f"pip install spatial-omics-toolkit[imaging]",
        }

    try:
        czi = aicspylibczi.CziFile(czi_path)
        stack, _ = czi.read_image()
        stack_squeezed = np.squeeze(stack)

        axes_labels = czi.dims.replace("X", "").replace("Y", "")
        root = ET.fromstring(czi.meta)
        raw_chan_names = [c.get("Name") for c in root.findall(".//Channels/Channel")]
        chan_names = [n or f"Ch{i}" for i, n in enumerate(raw_chan_names)]

        processed_stack = np.zeros_like(stack_squeezed, dtype=np.uint16)
        it = np.nditer(stack_squeezed[..., 0, 0], flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            processed_stack[idx] = enhance_image(stack_squeezed[idx], config.contrast_percentiles)

        outputs: List[str] = []
        tif_path = output_dir / f"{czi_path.stem}.tif"
        tifffile.imwrite(
            tif_path, processed_stack,
            metadata={"Axes": axes_labels, "Channel": {"Name": chan_names}}, imagej=True,
        )
        outputs.append(str(tif_path))

        if config.generate_mip and "Z" in axes_labels:
            z_axis = axes_labels.find("Z")
            mip_data = np.max(processed_stack, axis=z_axis)
            mip_axes = axes_labels.replace("Z", "")
            mip_path = output_dir / f"{czi_path.stem}_MIP.tif"
            tifffile.imwrite(
                mip_path, mip_data,
                metadata={"Axes": mip_axes, "Channel": {"Name": chan_names}}, imagej=True,
            )
            outputs.append(str(mip_path))

            if config.generate_composite and "C" in mip_axes:
                c_axis = mip_axes.find("C")
                mip_for_rgb = np.moveaxis(mip_data, c_axis, 0) if c_axis != 0 else mip_data
                rgb_comp = create_rgb_composite(mip_for_rgb, chan_names, config.custom_color_map)
                comp_path = output_dir / f"{czi_path.stem}_Composite.png"
                tifffile.imwrite(comp_path, rgb_comp)
                outputs.append(str(comp_path))

        return {"path": str(czi_path), "format": "czi", "status": "ok", "outputs": outputs, "error": None}
    except Exception as exc:
        return {"path": str(czi_path), "format": "czi", "status": "failed", "outputs": [], "error": str(exc)}


def _parse_nd2_channels(file_path: Path, nd2_file) -> List[str]:
    """Extract channel names from an ND2 file's text metadata.

    Ported unchanged from ``process_nd2.py::_parse_nd2_channels`` (adapted
    to take an already-open ``nd2_file`` handle rather than reopening the
    file a second time).
    """
    try:
        text_info = nd2_file.text_info
        channels: List[str] = []
        for line in (text_info or "").split("\n"):
            line = line.strip()
            if line.lower().startswith("channel"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    channels.append(parts[1].strip())
        return channels or [f"Ch{i}" for i in range(nd2_file.sizes.get("C", 1))]
    except Exception:
        return []


def convert_nd2_file(
    nd2_path,
    output_dir,
    config: ImageConversionConfig = PREVAIL_BARRETTS_CONVERSION_CONFIG,
) -> Dict:
    """Convert one ND2 file to OME-TIFF (+ optional MIP + pseudocolor composite).

    Ported from the per-file body of ``process_nd2.py::process_nd2s``.
    Requires the optional ``nd2`` dependency (lazy-imported); same
    structured-failure behavior as :func:`convert_czi_file`.

    Returns:
        ``{"path", "format", "status" ("ok"|"failed"), "outputs", "error"}``.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    try:
        import nd2
    except ImportError as e:
        return {
            "path": str(nd2_path), "format": "nd2", "status": "failed", "outputs": [],
            "error": f"nd2 not installed ({e}). Install with: pip install spatial-omics-toolkit[imaging]",
        }

    try:
        with nd2.ND2File(nd2_path) as nd2_file:
            stack = nd2_file.asarray()
            stack_squeezed = np.squeeze(stack)
            sizes = nd2_file.sizes
            ch_count = sizes.get("C", 1)
            chan_names = _parse_nd2_channels(nd2_path, nd2_file)
            if len(chan_names) != ch_count:
                chan_names = [f"Ch{i}" for i in range(ch_count)]
            raw_axes = nd2_file.dims
            squeezed_axes = "".join(a for a in raw_axes if stack_squeezed.shape[raw_axes.index(a)] > 1)

        while stack_squeezed.ndim < 5:
            stack_squeezed = stack_squeezed[np.newaxis, ...]
        t_dim, c_dim, z_dim = 0, 1, 2

        processed = np.zeros_like(stack_squeezed, dtype=np.uint16)
        for t in range(stack_squeezed.shape[t_dim]):
            for c in range(stack_squeezed.shape[c_dim]):
                for z in range(stack_squeezed.shape[z_dim]):
                    slice_2d = stack_squeezed[t, c, z]
                    if slice_2d.dtype == np.uint16:
                        processed[t, c, z] = enhance_image(slice_2d, config.contrast_percentiles)
                    else:
                        img = slice_2d.astype(np.float32)
                        rng = img.max() - img.min()
                        img = (img - img.min()) / (rng + 1e-8)
                        processed[t, c, z] = (img * 65535).astype(np.uint16)

        outputs: List[str] = []
        tif_path = output_dir / f"{nd2_path.stem}.tif"
        tifffile.imwrite(
            tif_path, processed,
            metadata={"Axes": squeezed_axes, "Channel": {"Name": chan_names}}, imagej=True,
        )
        outputs.append(str(tif_path))

        if config.generate_mip and processed.shape[z_dim] > 1:
            mip = np.max(processed, axis=z_dim)
            mip_axes = squeezed_axes.replace("Z", "")
            mip_path = output_dir / f"{nd2_path.stem}_MIP.tif"
            tifffile.imwrite(
                mip_path, mip,
                metadata={"Axes": mip_axes, "Channel": {"Name": chan_names}}, imagej=True,
            )
            outputs.append(str(mip_path))

            if config.generate_composite and processed.shape[c_dim] >= 2:
                t_idx = processed.shape[t_dim] - 1
                rgb_src = mip[t_idx] if mip.shape[t_dim] > 1 else mip[0]
                rgb_src = np.moveaxis(rgb_src, c_dim - 1, 0)
                rgb = create_rgb_composite(rgb_src, chan_names, config.custom_color_map)
                comp_path = output_dir / f"{nd2_path.stem}_Composite.png"
                tifffile.imwrite(comp_path, rgb)
                outputs.append(str(comp_path))

        return {"path": str(nd2_path), "format": "nd2", "status": "ok", "outputs": outputs, "error": None}
    except Exception as exc:
        return {"path": str(nd2_path), "format": "nd2", "status": "failed", "outputs": [], "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. In-process batch conversion (small inputs / the SLURM job's own body)
# ═══════════════════════════════════════════════════════════════════════════

def convert_raw_images_inprocess(
    input_folder,
    output_folder,
    config: ImageConversionConfig = PREVAIL_BARRETTS_CONVERSION_CONFIG,
    strict: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert every CZI/ND2 file in a folder, in the calling process.

    Ported and merged from ``process_czi.py::process_czis``,
    ``process_nd2.py::process_nd2s``, and ``batch.py::process_batch``.
    This is what a dispatched SLURM job actually runs (via this module's
    ``__main__`` CLI); call it directly only for inputs small/fast enough
    to not need SLURM (see the module docstring and
    :func:`step02_convert_raw_images`).

    Args:
        input_folder: directory containing ``.czi``/``.nd2`` files
            (extensions per ``config.czi_extension``/``config.nd2_extension``).
        output_folder: directory for outputs (created if missing).
        config: contrast/MIP/composite behavior.
        strict: if True (default), raise ``RuntimeError`` listing every
            failed file once the whole batch has been attempted, instead
            of the source scripts' behavior of printing a warning and
            silently moving on. If False, failures are only reported in
            the returned DataFrame.
        verbose: log a summary line via the module logger.

    Returns:
        DataFrame with columns ``path``, ``format``, ``status``,
        ``outputs``, ``error`` — one row per input file.

    Raises:
        FileNotFoundError: if ``input_folder`` does not exist.
        RuntimeError: if ``strict=True`` and any file failed.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    czi_files = sorted(input_path.glob(f"*{config.czi_extension}"))
    nd2_files = sorted(input_path.glob(f"*{config.nd2_extension}"))

    results: List[Dict] = []
    for f in czi_files:
        results.append(convert_czi_file(f, output_path, config))
    for f in nd2_files:
        results.append(convert_nd2_file(f, output_path, config))

    df = pd.DataFrame(results, columns=["path", "format", "status", "outputs", "error"])
    n_failed = int((df["status"] == "failed").sum()) if len(df) else 0
    if verbose:
        logger.info(f"Converted {len(df) - n_failed}/{len(df)} file(s) ({n_failed} failed)")

    if strict and n_failed:
        failed = df.loc[df["status"] == "failed", ["path", "error"]].to_dict("records")
        raise RuntimeError(f"{n_failed} file(s) failed conversion: {failed}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. SLURM job building & dispatch — the default path for step02
# ═══════════════════════════════════════════════════════════════════════════

def build_conversion_sbatch_script(
    input_folder,
    output_folder,
    slurm_config: SlurmDispatchConfig = PREVAIL_BARRETTS_SLURM_CONFIG,
    conversion_config: ImageConversionConfig = PREVAIL_BARRETTS_CONVERSION_CONFIG,
    log_dir: Optional[str] = None,
) -> str:
    """Build the ``sbatch`` script text for a step02 conversion job.

    New (not present in the source scripts, which were run directly or
    wrapped ad hoc by the analyst): dispatches
    :func:`convert_raw_images_inprocess` via this module's ``__main__``
    CLI, matching how ``mcmicro.py::build_mcmicro_cmd`` builds a command
    string rather than running the pipeline itself.

    Args:
        input_folder / output_folder: passed through to the CLI invocation.
        slurm_config: ``sbatch`` resource/partition/env settings.
        conversion_config: forwarded as ``--no-mip``/``--no-composite`` flags.
        log_dir: directory for ``%x_%j.out``/``.err``; defaults to
            ``{output_folder}/slurm_logs``.

    Returns:
        Complete ``sbatch`` script text (with shebang).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    log_path = Path(log_dir) if log_dir else output_folder / "slurm_logs"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_config.job_name}",
        f"#SBATCH --cpus-per-task={slurm_config.cpus_per_task}",
        f"#SBATCH --mem={slurm_config.mem_gb}G",
        f"#SBATCH --time={slurm_config.time_limit}",
        f"#SBATCH --output={log_path}/%x_%j.out",
        f"#SBATCH --error={log_path}/%x_%j.err",
    ]
    if slurm_config.partition:
        lines.append(f"#SBATCH --partition={slurm_config.partition}")
    lines.extend(slurm_config.extra_sbatch_lines)
    lines.append("")
    lines.append("set -euo pipefail")
    if slurm_config.conda_env:
        lines.append(f"conda activate {slurm_config.conda_env}")

    mip_flag = "" if conversion_config.generate_mip else " --no-mip"
    comp_flag = "" if conversion_config.generate_composite else " --no-composite"
    lines.append(
        f'python -m spatial_omics_toolkit.cycif.convert "{input_folder}" "{output_folder}"'
        f"{mip_flag}{comp_flag}"
    )
    return "\n".join(lines) + "\n"


def submit_conversion_job(script_path, sbatch_bin: str = "sbatch") -> int:
    """Submit a prebuilt sbatch script and return its job ID.

    Ported in spirit from ``mcmicro.py::run_mcmicro`` (subprocess ->
    ``RuntimeError`` on nonzero exit / unparseable output), adapted for
    ``sbatch``'s fire-and-forget semantics (it returns immediately with a
    job ID rather than blocking until the job finishes, unlike
    ``run_mcmicro``'s foreground ``nextflow run``).

    Args:
        script_path: path to a script from :func:`build_conversion_sbatch_script`.
        sbatch_bin: the ``sbatch`` executable to invoke.

    Returns:
        The submitted SLURM job ID.

    Raises:
        RuntimeError: if ``sbatch`` exits non-zero, or its stdout does not
            contain a parseable ``Submitted batch job <id>`` line.
    """
    result = subprocess.run([sbatch_bin, str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed (exit {result.returncode}) for {script_path}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        raise RuntimeError(f"sbatch exited 0 but job ID could not be parsed from stdout: {result.stdout!r}")
    return int(m.group(1))


# ═══════════════════════════════════════════════════════════════════════════
# 5. step02_convert_raw_images — the public step function
# ═══════════════════════════════════════════════════════════════════════════

def step02_convert_raw_images(
    input_folder,
    output_folder,
    dispatch: str = "slurm",
    conversion_config: ImageConversionConfig = PREVAIL_BARRETTS_CONVERSION_CONFIG,
    slurm_config: SlurmDispatchConfig = PREVAIL_BARRETTS_SLURM_CONFIG,
    jobs_dir: Optional[str] = None,
    submit: bool = False,
    strict: bool = True,
    verbose: bool = True,
):
    """Convert raw CZI/ND2 files to OME-TIFF (+ MIP + pseudocolor composite).

    Ported and generalized from ``process_czi.py``/``process_nd2.py``/
    ``batch.py`` — see the module docstring for the full list of changes,
    most importantly: **this does not process pixels in the calling
    process by default.** Gigapixel CZI/ND2 conversion is compute-heavy,
    multi-minute work that this lab's HPC policy and this package's own
    ``DESIGN.md`` (#10) require go to SLURM, never the login node or a
    notebook kernel. ``dispatch="slurm"`` (the default) builds an
    ``sbatch`` script (and submits it if ``submit=True``) instead of
    blocking; ``dispatch="inprocess"`` runs
    :func:`convert_raw_images_inprocess` directly and is intended only for
    small test/dev inputs.

    Args:
        input_folder: directory containing raw ``.czi``/``.nd2`` files.
        output_folder: directory for OME-TIFF/MIP/composite outputs and,
            for ``dispatch="slurm"``, the generated job script/logs
            (created if missing).
        dispatch: ``"slurm"`` (default, non-blocking — builds/submits a
            job) or ``"inprocess"`` (blocking — runs conversion directly;
            small inputs only).
        conversion_config: contrast/MIP/composite behavior.
        slurm_config: ``sbatch`` resource settings, used only when
            ``dispatch="slurm"``.
        jobs_dir: directory to write the ``.sbatch`` script into; defaults
            to ``{output_folder}/slurm_jobs``. Used only when
            ``dispatch="slurm"``.
        submit: if True and ``dispatch="slurm"``, actually call ``sbatch``
            and return the job ID; if False (default), only write the
            script and return its path so the caller/analyst can review
            or submit it themselves.
        strict: forwarded to :func:`convert_raw_images_inprocess` when
            ``dispatch="inprocess"``.
        verbose: log progress via the module logger.

    Returns:
        When ``dispatch="inprocess"``: the per-file results DataFrame from
        :func:`convert_raw_images_inprocess`.
        When ``dispatch="slurm"``: a dict — ``{"dispatch": "slurm",
        "submitted": bool, "script_path": str, "sbatch_cmd": str}``, plus
        ``"job_id": int`` if ``submitted`` is True.

    Raises:
        FileNotFoundError: if ``input_folder`` does not exist.
        ValueError: if ``dispatch`` is not ``"slurm"`` or ``"inprocess"``.
        RuntimeError: if ``dispatch="inprocess"`` and ``strict=True`` and
            any file failed; or if ``submit=True`` and ``sbatch`` fails.
    """
    if dispatch not in ("slurm", "inprocess"):
        raise ValueError(f"dispatch must be 'slurm' or 'inprocess', got {dispatch!r}")

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    if dispatch == "inprocess":
        return convert_raw_images_inprocess(
            input_path, output_path, config=conversion_config, strict=strict, verbose=verbose,
        )

    jobs_path = Path(jobs_dir) if jobs_dir else output_path / "slurm_jobs"
    jobs_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "slurm_logs"
    log_path.mkdir(parents=True, exist_ok=True)

    script_text = build_conversion_sbatch_script(
        input_path, output_path, slurm_config, conversion_config, log_dir=str(log_path),
    )
    script_path = jobs_path / f"{slurm_config.job_name}.sbatch"
    script_path.write_text(script_text)
    if verbose:
        logger.info(f"sbatch script written: {script_path}")

    if not submit:
        return {
            "dispatch": "slurm", "submitted": False,
            "script_path": str(script_path), "sbatch_cmd": f"sbatch {script_path}",
        }

    job_id = submit_conversion_job(script_path)
    if verbose:
        logger.info(f"Submitted SLURM job {job_id} ({script_path})")
    return {"dispatch": "slurm", "submitted": True, "script_path": str(script_path), "job_id": job_id}


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point — what the dispatched SLURM job actually runs
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert raw CZI/ND2 microscopy files to OME-TIFF (spatial_omics_toolkit step02).",
    )
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--no-mip", action="store_true", help="Skip maximum-intensity projection.")
    parser.add_argument("--no-composite", action="store_true", help="Skip pseudocolor RGB composite.")
    args = parser.parse_args()

    cli_config = ImageConversionConfig(
        generate_mip=not args.no_mip,
        generate_composite=not args.no_composite,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    convert_raw_images_inprocess(args.input_folder, args.output_folder, config=cli_config)
