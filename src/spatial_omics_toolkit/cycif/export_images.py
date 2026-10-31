"""step04_export_registered — export registered cycIF scenes + provenance.

Ported and generalized from
``02_spatial_proteomics/03_analysis/02_scripts/export.py`` (``save_scene_metadata``,
``build_composite_ometiff``, ``extract_pipex_channels``, ``save_border_masks``,
``qc_summary_table``, ``plot_dapi_qc``). Named ``export_images.py`` rather than
``export.py`` to avoid any risk of shadowing a stdlib/third-party ``export``
module on ``sys.path``.

What changed to generalize it
------------------------------
- All output filenames were string literals in the source script (e.g.
  ``f'{scene_label}_composite.ome.tif'``, ``f'{rnd}_valid_mask.tif'``). They
  are now templates on :class:`~spatial_omics_toolkit.cycif.config.ExportNamingConfig`,
  defaulting to the original strings.
- The five source functions (metadata / composite / PIPEX channels / masks /
  QC table) are preserved essentially unchanged as importable building
  blocks, plus one new orchestrating :func:`step04_export_registered` that
  calls all of them in the numbered-step calling convention this package
  uses elsewhere (``sot.step04_export_registered(...)``).
- No pixel-level computation was added or changed. As in the source script,
  this module is I/O and formatting only; registration itself (step03) is
  out of scope here.

Expected shape of ``result``
-----------------------------
This module consumes the same ``result`` dict produced by
``step03_register_rounds`` (SIFT+RANSAC multi-round registration, ported
from ``registration.py`` in the source project). The contract, unchanged
from the source script::

    {
        'pixel_size_um': float | None,
        'reference_shape': (H, W),
        'params': {...},                    # registration parameters, for provenance
        'round_order': ['R1', 'R2', ...],
        'rounds': {
            'R1': {
                'registered': bool,
                'status': 'reference' | 'ok' | '<failure reason>',
                'inliers': int, 'matches': int,
                'czi_path': str, 'czi_size_bytes': int,
                'mov_shape': (H, W) | None,
                'transform': np.ndarray (2x3) | None,
                'channels': np.ndarray (C, H, W) | None,
                'dapi': np.ndarray (H, W),
            },
            ...
        },
    }

Outputs (unchanged directory layout from the source script)
-------------------------------------------------------------
Per scene, under ``output_dir/{scene_label}/``:
    ``{round}_valid_mask.tif``        — per-round valid-pixel mask
    ``combined_valid_mask.tif``       — intersection across registered rounds
    ``{scene_label}_registration_metadata.json`` — full provenance record

Top-level ``output_dir/``:
    ``{scene_label}_composite.ome.tif`` — all rounds merged, calibrated OME-TIFF

``pipex_dir/{scene_label}/`` (only if ``write_pipex_channels=True``):
    ``{round}_{marker}.tif``          — one uint16 TIFF per channel
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from spatial_omics_toolkit.cycif.config import ExportNamingConfig

logger = logging.getLogger("spatial_omics_toolkit.cycif.export_images")

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------


def save_scene_metadata(
    result: Dict,
    scene_label: str,
    output_dir: PathLike,
    extra: Optional[Dict] = None,
    metadata_filename_template: str = "{scene_label}_registration_metadata.json",
) -> str:
    """Write a JSON provenance record for one registered scene.

    Records: UTC timestamp, software versions, all registration parameters,
    per-round CZI paths and file sizes, transform matrices, inlier rates,
    and pixel size. Unchanged from the source script's ``save_scene_metadata``
    other than the filename now coming from ``metadata_filename_template``
    instead of a hardcoded f-string.

    Args:
        result: dict from ``step03_register_rounds`` (see module docstring
            for the expected shape).
        scene_label: e.g. ``'hBE-TMA1-1_sceneA1'``.
        output_dir: directory to write the JSON into.
        extra: optional additional key-value pairs to include verbatim.
        metadata_filename_template: formatted with ``scene_label``.

    Returns:
        Absolute path to the written JSON file.
    """

    def _ver(name: str) -> str:
        try:
            m = __import__(name)
            return getattr(m, "__version__", "unknown")
        except ImportError:
            return "not installed"

    record = {
        "scene_label": scene_label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "opencv": _ver("cv2"),
            "numpy": _ver("numpy"),
            "tifffile": _ver("tifffile"),
            "scikit_image": _ver("skimage"),
            "aicspylibczi": _ver("aicspylibczi"),
        },
        "pixel_size_um": result.get("pixel_size_um"),
        "reference_shape": list(result["reference_shape"]),
        "params": result.get("params", {}),
        "rounds": {},
    }

    for rnd in result["round_order"]:
        rd = result["rounds"][rnd]
        T = rd.get("transform")
        entry = {
            "registered": rd["registered"],
            "status": rd["status"],
            "inliers": rd["inliers"],
            "matches": rd["matches"],
            "inlier_rate": (rd["inliers"] / rd["matches"]) if rd["matches"] else None,
            "czi_path": rd.get("czi_path", ""),
            "czi_size_bytes": rd.get("czi_size_bytes", 0),
            "mov_shape": list(rd["mov_shape"]) if rd.get("mov_shape") else None,
            "transform_2x3": T.tolist() if T is not None else None,
        }
        if T is not None:
            entry["scale"] = float(np.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2))
            entry["angle_deg"] = float(np.degrees(np.arctan2(T[0, 1], T[0, 0])))
            entry["shift_x_px"] = float(T[0, 2])
            entry["shift_y_px"] = float(T[1, 2])
        record["rounds"][rnd] = entry

    if extra:
        record.update(extra)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / metadata_filename_template.format(scene_label=scene_label)

    import json

    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    logger.info(f"  Metadata: {path.name}")
    return str(path)


# ---------------------------------------------------------------------------
# Composite OME-TIFF
# ---------------------------------------------------------------------------


def build_composite_ometiff(
    result: Dict,
    output_path: PathLike,
    marker_map: Dict[str, List[str]],
    pixel_size_um: Optional[float] = None,
) -> List[str]:
    """Merge all registered rounds into one OME-TIFF with correct metadata.

    Unchanged from the source script. Requires the ``imaging`` extra
    (``tifffile``); imported lazily so importing this module does not
    require it.

    Channel naming scheme: ``'{round}_{marker}'`` (e.g. ``'R1_DAPI'``,
    ``'R2_CK5'``). This avoids collisions when multiple rounds share the
    DAPI channel name.

    Args:
        result: dict from ``step03_register_rounds``.
        output_path: path ending in ``.ome.tif``.
        marker_map: ``{round_label: [ch0_name, ch1_name, ...]}``.
        pixel_size_um: physical pixel size in microns; overrides
            ``result['pixel_size_um']`` if given.

    Returns:
        Channel name strings in stack-index order.
    """
    import tifffile

    ps = pixel_size_um if pixel_size_um is not None else result.get("pixel_size_um")

    channels = []
    channel_names = []

    for rnd in result["round_order"]:
        rd = result["rounds"][rnd]
        if rd.get("channels") is None:
            logger.warning(
                f"  {rnd}: no data (registration failed) — excluded from composite"
            )
            continue
        n_ch = rd["channels"].shape[0]
        markers = marker_map.get(rnd, [f"Ch{c}" for c in range(n_ch)])
        for c in range(n_ch):
            channels.append(rd["channels"][c])
            marker = markers[c] if c < len(markers) else f"Ch{c}"
            channel_names.append(f"{rnd}_{marker}")

    if not channels:
        raise ValueError(
            "No registered channels available — cannot build composite OME-TIFF."
        )

    stack = np.stack(channels, axis=0).astype(np.uint16)  # (C, Y, X)

    metadata = {"axes": "CYX", "Channel": {"Name": channel_names}}
    if ps is not None:
        metadata["PhysicalSizeX"] = ps
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeY"] = ps
        metadata["PhysicalSizeYUnit"] = "µm"
    else:
        logger.warning(
            "  No pixel size available — OME-TIFF will lack spatial calibration"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        stack,
        photometric="minisblack",
        metadata=metadata,
        compression="zlib",
    )

    size_gb = os.path.getsize(str(output_path)) / 1024**3
    logger.info(
        f"  Composite: {output_path.name}  ({stack.shape[0]} ch, {size_gb:.2f} GB)"
    )
    return channel_names


# ---------------------------------------------------------------------------
# PIPEX single-channel TIFFs
# ---------------------------------------------------------------------------


def extract_pipex_channels(
    result: Dict,
    output_dir: PathLike,
    marker_map: Dict[str, List[str]],
    overwrite: bool = True,
    channel_filename_template: str = "{round}_{marker}.tif",
) -> List[str]:
    """Write one uint16 TIFF per channel for PIPEX segmentation and analysis.

    Unchanged from the source script other than the filename template
    (was a hardcoded ``f'{name}.tif'`` where ``name = f'{rnd}_{marker}'``).

    Args:
        result: dict from ``step03_register_rounds``.
        output_dir: flat folder; one TIFF per channel.
        marker_map: ``{round_label: [marker_names]}``.
        overwrite: always rewrite existing files (default True — prevents
            stale output from a previous run).
        channel_filename_template: formatted with ``round`` and ``marker``.

    Returns:
        List of written file paths.
    """
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for rnd in result["round_order"]:
        rd = result["rounds"].get(rnd)
        if rd is None or rd.get("channels") is None:
            logger.warning(f"  {rnd}: no data — PIPEX channels skipped")
            continue
        n_ch = rd["channels"].shape[0]
        markers = marker_map.get(rnd, [f"Ch{c}" for c in range(n_ch)])
        for c in range(n_ch):
            marker = markers[c] if c < len(markers) else f"Ch{c}"
            p = output_dir / channel_filename_template.format(round=rnd, marker=marker)
            if overwrite or not p.exists():
                tifffile.imwrite(str(p), rd["channels"][c].astype(np.uint16))
                written.append(str(p))

    total_mb = sum(os.path.getsize(p) for p in written) / 1024**2
    logger.info(f"  PIPEX: {len(written)} TIFFs -> {output_dir}  ({total_mb:.0f} MB)")
    return written


# ---------------------------------------------------------------------------
# Border masks
# ---------------------------------------------------------------------------


def save_border_masks(
    result: Dict,
    output_dir: PathLike,
    valid_mask_filename_template: str = "{round}_valid_mask.tif",
    combined_mask_filename: str = "combined_valid_mask.tif",
) -> Dict[str, np.ndarray]:
    """Generate and save per-round valid-pixel masks.

    Unchanged algorithm from the source script. ``warpAffine`` pads
    out-of-bounds regions with 0 (``BORDER_CONSTANT``); these zero-filled
    borders are artefacts, not real background signal. Without a mask,
    downstream segmentation may treat the border as tissue and produce
    truncated cells at the edge.

    Method: warp a uniform ones-image (same shape as the moving image)
    through the same transform. Pixels with warped value > 0.5 received
    real data.

    Args:
        result: dict from ``step03_register_rounds``.
        output_dir: directory to write masks into.
        valid_mask_filename_template: formatted with ``round``.
        combined_mask_filename: literal filename (no formatting fields —
            it names the intersection mask, not a per-round artefact).

    Returns:
        Dict mapping round_label -> binary mask array (0/1 uint8).
    """
    import cv2
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_shape = result["reference_shape"]
    ref_round = result["round_order"][0]
    masks = {}

    for rnd in result["round_order"]:
        rd = result["rounds"][rnd]

        if rnd == ref_round or rd["status"] == "reference":
            mask = np.ones(ref_shape, dtype=np.uint8)
        elif rd.get("transform") is None:
            mask = np.zeros(ref_shape, dtype=np.uint8)
        else:
            mov_shape = rd.get("mov_shape", ref_shape)
            ones = np.ones(mov_shape, dtype=np.float32)
            warped = cv2.warpAffine(
                ones,
                rd["transform"],
                (ref_shape[1], ref_shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            mask = (warped > 0.5).astype(np.uint8)

        tifffile.imwrite(
            str(output_dir / valid_mask_filename_template.format(round=rnd)), mask * 255
        )
        masks[rnd] = mask

    combined = np.ones(ref_shape, dtype=np.uint8)
    for rnd, mask in masks.items():
        if result["rounds"][rnd]["registered"]:
            combined &= mask
    tifffile.imwrite(str(output_dir / combined_mask_filename), combined * 255)

    parts = ", ".join(f"{r}={masks[r].mean()*100:.1f}%" for r in result["round_order"])
    logger.info(f"  Masks: {parts}")
    logger.info(f"  Combined valid area: {combined.mean()*100:.1f}% of mosaic")
    return masks


# ---------------------------------------------------------------------------
# QC display helpers (return DataFrames / figures for notebook display)
# ---------------------------------------------------------------------------


def qc_summary_table(result: Dict) -> "pd.DataFrame":
    """Return a DataFrame summarizing per-round registration quality.

    Unchanged from the source script.
    """
    import pandas as pd

    rows = []
    for rnd in result["round_order"]:
        rd = result["rounds"][rnd]
        T = rd.get("transform")

        if rd["status"] == "reference":
            rows.append(
                {
                    "Round": rnd,
                    "Status": "REFERENCE",
                    "Inliers": "—",
                    "Matches": "—",
                    "Rate": "—",
                    "Angle (deg)": "—",
                    "Scale": "—",
                    "Shift X (px)": "—",
                    "Shift Y (px)": "—",
                }
            )
        elif rd["registered"] and T is not None:
            scale = float(np.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2))
            angle = float(np.degrees(np.arctan2(T[0, 1], T[0, 0])))
            rate = 100.0 * rd["inliers"] / rd["matches"] if rd["matches"] else 0
            rows.append(
                {
                    "Round": rnd,
                    "Status": "OK",
                    "Inliers": rd["inliers"],
                    "Matches": rd["matches"],
                    "Rate": f"{rate:.1f}%",
                    "Angle (deg)": f"{angle:+.2f}",
                    "Scale": f"{scale:.4f}",
                    "Shift X (px)": f"{T[0,2]:+.1f}",
                    "Shift Y (px)": f"{T[1,2]:+.1f}",
                }
            )
        else:
            rows.append(
                {
                    "Round": rnd,
                    "Status": f'FAILED: {rd["status"]}',
                    "Inliers": rd["inliers"],
                    "Matches": rd["matches"],
                    "Rate": "—",
                    "Angle (deg)": "—",
                    "Scale": "—",
                    "Shift X (px)": "—",
                    "Shift Y (px)": "—",
                }
            )

    return pd.DataFrame(rows)


def plot_dapi_qc(
    result: Dict,
    scene_label: str,
    save_dir: Optional[PathLike] = None,
    downsample: int = 4,
) -> "plt.Figure":
    """Three-panel DAPI registration QC figure.

    Unchanged from the source script. This is a diagnostic QC figure, not a
    publication figure, so it does not follow this lab's publication plot
    defaults (5pt font / 0.5pt lines / no title) — those apply to figures an
    analyst builds for a manuscript, not to library-internal QC diagnostics.

    Panels: (1) reference DAPI, (2) cyan/red alignment overlay for the round
    with the largest Y-shift (the hardest case), (3) absolute-difference
    heatmap.

    Args:
        result: dict from ``step03_register_rounds``.
        scene_label: label for the figure title and saved filename.
        save_dir: if set, saves a PNG to this directory.
        downsample: pixel stride for display (default 4 — fast, avoids OOM
            on gigapixel scenes).

    Returns:
        matplotlib Figure, or None if there are no registered rounds to plot.
    """
    import matplotlib.pyplot as plt

    ref_round = result["round_order"][0]
    ref_dapi = result["rounds"][ref_round]["dapi"]

    candidates = [
        (rnd, abs(float(result["rounds"][rnd]["transform"][1, 2])))
        for rnd in result["round_order"][1:]
        if result["rounds"][rnd].get("registered")
        and result["rounds"][rnd].get("transform") is not None
    ]
    if not candidates:
        logger.warning(f"{scene_label}: no registered rounds to plot")
        return None

    worst_rnd = max(candidates, key=lambda x: x[1])[0]
    warped = result["rounds"][worst_rnd]["dapi"]

    def _norm(x: np.ndarray) -> np.ndarray:
        pos = x[x > 0]
        if pos.size == 0:
            return np.zeros(x.shape, dtype=float)
        p1, p99 = np.percentile(pos, (1, 99))
        return np.clip((x.astype(float) - p1) / (p99 - p1 + 1e-8), 0, 1)

    ds = downsample
    r = _norm(ref_dapi[::ds, ::ds])
    w = _norm(warped[::ds, ::ds])

    overlay = np.zeros((*r.shape, 3), dtype=float)
    overlay[..., 0] = w
    overlay[..., 1] = r
    overlay[..., 2] = r
    overlay = np.clip(overlay, 0, 1)

    diff = np.abs(r - w)
    vmax_diff = max(0.05, float(np.percentile(diff, 99)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(r, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Reference DAPI ({ref_round})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Registration overlay — {worst_rnd} -> {ref_round}\n"
        f"cyan = {ref_round} (ref)  |  red = {worst_rnd} (registered)  |  white = aligned",
        fontsize=10,
    )
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=vmax_diff)
    axes[2].set_title(f"|Difference|  (display max = {vmax_diff:.2f})", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    rd = result["rounds"][worst_rnd]
    T = rd["transform"]
    rate = f"{100*rd['inliers']/rd['matches']:.1f}%" if rd["matches"] else "n/a"
    scale = float(np.sqrt(T[0, 0] ** 2 + T[0, 1] ** 2))
    angle = float(np.degrees(np.arctan2(T[0, 1], T[0, 0])))
    annotation = (
        f"{worst_rnd}: {rd['inliers']}/{rd['matches']} inliers ({rate})  "
        f"angle={angle:+.2f} deg  scale={scale:.4f}  "
        f"shift=({T[0,2]:+.1f}, {T[1,2]:+.1f}) px"
    )
    fig.text(0.5, -0.02, annotation, ha="center", fontsize=9, color="#555555")

    plt.suptitle(f"{scene_label} — DAPI Registration QC", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(save_dir) / f"{scene_label}_dapi_qc.png", dpi=150, bbox_inches="tight"
        )
        logger.info(f"  QC plot: {scene_label}_dapi_qc.png")

    return fig


# ---------------------------------------------------------------------------
# step04 orchestrator
# ---------------------------------------------------------------------------


def step04_export_registered(
    result: Dict,
    scene_label: str,
    output_dir: PathLike,
    marker_map: Dict[str, List[str]],
    pixel_size_um: Optional[float] = None,
    pipex_dir: Optional[PathLike] = None,
    naming: ExportNamingConfig = ExportNamingConfig(),
    write_pipex_channels: bool = True,
    write_border_masks: bool = True,
    write_qc_plot: bool = True,
    qc_plot_downsample: int = 4,
    overwrite_pipex_channels: bool = True,
    extra_provenance: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Export a registered scene: composite OME-TIFF, PIPEX TIFFs, masks, provenance.

    Ported and generalized from
    ``02_spatial_proteomics/03_analysis/02_scripts/export.py``, which called
    ``save_scene_metadata`` / ``build_composite_ometiff`` /
    ``extract_pipex_channels`` / ``save_border_masks`` sequentially in a
    per-scene loop at the bottom of the registration driver script. This
    function is that orchestration, promoted to the numbered-step calling
    convention (``sot.step04_export_registered(...)``) and made independent
    of any hardcoded filename convention via ``naming``
    (:class:`~spatial_omics_toolkit.cycif.config.ExportNamingConfig`).

    No pixel-level computation was added or changed — see the individual
    ported functions in this module (``build_composite_ometiff``,
    ``extract_pipex_channels``, ``save_border_masks``, ``save_scene_metadata``)
    for the unchanged per-artefact logic.

    Requires the ``imaging`` extra (``tifffile``, ``opencv-python``) for the
    composite/mask/PIPEX-channel writers; the metadata JSON and QC table
    have no imaging dependency.

    Args:
        result: dict from ``step03_register_rounds`` (see module docstring
            for the expected shape).
        scene_label: e.g. ``'hBE-TMA1-1_sceneA1'``.
        output_dir: directory that receives the composite OME-TIFF at its
            top level and a ``{scene_label}/`` subfolder for masks +
            metadata (matches the source script's layout exactly).
        marker_map: ``{round_label: [ch0_name, ch1_name, ...]}`` — channel
            order must match the registration result's channel order.
        pixel_size_um: physical pixel size in microns; overrides
            ``result['pixel_size_um']`` if given.
        pipex_dir: root directory for PIPEX single-channel TIFFs; the
            scene's channels are written under ``pipex_dir/{scene_label}/``.
            Required if ``write_pipex_channels`` is True.
        naming: filename templates; see
            :class:`~spatial_omics_toolkit.cycif.config.ExportNamingConfig`.
        write_pipex_channels: write one TIFF per channel for PIPEX/step05.
        write_border_masks: write per-round + combined valid-pixel masks.
        write_qc_plot: also render and save the 3-panel DAPI QC figure.
        qc_plot_downsample: pixel stride for the QC figure (default 4).
        overwrite_pipex_channels: always rewrite existing PIPEX TIFFs
            (default True — prevents stale output from a previous run).
        extra_provenance: additional key-value pairs merged into the
            provenance JSON (e.g. upstream pipeline version, operator name).
        verbose: print a one-line summary of what was written.

    Returns:
        Dict with the paths and objects produced::

            {
                'scene_label': str,
                'composite_path': str,
                'channel_names': [str, ...],
                'metadata_path': str,
                'mask_paths': {round: path} | None,
                'pipex_channel_paths': [path, ...] | None,
                'qc_table': pd.DataFrame,
                'qc_plot_path': str | None,
            }

    Raises:
        ValueError: if ``write_pipex_channels`` is True but ``pipex_dir`` is
            not given, or if no registered channels are available at all.
    """
    if write_pipex_channels and pipex_dir is None:
        raise ValueError("pipex_dir is required when write_pipex_channels=True")

    output_dir = Path(output_dir)
    scene_dir = output_dir / scene_label
    scene_dir.mkdir(parents=True, exist_ok=True)

    composite_path = output_dir / naming.composite_filename_template.format(
        scene_label=scene_label
    )
    channel_names = build_composite_ometiff(
        result, composite_path, marker_map, pixel_size_um=pixel_size_um
    )

    metadata_path = save_scene_metadata(
        result,
        scene_label,
        scene_dir,
        extra=extra_provenance,
        metadata_filename_template=naming.metadata_filename_template,
    )

    mask_paths: Optional[Dict[str, str]] = None
    if write_border_masks:
        masks = save_border_masks(
            result,
            scene_dir,
            valid_mask_filename_template=naming.valid_mask_filename_template,
            combined_mask_filename=naming.combined_mask_filename,
        )
        mask_paths = {
            rnd: str(scene_dir / naming.valid_mask_filename_template.format(round=rnd))
            for rnd in masks
        }
        mask_paths["combined"] = str(scene_dir / naming.combined_mask_filename)

    pipex_channel_paths: Optional[List[str]] = None
    if write_pipex_channels:
        pipex_scene_dir = Path(pipex_dir) / scene_label
        pipex_channel_paths = extract_pipex_channels(
            result,
            pipex_scene_dir,
            marker_map,
            overwrite=overwrite_pipex_channels,
            channel_filename_template=naming.channel_filename_template,
        )

    qc_table = qc_summary_table(result)

    qc_plot_path: Optional[str] = None
    if write_qc_plot:
        fig = plot_dapi_qc(
            result, scene_label, save_dir=scene_dir, downsample=qc_plot_downsample
        )
        if fig is not None:
            qc_plot_path = str(scene_dir / f"{scene_label}_dapi_qc.png")
            import matplotlib.pyplot as plt

            plt.close(fig)

    if verbose:
        n_masks = len(mask_paths) - 1 if mask_paths else 0
        n_pipex = len(pipex_channel_paths) if pipex_channel_paths else 0
        print(
            f"step04_export_registered({scene_label}): "
            f"composite={len(channel_names)} ch, masks={n_masks} rounds, "
            f"pipex_tiffs={n_pipex}, metadata={Path(metadata_path).name}"
        )

    return {
        "scene_label": scene_label,
        "composite_path": str(composite_path),
        "channel_names": channel_names,
        "metadata_path": metadata_path,
        "mask_paths": mask_paths,
        "pipex_channel_paths": pipex_channel_paths,
        "qc_table": qc_table,
        "qc_plot_path": qc_plot_path,
    }
