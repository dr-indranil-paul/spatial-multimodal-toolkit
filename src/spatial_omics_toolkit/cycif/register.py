"""step03_register_rounds — SIFT+RANSAC multi-round cyclic-IF image registration.

Ported and generalized from
``02_spatial_proteomics/03_analysis/02_scripts/registration.py`` (rated the
best-engineered file in that source project: physically-motivated sanity
bounds on the fitted transform, explicit failure states, provenance-ready
output). Every registration *algorithm* decision — SIFT over SURF, FLANN
matching with Lowe's ratio test, ``estimateAffinePartial2D`` (4-DOF
similarity: translation + rotation + uniform scale, no shear), the RANSAC
parameters, and the sanity-bound thresholds on the fitted transform — is
preserved exactly, unchanged from the validated source. What changed:

- ``parse_czi_filename`` took a hardcoded regex for one manuscript's
  ``hBE-TMA`` filename convention. It now takes that regex as a ``pattern``
  parameter (default =
  :data:`spatial_omics_toolkit.cycif.config.PREVAIL_BARRETTS_CZI_FILENAME_PATTERN`,
  the exact pattern from the source file).
- ``_round_sort_key`` took a hardcoded ``R{n}(Q?)`` round-label regex. The
  numeric-aware sort logic itself (the actual bug fix: alphabetical sort
  breaks for R10+ and mis-orders R5 vs. R5Q) is unchanged; the regex is now a
  ``round_label_pattern`` parameter (default =
  :data:`spatial_omics_toolkit.cycif.config.PREVAIL_BARRETTS_ROUND_LABEL_PATTERN`).
- All SIFT/FLANN/RANSAC/sanity-bound numeric parameters are now bundled into
  :class:`spatial_omics_toolkit.cycif.config.RegistrationConfig` instead of
  being scattered default arguments, matching this package's config-dataclass
  convention — the numeric defaults inside that dataclass are copied verbatim
  from the source file, not re-derived.
- ``validate_registration`` (the source file's comparison against a specific
  collaborator's already-registered reference TIFFs, used one time during
  method validation) was intentionally **not** ported — it is a one-off QA
  script against another party's output naming convention, not part of the
  reusable registration algorithm, and is out of scope for a numbered
  notebook step.

Compute policy (this lab's binding HPC rule)
---------------------------------------------
This function does the real SIFT+RANSAC registration and warping when
called — it is a full port, not a stub. Registration of full-resolution,
multi-round, multi-gigapixel CZI mosaics is exactly the kind of
compute-heavy, memory-heavy (the docstring on the ``rounds`` result below
notes ~4-6 GB per scene at full resolution) workload this lab's HPC policy
requires running under a SLURM allocation, never on a login node:

    srun --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash

This package has no way to detect or enforce that a notebook cell is running
inside a SLURM allocation — it can only document the requirement. A notebook
user calling ``step03_register_rounds`` against real full-resolution
multi-round CZI data on a login node is violating lab policy, not a bug in
this function. For quick iteration/testing on a login node, pass
``scale_factor < 1.0`` to work on a downsampled mosaic, or call
:func:`register_round_pair` directly on small arrays (as this module's unit
tests do).
"""

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np

from spatial_omics_toolkit.cycif.config import (
    PREVAIL_BARRETTS_CZI_FILENAME_PATTERN,
    PREVAIL_BARRETTS_ROUND_LABEL_PATTERN,
    RegistrationConfig,
)


# ---------------------------------------------------------------------------
# Round ordering
# ---------------------------------------------------------------------------

def _round_sort_key(
    label: str,
    round_label_pattern: str = PREVAIL_BARRETTS_ROUND_LABEL_PATTERN,
) -> Tuple[int, int]:
    """Biological sort key: R1 < R2 < ... < R5 < R5Q (numeric-aware, not alphabetical).

    ``round_label_pattern`` must have exactly two capture groups: the numeric
    round number, and an optional non-empty suffix (e.g. "Q") that sorts
    after the un-suffixed round. Plain alphabetical sort fails for R10+
    (sorts before R2) and mis-orders R5Q relative to R5 in some locales;
    labels that do not match the pattern sort last (key ``(999, 0)``) so an
    unparseable label degrades to "put it at the end", not a crash, while
    still being visible in the returned round order for the caller to notice.
    """
    m = re.match(round_label_pattern, label)
    if m:
        return (int(m.group(1)), 1 if m.group(2) else 0)
    return (999, 0)


# ---------------------------------------------------------------------------
# CZI I/O
# ---------------------------------------------------------------------------

def get_czi_pixel_size(czi_path: str) -> Optional[float]:
    """Extract physical pixel size in micrometers from CZI XML metadata.

    Returns None if the value cannot be read — caller should warn and omit
    spatial calibration from outputs rather than silently using a wrong value.
    Requires the optional ``aicspylibczi`` dependency (``imaging`` extra).
    """
    try:
        import aicspylibczi
        import xml.etree.ElementTree as ET
        czi = aicspylibczi.CziFile(czi_path)
        meta = czi.meta
        root = ET.fromstring(meta) if isinstance(meta, str) else meta
        for dist in root.iter('Distance'):
            if dist.get('Id') == 'X':
                val = dist.find('Value')
                if val is not None:
                    return float(val.text) * 1e6   # meters -> micrometers
    except Exception:
        pass
    return None


def _get_czi_num_channels(czi_path: str, fallback: int = 5) -> int:
    """Auto-detect channel count from CZI metadata. Falls back to ``fallback``."""
    try:
        import aicspylibczi
        czi = aicspylibczi.CziFile(czi_path)
        dims = czi.get_dims_shape()
        for d in dims:
            if 'C' in d:
                _start, count = d['C']
                return count
    except Exception:
        pass
    return fallback


def read_czi_channel(czi_path: str, channel: int = 0, scale_factor: float = 1.0) -> np.ndarray:
    """Read one stitched channel from a per-scene mosaic CZI. Returns 2D uint16."""
    import aicspylibczi
    czi = aicspylibczi.CziFile(czi_path)
    img = czi.read_mosaic(C=channel, scale_factor=scale_factor)
    return np.squeeze(img)


def read_czi_all_channels(
    czi_path: str,
    num_channels: Optional[int] = None,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Read all channels from a per-scene mosaic CZI. Returns (C, Y, X) uint16.

    Channel count is auto-detected from metadata when ``num_channels`` is
    None. A channel that fails to read is filled with zeros and a warning is
    printed — it is never silently dropped, so downstream channel indexing
    stays consistent and the gap is visible to the caller.
    """
    import aicspylibczi
    czi = aicspylibczi.CziFile(czi_path)
    if num_channels is None:
        num_channels = _get_czi_num_channels(czi_path)

    channels = []
    for c in range(num_channels):
        try:
            img = czi.read_mosaic(C=c, scale_factor=scale_factor)
            channels.append(np.squeeze(img))
        except Exception as e:
            print(f'  Channel {c}: read failed ({e}) — filling with zeros')
            placeholder = np.zeros_like(channels[0]) if channels else np.zeros((1, 1), dtype=np.uint16)
            channels.append(placeholder)

    return np.stack(channels, axis=0)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile contrast-stretch to uint8 for feature detection.

    Uses 1st-99.5th percentile of non-zero pixels. Safe for blank/near-blank
    channels (e.g. autofluorescence-only "quench" rounds): falls back to a
    plain min-max stretch (or all-zero if the image has no dynamic range at
    all) rather than dividing by a near-zero percentile spread.
    """
    img_f = img.astype(np.float32)
    nonzero = img_f[img_f > 0]

    if nonzero.size < 100:
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx <= mn:
            return np.zeros_like(img_f, dtype=np.uint8)
        return ((img_f - mn) / (mx - mn) * 255).astype(np.uint8)

    p1, p99 = np.percentile(nonzero, (1.0, 99.5))
    clipped = np.clip(img_f, p1, p99)
    return ((clipped - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Transform validation
# ---------------------------------------------------------------------------

def _validate_transform(
    transform: np.ndarray,
    image_shape: Tuple[int, int],
    max_angle_deg: float = 5.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    max_shift_fraction: float = 0.5,
) -> Tuple[bool, str]:
    """Reject transforms that are physically implausible for same-slide TMA cycIF.

    Expected range for AxioScan re-staining cycles: scale ~= 1.0, angle ~= 0deg,
    shifts up to a few hundred pixels (tissue slippage between rounds).
    Anything outside these bounds indicates a registration failure, not a
    legitimate tissue deformation.

    Returns (is_valid, failure_reason).
    """
    scale = float(np.sqrt(transform[0, 0] ** 2 + transform[0, 1] ** 2))
    angle = float(np.degrees(np.arctan2(transform[0, 1], transform[0, 0])))
    tx, ty = float(transform[0, 2]), float(transform[1, 2])
    diag = float(np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2))

    if not (scale_range[0] <= scale <= scale_range[1]):
        return False, (f'scale={scale:.4f} outside [{scale_range[0]:.2f}, {scale_range[1]:.2f}]')
    if abs(angle) > max_angle_deg:
        return False, f'angle={angle:.2f} deg exceeds +/-{max_angle_deg} deg'
    max_shift = max_shift_fraction * diag
    if abs(tx) > max_shift or abs(ty) > max_shift:
        return False, (
            f'shift=({tx:+.1f}, {ty:+.1f}) px exceeds '
            f'{max_shift_fraction:.0%} of diagonal ({diag:.0f} px)'
        )
    return True, ''


# ---------------------------------------------------------------------------
# Feature-based registration (SIFT + RANSAC)
# ---------------------------------------------------------------------------

def register_round_pair(
    ref_dapi: np.ndarray,
    mov_dapi: np.ndarray,
    n_features: int = 10000,
    flann_checks: int = 100,
    lowe_ratio: float = 0.75,
    ransac_threshold: float = 3.0,
    ransac_confidence: float = 0.95,
    ransac_max_iters: int = 20000,
    max_angle_deg: float = 5.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    max_shift_fraction: float = 0.5,
) -> Tuple[Optional[np.ndarray], int, int, str]:
    """Estimate a similarity transform from moving DAPI to reference DAPI.

    Steps (unchanged from the validated source pipeline):
      1. Contrast-stretch both to uint8 (SIFT requires 8-bit)
      2. Detect SIFT keypoints and descriptors
      3. FLANN KD-tree matching (k=2 neighbours per query)
      4. Lowe's ratio test to filter ambiguous matches
      5. RANSAC estimateAffinePartial2D (4-DOF: tx, ty, rotation, scale)
      6. Sanity check on estimated transform parameters

    Parameters
    ----------
    ref_dapi, mov_dapi : 2D arrays (uint16 or uint8), same or different shape.
    n_features : max SIFT features per image.
    flann_checks : FLANN leaf checks — higher = better recall, slower.
    lowe_ratio : Lowe's ratio test threshold.
    ransac_threshold : reprojection error threshold in pixels.
    ransac_confidence : RANSAC confidence level.
    ransac_max_iters : RANSAC iteration cap.
    max_angle_deg : sanity limit for rotation.
    scale_range : sanity limits for scale factor.
    max_shift_fraction : sanity limit for translation, as a fraction of the
        reference image diagonal.

    Returns
    -------
    transform : (2x3) float64 ndarray mapping moving -> reference pixel
        coordinates, or None on failure.
    n_inliers : int
    n_matches : int (count after Lowe ratio test)
    status : 'ok' or a human-readable failure reason. Never silently
        returns a transform that failed the sanity check — callers must
        check ``transform is not None`` (equivalently ``status == 'ok'``)
        before using the result.
    """
    import cv2

    ref_u8 = _normalize_uint8(ref_dapi)
    mov_u8 = _normalize_uint8(mov_dapi)

    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_ref, des_ref = sift.detectAndCompute(ref_u8, None)
    kp_mov, des_mov = sift.detectAndCompute(mov_u8, None)

    n_kp_ref = len(kp_ref) if kp_ref else 0
    n_kp_mov = len(kp_mov) if kp_mov else 0

    if des_ref is None or des_mov is None or n_kp_ref < 4 or n_kp_mov < 4:
        return None, 0, 0, f'insufficient keypoints (ref={n_kp_ref}, mov={n_kp_mov})'

    flann = cv2.FlannBasedMatcher(
        {'algorithm': 1, 'trees': 5},   # FLANN_INDEX_KDTREE
        {'checks': flann_checks},
    )
    knn = flann.knnMatch(des_ref, des_mov, k=2)

    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)

    if len(good) < 4:
        return None, 0, len(good), f'too few matches after Lowe ratio test ({len(good)} < 4)'

    # src = moving keypoints, dst = reference keypoints
    # estimateAffinePartial2D finds T: src -> dst  (moving -> reference)
    src_pts = np.float32([kp_mov[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    transform, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iters,
    )

    if transform is None:
        return None, 0, len(good), 'RANSAC failed to find a consistent transform'

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    valid, reason = _validate_transform(
        transform, ref_dapi.shape, max_angle_deg, scale_range, max_shift_fraction,
    )
    if not valid:
        return None, n_inliers, len(good), f'transform sanity check failed: {reason}'

    return transform, n_inliers, len(good), 'ok'


def warp_image(
    img: np.ndarray,
    transform: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Apply a 2x3 similarity transform to map an image into the reference space.

    Out-of-bounds pixels are filled with 0 (BORDER_CONSTANT). Callers should
    generate and save a border mask to track these padding regions.
    """
    import cv2
    return cv2.warpAffine(
        img, transform,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# CZI filename parsing
# ---------------------------------------------------------------------------

def parse_czi_filename(
    filename: str,
    pattern: str = PREVAIL_BARRETTS_CZI_FILENAME_PATTERN,
) -> Optional[Dict]:
    """Parse a CZI filename into its round/marker/slide/date/scene components.

    ``pattern`` must be a regex with exactly 6 capture groups, in order:
    round, markers, slide, date, acq_id, scene. The default is the
    convention observed in the PREVAIL hBE-TMA cohort's raw CZI acquisitions:

        R{round}_{markers}_{slide}_{YYYY_MM_DD}__{acqid}-{scene}.czi

    Example: ``R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_2026_05_16__17930-sceneA1.czi``

    A different acquisition naming scheme (different institute, different
    scanner software) needs a different ``pattern``, not a code change.

    Returns a dict with keys: round, markers, slide, date, acq_id, scene.
    Returns None if the filename does not match ``pattern``.
    """
    m = re.match(pattern, filename)
    if m:
        return {
            'round':  m.group(1),
            'markers': m.group(2),
            'slide':   m.group(3),
            'date':    m.group(4).replace('_', '-'),
            'acq_id':  m.group(5),
            'scene':   m.group(6),
        }
    return None


def group_czis_by_scene(
    czi_dir: str,
    filename_pattern: str = PREVAIL_BARRETTS_CZI_FILENAME_PATTERN,
    round_label_pattern: str = PREVAIL_BARRETTS_ROUND_LABEL_PATTERN,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Group CZIs in a directory by (slide, scene) and sort rounds biologically.

    Filenames that do not match ``filename_pattern`` are skipped and reported
    (never silently dropped without a trace) via a printed warning listing up
    to 3 example unparsed names and the total count.

    Returns
    -------
    {(slide, scene): {round_label: czi_path, ...}}  — rounds in biological order
    """
    from collections import defaultdict
    groups: Dict = defaultdict(dict)
    unparsed = []

    for fname in sorted(os.listdir(czi_dir)):
        if not fname.lower().endswith('.czi'):
            continue
        info = parse_czi_filename(fname, pattern=filename_pattern)
        if info:
            key = (info['slide'], info['scene'])
            groups[key][info['round']] = os.path.join(czi_dir, fname)
        else:
            unparsed.append(fname)

    if unparsed:
        print(f'Skipped {len(unparsed)} unparseable filenames: {unparsed[:3]}{"..." if len(unparsed) > 3 else ""}')

    return {
        k: dict(sorted(v.items(), key=lambda x: _round_sort_key(x[0], round_label_pattern)))
        for k, v in sorted(groups.items())
    }


# ---------------------------------------------------------------------------
# step03_register_rounds
# ---------------------------------------------------------------------------

def step03_register_rounds(
    czis: Dict[str, str],
    config: Optional[RegistrationConfig] = None,
    ref_round: str = 'R1',
    dapi_channel: int = 0,
    num_channels: Optional[int] = None,
    scale_factor: float = 1.0,
    output_dir: Optional[str] = None,
    round_label_pattern: str = PREVAIL_BARRETTS_ROUND_LABEL_PATTERN,
    verbose: bool = True,
) -> Dict:
    """Register all imaging rounds of one TMA scene to a reference round's DAPI channel.

    Algorithm (unchanged from the validated source pipeline): reads the
    reference round's DAPI channel as the fixed coordinate system, then for
    every other round independently runs :func:`register_round_pair`
    (SIFT + FLANN + Lowe ratio test + RANSAC similarity fit + sanity check)
    and, on success, warps every channel of that round into the reference
    frame with :func:`warp_image`. A round whose fitted transform fails the
    physically-motivated sanity bounds in ``config`` is never silently
    warped or skipped — it is recorded with ``registered=False`` and a
    human-readable ``status`` explaining the failure, so the caller always
    sees it (in the returned dict and in the printed summary).

    HPC / SLURM requirement
    ------------------------
    Full-resolution multi-round CZI mosaics are gigapixel-scale; reading and
    registering them is compute- and memory-heavy (see the memory note
    below). Per this lab's HPC policy, calling this function against real
    full-resolution multi-round CZI data **must** happen inside a SLURM
    allocation (e.g. ``srun --cpus-per-task=8 --mem=32G --time=02:00:00
    --pty bash``), never on a login node. This function cannot detect or
    enforce that — it only documents it. Use ``scale_factor < 1.0`` for fast
    interactive iteration on a login node.

    Parameters
    ----------
    czis : dict — {round_label: czi_path}
    config : :class:`~spatial_omics_toolkit.cycif.config.RegistrationConfig`
        bundling the SIFT/FLANN/RANSAC parameters and transform sanity
        bounds. Defaults to ``RegistrationConfig()`` (the validated defaults)
        when omitted.
    ref_round : reference round label (default 'R1'); must be a key of ``czis``.
    dapi_channel : DAPI channel index within each CZI (default 0).
    num_channels : channels per round; auto-detected from the reference CZI's
        metadata when None.
    scale_factor : 1.0 = full resolution; < 1.0 to read a downsampled mosaic
        (fast iteration / login-node-safe testing).
    output_dir : if set, saves each round's registered channel stack as a
        TIFF here immediately after warping (supports streaming across scenes
        without holding every scene's pixels in memory at once).
    round_label_pattern : passed to :func:`_round_sort_key` for round ordering.
    verbose : print a per-round progress line and a final summary table.

    Returns
    -------
    dict with keys:
      'rounds'          : {round_label: {channels, dapi, transform, inliers,
                            matches, registered, status, mov_shape, czi_path,
                            czi_size_bytes}}
      'reference_shape' : (H, W)
      'round_order'     : rounds sorted in biological order
      'pixel_size_um'   : physical pixel size in micrometers, or None
      'params'          : all parameters used, as plain values (for provenance)

    A round's entry has ``channels=None`` and ``transform=None`` when
    ``registered`` is False — never a transform that failed validation.

    Memory note
    -----------
    The returned dict holds all pixel arrays for the scene (~4-6 GB at full
    resolution for a typical hBE-TMA scene / panel size). Callers should
    export what they need (e.g. write TIFFs via ``output_dir``, or extract
    only the channels needed downstream) then ``del result`` and
    ``gc.collect()`` before processing the next scene.
    """
    import tifffile

    if config is None:
        config = RegistrationConfig()

    rounds = sorted(czis.keys(), key=lambda r: _round_sort_key(r, round_label_pattern))
    if ref_round not in rounds:
        raise ValueError(f"Reference round '{ref_round}' not found. Available: {rounds}")

    if num_channels is None:
        num_channels = _get_czi_num_channels(czis[ref_round])
        if verbose:
            print(f'  Auto-detected {num_channels} channels in {ref_round}')

    pixel_size_um = get_czi_pixel_size(czis[ref_round])
    if verbose:
        if pixel_size_um:
            print(f'  Pixel size: {pixel_size_um:.4f} um/px')
        else:
            print('  Pixel size not found in CZI metadata — outputs will lack spatial calibration')

    nonref_rounds = [r for r in rounds if r != ref_round]

    if verbose:
        print(f'Reading reference round ({ref_round})...')
    ref_ch = read_czi_all_channels(czis[ref_round], num_channels, scale_factor)
    ref_dapi = ref_ch[dapi_channel]
    ref_shape = ref_dapi.shape
    if verbose:
        print(f'  Mosaic shape: {ref_shape[0]} x {ref_shape[1]} px')

    params = dict(
        ref_round=ref_round,
        dapi_channel=dapi_channel,
        num_channels=num_channels,
        scale_factor=scale_factor,
        n_features=config.n_features,
        flann_checks=config.flann_checks,
        lowe_ratio=config.lowe_ratio,
        ransac_threshold=config.ransac_threshold,
        ransac_confidence=config.ransac_confidence,
        ransac_max_iters=config.ransac_max_iters,
        max_angle_deg=config.max_angle_deg,
        scale_range=list(config.scale_range),
        max_shift_fraction=config.max_shift_fraction,
    )

    result: Dict = {
        'rounds': {
            ref_round: {
                'channels': ref_ch,
                'dapi': ref_dapi,
                'transform': np.eye(2, 3, dtype=np.float64),
                'inliers': 0,
                'matches': 0,
                'registered': True,
                'status': 'reference',
                'mov_shape': ref_shape,
                'czi_path': czis[ref_round],
                'czi_size_bytes': os.path.getsize(czis[ref_round]),
            }
        },
        'reference_shape': ref_shape,
        'round_order': rounds,
        'pixel_size_um': pixel_size_um,
        'params': params,
    }

    for rnd in nonref_rounds:
        if verbose:
            print(f'Registering {rnd}...')
        mov_ch = read_czi_all_channels(czis[rnd], num_channels, scale_factor)
        mov_dapi = mov_ch[dapi_channel]
        if verbose:
            print(f'  Mosaic shape: {mov_dapi.shape[0]} x {mov_dapi.shape[1]} px')

        transform, n_inliers, n_matches, status = register_round_pair(
            ref_dapi, mov_dapi,
            n_features=config.n_features,
            flann_checks=config.flann_checks,
            lowe_ratio=config.lowe_ratio,
            ransac_threshold=config.ransac_threshold,
            ransac_confidence=config.ransac_confidence,
            ransac_max_iters=config.ransac_max_iters,
            max_angle_deg=config.max_angle_deg,
            scale_range=config.scale_range,
            max_shift_fraction=config.max_shift_fraction,
        )

        if transform is None:
            if verbose:
                print(f'  FAILED — {status}')
            result['rounds'][rnd] = {
                'channels': None,
                'dapi': None,
                'transform': None,
                'inliers': 0,
                'matches': n_matches,
                'registered': False,
                'status': status,
                'mov_shape': mov_dapi.shape,
                'czi_path': czis[rnd],
                'czi_size_bytes': os.path.getsize(czis[rnd]),
            }
            continue

        if verbose:
            scale = float(np.sqrt(transform[0, 0] ** 2 + transform[0, 1] ** 2))
            angle = float(np.degrees(np.arctan2(transform[0, 1], transform[0, 0])))
            rate = 100.0 * n_inliers / n_matches if n_matches else 0.0
            print(
                f'  {n_inliers}/{n_matches} inliers ({rate:.1f}%)  '
                f'angle={angle:+.2f}deg  scale={scale:.4f}  '
                f'tx={transform[0,2]:+.1f}  ty={transform[1,2]:+.1f} px'
            )

        warped = np.stack(
            [warp_image(mov_ch[c], transform, ref_shape) for c in range(num_channels)],
            axis=0,
        )

        result['rounds'][rnd] = {
            'channels': warped,
            'dapi': warped[dapi_channel],
            'transform': transform,
            'inliers': n_inliers,
            'matches': n_matches,
            'registered': True,
            'status': status,
            'mov_shape': mov_dapi.shape,
            'czi_path': czis[rnd],
            'czi_size_bytes': os.path.getsize(czis[rnd]),
        }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for rnd in rounds:
            rd = result['rounds'].get(rnd)
            if rd and rd.get('channels') is not None:
                p = os.path.join(output_dir, f'{rnd}_registered.tif')
                tifffile.imwrite(p, rd['channels'].astype(np.uint16))
                if verbose:
                    print(f'  Saved: {os.path.basename(p)}')

    if verbose:
        print('\n' + '=' * 60)
        print('Registration summary:')
        for rnd in rounds:
            rd = result['rounds'][rnd]
            if rd['status'] == 'reference':
                print(f'  {rnd:6s}  REFERENCE  {ref_shape[0]}x{ref_shape[1]} px')
            elif rd['registered']:
                rate = 100.0 * rd['inliers'] / rd['matches'] if rd['matches'] else 0
                print(f'  {rnd:6s}  OK         {rd["inliers"]}/{rd["matches"]} inliers ({rate:.1f}%)')
            else:
                print(f'  {rnd:6s}  FAILED     {rd["status"]}')
        print('=' * 60)

    return result
