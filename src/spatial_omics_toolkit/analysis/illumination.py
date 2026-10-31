"""Flat-field shading and tile-edge illumination correction for spatial microscopy.

Implements shading / vignetting polynomial surface modeling and distance-transform
edge-gain compensation ported from DAS_v2 (stain_correction24_pTMA / tissue_edge_correction).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage.transform import resize

logger = logging.getLogger(__name__)


def _fit_2d_polynomial(
    image: np.ndarray,
    degree: int = 2,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit a 2D polynomial surface to an image."""
    H, W = image.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # Normalize coordinates to [-1, 1] for numerical stability
    x_norm = (2.0 * x_coords - (W - 1)) / max(W - 1, 1)
    y_norm = (2.0 * y_coords - (H - 1)) / max(H - 1, 1)

    # Build design matrix of polynomial terms
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            terms.append((x_norm**i) * (y_norm**j))

    A = np.column_stack([t.ravel() for t in terms])  # (H*W, n_terms)
    b = image.ravel().astype(float)

    if mask is not None:
        valid_idx = mask.ravel() > 0
        if np.sum(valid_idx) < len(terms):
            logger.warning(
                "Fewer valid masked pixels than polynomial terms; falling back to unmasked fit."
            )
            valid_idx = np.ones(len(b), dtype=bool)
        A_fit = A[valid_idx]
        b_fit = b[valid_idx]
    else:
        A_fit = A
        b_fit = b

    # Solve least squares: A_fit @ c = b_fit
    coeffs, _, _, _ = np.linalg.lstsq(A_fit, b_fit, rcond=None)
    surface = np.dot(A, coeffs).reshape(H, W)
    return surface


def correct_flatfield_shading(
    image: np.ndarray,
    degree: int = 2,
    tissue_mask: Optional[np.ndarray] = None,
    downsample: int = 4,
    clip_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct radial vignetting / illumination falloff using 2D polynomial surface modeling.

    Args:
        image: 2D single-channel image (H, W) (uint16, uint8, or float).
        degree: Degree of 2D polynomial surface (typically 2 for quadratic parabolic falloff).
        tissue_mask: Optional boolean tissue mask (True = tissue).
        downsample: Spatial downsampling factor to accelerate least-squares surface fitting.
        clip_min: Minimum value floor for output.

    Returns:
        Tuple of (corrected_image, estimated_shading_surface).
    """
    orig_shape = image.shape
    orig_dtype = image.dtype
    H, W = orig_shape

    # Downsample for robust and fast surface fitting
    if downsample > 1 and H > downsample * 16 and W > downsample * 16:
        sub_h, sub_w = H // downsample, W // downsample
        img_sub = resize(
            image.astype(float),
            (sub_h, sub_w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        mask_sub = (
            resize(
                tissue_mask.astype(float), (sub_h, sub_w), order=0, preserve_range=True
            )
            > 0.5
            if tissue_mask is not None
            else None
        )
    else:
        img_sub = image.astype(float)
        mask_sub = tissue_mask

    # Fit 2D polynomial surface
    surface_sub = _fit_2d_polynomial(img_sub, degree=degree, mask=mask_sub)

    # Upsample surface back to original resolution
    if surface_sub.shape != orig_shape:
        surface_full = resize(
            surface_sub, orig_shape, order=1, preserve_range=True, anti_aliasing=True
        )
    else:
        surface_full = surface_sub

    # Normalize surface by its median or mean
    ref_val = np.median(
        surface_full[tissue_mask] if tissue_mask is not None else surface_full
    )
    if ref_val <= 0 or np.isnan(ref_val):
        ref_val = np.mean(surface_full) + 1e-6

    shading_norm = surface_full / max(ref_val, 1e-6)
    shading_norm = np.clip(shading_norm, 0.1, 10.0)

    # Correct image by dividing out normalized illumination profile
    corrected = image.astype(float) / shading_norm
    corrected = np.clip(corrected, clip_min, None)

    if np.issubdtype(orig_dtype, np.integer):
        max_limit = np.iinfo(orig_dtype).max
        corrected = np.clip(corrected, 0, max_limit).astype(orig_dtype)

    return corrected, surface_full


def correct_tile_edge_gain(
    image: np.ndarray,
    tissue_mask: Optional[np.ndarray] = None,
    edge_width_pixels: int = 30,
    edge_gain_factor: float = 1.15,
) -> np.ndarray:
    """Compensate for tissue edge antibody drying or peripheral drop-off using distance transform.

    Args:
        image: 2D image (H, W).
        tissue_mask: Optional boolean tissue mask (if None, estimated from threshold).
        edge_width_pixels: Width of the outer edge band in pixels.
        edge_gain_factor: Gain scaling factor at the immediate edge boundary.

    Returns:
        Edge-gain compensated image.
    """
    if tissue_mask is None:
        thresh = np.percentile(image, 20)
        tissue_mask = image > thresh

    # Compute Euclidean distance to nearest background pixel
    dist = ndimage.distance_transform_edt(tissue_mask)

    # Smooth transition from edge_gain_factor down to 1.0 within edge_width_pixels
    dist_clamped = np.clip(dist, 0, edge_width_pixels)
    gain_field = 1.0 + (edge_gain_factor - 1.0) * (
        1.0 - dist_clamped / float(edge_width_pixels)
    )

    corrected = image.astype(float) * gain_field

    if np.issubdtype(image.dtype, np.integer):
        max_limit = np.iinfo(image.dtype).max
        corrected = np.clip(corrected, 0, max_limit).astype(image.dtype)

    return corrected
