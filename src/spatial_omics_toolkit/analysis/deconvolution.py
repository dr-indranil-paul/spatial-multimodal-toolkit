"""Ruifrok-Johnston optical density color deconvolution for brightfield multiplex IHC.

Separates overlapping chromogenic stains (AEC, Hematoxylin, DAB, Fast Red, CMYK)
from RGB brightfield histology into unmixed pseudo-fluorescent intensity channels.
Reference: Ruifrok & Johnston, Anal Quant Cytol Histol (2001).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Calibrated optical density vectors [Red, Green, Blue] for standard histochemical stains
STAIN_VECTORS: Dict[str, Tuple[float, float, float]] = {
    "Hematoxylin": (0.644211, 0.716556, 0.266844),
    "Eosin": (0.092789, 0.954111, 0.283111),
    "DAB": (0.268, 0.570, 0.776),
    "AEC": (0.2743, 0.6796, 0.6803),
    "FastRed": (0.214, 0.851, 0.478),
    "FastGreen": (0.98, 0.144, 0.133),
    "MethylGreen": (0.98, 0.144, 0.133),
    "Feulgen": (0.4642, 0.8300, 0.3082),
    "AlcianBlue": (0.874, 0.457, 0.158),
}

# Standard 3-stain preset matrices (3x3 normalized OD vectors)
PRESET_STAIN_MATRICES: Dict[str, List[str]] = {
    "HE": ["Hematoxylin", "Eosin"],
    "HDAB": ["Hematoxylin", "DAB"],
    "AEC_HEM": ["Hematoxylin", "AEC"],
}


def _normalize_vector(v: Tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / (norm if norm > 0 else 1.0)


def _find_orthogonal_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Find third orthogonal vector using cross product."""
    v3 = np.cross(v1, v2)
    norm = np.linalg.norm(v3)
    return v3 / (norm if norm > 0 else 1.0)


def build_stain_matrix(
    stains: Union[str, Sequence[str], np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Build a 3x3 normalized stain matrix from stain names or custom matrix.

    Args:
        stains: Preset name ("HE", "HDAB", "AEC_HEM"), list of 2 or 3 stain names, or 3x3 ndarray.

    Returns:
        Tuple of (3x3 stain matrix M, list of stain channel names).
    """
    if isinstance(stains, str):
        if stains in PRESET_STAIN_MATRICES:
            stain_names = PRESET_STAIN_MATRICES[stains]
        else:
            raise ValueError(
                f"Unknown stain preset '{stains}'. Choose from: {list(PRESET_STAIN_MATRICES.keys())}"
            )
    elif isinstance(stains, np.ndarray):
        if stains.shape != (3, 3):
            raise ValueError(
                f"Custom stain matrix must have shape (3, 3), got {stains.shape}"
            )
        return stains.astype(float), ["Stain_1", "Stain_2", "Stain_3"]
    else:
        stain_names = list(stains)

    if len(stain_names) == 2:
        s1 = _normalize_vector(STAIN_VECTORS[stain_names[0]])
        s2 = _normalize_vector(STAIN_VECTORS[stain_names[1]])
        s3 = _find_orthogonal_vector(s1, s2)
        M = np.vstack([s1, s2, s3])
        channel_names = [stain_names[0], stain_names[1], "Residual"]
    elif len(stain_names) == 3:
        s1 = _normalize_vector(STAIN_VECTORS[stain_names[0]])
        s2 = _normalize_vector(STAIN_VECTORS[stain_names[1]])
        s3 = _normalize_vector(STAIN_VECTORS[stain_names[2]])
        M = np.vstack([s1, s2, s3])
        channel_names = list(stain_names)
    else:
        raise ValueError(f"Expected 2 or 3 stains, got {len(stain_names)}")

    return M, channel_names


@dataclass
class DeconvolutionResult:
    """Result of color deconvolution."""

    channels: Dict[str, np.ndarray]
    stain_names: List[str]
    stain_matrix: np.ndarray
    optical_density: np.ndarray


def deconvolve_color_stains(
    rgb_image: np.ndarray,
    stains: Union[str, Sequence[str], np.ndarray] = "AEC_HEM",
    background_intensity: float = 255.0,
    output_dtype: str = "float32",
) -> DeconvolutionResult:
    """Perform Ruifrok-Johnston optical density color deconvolution on an RGB image.

    Args:
        rgb_image: (H, W, 3) RGB uint8 or float image.
        stains: Stain preset ("HE", "HDAB", "AEC_HEM") or custom stain list / 3x3 matrix.
        background_intensity: Transmitted illumination background (default 255 for 8-bit).
        output_dtype: "float32" or "uint16" (scales pseudo-fluorescence to 0..65535).

    Returns:
        DeconvolutionResult containing individual single-channel deconvolved images.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got shape {rgb_image.shape}")

    rgb = rgb_image[..., :3].astype(np.float64)
    rgb = np.clip(rgb, 1.0, background_intensity)

    # 1. Convert transmitted RGB intensity to Optical Density: OD = -log10(I / I0)
    od = -np.log10(rgb / background_intensity)  # (H, W, 3)
    H, W, _ = od.shape

    # 2. Invert stain matrix: C = OD @ M^-1
    M, channel_names = build_stain_matrix(stains)
    M_inv = np.linalg.pinv(M)  # (3, 3)

    od_flat = od.reshape(-1, 3)  # (N_pixels, 3)
    c_flat = np.dot(od_flat, M_inv)  # (N_pixels, 3)
    c_flat = np.clip(c_flat, 0.0, None)  # non-negative concentrations

    channels_dict: Dict[str, np.ndarray] = {}
    for i, name in enumerate(channel_names):
        ch_img = c_flat[:, i].reshape(H, W)
        if output_dtype == "uint16":
            # Map max concentration to 16-bit pseudo-fluorescence
            max_val = np.percentile(ch_img, 99.9) if np.max(ch_img) > 0 else 1.0
            scaled = np.clip(
                (ch_img / max(max_val, 1e-6)) * 65535.0, 0.0, 65535.0
            ).astype(np.uint16)
            channels_dict[name] = scaled
        else:
            channels_dict[name] = ch_img.astype(np.float32)

    return DeconvolutionResult(
        channels=channels_dict,
        stain_names=channel_names,
        stain_matrix=M,
        optical_density=od.astype(np.float32),
    )
