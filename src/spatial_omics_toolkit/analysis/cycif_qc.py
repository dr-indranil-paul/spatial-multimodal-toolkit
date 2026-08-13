"""CyCIF intensity QC, signal-to-noise calculation, and background signal filtering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_marker_snr(
    intensity_matrix: pd.DataFrame,
    background_percentile: float = 10.0,
    signal_percentile: float = 90.0
) -> Dict[str, float]:
    """Calculate Signal-to-Noise Ratio (SNR) for each marker intensity channel.
    
    Args:
        intensity_matrix: Cells x Markers dataframe of mean fluorescence intensity
        background_percentile: Percentile defining background noise baseline (default 10th)
        signal_percentile: Percentile defining specific staining signal (default 90th)
        
    Returns:
        Dict mapping marker_name -> SNR value (Signal / Background ratio)
    """
    snr_dict: Dict[str, float] = {}

    for col in intensity_matrix.select_dtypes(include=[np.number]).columns:
        values = intensity_matrix[col].dropna().values
        if len(values) == 0:
            snr_dict[col] = 0.0
            continue
            
        bg = float(np.percentile(values, background_percentile))
        sig = float(np.percentile(values, signal_percentile))
        
        snr = sig / max(bg, 1e-6)
        snr_dict[col] = float(np.round(snr, 3))

    return snr_dict


def filter_background_signal(
    intensity_matrix: pd.DataFrame,
    background_thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Subtract channel-specific background thresholds and floor at zero.
    
    Args:
        intensity_matrix: Cells x Markers dataframe
        background_thresholds: Dict of channel background cutoff values
        
    Returns:
        Cleaned intensity DataFrame with background subtracted
    """
    cleaned = intensity_matrix.copy()
    for col, bg_val in background_thresholds.items():
        if col in cleaned.columns:
            cleaned[col] = np.maximum(0.0, cleaned[col] - bg_val)
    return cleaned
