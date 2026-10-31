"""Empirical Bayes ComBat batch effect correction for multiplexed imaging and omics.

Implements Johnson et al. (Biostatistics, 2007) ComBat algorithm natively in Python with:
- Parametric and non-parametric empirical Bayes location/scale adjustments.
- Preservation of biological covariates (categorical and continuous).
- Confounding detection (enforcing the Universal STOP condition for perfect batch-condition confounding).
- Numerical stability safeguards for zero-variance and missing features.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.linalg as la
import pandas as pd

logger = logging.getLogger(__name__)


def _aprior(delta_hat: np.ndarray) -> float:
    """Estimate inverse-gamma alpha parameter via method of moments."""
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    if s2 == 0 or np.isnan(s2):
        return 1.0
    return float((2.0 * s2 + m**2) / s2)


def _bprior(delta_hat: np.ndarray) -> float:
    """Estimate inverse-gamma beta parameter via method of moments."""
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    if s2 == 0 or np.isnan(s2):
        return 1.0
    return float((m * s2 + m**3) / s2)


def _it_sol(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 0.0001,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative solver for parametric empirical Bayes estimators."""
    n = sdat.shape[1]
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(max_iter):
        # Update gamma*
        g_new = (n * t2 * g_hat + d_old * g_bar) / (n * t2 + d_old)
        # Update delta*
        sum_sq = np.sum((sdat - g_new[:, np.newaxis]) ** 2, axis=1)
        d_new = (2.0 * b + sum_sq) / (2.0 * (a + 1.0) + n)

        change = np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)) + np.max(
            np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8)
        )
        if change < conv:
            return g_new, d_new
        g_old = g_new
        d_old = d_new

    return g_old, d_old


def _nonparametric_eb(
    sdat: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-parametric empirical Bayes adjustment via kernel likelihood integration."""
    n_genes, n_samples = sdat.shape
    gamma_star = np.zeros(n_genes)
    delta_star = np.zeros(n_genes)

    for i in range(n_genes):
        # Vectorized likelihood across all candidate (g_hat, d_hat) pairs
        temp_g = g_hat
        temp_d = d_hat

        # Log-likelihood across samples
        resid = sdat[i, :] - temp_g[:, np.newaxis]  # (n_cand, n_samples)
        ll = -0.5 * n_samples * np.log(np.clip(temp_d, 1e-8, None)) - np.sum(
            resid**2, axis=1
        ) / (2.0 * np.clip(temp_d, 1e-8, None))
        weights = np.exp(ll - np.max(ll))
        w_sum = np.sum(weights) + 1e-12
        weights /= w_sum

        gamma_star[i] = np.sum(weights * temp_g)
        delta_star[i] = np.sum(weights * temp_d)

    return gamma_star, delta_star


def correct_batch_combat(
    df: pd.DataFrame,
    batch: Union[str, Sequence[object], pd.Series],
    covariates: Optional[Union[Sequence[str], pd.DataFrame]] = None,
    parametric: bool = True,
    mean_only: bool = False,
    min_batch_size: int = 2,
) -> pd.DataFrame:
    """Batch-effect correction using Empirical Bayes ComBat.

    Standardizes data across batches while preserving specified biological covariates.

    Args:
        df: Expression / intensity DataFrame (n_samples x n_features).
        batch: Column name in df, or separate array/Series containing batch identifiers.
        covariates: Optional list of column names or DataFrame of biological covariates to preserve.
        parametric: If True, uses parametric (inverse-gamma) priors; if False, non-parametric.
        mean_only: If True, only corrects location/mean shift without scaling variances.
        min_batch_size: Minimum number of samples required per batch.

    Returns:
        Batch-corrected pd.DataFrame with identical index and columns to df.

    Raises:
        ValueError: If perfect batch-condition confounding is detected or batch has < min_batch_size samples.
    """
    if isinstance(batch, str):
        if batch not in df.columns:
            raise ValueError(f"Batch column '{batch}' not found in DataFrame.")
        batch_series = df[batch].astype(str)
        data = df.drop(columns=[batch])
    else:
        batch_series = pd.Series(batch, index=df.index).astype(str)
        data = df.copy()

    # Identify numeric features for correction
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    non_numeric_cols = [c for c in data.columns if c not in numeric_cols]

    if not numeric_cols:
        raise ValueError("No numeric columns found in DataFrame for ComBat correction.")

    # Check batch structure
    batch_counts = batch_series.value_counts()
    unique_batches = list(batch_counts.index)

    if len(unique_batches) <= 1:
        logger.warning("ComBat called with <= 1 batch; returning unchanged data.")
        return df.copy()

    small_batches = batch_counts[batch_counts < min_batch_size]
    if len(small_batches) > 0:
        raise ValueError(
            f"Batches with fewer than {min_batch_size} samples detected: {dict(small_batches)}. "
            "ComBat requires at least 2 samples per batch."
        )

    # Build design matrix for batches and covariates
    batch_dummies = pd.get_dummies(batch_series, drop_first=False, dtype=float)
    n_batches = batch_dummies.shape[1]
    n_samples = len(data)

    cov_df = pd.DataFrame(index=data.index)
    if covariates is not None:
        if isinstance(covariates, (list, tuple)):
            for c in covariates:
                if c in data.columns:
                    cov_df[c] = data[c]
                elif c in df.columns:
                    cov_df[c] = df[c]
                else:
                    raise ValueError(f"Covariate '{c}' not found in data.")
        elif isinstance(covariates, pd.DataFrame):
            cov_df = covariates.loc[data.index].copy()

    # Confounding check: check rank of design matrix
    cov_dummies_list = []
    for col in cov_df.columns:
        if pd.api.types.is_numeric_dtype(cov_df[col]):
            cov_dummies_list.append(cov_df[[col]].astype(float))
        else:
            d = pd.get_dummies(cov_df[col], drop_first=True, dtype=float)
            cov_dummies_list.append(d)

    if cov_dummies_list:
        full_design = pd.concat([batch_dummies] + cov_dummies_list, axis=1)
    else:
        full_design = batch_dummies

    # Check rank for perfect confounding (Universal STOP condition)
    rank = la.matrix_rank(full_design.values)
    if rank < full_design.shape[1]:
        raise ValueError(
            "STOP: NON-IDENTIFIABLE DESIGN: Perfect collinearity / confounding detected between "
            "batches and biological covariates. ComBat cannot distinguish batch from biology."
        )

    # Transpose data to (n_features x n_samples) for ComBat standard algebra
    Y = data[numeric_cols].values.T.astype(float)  # (G, N)
    X = full_design.values  # (N, p)

    # 1. Standardize data across features
    # B_hat = (X^T X)^-1 X^T Y^T  -> shape (p, G)
    XtX_inv = la.pinv(np.dot(X.T, X))
    B_hat = np.dot(np.dot(XtX_inv, X.T), Y.T)  # (p, G)

    # Grand mean: weighted mean of batch effects
    batch_sample_counts = batch_dummies.sum(axis=0).values  # (n_batches,)
    grand_mean = np.dot(batch_sample_counts / n_samples, B_hat[:n_batches, :])  # (G,)

    # Variance pooled across samples
    fitted = np.dot(X, B_hat).T  # (G, N)
    var_pooled = np.sum((Y - fitted) ** 2, axis=1) / float(n_samples)  # (G,)
    var_pooled = np.where(var_pooled <= 1e-12, 1e-12, var_pooled)

    # Standardized mean (including covariate effects)
    stand_mean = np.dot(grand_mean[:, np.newaxis], np.ones((1, n_samples)))  # (G, N)
    if full_design.shape[1] > n_batches:
        X_cov = X.copy()
        X_cov[:, :n_batches] = 0.0
        stand_mean += np.dot(X_cov, B_hat).T

    # Standardized data Z_ij
    s_data = (Y - stand_mean) / np.sqrt(var_pooled)[:, np.newaxis]  # (G, N)

    # 2. Fit batch location/scale parameters
    gamma_star_list = []
    delta_star_list = []

    X_batch = batch_dummies.values
    batch_idx_groups = [np.where(X_batch[:, b] == 1.0)[0] for b in range(n_batches)]

    for b, idxs in enumerate(batch_idx_groups):
        s_batch = s_data[:, idxs]  # (G, n_b)
        n_b = len(idxs)

        # Naive estimators
        gamma_hat = np.mean(s_batch, axis=1)  # (G,)
        delta_hat = (
            np.var(s_batch, axis=1, ddof=1) if n_b > 1 else np.ones(len(gamma_hat))
        )  # (G,)
        delta_hat = np.where(delta_hat <= 1e-8, 1.0, delta_hat)

        if mean_only:
            gamma_star = gamma_hat
            delta_star = np.ones_like(delta_hat)
        elif parametric:
            g_bar = float(np.mean(gamma_hat))
            t2 = float(np.var(gamma_hat, ddof=1)) if len(gamma_hat) > 1 else 1.0
            t2 = max(t2, 1e-8)
            a = _aprior(delta_hat)
            b_val = _bprior(delta_hat)
            gamma_star, delta_star = _it_sol(
                s_batch, gamma_hat, delta_hat, g_bar, t2, a, b_val
            )
        else:
            gamma_star, delta_star = _nonparametric_eb(s_batch, gamma_hat, delta_hat)

        gamma_star_list.append(gamma_star)
        delta_star_list.append(delta_star)

    # 3. Adjust standardized data
    bayes_data = s_data.copy()
    for b, idxs in enumerate(batch_idx_groups):
        g_s = gamma_star_list[b][:, np.newaxis]
        d_s = np.sqrt(np.clip(delta_star_list[b][:, np.newaxis], 1e-8, None))
        bayes_data[:, idxs] = (bayes_data[:, idxs] - g_s) / d_s

    # 4. Re-scale to original measurement scale
    adj_data = bayes_data * np.sqrt(var_pooled)[:, np.newaxis] + stand_mean

    # Reconstruct output DataFrame
    corrected_df = pd.DataFrame(adj_data.T, index=data.index, columns=numeric_cols)
    if non_numeric_cols:
        for c in non_numeric_cols:
            corrected_df[c] = data[c]
    if isinstance(batch, str):
        corrected_df[batch] = batch_series

    return corrected_df
