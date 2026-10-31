"""Vector loss-weight cell typing and SVM active-learning cell classification engines.

Implements and modernizes the Maxey loss-weight matrix phenotyping and SVM classification
paradigms from DAS_v2 (Coussens/Chin Lab, OHSU) into clean, stateless, type-checked
analytical workflows with confidence margins and uncertainty sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Standard Maxey weight matrices ported from DAS_v2
MAXEY_PRIMARY_MATRIX: Dict[str, Dict[str, float]] = {
    "1: endothelial": {"CD31": 2.0},
    "2: immune": {
        "CD45": 1.0,
        "CD3": 1.0,
        "CD4": 1.0,
        "CD8": 1.0,
        "CD68": 1.0,
        "CD20": 1.0,
    },
    "3: epithelial": {
        "panCK": 1.0,
        "pan-CK": 1.0,
        "CK19": 1.0,
        "CK5": 1.0,
        "CK7": 1.0,
        "CK8": 1.0,
        "CK14": 1.0,
        "Ecad": 5.0,
        "MUC1": 1.0,
    },
    "4: active fibroblast": {"aSMA": 1.0, "Vim": 1.0, "ColI": 1.0, "FAP": 1.0},
    "5: stromal": {"Vim": 1.0, "ColI": 1.0, "CD90": 1.0},
}

MAXEY_IMMUNE_MATRIX: Dict[str, Dict[str, float]] = {
    "CD4_T_cell": {"CD3": 1.0, "CD4": 2.0, "CD45": 1.0},
    "CD8_T_cell": {"CD3": 1.0, "CD8": 2.0, "CD45": 1.0},
    "B_cell": {"CD20": 2.0, "CD45": 1.0},
    "Macrophage_M1": {"CD68": 2.0, "HLA-DR": 1.5, "CD45": 1.0},
    "Macrophage_M2": {"CD68": 1.5, "CD163": 2.0, "CD45": 1.0},
    "Dendritic_cell": {"CD11c": 2.0, "HLA-DR": 1.5, "CD45": 1.0},
    "NK_cell": {"CD56": 2.0, "CD45": 1.0},
}

MAXEY_TUMOR_MATRIX: Dict[str, Dict[str, float]] = {
    "Epithelial_differentiated": {"Ecad": 2.0, "panCK": 1.5, "CK19": 1.5},
    "EMT_hybrid": {"Ecad": 1.0, "Vim": 1.0, "panCK": 1.0, "aSMA": 1.0},
    "Mesenchymal_like": {"Vim": 2.0, "aSMA": 1.5, "CD44": 1.0},
    "Proliferative_tumor": {"Ki67": 2.0, "PCNA": 1.5, "panCK": 1.0},
}


def _marker_root(name: object) -> str:
    """Extract base marker name by stripping common intensity / compartment suffixes."""
    s = str(name).strip()
    # Handle patterns like CD31_mean, CD45_nuclei, Ecad_cytoplasm, R2_CD45
    parts = s.split("_")
    if len(parts) > 1 and parts[0].startswith("R") and parts[0][1:].isdigit():
        # Round prefix e.g. R2_CD45 -> CD45
        return parts[1]
    return parts[0]


def loss_weights_to_dataframe(
    weights: Union[pd.DataFrame, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """Convert a dictionary or DataFrame into a standardized (markers x celltypes) weight matrix.

    Args:
        weights: Dictionary mapping celltype -> {marker: weight} or DataFrame.

    Returns:
        pd.DataFrame where index is marker names and columns are celltype names.
    """
    if isinstance(weights, dict):
        df_w = pd.DataFrame(weights).fillna(0.0)
    elif isinstance(weights, pd.DataFrame):
        df_w = weights.copy().fillna(0.0)
    else:
        raise TypeError(
            f"weights must be a dict or DataFrame, got {type(weights).__name__}"
        )

    df_w.index = df_w.index.astype(str)
    df_w.columns = df_w.columns.astype(str)
    return df_w.astype(float)


def align_features_to_weights(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    match_mode: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match DataFrame columns to loss-weight matrix markers.

    Args:
        df: Expression matrix (cells x features).
        weights_df: Loss-weight matrix (markers x celltypes).
        match_mode: "auto", "exact", or "prefix".

    Returns:
        Tuple of (aligned_df_subset, aligned_weights_subset).
    """
    df_cols = list(df.columns)
    weight_markers = list(weights_df.index)

    exact_matches = [m for m in weight_markers if m in df_cols]

    if match_mode == "exact" or (match_mode == "auto" and len(exact_matches) > 0):
        matched_markers = exact_matches
        matched_cols = exact_matches
    elif match_mode in ("prefix", "auto"):
        matched_markers = []
        matched_cols = []
        for m in weight_markers:
            m_root = _marker_root(m).lower()
            candidates = [c for c in df_cols if _marker_root(c).lower() == m_root]
            if len(candidates) == 1:
                matched_markers.append(m)
                matched_cols.append(candidates[0])
            elif len(candidates) > 1:
                # Pick shortest or direct match
                exact = [c for c in candidates if c == m]
                if exact:
                    matched_markers.append(m)
                    matched_cols.append(exact[0])
                else:
                    matched_markers.append(m)
                    matched_cols.append(candidates[0])
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}")

    if not matched_markers:
        raise ValueError(
            f"No columns in DataFrame matched loss weight markers: {weight_markers[:8]}"
        )

    sub_df = df[matched_cols].copy()
    sub_df.columns = matched_markers
    sub_weights = weights_df.loc[matched_markers].copy()

    return sub_df, sub_weights


def scale_features(
    df: pd.DataFrame,
    method: str = "zscore",
    core_col: Optional[str] = None,
    full_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Scale expression features across cells or per-core.

    Args:
        df: Feature numeric DataFrame.
        method: "zscore", "log2-zscore", "rank", "minmax", or "none".
        core_col: Optional column name for per-core grouping.
        full_df: Full DataFrame containing core_col if df is a subset.

    Returns:
        Scaled DataFrame matching df shape and index.
    """
    data = df.astype(float).copy()
    if method == "none":
        return data

    if method == "log2-zscore":
        data = np.log2(np.clip(data, 1.0, None))
        method = "zscore"

    if core_col is not None and full_df is not None and core_col in full_df.columns:
        cores = full_df.loc[data.index, core_col]
        if method == "zscore":
            scaled = data.copy()
            for col in data.columns:
                mean = data.groupby(cores)[col].transform("mean")
                std = data.groupby(cores)[col].transform("std").replace(0.0, np.nan)
                scaled[col] = ((data[col] - mean) / std).fillna(0.0)
            return scaled
        elif method == "rank":
            scaled = data.copy()
            for col in data.columns:
                scaled[col] = data.groupby(cores)[col].rank(pct=True)
            return scaled

    # Global scaling
    if method == "zscore":
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=0).replace(0.0, np.nan)
        return ((data - mean) / std).fillna(0.0)
    elif method == "rank":
        denom = max(len(data) - 1, 1)
        return (data.rank(axis=0, method="average") - 1) / denom
    elif method == "minmax":
        min_v = data.min(axis=0)
        rng = (data.max(axis=0) - min_v).replace(0.0, np.nan)
        return ((data - min_v) / rng).fillna(0.0)
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def vector_classify_cells(
    df: pd.DataFrame,
    loss_weights: Union[
        pd.DataFrame, Dict[str, Dict[str, float]]
    ] = MAXEY_PRIMARY_MATRIX,
    scaling: str = "zscore",
    match_mode: str = "auto",
    confidence_threshold: float = 0.0,
    margin_threshold: float = 0.0,
    core_col: Optional[str] = None,
    output_col: str = "vector_celltype",
    confidence_col: str = "vector_confidence",
    margin_col: str = "vector_margin",
    unidentified_label: str = "Unidentified",
) -> pd.DataFrame:
    """Assign cell phenotypes via Maxey loss-weight matrix scoring with confidence margins.

    Args:
        df: Cells DataFrame with marker intensity columns.
        loss_weights: Matrix or dict of (markers x celltypes) importance weights.
        scaling: "zscore", "log2-zscore", "rank", "minmax", or "none".
        match_mode: Column matching mode ("auto", "exact", "prefix").
        confidence_threshold: Minimum winning softmax probability required (0.0 to 1.0).
        margin_threshold: Minimum difference between 1st and 2nd highest scores.
        core_col: Optional column name for per-core normalization.
        output_col: Resulting classification column name.
        confidence_col: Resulting confidence probability column name.
        margin_col: Resulting margin score column name.
        unidentified_label: Label for low-confidence or ambiguous cells.

    Returns:
        DataFrame with classification, confidence, and margin columns appended.
    """
    weights_df = loss_weights_to_dataframe(loss_weights)
    sub_df, sub_weights = align_features_to_weights(
        df, weights_df, match_mode=match_mode
    )
    scaled_df = scale_features(sub_df, method=scaling, core_col=core_col, full_df=df)

    # Compute dot-product match scores: S = X @ W (shape: n_cells x n_celltypes)
    scores = np.dot(scaled_df.values, sub_weights.values)  # (N, C)
    celltypes = list(sub_weights.columns)

    # Softmax probabilities over cell types
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Sorted score analysis for confidence margins
    sorted_score_indices = np.argsort(-scores, axis=1)
    top1_idx = sorted_score_indices[:, 0]
    top2_idx = sorted_score_indices[:, 1] if scores.shape[1] > 1 else top1_idx

    top1_scores = scores[np.arange(len(scores)), top1_idx]
    top2_scores = scores[np.arange(len(scores)), top2_idx]
    margins = top1_scores - top2_scores if scores.shape[1] > 1 else top1_scores

    top1_probs = probs[np.arange(len(probs)), top1_idx]
    assigned_types = [celltypes[i] for i in top1_idx]

    # Apply thresholds
    final_types = []
    for i, ctype in enumerate(assigned_types):
        if top1_probs[i] < confidence_threshold or margins[i] < margin_threshold:
            final_types.append(unidentified_label)
        else:
            final_types.append(ctype)

    result = df.copy()
    result[output_col] = final_types
    result[confidence_col] = np.round(top1_probs, 4)
    result[margin_col] = np.round(margins, 4)

    # Also attach individual celltype probability scores
    for j, ctype in enumerate(celltypes):
        result[f"prob_{ctype}"] = np.round(probs[:, j], 4)

    return result


@dataclass
class SVMClassificationResult:
    """Result bundle from SVM cell classification."""

    model: Any
    scaler: Optional[StandardScaler]
    feature_columns: List[str]
    classes: List[str]
    training_accuracy: float


def train_celltype_svm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    kernel: str = "rbf",
    C: float = 1.0,
    class_weight: str = "balanced",
    probability: bool = True,
) -> SVMClassificationResult:
    """Train an SVM classifier on annotated cell phenotypes.

    Args:
        df: Annotated training cells DataFrame.
        target_col: Column with ground-truth cell phenotype labels.
        feature_cols: Optional list of feature columns (defaults to numeric columns).
        kernel: SVM kernel ("rbf", "linear", "poly").
        C: Regularization parameter.
        class_weight: "balanced" or None.
        probability: Whether to enable probability calibration.

    Returns:
        SVMClassificationResult containing fitted model, scaler, and metadata.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in training DataFrame."
        )

    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) == 0:
        raise ValueError(
            "Training DataFrame contains 0 rows after dropping NaN targets."
        )

    if feature_cols is None:
        feature_cols = [
            c
            for c in clean_df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
    else:
        feature_cols = list(feature_cols)

    missing_cols = [c for c in feature_cols if c not in clean_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing feature columns in training DataFrame: {missing_cols}"
        )

    X = clean_df[feature_cols].astype(float).fillna(0.0).values
    y = clean_df[target_col].astype(str).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = svm.SVC(
        C=C,
        kernel=kernel,
        class_weight=class_weight,
        probability=probability,
        random_state=42,
    )
    clf.fit(X_scaled, y)
    acc = float(clf.score(X_scaled, y))

    logger.info(
        f"Trained SVM ({kernel}) on {len(clean_df)} cells across {len(clf.classes_)} classes. Acc: {acc:.4f}"
    )
    return SVMClassificationResult(
        model=clf,
        scaler=scaler,
        feature_columns=feature_cols,
        classes=list(clf.classes_),
        training_accuracy=acc,
    )


def predict_celltype_svm(
    svm_result: SVMClassificationResult,
    df: pd.DataFrame,
    confidence_threshold: float = 0.0,
    output_col: str = "svm_celltype",
    confidence_col: str = "svm_confidence",
    uncertain_label: str = "Uncertain",
) -> pd.DataFrame:
    """Predict cell phenotypes on unannotated cells using a trained SVM.

    Args:
        svm_result: Trained SVMClassificationResult.
        df: Unannotated cells DataFrame.
        confidence_threshold: Minimum probability to assign a class.
        output_col: Output classification column name.
        confidence_col: Output probability confidence column.
        uncertain_label: Label for cells below confidence threshold.

    Returns:
        DataFrame with predictions, confidences, and uncertainty flags.
    """
    feature_cols = svm_result.feature_columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required SVM features: {missing}")

    X = df[feature_cols].astype(float).fillna(0.0).values
    X_scaled = svm_result.scaler.transform(X) if svm_result.scaler is not None else X

    preds = svm_result.model.predict(X_scaled)
    result = df.copy()

    if hasattr(svm_result.model, "predict_proba"):
        probs = svm_result.model.predict_proba(X_scaled)
        max_probs = np.max(probs, axis=1)

        final_preds = []
        for i, p in enumerate(preds):
            if max_probs[i] < confidence_threshold:
                final_preds.append(uncertain_label)
            else:
                final_preds.append(p)

        result[output_col] = final_preds
        result[confidence_col] = np.round(max_probs, 4)

        for j, cls_name in enumerate(svm_result.classes):
            result[f"svm_prob_{cls_name}"] = np.round(probs[:, j], 4)
    else:
        result[output_col] = preds
        result[confidence_col] = 1.0

    return result


def uncertainty_sampling(
    svm_result: SVMClassificationResult,
    df: pd.DataFrame,
    n_samples: int = 50,
) -> pd.DataFrame:
    """Active learning helper: identify cells with the highest classification uncertainty.

    Ranks cells by prediction entropy or smallest probability margin for pathologist review.

    Args:
        svm_result: Trained SVMClassificationResult.
        df: Candidate cells DataFrame.
        n_samples: Number of most uncertain cells to return.

    Returns:
        DataFrame containing candidate cells sorted by uncertainty score descending.
    """
    if not hasattr(svm_result.model, "predict_proba"):
        raise ValueError(
            "SVM model must be trained with probability=True for uncertainty sampling."
        )

    feature_cols = svm_result.feature_columns
    X = df[feature_cols].astype(float).fillna(0.0).values
    X_scaled = svm_result.scaler.transform(X) if svm_result.scaler is not None else X
    probs = svm_result.model.predict_proba(X_scaled)

    # Prediction entropy: H = - sum(p * log(p))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    # Margin: difference between top-1 and top-2
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - (sorted_probs[:, 1] if probs.shape[1] > 1 else 0.0)

    work = df.copy()
    work["uncertainty_entropy"] = np.round(entropy, 4)
    work["uncertainty_margin"] = np.round(margin, 4)
    top_indices = np.argmax(probs, axis=1)
    work["top_predicted_type"] = [svm_result.classes[i] for i in top_indices]
    ranked = work.sort_values(
        by=["uncertainty_entropy", "uncertainty_margin"], ascending=[False, True]
    )
    return ranked.head(n_samples)
