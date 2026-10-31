"""Spatial interactions, biomarker diffusion lag, virtual therapy simulations, and ROI filtering.

Implements and extends spatial analysis capabilities from DAS_v2 (RAT2 / cmifAnalysis50):
- Spatial lag / neighborhood biomarker smoothing via k-d tree radius and kNN graphs.
- Pairwise cell-cell contact / proximity interaction matrices with null permutation z-scores.
- Virtual therapy / targeted cell ablation simulation with Shannon entropy and niche disruption metrics.
- Arbitrary polygon ROI spatial filtering and Euclidean distance-to-boundary calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def compute_spatial_lag_features(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    radius: float = 50.0,
    k_neighbors: Optional[int] = None,
    core_col: Optional[str] = None,
    include_self: bool = False,
    suffix: str = "_neighbor_mean",
) -> pd.DataFrame:
    """Compute smoothed neighborhood biomarker expressions (spatial lag) for every cell.

    Args:
        df: Cells DataFrame with coordinates and biomarker columns.
        feature_cols: Columns to average over spatial neighborhoods (defaults to numeric).
        x_col: X coordinate column name.
        y_col: Y coordinate column name.
        radius: Spatial neighborhood radius in pixels.
        k_neighbors: Optional fixed k-nearest neighbors count (overrides radius if provided).
        core_col: Optional grouping column (e.g. "slide_scene") to prevent cross-core neighbors.
        include_self: Whether to include the cell's own expression in the neighborhood mean.
        suffix: Suffix to append to newly created neighborhood feature columns.

    Returns:
        DataFrame with neighborhood-smoothed feature columns appended.
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame.")

    if feature_cols is None:
        exclude = {x_col, y_col}
        if core_col:
            exclude.add(core_col)
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]
    else:
        feature_cols = list(feature_cols)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    result = df.copy()
    new_cols = {f"{c}{suffix}": np.zeros(len(df), dtype=float) for c in feature_cols}

    groups = (
        [(None, df)]
        if core_col is None or core_col not in df.columns
        else df.groupby(core_col)
    )

    for _, sub_df in groups:
        if len(sub_df) == 0:
            continue
        coords = sub_df[[x_col, y_col]].values
        tree = KDTree(coords)
        feat_matrix = sub_df[feature_cols].astype(float).fillna(0.0).values
        sub_indices = sub_df.index

        if k_neighbors is not None:
            # Query k nearest neighbors
            k = min(k_neighbors + (1 if not include_self else 0), len(sub_df))
            _, neighbor_indices = tree.query(coords, k=k)
            if k == 1:
                neighbor_indices = neighbor_indices[:, np.newaxis]

            for i, loc_idx in enumerate(sub_indices):
                n_idxs = neighbor_indices[i]
                if not include_self and len(n_idxs) > 1:
                    n_idxs = n_idxs[n_idxs != i]
                means = np.mean(feat_matrix[n_idxs], axis=0)
                for f_idx, col_name in enumerate(feature_cols):
                    new_cols[f"{col_name}{suffix}"][df.index.get_loc(loc_idx)] = means[
                        f_idx
                    ]
        else:
            # Query within radius
            neighbors_list = tree.query_ball_tree(tree, r=radius)
            for i, loc_idx in enumerate(sub_indices):
                n_idxs = neighbors_list[i]
                if not include_self and len(n_idxs) > 1:
                    n_idxs = [idx for idx in n_idxs if idx != i]
                if len(n_idxs) == 0:
                    means = feat_matrix[i]  # fallback to self if isolated
                else:
                    means = np.mean(feat_matrix[n_idxs], axis=0)
                for f_idx, col_name in enumerate(feature_cols):
                    new_cols[f"{col_name}{suffix}"][df.index.get_loc(loc_idx)] = means[
                        f_idx
                    ]

    for col_name, values in new_cols.items():
        result[col_name] = values

    return result


@dataclass
class CellInteractionResult:
    """Pairwise cell-type spatial interaction analysis result."""

    contact_counts: pd.DataFrame
    interaction_frequencies: pd.DataFrame
    enrichment_zscores: Optional[pd.DataFrame] = None
    p_values: Optional[pd.DataFrame] = None


def compute_cell_interaction_matrix(
    df: pd.DataFrame,
    type_col: str = "cell_type",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    radius: float = 50.0,
    core_col: Optional[str] = None,
    n_permutations: int = 0,
    random_seed: int = 42,
) -> CellInteractionResult:
    """Compute pairwise cell-type spatial contact frequency matrix with permutation null tests.

    Args:
        df: Cells DataFrame.
        type_col: Column with cell phenotype labels.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        radius: Distance threshold for spatial adjacency.
        core_col: Optional core identifier column.
        n_permutations: Number of label permutations for spatial null tests (0 = skip).
        random_seed: Random seed for reproducibility.

    Returns:
        CellInteractionResult containing count matrix, normalized frequencies, and z-scores.
    """
    for col in (type_col, x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")

    unique_types = sorted(df[type_col].dropna().unique())
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    def _count_contacts(
        cell_types_arr: np.ndarray, sub_coords: np.ndarray
    ) -> np.ndarray:
        tree = KDTree(sub_coords)
        pairs = tree.query_pairs(r=radius)
        counts = np.zeros((n_types, n_types), dtype=float)
        for i, j in pairs:
            t1 = cell_types_arr[i]
            t2 = cell_types_arr[j]
            if t1 in type_to_idx and t2 in type_to_idx:
                idx1 = type_to_idx[t1]
                idx2 = type_to_idx[t2]
                counts[idx1, idx2] += 1.0
                counts[idx2, idx1] += 1.0
        return counts

    # Observed contact matrix
    observed_counts = np.zeros((n_types, n_types), dtype=float)
    groups = (
        [(None, df)]
        if core_col is None or core_col not in df.columns
        else df.groupby(core_col)
    )

    for _, sub_df in groups:
        if len(sub_df) < 2:
            continue
        c_types = sub_df[type_col].values
        coords = sub_df[[x_col, y_col]].values
        observed_counts += _count_contacts(c_types, coords)

    counts_df = pd.DataFrame(observed_counts, index=unique_types, columns=unique_types)
    total_contacts = np.sum(observed_counts)
    freq_df = counts_df / (total_contacts if total_contacts > 0 else 1.0)

    # Permutation null testing if requested
    z_df: Optional[pd.DataFrame] = None
    p_df: Optional[pd.DataFrame] = None

    if n_permutations > 0:
        rng = np.random.RandomState(random_seed)
        perm_counts = np.zeros((n_permutations, n_types, n_types), dtype=float)

        for p in range(n_permutations):
            for _, sub_df in groups:
                if len(sub_df) < 2:
                    continue
                shuffled_types = rng.permutation(sub_df[type_col].values)
                coords = sub_df[[x_col, y_col]].values
                perm_counts[p] += _count_contacts(shuffled_types, coords)

        perm_mean = np.mean(perm_counts, axis=0)
        perm_std = np.std(perm_counts, axis=0, ddof=1)
        perm_std = np.where(perm_std == 0, np.nan, perm_std)

        z_scores = (observed_counts - perm_mean) / perm_std
        z_scores = np.nan_to_num(z_scores, nan=0.0)

        # Empirical two-sided p-values
        more_extreme = np.sum(
            np.abs(perm_counts - perm_mean) >= np.abs(observed_counts - perm_mean),
            axis=0,
        )
        p_vals = (more_extreme + 1.0) / (n_permutations + 1.0)

        z_df = pd.DataFrame(
            np.round(z_scores, 4), index=unique_types, columns=unique_types
        )
        p_df = pd.DataFrame(
            np.round(p_vals, 4), index=unique_types, columns=unique_types
        )

    return CellInteractionResult(
        contact_counts=counts_df,
        interaction_frequencies=freq_df,
        enrichment_zscores=z_df,
        p_values=p_df,
    )


@dataclass
class VirtualTherapyResult:
    """Result of virtual targeted cell depletion / therapy simulation."""

    retained_cells: pd.DataFrame
    depleted_count: int
    baseline_entropy: float
    post_knockout_entropy: float
    entropy_delta: float
    disruption_score: float
    per_type_neighborhood_loss: Dict[str, float]


def simulate_cell_knockout(
    df: pd.DataFrame,
    target_types: Union[str, Sequence[str]],
    type_col: str = "cell_type",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    radius: float = 50.0,
    core_col: Optional[str] = None,
) -> VirtualTherapyResult:
    """Virtual therapy simulation: ablate target cell types and quantify niche disruption.

    Args:
        df: Baseline cells DataFrame.
        target_types: Cell phenotype name(s) to virtually knockout/ablate.
        type_col: Cell type column name.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        radius: Neighborhood radius in pixels.
        core_col: Optional core identifier column.

    Returns:
        VirtualTherapyResult with entropy changes, disruption score, and retained cells.
    """
    if isinstance(target_types, str):
        targets = {target_types}
    else:
        targets = set(target_types)

    for col in (type_col, x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from DataFrame.")

    # Baseline Shannon entropy over cell phenotype proportions
    baseline_counts = df[type_col].value_counts()
    p_base = baseline_counts.values / float(len(df))
    base_entropy = float(-np.sum(p_base * np.log(np.clip(p_base, 1e-12, 1.0))))

    # Deplete cells
    knockout_mask = df[type_col].isin(targets)
    depleted_count = int(np.sum(knockout_mask))
    retained_df = df[~knockout_mask].copy()

    if len(retained_df) == 0:
        return VirtualTherapyResult(
            retained_cells=retained_df,
            depleted_count=depleted_count,
            baseline_entropy=base_entropy,
            post_knockout_entropy=0.0,
            entropy_delta=-base_entropy,
            disruption_score=1.0,
            per_type_neighborhood_loss={},
        )

    # Post-knockout entropy
    post_counts = retained_df[type_col].value_counts()
    p_post = post_counts.values / float(len(retained_df))
    post_entropy = float(-np.sum(p_post * np.log(np.clip(p_post, 1e-12, 1.0))))

    # Quantify neighborhood contact loss for surviving cells
    coords_all = df[[x_col, y_col]].values
    tree_all = KDTree(coords_all)

    # Count how many target neighbors were adjacent to each remaining cell
    target_indices = set(np.where(knockout_mask.values)[0])
    surviving_indices = np.where((~knockout_mask).values)[0]

    all_neighbors = tree_all.query_ball_point(coords_all[surviving_indices], r=radius)
    loss_by_type: Dict[str, List[float]] = {
        t: [] for t in retained_df[type_col].unique()
    }

    total_lost_neighbors = 0
    for i, neighbors in enumerate(all_neighbors):
        cell_type = retained_df.iloc[i][type_col]
        lost_in_niche = len(set(neighbors).intersection(target_indices))
        loss_by_type[cell_type].append(lost_in_niche)
        total_lost_neighbors += lost_in_niche

    avg_loss_per_type = {
        t: float(np.mean(vals)) if vals else 0.0 for t, vals in loss_by_type.items()
    }
    avg_lost = float(total_lost_neighbors) / max(len(retained_df), 1)
    disruption_score = float(1.0 - np.exp(-avg_lost / 5.0))

    return VirtualTherapyResult(
        retained_cells=retained_df,
        depleted_count=depleted_count,
        baseline_entropy=base_entropy,
        post_knockout_entropy=post_entropy,
        entropy_delta=post_entropy - base_entropy,
        disruption_score=round(disruption_score, 4),
        per_type_neighborhood_loss=avg_loss_per_type,
    )


def filter_cells_by_polygon(
    df: pd.DataFrame,
    polygon_vertices: Sequence[Tuple[float, float]],
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    inside: bool = True,
) -> pd.DataFrame:
    """Filter cells inside or outside an arbitrary 2D polygon ROI.

    Args:
        df: Cells DataFrame with coordinates.
        polygon_vertices: Sequence of (x, y) coordinates defining polygon vertices.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        inside: If True, keep cells inside polygon; if False, keep cells outside.

    Returns:
        Filtered DataFrame containing only cells matching polygon ROI condition.
    """
    if len(polygon_vertices) < 3:
        raise ValueError("Polygon must contain at least 3 vertices.")

    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame.")

    poly_path = MplPath(np.asarray(polygon_vertices, dtype=float))
    pts = df[[x_col, y_col]].values
    mask = poly_path.contains_points(pts)

    if not inside:
        mask = ~mask

    return df[mask].copy()


def compute_distance_to_boundary(
    df: pd.DataFrame,
    boundary_points: Union[np.ndarray, Sequence[Tuple[float, float]]],
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    output_col: str = "distance_to_boundary",
) -> pd.DataFrame:
    """Compute minimum Euclidean distance from each cell to a boundary or tumor margin.

    Args:
        df: Cells DataFrame.
        boundary_points: Array or list of (x, y) coordinates forming the boundary curve/points.
        x_col: X coordinate column.
        y_col: Y coordinate column.
        output_col: Name of resulting distance column.

    Returns:
        DataFrame with distance_to_boundary column appended.
    """
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame.")

    boundary_pts = np.asarray(boundary_points, dtype=float)
    if len(boundary_pts) == 0:
        raise ValueError("boundary_points cannot be empty.")

    tree = KDTree(boundary_pts)
    cell_pts = df[[x_col, y_col]].values
    distances, _ = tree.query(cell_pts)

    result = df.copy()
    result[output_col] = np.round(distances, 2)
    return result
