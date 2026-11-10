"""Continuous HS–cycIF spatial correspondence and grouped marker-image recovery.

Consumes source-bound registered images from clinical_roi_valis.
Image-reviewed exploratory pairs remain distinct from landmark-validated pairs.
No segmentation, global predictive PCA, P/NP predictors, or pixel-level P values.
Raw files and upstream arrays are never modified. Figures are internal review.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REQUIRED_COLUMNS = {
    "assigned_tma",
    "confirmed_core_id",
    "marker",
    "image_path",
    "support_path",
    "hs_features_path",
    "geometry_path",
    "geometry_sha256",
    "grid_id",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "registration_receipt_path",
    "registration_receipt_sha256",
    "registration_accepted",
    "section_relationship",
    "section_evidence",
    "group_id",
    "registration_driver",
}
GEOMETRY_KEYS = {
    "axes",
    "native_shape_yx",
    "native_origin_xy",
    "downsample_factor",
    "block_center_offset_native_xy",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "assigned_tma",
    "confirmed_core_id",
    "hs_source_sha256",
}
MODEL_NAMES = (
    "training_mean",
    "total_only",
    "l1_ridge",
    "l1_logtotal_ridge",
    "l1_gradient_boosting",
)

_FILE_HASH_CACHE: dict[str, tuple[tuple[int, ...], str]] = {}


def _stage_identity(frame: pd.DataFrame, parameters: dict[str, Any]) -> dict[str, Any]:
    """Bind an analysis cache to inputs, settings, source code and runtime."""
    return {
        "input_rows": frame.to_dict("records"),
        "parameters": parameters,
        "code_sha256": file_sha256(__file__),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
        },
    }


def _matching_cache(
    receipt_path: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a cache only when every declared output still matches its digest."""
    if not receipt_path.is_file():
        return None
    receipt = _json(receipt_path)
    if receipt.get("identity") != identity or not receipt.get("artifacts"):
        return None
    for item in receipt["artifacts"]:
        path = Path(item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            return None
    return receipt


def file_sha256(path: str | Path) -> str:
    """Hash complete bytes once per process, guarded by immutable file state.

    The cache is never persisted. A new process, inode, size, mtime or ctime
    requires a fresh complete digest. A change during hashing is an error.
    """
    source = Path(path).resolve()
    with source.open("rb") as handle:
        # Opening refreshes network-filesystem attributes after a recent write.
        # Both snapshots refer to this same open descriptor, not stale pre-open
        # path metadata. No input file is modified by this operation.
        before = os.fstat(handle.fileno())
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        previous = _FILE_HASH_CACHE.get(str(source))
        if previous is not None and previous[0] == signature:
            digest = previous[1]
        else:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = os.fstat(handle.fileno())
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_after = source.stat()
    if (
        signature != after_signature
        or (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        != after_signature
    ):
        raise RuntimeError(f"Source changed while hashing: {source}")
    _FILE_HASH_CACHE[str(source)] = (signature, digest)
    return digest


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def strict_bool(value: object) -> bool:
    """Parse exported booleans without accepting nonempty strings as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid explicit boolean: {value!r}")


def _array(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source, mmap_mode="r", allow_pickle=False)
    if source.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        return tifffile.imread(source)
    raise ValueError(f"Unsupported quantitative array format: {source}")


def _core_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["assigned_tma"]), str(row["confirmed_core_id"])


def _resolve(base: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (base / path).resolve()


def audit_registered_manifest(
    manifest_path: str | Path,
    *,
    allow_exploratory: bool = False,
) -> pd.DataFrame:
    """Report eligibility; a rejected or unresolved row is never silently usable."""
    path = Path(manifest_path)
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Registration handoff is missing columns: {sorted(missing)}")
    keys = ["assigned_tma", "confirmed_core_id", "marker"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate TMA/core/marker handoff rows")
    if frame.empty:
        raise ValueError("Registration handoff has no accepted marker images")
    audit_rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        reasons: list[str] = []
        if not strict_bool(row["registration_accepted"]):
            reasons.append("registration_not_accepted")
        exploratory = (
            allow_exploratory
            and row.get("acceptance_basis") == "agent_reviewed_exploratory_images"
            and bool(str(row.get("validation_scope", "")).strip())
        )
        if (
            row["section_relationship"] not in {"same_section", "serial_section"}
            and not exploratory
        ):
            reasons.append("section_relationship_unresolved")
        if not str(row["section_evidence"]).strip() and not exploratory:
            reasons.append("section_evidence_missing")
        if strict_bool(row["registration_driver"]):
            reasons.append("registration_driver_not_independent_endpoint")
        section_note = {
            "same_section": "source-described same-section pair",
            "serial_section": "source-described serial-section regional correspondence",
        }.get(row["section_relationship"], "physical-section relationship unresolved")
        audit_rows.append(
            {
                **{key: row[key] for key in keys},
                "eligible": not reasons,
                "reason": ";".join(reasons),
                "group_id": row["group_id"],
                "section_relationship": row["section_relationship"],
                "acceptance_basis": row.get(
                    "acceptance_basis", "documented_registration_acceptance"
                ),
                "validation_scope": row.get("validation_scope", ""),
                "interpretation": (
                    f"{section_note}; image-reviewed fit without independent landmark TRE"
                    if exploratory
                    else "Source-documented registration and section relationship"
                ),
            }
        )
    return pd.DataFrame(audit_rows)


def load_registered_manifest(
    manifest_path: str | Path,
    *,
    allow_exploratory: bool = False,
) -> pd.DataFrame:
    """Validate accepted pairs, exact geometry, receipt authorization and all artifacts."""
    path = Path(manifest_path).resolve()
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    audit = audit_registered_manifest(path, allow_exploratory=allow_exploratory)
    eligible_keys = set(
        map(
            tuple,
            audit.loc[
                audit["eligible"], ["assigned_tma", "confirmed_core_id", "marker"]
            ].to_numpy(),
        )
    )
    selected = frame.loc[
        [
            tuple(row) in eligible_keys
            for row in frame[["assigned_tma", "confirmed_core_id", "marker"]].to_numpy()
        ]
    ].copy()
    if selected.empty:
        raise ValueError(
            "No eligible marker pairs in the notebook-05 registration handoff"
        )
    verified: dict[Path, str] = {}
    receipts: dict[Path, dict[str, Any]] = {}
    geometries: dict[Path, dict[str, Any]] = {}
    path_columns = [
        "image_path",
        "support_path",
        "hs_features_path",
        "geometry_path",
        "registration_receipt_path",
    ]
    for index, record in selected.iterrows():
        row = record.to_dict()
        for column in path_columns:
            row[column] = str(_resolve(path.parent, row[column]))
            selected.at[index, column] = row[column]
        receipt_path = Path(row["registration_receipt_path"])
        if receipt_path not in verified:
            verified[receipt_path] = file_sha256(receipt_path)
        digest = verified[receipt_path]
        if digest != str(row["registration_receipt_sha256"]):
            raise ValueError(f"Stale or tampered registration receipt: {receipt_path}")
        if receipt_path not in receipts:
            receipt = _json(receipt_path)
            artifacts = receipt.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError(
                    "Registration receipt has no hash-bound artifact inventory"
                )
            bindings: dict[str, str] = {}
            for artifact in artifacts:
                artifact_path = _resolve(receipt_path.parent, artifact["path"])
                expected = str(artifact["sha256"])
                actual = file_sha256(artifact_path)
                if actual != expected:
                    raise ValueError(
                        f"Stale or tampered registered artifact: {artifact_path}"
                    )
                if (
                    str(artifact_path) in bindings
                    and bindings[str(artifact_path)] != expected
                ):
                    raise ValueError("Conflicting registration artifact bindings")
                bindings[str(artifact_path)] = expected
            receipts[receipt_path] = {**receipt, "_artifact_bindings": bindings}
        receipt = receipts[receipt_path]
        accepted = {
            tuple(map(str, key)) for key in receipt.get("accepted_core_keys", [])
        }
        if _core_key(row) not in accepted:
            raise ValueError(
                "Core is not authorized by the computed registration acceptance receipt"
            )
        bound_rows = [
            item
            for item in receipt.get("canonical_rows", [])
            if _core_key(item) == _core_key(row) and item.get("marker") == row["marker"]
        ]
        if len(bound_rows) != 1:
            raise ValueError(
                "Receipt does not uniquely bind this core/marker handoff row"
            )
        bound = bound_rows[0]
        for column in (
            "image_path",
            "support_path",
            "hs_features_path",
            "geometry_path",
        ):
            if _resolve(receipt_path.parent, bound.get(column, "")) != Path(
                row[column]
            ):
                raise ValueError(
                    f"Handoff {column} differs from its accepted registration record"
                )
        for column in (
            "grid_id",
            "geometry_sha256",
            "section_relationship",
            "section_evidence",
            "group_id",
            "group_evidence",
        ):
            if str(bound.get(column, "")) != str(row.get(column, "")):
                raise ValueError(
                    f"Handoff {column} differs from its accepted registration record"
                )
        for column in ("acceptance_basis", "validation_scope"):
            if column in row and str(bound.get(column, "")) != str(row[column]):
                raise ValueError(
                    f"Handoff {column} differs from its registration review record"
                )
        for column in (
            "fitted_feature_count",
            "fitted_feature_median_um",
            "fitted_feature_p95_um",
            "independent_validation_n",
            "independent_tre_median_um",
            "independent_tre_p95_um",
            "registration_model",
            "valis_version",
            "simpleitk_version",
            "mi_cost_before",
            "mi_cost_after",
            "mi_optimizer_success",
            "mi_at_search_bound",
            "fine_inverse_angle_degrees",
            "fine_inverse_translation_xy_um",
            "fine_max_corner_displacement_um",
            "supplied_transform_parity_max_error_pixels",
            "unused_descriptor_match_count",
            "visual_review_notes",
        ):
            if column not in row:
                continue
            expected, actual = bound.get(column), row[column]
            if expected is None:
                matches = actual is None or str(actual).strip() == ""
            elif isinstance(expected, bool):
                try:
                    matches = strict_bool(expected) == strict_bool(actual)
                except ValueError:
                    matches = False
            elif isinstance(expected, (list, dict)):
                try:
                    decoded = json.loads(actual) if isinstance(actual, str) else actual
                    matches = expected == decoded
                except (TypeError, ValueError):
                    matches = False
            elif isinstance(expected, (int, float)):
                try:
                    matches = math.isclose(
                        float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = str(expected) == str(actual)
            if not matches:
                raise ValueError(
                    f"Handoff {column} differs from its source-bound registration quality"
                )
        for column in ("registration_accepted", "registration_driver"):
            if strict_bool(bound.get(column)) != strict_bool(row[column]):
                raise ValueError(
                    f"Handoff {column} differs from its accepted registration record"
                )
        for column in (
            "image_path",
            "support_path",
            "hs_features_path",
            "geometry_path",
        ):
            if str(Path(row[column]).resolve()) not in receipt["_artifact_bindings"]:
                raise ValueError(f"{column} is not bound to this registration receipt")
        geometry_path = Path(row["geometry_path"])
        if file_sha256(geometry_path) != str(row["geometry_sha256"]):
            raise ValueError("Geometry metadata hash mismatch")
        geometry = geometries.setdefault(geometry_path, _json(geometry_path))
        if GEOMETRY_KEYS - set(geometry):
            raise ValueError(
                f"Missing physical geometry fields: {sorted(GEOMETRY_KEYS - set(geometry))}"
            )
        canonical = json.dumps(
            {key: geometry[key] for key in GEOMETRY_KEYS},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != str(row["grid_id"]):
            raise ValueError("Marker and HS grid identity do not match")
        if geometry["axes"] != "YXC" or _core_key(geometry) != _core_key(row):
            raise ValueError("Physical geometry axes/core identity mismatch")
        for axis in ("x", "y"):
            scale = float(row[f"pixel_size_{axis}_um"])
            if (
                not math.isfinite(scale)
                or scale <= 0
                or not np.isclose(scale, float(geometry[f"pixel_size_{axis}_um"]))
            ):
                raise ValueError("Invalid or mismatched physical pixel pitch")
        factor = geometry["downsample_factor"]
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("Invalid integer downsample factor")
        if not np.allclose(
            geometry["block_center_offset_native_xy"], [(factor - 1) / 2] * 2
        ):
            raise ValueError("Unsupported native block-center convention")
        selected.at[index, "native_hs_pitch_x_um"] = (
            float(geometry["pixel_size_x_um"]) / factor
        )
        selected.at[index, "native_hs_pitch_y_um"] = (
            float(geometry["pixel_size_y_um"]) / factor
        )
    selected.attrs["manifest_path"] = str(path)
    selected.attrs["manifest_sha256"] = file_sha256(path)
    selected.attrs["exclusions"] = audit.loc[~audit["eligible"]].to_dict("records")
    selected.attrs["allow_exploratory"] = allow_exploratory
    return selected.reset_index(drop=True)


def load_registered_pair(
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one registered marker and matched HS features; return valid shared tissue."""
    with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
        l1 = np.array(bundle["l1"], dtype=np.float32)
        total = np.array(bundle["total"], dtype=np.float32)
        hs_support = np.array(bundle["support"])
    marker = np.asarray(_array(row["image_path"]), dtype=np.float32)
    marker_support = np.asarray(_array(row["support_path"]))
    if l1.ndim != 3 or l1.shape[2] != 20 or total.shape != l1.shape[:2]:
        raise ValueError("HS feature bundle requires matching HWC20 and HW arrays")
    if (
        marker.shape != total.shape
        or marker_support.shape != total.shape
        or hs_support.shape != total.shape
    ):
        raise ValueError(
            "Registered marker/support arrays are not on the exact HS anchor grid"
        )
    geometry = _json(row["geometry_path"])
    native_shape = geometry["native_shape_yx"]
    factor = geometry["downsample_factor"]
    expected_shape = tuple(int(value) // factor for value in native_shape)
    if total.shape != expected_shape:
        raise ValueError(
            "Feature bundle shape disagrees with its native-grid downsampling geometry"
        )
    if hs_support.dtype != np.bool_ or marker_support.dtype != np.bool_:
        raise ValueError(
            "Shared support must be categorical boolean, never interpolated intensities"
        )
    valid = hs_support & marker_support & np.isfinite(marker) & np.isfinite(total)
    valid &= np.isfinite(l1).all(axis=2) & (total > 0) & (l1 >= 0).all(axis=2)
    if not valid.any():
        raise ValueError("Pair has no finite observed shared tissue support")
    return l1, total, marker, valid


def load_domain_marker_pair(
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a genuine joint summed-HS anchor without inventing spectral features."""
    if row.get("domain_anchor_kind") != "joint_sum_hs":
        _, total, marker, valid = load_registered_pair(row)
        return total, marker, valid
    with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
        total = np.asarray(bundle["total"], dtype=np.float32)
        hs_support = np.asarray(bundle["support"])
    marker = np.asarray(_array(row["image_path"]), dtype=np.float32)
    marker_support = np.asarray(_array(row["support_path"]))
    geometry = _json(row["geometry_path"])
    if geometry.get("axes") != "YX" or total.ndim != 2:
        raise ValueError("Joint-domain anchor requires explicit YX geometry")
    if tuple(geometry["array_shape_yx"]) != total.shape:
        raise ValueError("Joint-domain summed anchor disagrees with exact crop geometry")
    if any(array.shape != total.shape for array in (marker, hs_support, marker_support)):
        raise ValueError("Joint-domain marker and supports must match the exact anchor grid")
    if hs_support.dtype != np.bool_ or marker_support.dtype != np.bool_:
        raise ValueError("Joint-domain support must be categorical boolean")
    valid = hs_support & marker_support & np.isfinite(marker) & np.isfinite(total) & (total > 0)
    if not valid.any():
        raise ValueError("Joint-domain pair has no finite observed shared tissue")
    return total, marker, valid


def continuous_pattern_metrics(
    observed: np.ndarray, predicted: np.ndarray, reference_mean: float
) -> dict[str, Any]:
    """Continuous image metrics; no pixel-level significance or constant-image score fabrication."""
    y = np.asarray(observed, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if (
        y.shape != p.shape
        or y.size == 0
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Metrics require nonempty, matched finite image observations")
    y_constant = bool(np.ptp(y) == 0)
    p_constant = bool(np.ptp(p) == 0)
    reason = (
        "constant_observed"
        if y_constant
        else ("constant_prediction" if p_constant else "")
    )
    rho = float("nan") if reason else float(spearmanr(y, p).statistic)
    squared_error = float(np.mean((y - p) ** 2))
    variance = float(np.mean((y - y.mean()) ** 2))
    return {
        "spearman": rho,
        "spearman_reason": reason,
        "r2": float("nan") if y_constant else 1.0 - squared_error / variance,
        "r2_reason": "constant_observed" if y_constant else "",
        "mse": squared_error,
        "training_mean_baseline_mse": float(np.mean((y - reference_mean) ** 2)),
        "spatial_observations": int(y.size),
    }


def _plot_style() -> None:
    utilities = Path(__file__).resolve().parents[1] / "01_utils"
    if str(utilities) not in sys.path:
        sys.path.insert(0, str(utilities))
    from plotnine_style_helper import apply_matplotlib_style

    apply_matplotlib_style()


def _display_range(values: np.ndarray) -> tuple[float, float]:
    lo, hi = map(float, np.percentile(values, [1, 99]))
    if hi <= lo:
        hi = lo + 1.0  # display-only range for constant images
    return lo, hi


def render_prediction_panel(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
    destination: str | Path,
) -> Path:
    """Render actual measured/predicted/residual images on common display scales."""
    import matplotlib.pyplot as plt

    _plot_style()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = _display_range(observed[valid])
    residual = observed - predicted
    half_range = float(np.max(np.abs(residual[valid]))) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 72 / 25.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.15, wspace=0.08)
    stride = max(1, int(np.ceil(max(valid.shape) / 768)))
    view_valid = valid[::stride, ::stride]
    for axis, image, name in zip(
        axes,
        (observed, predicted, residual),
        ("Measured cycIF", "Held-out prediction", "Measured minus predicted"),
    ):
        kwargs = (
            {"vmin": -half_range, "vmax": half_range, "cmap": "RdBu_r"}
            if name.startswith("Measured minus")
            else {"vmin": lo, "vmax": hi, "cmap": "magma"}
        )
        axis.imshow(
            np.ma.array(image[::stride, ::stride], mask=~view_valid),
            interpolation="nearest",
            **kwargs,
        )
        axis.axis("off")
        box = axis.get_position()
        fig.text(box.x0, 0.87, name, fontsize=5)
    fig.text(0.02, 0.96, label, fontsize=5)
    fig.text(
        0.02,
        0.06,
        "Matched physical grid; shared tissue only; display-only contrast. Serial sections: regional patterns.",
        fontsize=5,
    )
    fig.text(0.02, 0.025, "Generator: clinical_roi_pattern_recovery.py", fontsize=5)
    fig.savefig(path)
    plt.close(fig)
    return path


def describe_spatial_correspondence(
    manifest: pd.DataFrame,
    *,
    table_dir: str | Path,
    plot_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Describe marker–spectrum image correspondence, without fitting a recovery model."""
    from scipy.ndimage import sobel
    import matplotlib.pyplot as plt

    tables, plots = Path(table_dir), Path(plot_dir)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    output = tables / "hs_cycif_spatial_correspondence.csv"
    receipt_path = tables / "spatial_correspondence_receipt.json"
    identity = _stage_identity(
        manifest, {"method": "within-image Spearman per L1 band"}
    )
    previous = None if force else _matching_cache(receipt_path, identity)
    if previous is not None:
        print("CACHE_HIT descriptive spatial correspondence")
        return {
            "summary": pd.read_csv(output, float_precision="round_trip"),
            "pdfs": [Path(path) for path in previous["pdfs"]],
            "table_path": output,
            "receipt_path": receipt_path,
            "cache_status": "CACHE_HIT",
        }
    _plot_style()
    records: list[dict[str, Any]] = []
    galleries: list[Path] = []
    for row in manifest.to_dict("records"):
        l1, total, marker, valid = load_registered_pair(row)
        y = marker[valid]
        for channel in range(20):
            values = l1[:, :, channel][valid]
            result = continuous_pattern_metrics(y, values, float(y.mean()))
            records.append(
                {
                    "assigned_tma": row["assigned_tma"],
                    "confirmed_core_id": row["confirmed_core_id"],
                    "marker": row["marker"],
                    "channel_index": channel,
                    "spearman": result["spearman"],
                    "reason": result["spearman_reason"],
                    "spatial_observations": result["spatial_observations"],
                    "claim_label": "Exploratory",
                    "interpretation": "within-image spatial association, not held-out recovery",
                }
            )
        stride = max(1, int(np.ceil(max(marker.shape) / 768)))
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 72 / 25.4))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.15, wspace=0.08)
        for axis, image, title in zip(
            axes,
            (np.log1p(total), marker, np.hypot(sobel(marker, 0), sobel(marker, 1))),
            ("HS log total", "Registered cycIF marker", "Marker edges (display only)"),
        ):
            lo, hi = _display_range(image[valid])
            axis.imshow(
                np.ma.array(image[::stride, ::stride], mask=~valid[::stride, ::stride]),
                cmap="magma",
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
            )
            axis.axis("off")
            fig.text(axis.get_position().x0, 0.87, title, fontsize=5)
        label = f"{row['assigned_tma']} {row['confirmed_core_id']} {row['marker']}"
        fig.text(0.02, 0.96, label, fontsize=5)
        fig.text(
            0.02,
            0.05,
            "Descriptive spatial correspondence only; independent scaling for different measurements.",
            fontsize=5,
        )
        fig.text(0.02, 0.025, "Generator: clinical_roi_pattern_recovery.py", fontsize=5)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        path = plots / f"{safe}_registered_correspondence.pdf"
        fig.savefig(path)
        plt.close(fig)
        galleries.append(path)
    summary = pd.DataFrame(records)
    summary.to_csv(output, index=False)
    _save_json(
        receipt_path,
        {
            "identity": identity,
            "pdfs": [str(path.resolve()) for path in galleries],
            "artifacts": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in [output, *galleries]
            ],
        },
    )
    print("RECOMPUTE descriptive spatial correspondence")
    return {
        "summary": summary,
        "pdfs": galleries,
        "table_path": output,
        "receipt_path": receipt_path,
        "cache_status": "COMPUTED",
    }


def cluster_marker_correspondence(
    row: dict[str, Any],
    labels_path: str | Path,
    *,
    label_geometry_path: str | Path,
    palette_path: str | Path,
    cluster_count: int,
    table_dir: str | Path,
    plot_dir: str | Path,
) -> dict[str, Any]:
    """Measure continuous marker enrichment within actual same-grid cluster domains."""
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    _plot_style()
    geometry = _json(label_geometry_path)
    marker_geometry = _json(row["geometry_path"])
    if geometry != marker_geometry:
        raise ValueError(
            "Cluster labels do not carry the exact registration-anchor geometry"
        )
    labels = np.asarray(_array(labels_path))
    _, marker, valid = load_domain_marker_pair(row)
    if (
        labels.shape != marker.shape
        or not np.issubdtype(labels.dtype, np.integer)
        or (labels < 0).any()
    ):
        raise ValueError("Labels must be nonnegative integers on the exact shared grid")
    palette = np.asarray(np.load(palette_path, allow_pickle=False), dtype=float)
    if (
        not isinstance(cluster_count, int)
        or cluster_count < 2
        or palette.shape != (cluster_count + 1, 3)
        or not np.isfinite(palette).all()
        or (palette < 0).any()
        or (palette > 1).any()
        or int(labels.max()) > cluster_count
    ):
        raise ValueError(
            "Supply the frozen model palette as (K+1,3) RGB values in [0,1], including ID0"
        )
    color_map = ListedColormap(palette)
    color_norm = BoundaryNorm(np.arange(cluster_count + 2) - 0.5, color_map.N)
    valid &= labels > 0
    if not valid.any():
        raise ValueError("No observed cluster domain intersects shared tissue")
    tissue_mean = float(marker[valid].mean())
    records = []
    for cluster_id in np.unique(labels[valid]):
        observations = marker[valid & (labels == cluster_id)]
        mean = float(observations.mean())
        records.append(
            {
                "cluster_id": int(cluster_id),
                "mean_marker_intensity": mean,
                "median_marker_intensity": float(np.median(observations)),
                "tissue_mean_marker_intensity": tissue_mean,
                "enrichment_ratio": mean / tissue_mean
                if tissue_mean > 0
                else float("nan"),
                "valid_spatial_observations": int(observations.size),
                "interpretation": "descriptive continuous intensity; no marker-positive threshold",
            }
        )
    tables, plots = Path(table_dir), Path(plot_dir)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    prefix = f"{row['assigned_tma']}_{row['confirmed_core_id']}_{row['marker']}_{file_sha256(labels_path)[:12]}"
    prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)
    frame = pd.DataFrame(records)
    frame.to_csv(tables / f"{prefix}_cluster_marker_enrichment.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 72 / 25.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.29)
    lo, hi = _display_range(marker[valid])
    for axis in axes:
        axis.axis("off")
    marker_artist = axes[0].imshow(np.ma.array(marker, mask=~valid), cmap="magma", vmin=lo, vmax=hi)
    axes[1].imshow(
        np.ma.array(labels, mask=~valid),
        cmap=color_map,
        norm=color_norm,
        interpolation="nearest",
    )
    axes[2].imshow(np.ma.array(marker, mask=~valid), cmap="gray", vmin=lo, vmax=hi)
    boundary = find_boundaries(labels, mode="inner") & valid
    rgba = np.zeros((*boundary.shape, 4), dtype=np.float32)
    rgba[boundary] = (0, 1, 1, 0.2)
    axes[2].imshow(rgba, interpolation="nearest")
    for axis, text in zip(
        axes, ("Measured marker", row.get("domain_label", "Current HS labels"), "Domain boundaries on marker")
    ):
        fig.text(axis.get_position().x0, 0.87, text, fontsize=5)
    fig.text(0.02, 0.96, prefix, fontsize=5)
    colorbar_axis = fig.add_axes((0.04, 0.19, 0.23, 0.025))
    colorbar = fig.colorbar(marker_artist, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Continuous marker intensity (a.u.)", fontsize=5)
    colorbar.ax.tick_params(labelsize=5, width=0.5, length=2, pad=1)
    colorbar.outline.set_linewidth(0.5)
    fig.legend(
        handles=[
            Patch(facecolor=palette[domain], edgecolor="#444444", linewidth=0.5, label=str(domain))
            for domain in range(1, cluster_count + 1)
        ],
        title="Domain ID", loc="lower center", bbox_to_anchor=(0.65, 0.13),
        ncol=min(cluster_count, 9), frameon=False, fontsize=5, title_fontsize=5,
        handlelength=1, columnspacing=0.8,
    )
    fig.text(
        0.02, 0.065,
        "Dense native domain boundaries shown at 20% opacity; labels and quantitative marker arrays unchanged.",
        fontsize=5,
    )
    fig.text(
        0.02,
        0.025,
        "Generator: clinical_roi_pattern_recovery.py; descriptive, not validated cell identity",
        fontsize=5,
    )
    pdf = plots / f"{prefix}_cluster_marker_overlay.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return {"summary": frame, "pdf": pdf}


def run_frozen_domain_atlas(
    manifest: pd.DataFrame,
    labels_manifest: pd.DataFrame,
    *,
    table_dir: str | Path,
    plot_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Describe all frozen-domain/marker pairs without selecting or refitting K."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    tables, plots = Path(table_dir), Path(plot_dir)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    identity = _stage_identity(
        manifest,
        {
            "labels": labels_manifest.to_dict("records"),
            "enrichment": "domain mean / mean over label-positive shared tissue",
        },
    )
    receipt_path = tables / "frozen_domain_atlas_receipt.json"
    summary_path = tables / "frozen_domain_marker_enrichment.csv"
    previous = None if force else _matching_cache(receipt_path, identity)
    if previous is not None:
        print("CACHE_HIT frozen-domain marker atlas")
        return {
            "summary": pd.read_csv(summary_path, float_precision="round_trip"),
            "pdfs": [Path(path) for path in previous["pdfs"]],
            "heatmap": Path(previous["heatmap"]),
            "receipt_path": receipt_path,
            "cache_status": "CACHE_HIT",
        }
    summaries: list[pd.DataFrame] = []
    pdfs: list[Path] = []
    for label in labels_manifest.to_dict("records"):
        for path_key, hash_key in (
            ("labels_path", "labels_sha256"),
            ("palette_path", "palette_sha256"),
            ("label_geometry_path", "geometry_sha256"),
        ):
            if file_sha256(label[path_key]) != label[hash_key]:
                raise ValueError(f"Frozen-domain input changed: {label[path_key]}")
        matches = manifest.loc[
            manifest["assigned_tma"].eq(label["assigned_tma"])
            & manifest["confirmed_core_id"].eq(label["confirmed_core_id"])
        ]
        for row in matches.to_dict("records"):
            if row["grid_id"] != label["grid_id"]:
                raise ValueError("Frozen HS domain and marker grid IDs differ")
            result = cluster_marker_correspondence(
                row,
                label["labels_path"],
                label_geometry_path=label["label_geometry_path"],
                palette_path=label["palette_path"],
                cluster_count=int(label["cluster_count"]),
                table_dir=tables,
                plot_dir=plots,
            )
            summaries.append(
                result["summary"].assign(
                    assigned_tma=row["assigned_tma"],
                    confirmed_core_id=row["confirmed_core_id"],
                    marker=row["marker"],
                    method=label["method"],
                    cluster_count=int(label["cluster_count"]),
                    claim_label="Exploratory",
                )
            )
            pdfs.append(result["pdf"])
    if not summaries:
        raise ValueError("No frozen HS domains match the registered marker inputs")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(summary_path, index=False)
    heatmap = plots / "frozen_domain_marker_enrichment_heatmaps.pdf"
    _plot_style()
    with PdfPages(heatmap) as document:
        for (method, k, marker), group in summary.groupby(
            ["method", "cluster_count", "marker"],
            sort=True,
        ):
            group = group.assign(
                core=group["assigned_tma"] + ":" + group["confirmed_core_id"]
            )
            values = group.pivot(
                index="core", columns="cluster_id", values="enrichment_ratio"
            ).reindex(columns=range(1, int(k) + 1))
            matrix = values.to_numpy(dtype=float)
            logged = np.full(matrix.shape, np.nan)
            positive = np.isfinite(matrix) & (matrix > 0)
            logged[positive] = np.log2(matrix[positive])
            finite = np.abs(logged[np.isfinite(logged)])
            extent = max(float(finite.max()) if finite.size else 0.0, 1e-6)
            fig, ax = plt.subplots(figsize=(150 / 25.4, 120 / 25.4))
            fig.subplots_adjust(left=0.20, right=0.80, bottom=0.18, top=0.88)
            palette = plt.colormaps["RdBu_r"].copy()
            palette.set_bad("#D9D9D9")
            nrows, ncols = logged.shape
            artist = ax.pcolormesh(
                np.arange(ncols + 1) - 0.5,
                np.arange(nrows + 1) - 0.5,
                np.ma.masked_invalid(logged),
                cmap=palette,
                vmin=-extent,
                vmax=extent,
                shading="flat",
                edgecolors="none",
                linewidth=0,
                antialiased=False,
                rasterized=False,
                snap=True,
            )
            ax.set_xlim(-0.5, ncols - 0.5)
            ax.set_ylim(nrows - 0.5, -0.5)
            ax.grid(False)
            ax.set_xticks(np.arange(len(values.columns)), labels=values.columns)
            ax.set_yticks(np.arange(len(values.index)), labels=values.index)
            ax.set_xlabel("Frozen spectral domain ID")
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.04, pad=0.04)
            if colorbar.solids is not None:
                colorbar.solids.set_rasterized(False)
            colorbar.set_label("log2(domain mean / shared-tissue mean)")
            fig.text(0.02, 0.96, f"{marker}; {method}; K={k}", fontsize=5)
            fig.text(
                0.02,
                0.055,
                "Descriptive continuous signal; gray = absent/undefined. Domains are not cell-type labels.",
                fontsize=5,
            )
            fig.text(
                0.02, 0.025, "Generator: clinical_roi_pattern_recovery.py", fontsize=5
            )
            document.savefig(fig)
            plt.close(fig)
    _save_json(
        receipt_path,
        {
            "identity": identity,
            "pdfs": [str(path.resolve()) for path in pdfs],
            "heatmap": str(heatmap.resolve()),
            "artifacts": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in [summary_path, heatmap, *pdfs]
            ],
        },
    )
    print("RECOMPUTE frozen-domain marker atlas; labels and palette unchanged")
    return {
        "summary": summary,
        "pdfs": pdfs,
        "heatmap": heatmap,
        "receipt_path": receipt_path,
        "cache_status": "COMPUTED",
    }


def grouped_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave one entire verified biological group out, never split neighboring pixels."""
    values = np.asarray(groups).astype(str)
    unique = np.unique(values)
    if len(unique) < 3 or any(not value.strip() for value in unique):
        raise ValueError(
            "Held-out recovery requires at least three explicitly verified independent groups"
        )
    return [
        (np.flatnonzero(values != group), np.flatnonzero(values == group))
        for group in unique
    ]


def prepare_validation_groups(
    manifest: pd.DataFrame,
    *,
    grouping_policy: str,
) -> pd.DataFrame:
    """Separate a computational holdout unit from verified biological identity."""
    result = manifest.copy()
    if grouping_policy == "verified_groups":
        if "group_evidence" not in result:
            raise ValueError("Verified groups require source-backed group_evidence")
        for column in ("group_id", "group_evidence"):
            if result[column].astype(str).str.strip().eq("").any():
                raise ValueError(
                    "Verified group_id and group_evidence must not be empty"
                )
        result["validation_group_id"] = result["group_id"].astype(str)
        result["validation_unit"] = "source_verified_group"
        result["independence_status"] = "as documented in group_evidence"
    elif grouping_policy == "core_blocked_descriptive":
        result["validation_group_id"] = (
            result["assigned_tma"].astype(str)
            + ":"
            + result["confirmed_core_id"].astype(str)
        )
        result["validation_unit"] = "TMA-qualified core image"
        result["independence_status"] = (
            "Patient identity/independence not established; descriptive transfer between core images"
        )
    else:
        raise ValueError("Use verified_groups or core_blocked_descriptive")
    counts = result.groupby(["assigned_tma", "confirmed_core_id"])[
        "validation_group_id"
    ].nunique()
    if (counts != 1).any():
        raise ValueError("One core cannot appear in multiple validation groups")
    return result


def _model_features(l1: np.ndarray, total: np.ndarray, name: str) -> np.ndarray:
    if name == "total_only":
        return np.log1p(total).reshape(-1, 1)
    if name == "l1_logtotal_ridge":
        return np.column_stack((l1, np.log1p(total)))
    return l1


def _fit_predictor(
    x: np.ndarray, y: np.ndarray, name: str, *, ridge_alpha: float, random_state: int
) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "l1_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        )
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
    return model.fit(x, y)


def run_grouped_marker_recovery(
    manifest: pd.DataFrame,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    plot_dir: str | Path,
    samples_per_core: int = 2000,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    models: tuple[str, ...] = MODEL_NAMES,
    prediction_chunk: int = 65536,
    grouping_policy: str = "verified_groups",
    allow_empty: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Predict complete held-out marker images from native-derived L1 spectra.

    Model settings are fixed before inspecting held-out images. All learned
    preprocessing is fit on training groups; no globally fit PCA is consumed.
    Per-core sampling balances training contributions. No pixel-based inference.
    """
    from threadpoolctl import threadpool_limits

    if (
        samples_per_core < 1
        or prediction_chunk < 1
        or ridge_alpha <= 0
        or not math.isfinite(ridge_alpha)
    ):
        raise ValueError(
            "Positive sampling/chunk limits and finite ridge alpha are required"
        )
    if (
        not models
        or len(set(models)) != len(models)
        or not set(models) <= set(MODEL_NAMES)
    ):
        raise ValueError(
            f"Model names must be a distinct nonempty subset of {MODEL_NAMES}"
        )
    manifest = prepare_validation_groups(manifest, grouping_policy=grouping_policy)
    objects, tables, plots = Path(object_dir), Path(table_dir), Path(plot_dir)
    for directory in (objects, tables, plots):
        directory.mkdir(parents=True, exist_ok=True)
    # Revalidate through the upstream manifest, including every bound artifact,
    # before considering persisted predictions reusable.
    source_manifest = manifest.attrs.get("manifest_path")
    if source_manifest:
        current = load_registered_manifest(
            source_manifest,
            allow_exploratory=manifest.attrs.get("allow_exploratory", False),
        )
        if current.attrs["manifest_sha256"] != manifest.attrs.get("manifest_sha256"):
            raise ValueError("Registration manifest changed after it was loaded")
    parameters = {
        "models": list(models),
        "ridge_alpha": ridge_alpha,
        "samples_per_core": samples_per_core,
        "random_state": random_state,
        "prediction_chunk": prediction_chunk,
        "grouping_policy": grouping_policy,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn")
        },
        "primary_endpoint": "held-out image Spearman over valid shared tissue",
    }
    input_identity = hashlib.sha256(
        json.dumps(
            manifest.to_dict("records"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    code_identity = file_sha256(__file__)
    receipt_path = tables / "heldout_marker_recovery_receipt.json"
    metric_path = tables / "heldout_marker_recovery_metrics.csv"
    prediction_table = tables / "heldout_marker_recovery_predictions.csv"
    summary_path = tables / "heldout_marker_recovery_by_tma.csv"
    if receipt_path.is_file() and not force:
        previous = _json(receipt_path)
        if (
            previous.get("parameters") == parameters
            and previous.get("input_identity") == input_identity
            and previous.get("code_sha256") == code_identity
            and previous.get("artifacts")
            and all(
                Path(item["path"]).is_file()
                and file_sha256(item["path"]) == item["sha256"]
                for item in previous["artifacts"]
            )
        ):
            return {
                "metrics": pd.read_csv(metric_path, float_precision="round_trip"),
                "predictions": pd.read_csv(
                    prediction_table, float_precision="round_trip"
                ),
                "summary": pd.read_csv(summary_path, float_precision="round_trip"),
                "receipt_path": receipt_path,
                "cache_status": "CACHE_HIT",
            }
    metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    rng = np.random.default_rng(random_state)
    for marker, marker_frame in manifest.groupby("marker", sort=True):
        rows = marker_frame.to_dict("records")
        groups = np.array([str(row["validation_group_id"]) for row in rows])
        minimum_groups = 3 if grouping_policy == "verified_groups" else 2
        if len(np.unique(groups)) < minimum_groups:
            skipped.append(
                {
                    "marker": str(marker),
                    "reason": f"fewer_than_{minimum_groups}_available_holdout_groups",
                }
            )
            continue
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row in rows:
            l1, total, observed, valid = load_registered_pair(row)
            population = np.flatnonzero(valid)
            indices = rng.choice(
                population, size=min(samples_per_core, len(population)), replace=False
            )
            samples.append(
                (
                    l1.reshape(-1, 20)[indices],
                    total.ravel()[indices],
                    observed.ravel()[indices],
                )
            )
        splits = [
            (np.flatnonzero(groups != group), np.flatnonzero(groups == group))
            for group in np.unique(groups)
        ]
        for train_indices, test_indices in splits:
            print(
                f"{marker}: held-out group {groups[test_indices[0]]}; "
                f"training core images {len(train_indices)}; model arms {len(models)}",
                flush=True,
            )
            # Equal contribution per core, including when one core has fewer valid observations.
            fit_count = min(len(samples[index][2]) for index in train_indices)
            train_l1 = np.concatenate(
                [samples[index][0][:fit_count] for index in train_indices]
            )
            train_total = np.concatenate(
                [samples[index][1][:fit_count] for index in train_indices]
            )
            train_y = np.concatenate(
                [samples[index][2][:fit_count] for index in train_indices]
            )
            reference_mean = float(train_y.mean())
            fitted: dict[str, Any] = {}
            with threadpool_limits(limits=1):
                for name in models:
                    if name != "training_mean":
                        fitted[name] = _fit_predictor(
                            _model_features(train_l1, train_total, name),
                            train_y,
                            name,
                            ridge_alpha=ridge_alpha,
                            random_state=random_state,
                        )
            for index in test_indices:
                row = rows[index]
                l1, total, observed, valid = load_registered_pair(row)
                spatial_indices = np.flatnonzero(valid)
                for name in models:
                    prediction = np.full(observed.shape, np.nan, dtype=np.float32)
                    with threadpool_limits(limits=1):
                        for start in range(0, len(spatial_indices), prediction_chunk):
                            chosen = spatial_indices[start : start + prediction_chunk]
                            if name == "training_mean":
                                predicted = np.full(len(chosen), reference_mean)
                            else:
                                x = _model_features(
                                    l1.reshape(-1, 20)[chosen],
                                    total.ravel()[chosen],
                                    name,
                                )
                                predicted = fitted[name].predict(x)
                            prediction.ravel()[chosen] = predicted
                    result = continuous_pattern_metrics(
                        observed[valid], prediction[valid], reference_mean
                    )
                    record = {
                        "assigned_tma": row["assigned_tma"],
                        "confirmed_core_id": row["confirmed_core_id"],
                        "group_id": row["group_id"],
                        "marker": marker,
                        "model": name,
                        "validation_unit": row["validation_unit"],
                        "independence_status": row["independence_status"],
                        "heldout_group": groups[index],
                        "training_groups": json.dumps(
                            sorted(set(groups[train_indices]))
                        ),
                        "training_pixels_per_core": fit_count,
                        "training_core_count": len(train_indices),
                        "pixel_size_x_um": row["pixel_size_x_um"],
                        "pixel_size_y_um": row["pixel_size_y_um"],
                        "native_hs_pitch_x_um": row.get(
                            "native_hs_pitch_x_um",
                            float(row["pixel_size_x_um"])
                            / _json(row["geometry_path"])["downsample_factor"],
                        ),
                        "section_relationship": row["section_relationship"],
                        "claim_label": "Exploratory",
                        **result,
                    }
                    metric_records.append(record)
                    prefix = f"{row['assigned_tma']}_{row['confirmed_core_id']}_{marker}_{name}"
                    prefix = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in prefix
                    )
                    prediction_path = objects / f"{prefix}_heldout_prediction.npy"
                    np.save(prediction_path, prediction)
                    pdf = render_prediction_panel(
                        observed,
                        prediction,
                        valid,
                        label=prefix,
                        destination=plots / f"{prefix}_heldout_recovery.pdf",
                    )
                    prediction_records.append(
                        {
                            **{
                                key: record[key]
                                for key in (
                                    "assigned_tma",
                                    "confirmed_core_id",
                                    "group_id",
                                    "marker",
                                    "model",
                                )
                            },
                            "prediction_path": str(prediction_path.resolve()),
                            "pdf_path": str(pdf.resolve()),
                            "grid_id": row["grid_id"],
                        }
                    )
                    for artifact in (prediction_path, pdf):
                        artifacts.append(
                            {
                                "path": str(artifact.resolve()),
                                "sha256": file_sha256(artifact),
                            }
                        )
                    del prediction
    metrics = pd.DataFrame(metric_records)
    predictions = pd.DataFrame(prediction_records)
    pd.DataFrame(skipped, columns=["marker", "reason"]).to_csv(
        tables / "marker_recovery_exclusions.csv", index=False
    )
    if metrics.empty and not allow_empty:
        raise ValueError(
            "No marker has enough verified independent paired groups; inspect marker_recovery_exclusions.csv"
        )
    if metrics.empty:
        metrics = pd.DataFrame(
            columns=[
                "assigned_tma",
                "confirmed_core_id",
                "marker",
                "model",
                "heldout_group",
                "spearman",
                "r2",
                "mse",
                "native_hs_pitch_x_um",
            ]
        )
        predictions = pd.DataFrame(
            columns=[
                "assigned_tma",
                "confirmed_core_id",
                "group_id",
                "marker",
                "model",
                "prediction_path",
                "pdf_path",
                "grid_id",
            ]
        )
    # Preserve every tested marker/model and held-out sample, not only winners.
    metrics.to_csv(metric_path, index=False)
    predictions.to_csv(prediction_table, index=False)
    summary = (
        metrics.groupby(
            ["assigned_tma", "native_hs_pitch_x_um", "marker", "model"], dropna=False
        )
        .agg(
            heldout_cores=("confirmed_core_id", "count"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            mean_r2=("r2", "mean"),
            mean_mse=("mse", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    receipt = {
        "schema": "clinical_hs_cycif_pattern_recovery_v1",
        "claim_label": "Exploratory",
        "parameters": parameters,
        "input_identity": input_identity,
        "code_sha256": file_sha256(__file__),
        "input_rows": manifest.to_dict("records"),
        "validation": f"{grouping_policy}; entire group held out; training-only scaling; fixed hyperparameters",
        "limitations": "No patient independence inferred from core names; no calibrated protein recovery or same-cell identity inferred; no pixel-level P values.",
        "artifacts": artifacts
        + [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in (metric_path, prediction_table, summary_path)
        ],
    }
    _save_json(receipt_path, receipt)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "summary": summary,
        "receipt_path": receipt_path,
        "cache_status": "COMPUTED",
    }


def run_checkpointed_marker_recovery(
    manifest: pd.DataFrame,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    plot_dir: str | Path,
    samples_per_core: int = 2000,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    models: tuple[str, ...] = MODEL_NAMES,
    grouping_policy: str = "verified_groups",
    force: bool = False,
) -> dict[str, Any]:
    """Checkpoint each marker independently while reusing the canonical fitter."""
    objects, tables, plots = map(Path, (object_dir, table_dir, plot_dir))
    for directory in (objects, tables, plots):
        directory.mkdir(parents=True, exist_ok=True)
    source_manifest = manifest.attrs.get("manifest_path")
    if source_manifest:
        current = load_registered_manifest(
            source_manifest,
            allow_exploratory=manifest.attrs.get("allow_exploratory", False),
        )
        if current.attrs["manifest_sha256"] != manifest.attrs.get("manifest_sha256"):
            raise ValueError("Registration manifest changed before checkpoint recovery")
    outputs = {
        "metrics": tables / "all_marker_heldout_metrics.csv",
        "predictions": tables / "all_marker_heldout_predictions.csv",
        "summary": tables / "all_marker_heldout_summary.csv",
    }
    receipt_path = tables / "checkpointed_marker_recovery_receipt.json"
    identity = _stage_identity(
        manifest,
        {
            "samples_per_core": samples_per_core,
            "random_state": random_state,
            "ridge_alpha": ridge_alpha,
            "models": list(models),
            "grouping_policy": grouping_policy,
        },
    )
    previous = None if force else _matching_cache(receipt_path, identity)
    if previous is not None:
        checkpoints_valid = all(
            Path(item["path"]).is_file()
            and file_sha256(item["path"]) == item["sha256"]
            and all(
                Path(artifact["path"]).is_file()
                and file_sha256(artifact["path"]) == artifact["sha256"]
                for artifact in _json(item["path"])["artifacts"]
            )
            for item in previous["marker_receipts"]
        )
        if checkpoints_valid:
            print("CACHE_HIT all marker checkpoints and aggregate tables", flush=True)
            return {
                **{
                    name: pd.read_csv(path, float_precision="round_trip")
                    for name, path in outputs.items()
                },
                "receipt_path": receipt_path,
                "cache_status": "CACHE_HIT",
            }
    blocks: list[dict[str, Any]] = []
    for marker, subset in manifest.groupby("marker", sort=True):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(marker))
        # The digest prevents two distinct biological marker names collapsing
        # into one filesystem name after punctuation replacement.
        safe += "_" + hashlib.sha256(str(marker).encode()).hexdigest()[:8]
        print(f"Whole-group recovery checkpoint: {marker}", flush=True)
        block = run_grouped_marker_recovery(
            subset.copy(),
            object_dir=objects / safe,
            table_dir=tables / safe,
            plot_dir=plots / safe,
            samples_per_core=samples_per_core,
            random_state=random_state,
            ridge_alpha=ridge_alpha,
            models=models,
            grouping_policy=grouping_policy,
            allow_empty=True,
            force=force,
        )
        print(f"{marker}: {block['cache_status']}", flush=True)
        blocks.append(block)
    if not blocks:
        raise ValueError("No marker inputs supplied")
    metrics = pd.concat([block["metrics"] for block in blocks], ignore_index=True)
    predictions = pd.concat(
        [block["predictions"] for block in blocks], ignore_index=True
    )
    summary = pd.concat([block["summary"] for block in blocks], ignore_index=True)
    for name, frame in (
        ("metrics", metrics),
        ("predictions", predictions),
        ("summary", summary),
    ):
        frame.to_csv(outputs[name], index=False)
    _save_json(
        receipt_path,
        {
            "identity": identity,
            "claim_label": "Exploratory",
            "checkpoint_unit": "one complete marker across all its held-out groups",
            "marker_receipts": [
                {
                    "path": str(block["receipt_path"]),
                    "sha256": file_sha256(block["receipt_path"]),
                }
                for block in blocks
            ],
            "artifacts": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in outputs.values()
            ],
        },
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "summary": summary,
        "receipt_path": receipt_path,
        "cache_status": "CACHE_HIT"
        if all(block["cache_status"] == "CACHE_HIT" for block in blocks)
        else "COMPUTED",
    }


def export_current_hs_cluster_maps(
    hs_results: dict[str, Any],
    cohort: dict[str, Any],
    registered_manifest: pd.DataFrame,
    *,
    object_dir: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """Export current notebook-03 frozen labels into accepted registration grids.

    Call with its actual result dictionaries or the saved-receipt adapter below.
    No fitting, spectral warping, or reconstruction of historical label maps.
    The current HS producer samples native indices 0, stride, 2*stride, ...;
    use that convention explicitly to map labels with nearest interpolation.
    """
    from scipy.ndimage import map_coordinates
    import matplotlib.pyplot as plt

    maps = hs_results.get("label_maps")
    if not isinstance(maps, dict) or not maps:
        raise ValueError(
            "Current HS result has no actual label_maps; export after its real clustering cell, not a receipt-only cache return"
        )
    display_inputs = cohort["display_inputs"]
    summary = cohort["summary"].set_index(
        ["assigned_tma", "confirmed_core_id"], verify_integrity=True
    )
    stride = int(cohort["parameters"]["display_stride"])
    if stride < 1:
        raise ValueError("Invalid recorded HS display stride")
    directory = Path(object_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    targets = registered_manifest.drop_duplicates(["assigned_tma", "confirmed_core_id"])
    for (method, k), core_maps in maps.items():
        k = int(k)
        # Exact palette of the current notebook-03 producer, not a per-core remap.
        palette = np.vstack(
            (np.zeros((1, 3)), np.asarray(plt.get_cmap("tab20").colors[:k]))
        )
        if palette.shape != (k + 1, 3):
            raise ValueError("Cluster count exceeds the frozen producer palette")
        safe_method = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(method)
        )
        palette_path = directory / f"{safe_method}_k{k}_frozen_palette.npy"
        np.save(palette_path, palette)
        for target in targets.to_dict("records"):
            tma, core = _core_key(target)
            key = f"{tma}_{core}"
            if key not in core_maps or key not in display_inputs:
                raise ValueError(f"Current frozen HS model has no map for {key}")
            source_geometry = display_inputs[key]
            geometry = _json(target["geometry_path"])
            if str(summary.loc[(tma, core), "full_file_sha256"]) != str(
                geometry["hs_source_sha256"]
            ):
                raise ValueError(
                    "Frozen cluster and registration use different raw HS source bytes"
                )
            if not np.allclose(
                [source_geometry["mosaic_x"], source_geometry["mosaic_y"]],
                geometry["native_origin_xy"],
            ):
                raise ValueError(
                    "Frozen cluster and registration have different native mosaic origins"
                )
            if float(source_geometry["scale_factor"]) != 1.0:
                raise ValueError(
                    "Unsupported HS producer source scale; do not guess geometry"
                )
            labels = np.asarray(core_maps[key])
            if labels.shape != tuple(source_geometry["shape"]) or not np.issubdtype(
                labels.dtype, np.integer
            ):
                raise ValueError(
                    "Frozen cluster map shape/dtype differs from its source geometry"
                )
            if labels.min() < 0 or labels.max() > k:
                raise ValueError("Frozen label exceeds declared cluster count")
            factor = int(geometry["downsample_factor"])
            height, width = (int(v) // factor for v in geometry["native_shape_yx"])
            offset_x, offset_y = geometry["block_center_offset_native_xy"]
            yy, xx = np.meshgrid(
                (np.arange(height) * factor + offset_y) / stride,
                (np.arange(width) * factor + offset_x) / stride,
                indexing="ij",
            )
            aligned = map_coordinates(
                labels,
                np.stack((yy, xx)),
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(labels.dtype)
            label_path = (
                directory / f"{tma}_{core}_{safe_method}_k{k}_registered_labels.npy"
            )
            np.save(label_path, aligned)
            rows.append(
                {
                    "assigned_tma": tma,
                    "confirmed_core_id": core,
                    "method": method,
                    "cluster_count": k,
                    "labels_path": str(label_path.resolve()),
                    "labels_sha256": file_sha256(label_path),
                    "label_geometry_path": target["geometry_path"],
                    "geometry_sha256": target["geometry_sha256"],
                    "grid_id": target["grid_id"],
                    "palette_path": str(palette_path.resolve()),
                    "palette_sha256": file_sha256(palette_path),
                    "source_native_stride": stride,
                    "source_label_pitch_x_um": float(geometry["pixel_size_x_um"])
                    * stride
                    / factor,
                    "source_label_pitch_y_um": float(geometry["pixel_size_y_um"])
                    * stride
                    / factor,
                    "interpretation": "current shared-model labels resampled nearest to accepted HS anchor; not historical original K9",
                }
            )
    frame = pd.DataFrame(rows)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    _save_json(
        destination.with_suffix(".json"),
        {
            "schema": "current_hs_cluster_grid_export_v1",
            "code_sha256": file_sha256(__file__),
            "source_stride": stride,
            "source_sample_center_native_xy": [0, 0],
            "rows": rows,
            "manifest_sha256": file_sha256(destination),
        },
    )
    return frame


def export_saved_hs_cluster_maps(
    hs_review_receipt: str | Path,
    registered_manifest: pd.DataFrame,
    *,
    object_dir: str | Path,
    manifest_path: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    """Reuse saved current HS labels without fitting or changing accepted maps.

    The source review stores the actual labels, native sample stride, source
    image identities, mosaic origins and shared palette specification. This
    adapter restores only those exporter inputs, then uses the existing
    nearest-neighbor geometry conversion.
    """
    import inspect

    source_path = Path(hs_review_receipt).resolve()
    source = _json(source_path)
    for item in source["artifacts"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen HS artifact changed: {item['path']}")
    source_identity = source["identity"]
    parameters = source_identity["parameters"]
    destination = Path(manifest_path)
    receipt_path = destination.with_suffix(".json")
    identity = {
        "source_receipt_sha256": file_sha256(source_path),
        "registered_rows": registered_manifest.to_dict("records"),
        "code_sha256": hashlib.sha256(
            (
                inspect.getsource(export_saved_hs_cluster_maps)
                + inspect.getsource(export_current_hs_cluster_maps)
            ).encode()
        ).hexdigest(),
    }
    if destination.is_file() and receipt_path.is_file() and not force:
        prior = _json(receipt_path)
        if (
            prior.get("saved_source_identity") == identity
            and prior.get("manifest_sha256") == file_sha256(destination)
            and all(
                file_sha256(row[path_key]) == row[hash_key]
                for row in prior["rows"]
                for path_key, hash_key in (
                    ("labels_path", "labels_sha256"),
                    ("palette_path", "palette_sha256"),
                    ("label_geometry_path", "geometry_sha256"),
                )
            )
        ):
            print("CACHE_HIT frozen current HS domain export")
            return pd.read_csv(
                destination, keep_default_na=False, float_precision="round_trip"
            )
    labels: dict[str, np.ndarray] = {}
    for path in source["object_paths"]:
        with np.load(path, allow_pickle=False) as bundle:
            if str(bundle["identity_sha256"]) != source["identity_sha256"]:
                raise ValueError("Saved HS labels belong to another fitted model")
            if str(bundle["axes"]) != "YX":
                raise ValueError("Unsupported saved HS label axes")
            if int(bundle["display_stride"]) != int(parameters["display_stride"]):
                raise ValueError("Saved HS stride differs from its source receipt")
            labels[str(bundle["core_key"])] = np.array(bundle["labels"])
    summaries: list[dict[str, Any]] = []
    for item in source_identity["sources"]:
        tma, core = item["core_key"].split("_", 1)
        summaries.append(
            {
                "assigned_tma": tma,
                "confirmed_core_id": core,
                "full_file_sha256": item["sha256"],
            }
        )
    cohort = {
        "parameters": {"display_stride": int(parameters["display_stride"])},
        "display_inputs": source_identity["display_inputs"],
        "summary": pd.DataFrame(summaries),
    }
    result = export_current_hs_cluster_maps(
        {
            "label_maps": {
                ("frozen_l1_minibatch_kmeans", int(parameters["cluster_count"])): labels
            }
        },
        cohort,
        registered_manifest,
        object_dir=object_dir,
        manifest_path=destination,
    )
    receipt = _json(receipt_path)
    receipt["saved_source_identity"] = identity
    receipt["source_receipt_path"] = str(source_path)
    _save_json(receipt_path, receipt)
    print("RECOMPUTE frozen current HS domain export; source labels unchanged")
    return result


def compare_recovery_baselines(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every held-out image to its own brightness and mean baselines."""
    if metrics.empty:
        return pd.DataFrame()
    keys = ["assigned_tma", "confirmed_core_id", "marker", "heldout_group"]
    if "spatial_fold" in metrics:
        keys.append("spatial_fold")
    baseline = metrics.loc[
        metrics["model"].eq("total_only"), keys + ["spearman", "mse"]
    ].rename(
        columns={
            "spearman": "brightness_baseline_spearman",
            "mse": "brightness_baseline_mse",
        }
    )
    if baseline.duplicated(keys).any():
        raise ValueError("Brightness baseline is not unique for each held-out image")
    result = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
    result["spearman_gain_over_brightness"] = (
        result["spearman"] - result["brightness_baseline_spearman"]
    )
    result["mse_reduction_vs_brightness"] = (
        result["brightness_baseline_mse"] - result["mse"]
    )
    result["mse_reduction_vs_training_mean"] = (
        result["training_mean_baseline_mse"] - result["mse"]
    )
    result["claim_label"] = "Exploratory"
    return result


def render_recovery_scorecard(
    metrics: pd.DataFrame,
    *,
    destination: str | Path,
    force: bool = False,
) -> Path | None:
    """Show every held-out image/model score; undefined correlations remain gray."""
    if metrics.empty:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    _plot_style()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(".json")
    identity = _stage_identity(
        metrics.astype(object).where(metrics.notna(), None),
        {"metric": "Spearman", "fixed_limits": [-1, 1]},
    )
    if not force and _matching_cache(receipt_path, identity) is not None:
        print("CACHE_HIT recovery scorecard")
        return path
    with PdfPages(path) as document:
        for marker, frame in metrics.groupby("marker", sort=True):
            frame = frame.assign(
                sample=frame["assigned_tma"] + ":" + frame["confirmed_core_id"]
            )
            if "spatial_fold" in frame:
                frame["sample"] += ":q" + frame["spatial_fold"].astype(str)
            values = frame.pivot(index="sample", columns="model", values="spearman")
            columns = [name for name in MODEL_NAMES if name in values]
            values = values.reindex(columns=columns)
            figure, axis = plt.subplots(figsize=(170 / 25.4, 130 / 25.4))
            figure.subplots_adjust(left=0.20, right=0.80, bottom=0.22, top=0.90)
            palette = plt.colormaps["RdBu_r"].copy()
            palette.set_bad("#D9D9D9")
            artist = axis.imshow(
                np.ma.masked_invalid(values.to_numpy(dtype=float)),
                cmap=palette,
                vmin=-1,
                vmax=1,
                aspect="auto",
                interpolation="none",
            )
            axis.set_xticks(
                np.arange(len(columns)), labels=columns, rotation=35, ha="right"
            )
            axis.set_yticks(np.arange(len(values)), labels=values.index)
            colorbar = figure.colorbar(artist, ax=axis, fraction=0.04, pad=0.04)
            colorbar.set_label("Held-out image Spearman")
            figure.text(0.02, 0.96, str(marker), fontsize=5)
            figure.text(
                0.02,
                0.065,
                "Every retained sample and declared model; gray = undefined correlation, not zero.",
                fontsize=5,
            )
            figure.text(
                0.02,
                0.025,
                "Generator: clinical_roi_pattern_recovery.py; exploratory spatial pattern recovery.",
                fontsize=5,
            )
            document.savefig(figure)
            plt.close(figure)
    _save_json(
        receipt_path,
        {
            "identity": identity,
            "artifacts": [{"path": str(path.resolve()), "sha256": file_sha256(path)}],
        },
    )
    print("RECOMPUTE recovery scorecard")
    return path


def render_compact_marker_atlas(
    manifest: pd.DataFrame,
    *,
    plot_dir: str | Path,
    labels_manifest: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    validation_label: str = "descriptive correspondence",
    rows_per_page: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Keep notebooks compact while displaying every core for each marker.

    Full individual overlays and every model's predicted arrays/PDFs remain
    separate artifacts. This overview uses either fixed HS domains or the
    prespecified primary model and brightness baseline; it never selects a
    model by its held-out performance.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if (labels_manifest is None) == (predictions is None):
        raise ValueError("Supply exactly one of frozen domains or predictions")
    if rows_per_page < 1 or rows_per_page > 6:
        raise ValueError("Use 1–6 rows per atlas page")
    directory = Path(plot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    extra = labels_manifest if labels_manifest is not None else predictions
    assert extra is not None
    parameters: dict[str, Any] = {
        "extra_rows": extra.to_dict("records"),
        "validation_label": validation_label,
        "rows_per_page": rows_per_page,
    }
    if predictions is not None:
        parameters["prediction_sha256"] = {
            path: file_sha256(path) for path in predictions["prediction_path"]
        }
    identity = _stage_identity(manifest, parameters)
    receipt_path = directory / "compact_atlas_receipt.json"
    previous = None if force else _matching_cache(receipt_path, identity)
    if previous is not None:
        print("CACHE_HIT compact marker atlas")
        return {
            "pdfs": [Path(path) for path in previous["pdfs"]],
            "receipt_path": receipt_path,
        }
    _plot_style()
    pdfs: list[Path] = []
    for marker, marker_frame in manifest.groupby("marker", sort=True):
        rows = marker_frame.sort_values(["assigned_tma", "confirmed_core_id"]).to_dict(
            "records"
        )
        if predictions is not None:
            available = set(
                map(
                    tuple,
                    predictions.loc[
                        predictions["marker"].eq(marker)
                        & predictions["model"].eq("l1_ridge"),
                        ["assigned_tma", "confirmed_core_id"],
                    ].to_numpy(),
                )
            )
            rows = [row for row in rows if _core_key(row) in available]
        if not rows:
            continue
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(marker))
        safe += "_" + hashlib.sha256(str(marker).encode()).hexdigest()[:8]
        pdf_path = directory / f"{safe}_all_core_atlas.pdf"
        with PdfPages(pdf_path) as document:
            for start in range(0, len(rows), rows_per_page):
                figure, axes = plt.subplots(
                    rows_per_page,
                    3,
                    squeeze=False,
                    figsize=(210 / 25.4, 275 / 25.4),
                )
                figure.subplots_adjust(
                    left=0.04,
                    right=0.98,
                    bottom=0.09,
                    top=0.91,
                    hspace=0.18,
                    wspace=0.04,
                )
                headings = (
                    (
                        "HS log brightness",
                        "Measured marker",
                        "Frozen HS spectral domains",
                    )
                    if labels_manifest is not None
                    else (
                        "Measured marker",
                        "Primary: L1 ridge prediction",
                        "Control: brightness prediction",
                    )
                )
                figure.text(0.04, 0.975, f"{marker} | {validation_label}", fontsize=5)
                for column, heading in enumerate(headings):
                    bounds = axes[0, column].get_position()
                    figure.text(bounds.x0, 0.945, heading, fontsize=5)
                for offset in range(rows_per_page):
                    if start + offset >= len(rows):
                        for axis in axes[offset]:
                            axis.set_axis_off()
                        continue
                    row = rows[start + offset]
                    _, total, measured, valid = load_registered_pair(row)
                    lo, hi = _display_range(measured[valid])
                    settings: list[dict[str, Any]] = []
                    if labels_manifest is not None:
                        match = labels_manifest.loc[
                            labels_manifest["assigned_tma"].eq(row["assigned_tma"])
                            & labels_manifest["confirmed_core_id"].eq(
                                row["confirmed_core_id"]
                            )
                        ]
                        if len(match) != 1:
                            raise ValueError(
                                "Compact domain atlas needs one frozen model per core"
                            )
                        label = match.iloc[0]
                        domains = np.load(label["labels_path"], mmap_mode="r")
                        palette = np.load(label["palette_path"])
                        log_total = np.log1p(total)
                        low_hs, high_hs = _display_range(log_total[valid])
                        views = [log_total, measured, domains]
                        settings = [
                            {"cmap": "gray", "vmin": low_hs, "vmax": high_hs},
                            {"cmap": "magma", "vmin": lo, "vmax": hi},
                            {
                                "cmap": ListedColormap(palette),
                                "norm": BoundaryNorm(
                                    np.arange(len(palette) + 1) - 0.5, len(palette)
                                ),
                            },
                        ]
                    else:
                        assert predictions is not None
                        match = predictions.loc[
                            predictions["assigned_tma"].eq(row["assigned_tma"])
                            & predictions["confirmed_core_id"].eq(
                                row["confirmed_core_id"]
                            )
                            & predictions["marker"].eq(marker)
                        ].set_index("model", verify_integrity=True)
                        primary = np.load(
                            match.loc["l1_ridge", "prediction_path"], mmap_mode="r"
                        )
                        brightness = np.load(
                            match.loc["total_only", "prediction_path"], mmap_mode="r"
                        )
                        if (
                            primary.shape != measured.shape
                            or brightness.shape != measured.shape
                        ):
                            raise ValueError(
                                "Prediction atlas grid differs from measured marker"
                            )
                        valid &= np.isfinite(primary) & np.isfinite(brightness)
                        views = [measured, primary, brightness]
                        settings = [{"cmap": "magma", "vmin": lo, "vmax": hi}] * 3
                    stride = max(1, int(math.ceil(max(valid.shape) / 512)))
                    for axis, values, style in zip(axes[offset], views, settings):
                        axis.imshow(
                            np.ma.array(
                                values[::stride, ::stride],
                                mask=~valid[::stride, ::stride],
                            ),
                            interpolation="none",
                            **style,
                        )
                        axis.set_axis_off()
                    bounds = axes[offset, 0].get_position()
                    figure.text(
                        bounds.x0,
                        bounds.y1 + 0.009,
                        f"{row['assigned_tma']}:{row['confirmed_core_id']} | grid {row['pixel_size_x_um']:.3f} × {row['pixel_size_y_um']:.3f} µm",
                        fontsize=5,
                    )
                figure.text(
                    0.04,
                    0.045,
                    "Every available core; shared observed tissue. Prediction colors use each row’s measured-marker limits.",
                    fontsize=5,
                )
                figure.text(
                    0.04,
                    0.025,
                    "Generator: clinical_roi_pattern_recovery.py. Complete individual maps and residuals are saved separately.",
                    fontsize=5,
                )
                document.savefig(figure)
                plt.close(figure)
        pdfs.append(pdf_path)
    _save_json(
        receipt_path,
        {
            "identity": identity,
            "pdfs": [str(path.resolve()) for path in pdfs],
            "artifacts": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in pdfs
            ],
        },
    )
    print("RECOMPUTE compact marker atlas")
    return {"pdfs": pdfs, "receipt_path": receipt_path}


def run_spatial_block_marker_recovery(
    manifest: pd.DataFrame,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    plot_dir: str | Path,
    samples_per_training_region: int = 2000,
    guard_fraction: float = 0.05,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
    models: tuple[str, ...] = MODEL_NAMES,
    force: bool = False,
) -> dict[str, Any]:
    """Predict held-out quadrants when cross-core validation is unavailable.

    The image is divided into four fixed spatial quadrants. For each test
    quadrant, neighboring training pixels within a declared geometric guard
    are excluded. This tests interpolation into a spatial region of the same
    registered image, not transfer to a new patient or independent tissue.
    The guard is a leakage precaution, not a measured registration accuracy.
    """
    from scipy.ndimage import binary_dilation
    from threadpoolctl import threadpool_limits

    if not 0 < guard_fraction < 0.25:
        raise ValueError("Spatial guard_fraction must lie between 0 and 0.25")
    if samples_per_training_region < 1 or ridge_alpha <= 0:
        raise ValueError("Positive training sample cap and ridge alpha required")
    if not models or not set(models) <= set(MODEL_NAMES):
        raise ValueError("Unsupported model arms")
    objects, tables, plots = map(Path, (object_dir, table_dir, plot_dir))
    for directory in (objects, tables, plots):
        directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        "partition": "four fixed image quadrants",
        "guard_fraction_of_short_axis": guard_fraction,
        "samples_per_training_region": samples_per_training_region,
        "random_state": random_state,
        "ridge_alpha": ridge_alpha,
        "models": list(models),
    }
    identity = {
        "input_rows": manifest.to_dict("records"),
        "parameters": parameters,
        "code_sha256": file_sha256(__file__),
    }
    receipt_path = tables / "spatial_block_recovery_receipt.json"
    metric_path = tables / "spatial_block_recovery_metrics.csv"
    prediction_path = tables / "spatial_block_recovery_predictions.csv"
    exclusions_path = tables / "spatial_block_recovery_exclusions.csv"
    if receipt_path.is_file() and not force:
        previous = _json(receipt_path)
        if previous.get("identity") == identity and all(
            Path(item["path"]).is_file() and file_sha256(item["path"]) == item["sha256"]
            for item in previous["artifacts"]
        ):
            print("CACHE_HIT spatial-block recovery")
            return {
                "metrics": pd.read_csv(metric_path, float_precision="round_trip"),
                "predictions": pd.read_csv(
                    prediction_path, float_precision="round_trip"
                ),
                "exclusions": pd.read_csv(
                    exclusions_path, float_precision="round_trip"
                ),
                "receipt_path": receipt_path,
                "cache_status": "CACHE_HIT",
            }
    rng = np.random.default_rng(random_state)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    for row in manifest.to_dict("records"):
        l1, total, observed, valid = load_registered_pair(row)
        height, width = valid.shape
        yy, xx = np.indices(valid.shape)
        quadrants = (yy >= height // 2).astype(np.uint8) * 2
        quadrants += (xx >= width // 2).astype(np.uint8)
        guard = max(1, int(math.ceil(min(height, width) * guard_fraction)))
        predictions = {
            name: np.full(observed.shape, np.nan, dtype=np.float32) for name in models
        }
        tested = np.zeros(valid.shape, dtype=bool)
        for fold in range(4):
            region = quadrants == fold
            test = valid & region
            training = valid & ~binary_dilation(region, iterations=guard)
            if int(test.sum()) < 10 or int(training.sum()) < 20:
                exclusions.append(
                    {
                        "assigned_tma": row["assigned_tma"],
                        "confirmed_core_id": row["confirmed_core_id"],
                        "marker": row["marker"],
                        "spatial_fold": fold,
                        "reason": "insufficient observed tissue after spatial guard",
                    }
                )
                continue
            train_indices = np.flatnonzero(training)
            chosen = rng.choice(
                train_indices,
                size=min(samples_per_training_region, len(train_indices)),
                replace=False,
            )
            train_y = observed.ravel()[chosen]
            train_l1 = l1.reshape(-1, 20)[chosen]
            train_total = total.ravel()[chosen]
            reference_mean = float(train_y.mean())
            test_indices = np.flatnonzero(test)
            for name in models:
                with threadpool_limits(limits=1):
                    model = (
                        None
                        if name == "training_mean"
                        else _fit_predictor(
                            _model_features(train_l1, train_total, name),
                            train_y,
                            name,
                            ridge_alpha=ridge_alpha,
                            random_state=random_state,
                        )
                    )
                    for start in range(0, len(test_indices), 65536):
                        index = test_indices[start : start + 65536]
                        if model is None:
                            values = np.full(len(index), reference_mean)
                        else:
                            values = model.predict(
                                _model_features(
                                    l1.reshape(-1, 20)[index],
                                    total.ravel()[index],
                                    name,
                                )
                            )
                        predictions[name].ravel()[index] = values
                scores = continuous_pattern_metrics(
                    observed[test],
                    predictions[name][test],
                    reference_mean,
                )
                metric_rows.append(
                    {
                        "assigned_tma": row["assigned_tma"],
                        "confirmed_core_id": row["confirmed_core_id"],
                        "marker": row["marker"],
                        "model": name,
                        "heldout_group": f"{row['assigned_tma']}:{row['confirmed_core_id']}:quadrant{fold}",
                        "spatial_fold": fold,
                        "guard_pixels": guard,
                        "guard_x_um": guard * float(row["pixel_size_x_um"]),
                        "guard_y_um": guard * float(row["pixel_size_y_um"]),
                        "training_pixels": len(chosen),
                        "validation_unit": "spatial quadrant within the same registered image",
                        "claim_label": "Exploratory",
                        **scores,
                    }
                )
            tested |= test
        for name in models:
            if not tested.any():
                continue
            label = f"{row['assigned_tma']}_{row['confirmed_core_id']}_{row['marker']}_{name}_spatial_blocks"
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
            array_path = objects / f"{safe}.npy"
            np.save(array_path, predictions[name])
            pdf = render_prediction_panel(
                observed,
                predictions[name],
                tested,
                label=label,
                destination=plots / f"{safe}.pdf",
            )
            artifacts.extend([array_path, pdf])
            prediction_rows.append(
                {
                    "assigned_tma": row["assigned_tma"],
                    "confirmed_core_id": row["confirmed_core_id"],
                    "marker": row["marker"],
                    "model": name,
                    "prediction_path": str(array_path.resolve()),
                    "pdf_path": str(pdf.resolve()),
                    "grid_id": row["grid_id"],
                }
            )
    metrics = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=[
            "assigned_tma",
            "confirmed_core_id",
            "marker",
            "spatial_fold",
            "reason",
        ],
    )
    if metrics.empty:
        metrics = pd.DataFrame(
            columns=[
                "assigned_tma",
                "confirmed_core_id",
                "marker",
                "model",
                "heldout_group",
                "spatial_fold",
                "spearman",
                "mse",
            ]
        )
    if predictions_frame.empty:
        predictions_frame = pd.DataFrame(
            columns=[
                "assigned_tma",
                "confirmed_core_id",
                "marker",
                "model",
                "prediction_path",
                "pdf_path",
                "grid_id",
            ]
        )
    metrics.to_csv(metric_path, index=False)
    predictions_frame.to_csv(prediction_path, index=False)
    exclusion_frame.to_csv(exclusions_path, index=False)
    artifacts.extend([metric_path, prediction_path, exclusions_path])
    _save_json(
        receipt_path,
        {
            "schema": "spatial_block_marker_recovery_v1",
            "identity": identity,
            "claim_label": "Exploratory",
            "limitations": (
                "Within-image spatial holdout, not independent patient validation. "
                "Geometric guard does not establish registration accuracy."
            ),
            "artifacts": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for path in artifacts
            ],
        },
    )
    print("RECOMPUTE spatial-block recovery")
    return {
        "metrics": metrics,
        "predictions": predictions_frame,
        "exclusions": exclusion_frame,
        "receipt_path": receipt_path,
        "cache_status": "COMPUTED",
    }


def render_joint_cluster_marker_summary(
    summary: pd.DataFrame,
    *,
    table_dir: str | Path,
    plot_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Describe serial-section joint-domain profiles without biological inference.

    Cores contribute equally after within-core log2 enrichment. Missing, zero,
    negative and unsupported measurements remain undefined, without pseudocounts.
    PCA uses complete, nonconstant marker dimensions only; Spearman compares
    marker profiles across unordered cluster categories, not numeric cluster IDs.
    Registration and structural orientation acceptance are upstream prerequisites.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    keys = ["assigned_tma", "method", "cluster_count"]
    core_keys = [*keys, "confirmed_core_id", "cluster_id", "marker"]
    required = {
        *core_keys, "mean_marker_intensity", "tissue_mean_marker_intensity",
        "enrichment_ratio", "valid_spatial_observations", "claim_label",
    }
    if missing := required.difference(summary.columns):
        raise ValueError(f"Joint profile summary missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("No accepted joint cluster/marker observations")
    if summary[core_keys].isna().any().any():
        raise ValueError("Joint profile identity columns must not be missing")
    if summary.duplicated(core_keys).any():
        raise ValueError("Duplicate core/cluster/marker profile measurements")
    cluster_ids = pd.to_numeric(summary["cluster_id"], errors="raise").to_numpy(float)
    cluster_counts = pd.to_numeric(summary["cluster_count"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(cluster_ids).all() or not np.isfinite(cluster_counts).all()
        or np.any(cluster_ids != np.floor(cluster_ids))
        or np.any(cluster_counts != np.floor(cluster_counts))
        or np.any(cluster_ids < 1) or np.any(cluster_ids > cluster_counts)
        or np.any(cluster_counts < 2)
    ):
        raise ValueError("Joint cluster identities must be integer IDs within declared K")
    frame = summary.sort_values(core_keys).reset_index(drop=True).copy()
    frame["cluster_id"] = pd.to_numeric(frame["cluster_id"]).astype(int)
    frame["cluster_count"] = pd.to_numeric(frame["cluster_count"]).astype(int)
    tables, plots = Path(table_dir).resolve(), Path(plot_dir).resolve()
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    style_path = Path(__file__).resolve().parents[1] / "01_utils" / "plotnine_style_helper.py"
    identity = _stage_identity(frame, {
        "analysis": "joint_serial_section_marker_profiles",
        "version": 1,
        "style_sha256": file_sha256(style_path),
        "table_dir": str(tables), "plot_dir": str(plots),
        "source_identity": json.loads(json.dumps(summary.attrs, default=str)),
        "aggregation": "equal-core mean of finite within-core log2 enrichment",
        "pca": "centered unscaled complete finite nonconstant equal-core marker profiles",
        "similarity": "pairwise-complete Spearman across unordered equal-core cluster profiles",
        "canvas_mm": [180, 140],
    })
    # Normalize nonfinite identity values to JSON null for stable cache comparison.
    identity = json.loads(json.dumps(identity, default=str), parse_constant=lambda _: None)
    receipt_path = tables / "joint_cluster_marker_summary_receipt.json"
    names = (
        "within_core", "equal_core", "coverage", "ledger", "pca_scores",
        "pca_loadings", "pca_variance", "marker_similarity",
    )
    paths = {name: tables / f"joint_cluster_marker_{name}.csv" for name in names}
    previous = None if force else _matching_cache(receipt_path, identity)
    if previous is not None:
        return {
            "pdfs": [Path(path) for path in previous["pdfs"]],
            "receipt_path": receipt_path, "summary_paths": paths,
            "summary": pd.read_csv(paths["equal_core"]),
            "cache_status": "CACHE_HIT",
        }
    for column in (
        "mean_marker_intensity", "tissue_mean_marker_intensity",
        "enrichment_ratio", "valid_spatial_observations",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numerator = frame["mean_marker_intensity"].to_numpy(float)
    denominator = frame["tissue_mean_marker_intensity"].to_numpy(float)
    support = frame["valid_spatial_observations"].to_numpy(float)
    ratio = frame["enrichment_ratio"].to_numpy(float)
    good = (
        np.isfinite(numerator) & np.isfinite(denominator) & np.isfinite(ratio)
        & np.isfinite(support) & (support > 0) & (numerator > 0)
        & (denominator > 0) & (ratio > 0)
    )
    expected = np.full(len(frame), np.nan)
    np.divide(numerator, denominator, out=expected, where=denominator > 0)
    if np.any(good & ~np.isclose(ratio, expected, rtol=1e-6, atol=1e-12)):
        raise ValueError("Enrichment ratios disagree with measured numerator/denominator")
    frame["log2_enrichment"] = np.nan
    frame.loc[good, "log2_enrichment"] = np.log2(ratio[good])
    frame["measurement_status"] = [
        "finite_positive_supported" if valid else
        "missing_or_nonfinite" if not np.isfinite([n, d, s]).all() else
        "no_spatial_support" if s <= 0 else
        "negative_signal_undefined_log" if n < 0 or d < 0 else
        "zero_signal_no_pseudocount" if n == 0 or d == 0 else
        "undefined_derived_ratio" if not np.isfinite(r) else
        "zero_signal_no_pseudocount" if r == 0 else
        "negative_signal_undefined_log"
        for valid, n, d, r, s in zip(good, numerator, denominator, ratio, support)
    ]
    # Explicitly enumerate missing core/cluster/marker combinations within strata.
    expanded: list[pd.DataFrame] = []
    for _, group in frame.groupby(keys, sort=True, dropna=False):
        grid = pd.MultiIndex.from_product(
            [group["confirmed_core_id"].unique(), range(1, int(group["cluster_count"].iloc[0]) + 1),
             group["marker"].unique()],
            names=["confirmed_core_id", "cluster_id", "marker"],
        )
        complete = group.set_index(["confirmed_core_id", "cluster_id", "marker"]).reindex(grid)
        for key in keys:
            complete[key] = group[key].iloc[0]
        complete["measurement_status"] = complete["measurement_status"].fillna("missing_combination")
        complete["claim_label"] = "Exploratory"
        expanded.append(complete.reset_index())
    frame = pd.concat(expanded, ignore_index=True)
    group_keys = [*keys, "cluster_id", "marker"]
    equal = frame.groupby(group_keys, sort=True, dropna=False).agg(
        mean_log2_enrichment=("log2_enrichment", "mean"),
        sd_log2_enrichment=("log2_enrichment", "std"),
        min_log2_enrichment=("log2_enrichment", "min"),
        max_log2_enrichment=("log2_enrichment", "max"),
        finite_core_count=("log2_enrichment", "count"),
        eligible_core_count=("confirmed_core_id", "nunique"),
        valid_spatial_observations=("valid_spatial_observations", lambda x: x.sum(min_count=1)),
    ).reset_index()
    equal["core_coverage_fraction"] = equal["finite_core_count"] / equal["eligible_core_count"]
    equal["claim_label"] = "Exploratory"
    equal["interpretation"] = "Serial-section regional correspondence; equal core weight"
    ledger = frame[[*core_keys, "measurement_status"]].copy()
    ledger["analysis"] = "within_core_log2"
    extra_ledger: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    loadings: list[dict[str, Any]] = []
    variances: list[dict[str, Any]] = []
    similarities: list[dict[str, Any]] = []
    _plot_style()
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 5, "font.weight": "normal",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
        "lines.linewidth": 0.5, "grid.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    })
    pdfs: list[Path] = []

    def save_heatmap(
        document: PdfPages, values: pd.DataFrame, label: str, note: str,
        *, limit: float | None = None, y_label: str = "Joint HS+FLIM cluster",
    ) -> None:
        """Write a fixed-canvas heatmap without changing effect magnitudes."""
        figure, axis = plt.subplots(figsize=(180 / 25.4, 140 / 25.4))
        figure.subplots_adjust(left=0.20, right=0.88, bottom=0.30, top=0.89)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#D1D5DB")
        array = values.to_numpy(float)
        finite = np.abs(array[np.isfinite(array)])
        bound = limit if limit is not None else (float(finite.max()) if finite.size else 1.0)
        bound = bound if bound > 0 else 1.0  # display scale only, never modifies data
        nrows, ncols = array.shape
        artist = axis.pcolormesh(
            np.arange(ncols + 1) - 0.5,
            np.arange(nrows + 1) - 0.5,
            np.ma.masked_invalid(array),
            cmap=cmap, vmin=-bound, vmax=bound,
            shading="flat", edgecolors="none", linewidth=0,
            antialiased=False, rasterized=False, snap=True,
        )
        axis.set_xlim(-0.5, ncols - 0.5)
        axis.set_ylim(nrows - 0.5, -0.5)
        axis.grid(False)
        axis.set_xticks(np.arange(len(values.columns)), labels=values.columns, rotation=90)
        axis.set_yticks(np.arange(len(values.index)), labels=values.index)
        axis.set_ylabel(y_label)
        axis.tick_params(width=0.5, length=2)
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.035, pad=0.03)
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
        colorbar.set_label(label)
        figure.text(0.04, 0.95, note, fontsize=5)
        figure.text(0.04, 0.065, "Gray = undefined; no imputation. Serial sections: regional correspondence, not identical cells.", fontsize=5)
        figure.text(0.04, 0.035, "Generator: clinical_roi_pattern_recovery.py", fontsize=5)
        document.savefig(figure)
        plt.close(figure)

    heat_path = plots / "joint_cluster_marker_equal_core_log2_heatmaps.pdf"
    core_path = plots / "joint_cluster_marker_per_core_log2_heatmaps.pdf"
    profile_path = plots / "joint_cluster_marker_sample_resolved_profiles.pdf"
    similarity_path = plots / "joint_cluster_marker_descriptive_spearman.pdf"
    pca_path = plots / "joint_cluster_marker_profile_pca.pdf"
    # Delay optional documents until a genuine finite analysis exists.
    pca_panels: list[tuple[pd.DataFrame, str, float, float]] = []
    similarity_panels: list[tuple[pd.DataFrame, str]] = []
    with PdfPages(heat_path) as heat_doc, PdfPages(core_path) as core_doc, PdfPages(profile_path) as profile_doc:
        for stratum, group in equal.groupby(keys, sort=True, dropna=False):
            metadata = dict(zip(keys, stratum))
            matrix = group.pivot(index="cluster_id", columns="marker", values="mean_log2_enrichment")
            source = frame
            for key, value in metadata.items():
                source = source.loc[source[key] == value]
            source_cores = sorted(source["confirmed_core_id"].unique(), key=str)
            label = (
                f"{metadata['assigned_tma']} | Joint HS+FLIM clusters | "
                f"K={metadata['cluster_count']} | n={len(source_cores)} "
                f"{'core' if len(source_cores) == 1 else 'cores'}: "
                + ", ".join(str(core) for core in source_cores)
            )
            source_finite = np.abs(source["log2_enrichment"].dropna().to_numpy(float))
            bound = float(source_finite.max()) if source_finite.size else 1.0
            save_heatmap(heat_doc, matrix, "Equal-core mean log2 enrichment", label + "; marker coverage in source tables", limit=bound)
            for core, core_group in source.groupby("confirmed_core_id", sort=True):
                core_matrix = core_group.pivot(index="cluster_id", columns="marker", values="log2_enrichment").reindex(index=matrix.index, columns=matrix.columns)
                save_heatmap(core_doc, core_matrix, "Within-core log2 enrichment", f"{label} | core={core}", limit=bound)
            # One marker/page: unordered clusters are categorical positions, no connecting lines.
            for marker, marker_group in source.groupby("marker", sort=True):
                figure, axis = plt.subplots(figsize=(180 / 25.4, 140 / 25.4))
                figure.subplots_adjust(left=0.16, right=0.72, bottom=0.20, top=0.88)
                clusters = list(matrix.index)
                cores = sorted(marker_group["confirmed_core_id"].unique(), key=str)
                for index, core in enumerate(cores):
                    part = marker_group.loc[marker_group["confirmed_core_id"] == core].set_index("cluster_id").reindex(clusters)
                    offset = (index - (len(cores) - 1) / 2) * min(0.08, 0.6 / len(cores))
                    axis.scatter(np.arange(len(clusters)) + offset, part["log2_enrichment"], s=3.14, linewidths=0.5, label=str(core))
                axis.axhline(0, color="#777777", linewidth=0.5)
                axis.set_xticks(np.arange(len(clusters)), labels=clusters)
                axis.set_xlabel("Unordered joint cluster category")
                axis.set_ylabel("Within-core log2 enrichment")
                axis.grid(True, color="#E5E7EB", linewidth=0.5)
                axis.set_axisbelow(True)
                axis.legend(title="Core", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                figure.text(0.04, 0.95, f"{label} | {marker}; n={len(cores)} cores (not necessarily independent patients)", fontsize=5)
                figure.text(0.04, 0.065, "Sample-resolved descriptive regional profiles; missing/zero values not plotted. No P values.", fontsize=5)
                figure.text(0.04, 0.035, "Generator: clinical_roi_pattern_recovery.py", fontsize=5)
                profile_doc.savefig(figure)
                plt.close(figure)
            correlation = pd.DataFrame(np.nan, index=matrix.columns, columns=matrix.columns)
            for first in matrix.columns:
                for second in matrix.columns:
                    pair = matrix[[first]].rename(columns={first: "first"}).join(matrix[[second]].rename(columns={second: "second"})).dropna()
                    status = "finite"
                    rho = np.nan
                    if len(pair) < 3:
                        status = "fewer_than_three_complete_cluster_profiles"
                    elif pair.nunique().min() < 2:
                        status = "constant_marker_profile"
                    else:
                        rho = float(spearmanr(pair["first"], pair["second"]).statistic)
                    correlation.loc[first, second] = rho
                    similarities.append({**metadata, "marker_a": first, "marker_b": second, "spearman_rho": rho, "complete_cluster_profiles": len(pair), "status": status, "claim_label": "Exploratory"})
            if np.isfinite(correlation.to_numpy()).any():
                similarity_panels.append((correlation, label))
            observed_matrix = matrix.dropna(axis=0, how="all")
            for cluster in matrix.index.difference(observed_matrix.index):
                extra_ledger.append({**metadata, "cluster_id": cluster, "analysis": "pca", "measurement_status": "cluster_without_finite_profiles"})
            eligible = observed_matrix.notna().all(axis=0) & (observed_matrix.nunique(dropna=True) > 1)
            for marker in matrix.columns:
                extra_ledger.append({**metadata, "marker": marker, "analysis": "pca", "measurement_status": "included" if eligible[marker] else "incomplete_or_constant_marker_profile"})
            selected = observed_matrix.loc[:, eligible]
            if selected.shape[0] < 3 or selected.shape[1] < 2:
                extra_ledger.append({**metadata, "analysis": "pca", "measurement_status": "skipped_insufficient_complete_dimensions"})
                continue
            centered = selected.to_numpy(float) - selected.to_numpy(float).mean(axis=0)
            u, singular, vt = np.linalg.svd(centered, full_matrices=False)
            tolerance = np.finfo(float).eps * max(centered.shape) * singular[0]
            rank = int(np.count_nonzero(singular > tolerance))
            if rank < 2:
                extra_ledger.append({**metadata, "analysis": "pca", "measurement_status": "skipped_rank_below_two"})
                continue
            fractions = singular ** 2 / np.sum(singular ** 2)
            coordinates = u * singular
            for component in range(rank):
                component_name = f"PC{component + 1}"
                variances.append({**metadata, "component": component_name, "explained_variance": singular[component] ** 2 / (len(selected) - 1), "explained_variance_ratio": fractions[component], "cluster_profiles": len(selected), "markers": selected.shape[1]})
                for cluster, value in zip(selected.index, coordinates[:, component]):
                    scores.append({**metadata, "cluster_id": cluster, "component": component_name, "score": value})
                for marker, value in zip(selected.columns, vt[component]):
                    loadings.append({**metadata, "marker": marker, "component": component_name, "loading": value})
            pca_panels.append((pd.DataFrame(coordinates[:, :2], index=selected.index), label, float(fractions[0]), float(fractions[1])))
    pdfs.extend([heat_path, core_path, profile_path])
    if similarity_panels:
        with PdfPages(similarity_path) as document:
            for correlation, label in similarity_panels:
                save_heatmap(document, correlation, "Descriptive Spearman rho", label + "; across cluster profiles, not patients", limit=1.0, y_label="Antibody marker")
        pdfs.append(similarity_path)
    if pca_panels:
        with PdfPages(pca_path) as document:
            for coordinates, label, first, second in pca_panels:
                figure, axis = plt.subplots(figsize=(180 / 25.4, 140 / 25.4))
                figure.subplots_adjust(left=0.16, right=0.92, bottom=0.20, top=0.88)
                axis.scatter(coordinates[0], coordinates[1], s=3.14, linewidths=0.5)
                for cluster, row in coordinates.iterrows():
                    axis.annotate(str(cluster), (row[0], row[1]), xytext=(3, 3), textcoords="offset points", fontsize=5)
                axis.set_xlabel(f"PC1 ({first:.1%} variance)")
                axis.set_ylabel(f"PC2 ({second:.1%} variance)")
                axis.grid(True, color="#E5E7EB", linewidth=0.5)
                axis.set_axisbelow(True)
                figure.text(0.04, 0.95, label, fontsize=5)
                figure.text(0.04, 0.065, "Centered, unscaled equal-core log2 profiles; complete nonconstant markers only. Descriptive, no biological replication.", fontsize=5)
                figure.text(0.04, 0.035, "Generator: clinical_roi_pattern_recovery.py", fontsize=5)
                document.savefig(figure)
                plt.close(figure)
        pdfs.append(pca_path)
    ledger = pd.concat([ledger, pd.DataFrame(extra_ledger)], ignore_index=True)
    outputs = {
        "within_core": frame, "equal_core": equal,
        "coverage": equal[[*group_keys, "finite_core_count", "eligible_core_count", "core_coverage_fraction", "valid_spatial_observations"]],
        "ledger": ledger,
        "pca_scores": pd.DataFrame(scores, columns=[*keys, "cluster_id", "component", "score"]),
        "pca_loadings": pd.DataFrame(loadings, columns=[*keys, "marker", "component", "loading"]),
        "pca_variance": pd.DataFrame(variances, columns=[*keys, "component", "explained_variance", "explained_variance_ratio", "cluster_profiles", "markers"]),
        "marker_similarity": pd.DataFrame(similarities, columns=[*keys, "marker_a", "marker_b", "spearman_rho", "complete_cluster_profiles", "status", "claim_label"]),
    }
    for name, output in outputs.items():
        output.to_csv(paths[name], index=False)
    _save_json(receipt_path, {
        "identity": identity, "pdfs": [str(path) for path in pdfs],
        "summary_paths": {key: str(path) for key, path in paths.items()},
        "limitations": "Descriptive regional serial-section correspondence; cores are not assumed independent patients. No P values or numerical cluster-ID correlations.",
        "artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in [*paths.values(), *pdfs]],
    })
    return {
        "pdfs": pdfs, "receipt_path": receipt_path, "summary_paths": paths,
        "summary": equal, "cache_status": "COMPUTED",
    }
