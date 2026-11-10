"""Orientation-aware clinical HS/cycIF VALIS integration and acceptance QC.

Reuses the native HS loader, image-only orientation mathematics, structural
masks and installed VALIS. No cell segmentation or invented registration.
Heavy VALIS fitting/warping runs in the existing isolated guarded worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, optimize

GEOMETRY_KEYS = (
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
)
GUARDED_PYTHON = Path("/home/groups/EmiliLab/pauli/00_environments/bin/guarded-python")
COMPUTE_GUARD = Path(
    "/home/groups/EmiliLab/pauli/00_environments/bin/require-compute.sh"
)
_FILE_HASH_CACHE: dict[str, tuple[tuple[int, ...], str]] = {}


def compute_file_sha256(path: str | Path) -> str:
    source = Path(path).resolve()
    # A newly flushed network file may expose final timestamps on its first read.
    # Retry the complete hash; never reuse a digest across a changed signature.
    for _ in range(3):
        before = source.stat()
        signature = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        cached = _FILE_HASH_CACHE.get(str(source))
        if cached is not None and cached[0] == signature:
            return cached[1]
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = source.stat()
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if signature == after_signature:
            _FILE_HASH_CACHE[str(source)] = (signature, digest)
            return digest
    raise ValueError(f"Source repeatedly changed while hashing: {source}")


@lru_cache(maxsize=1)
def worker_runtime_identity() -> dict[str, Any]:
    """Read isolated package versions before deciding whether a fitted model is reusable."""
    code = (
        "import importlib.metadata as m,json,sys; "
        'print(json.dumps({"python":sys.version,"executable":sys.executable,'
        '"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'
    )
    result = subprocess.run(
        [str(GUARDED_PYTHON), "spatial-valis", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(result.stdout)
    identity["launcher_sha256"] = compute_file_sha256(GUARDED_PYTHON)
    return identity


def registration_code_bindings() -> dict[str, str]:
    """Bind fitting/warping computation independently of figure and review formatting."""
    functions = (
        _worker,
        bounded_mattes_correction,
        _scale_structural_proxy,
        fit_and_warp_pairs,
        warp_image_yx,
        warp_continuous_yx,
        coarse_antialias_sigma,
        transform_points_yx,
        _read_array,
        _run_isolated,
        worker_runtime_identity,
    )
    result = {
        function.__name__: hashlib.sha256(
            inspect.getsource(function).encode()
        ).hexdigest()
        for function in functions
    }
    result["dylan_c2_image_orientation.py"] = compute_file_sha256(
        Path(__file__).parent / "dylan_c2_image_orientation.py"
    )
    result["valis_supplied_transform.py"] = compute_file_sha256(
        Path(__file__).parent / "valis_supplied_transform.py"
    )
    return result


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def guard_compute() -> None:
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && sp_require_slurm',
            "compute-guard",
            str(COMPUTE_GUARD),
        ],
        check=True,
    )


def _binding(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    return {"path": str(source), "sha256": compute_file_sha256(source)}


def _verify_artifacts(items: list[dict[str, str]]) -> bool:
    return bool(items) and all(
        Path(item["path"]).is_file()
        and compute_file_sha256(item["path"]) == item["sha256"]
        for item in items
    )


def _code_bindings() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: compute_file_sha256(directory / name)
        for name in (
            "clinical_roi_valis.py",
            "valis_supplied_transform.py",
            "clinical_roi_cycif.py",
            "clinical_roi_hyperspectral.py",
            "dylan_c2_image_orientation.py",
            "guided_serial_registration.py",
        )
    }


def strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid explicit boolean: {value!r}")


def _read_array(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source, mmap_mode="r", allow_pickle=False)
    import tifffile

    return tifffile.imread(source)


def _tissue(image: np.ndarray) -> np.ndarray:
    from guided_serial_registration import tissue_mask

    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Structural input must be finite and two-dimensional")
    if np.ptp(image) == 0:
        raise ValueError("Constant structural image has no orientation evidence")
    mask = tissue_mask(image, percentile=60.0, closing_radius=3).astype(bool)
    if not mask.any():
        raise ValueError("Structural tissue mask is empty")
    return mask


def _style() -> None:
    utilities = Path(__file__).resolve().parents[1] / "01_utils"
    if str(utilities) not in sys.path:
        sys.path.insert(0, str(utilities))
    from plotnine_style_helper import apply_matplotlib_style

    apply_matplotlib_style()


def derive_native_l1_features(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize in the native frame, before averaging or any spatial transform."""
    array = np.asarray(cube)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[0] != 20:
        raise ValueError("Require native C20,Y,X or C20,1,Y,X cube")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Native spectral intensities must be finite and nonnegative")
    total = array.sum(axis=0, dtype=np.float32)
    l1 = np.zeros(array.shape, dtype=np.float32)
    np.divide(array, total[None], out=l1, where=total[None] > 0)
    return l1, total


def area_average_downsample_l1(
    l1_cyx: np.ndarray, total_yx: np.ndarray, factor: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average complete native blocks; preserve the documented block-center offset."""
    if (
        not isinstance(factor, int)
        or factor < 1
        or l1_cyx.shape != (20, *total_yx.shape)
    ):
        raise ValueError("Require integer factor and matching native C20/HW arrays")
    height, width = (size // factor for size in total_yx.shape)
    if min(height, width) < 1:
        raise ValueError("Downsample factor is larger than the native image")
    native_total = total_yx[: height * factor, : width * factor]
    l1 = (
        l1_cyx[:, : height * factor, : width * factor]
        .reshape(20, height, factor, width, factor)
        .mean(axis=(2, 4))
    )
    total = native_total.reshape(height, factor, width, factor).mean(axis=(1, 3))
    tissue = (
        _tissue(native_total).reshape(height, factor, width, factor).mean(axis=(1, 3))
        >= 0.5
    )
    support = tissue & (total > 0) & np.isfinite(total)
    return l1.transpose(1, 2, 0).astype(np.float32), total.astype(np.float32), support


def geometry_grid_id(metadata: dict[str, Any]) -> str:
    missing = set(GEOMETRY_KEYS) - set(metadata)
    if missing:
        raise ValueError(f"Missing native geometry fields: {sorted(missing)}")
    canonical = json.dumps(
        {key: metadata[key] for key in GEOMETRY_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_cycif_handoff(path: str | Path) -> pd.DataFrame:
    """Validate notebook-02 arrays and source receipts, not just CSV presence."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "assigned_tma",
        "confirmed_core_id",
        "marker",
        "image_path",
        "support_path",
        "tissue_mask_path",
        "reference_image_path",
        "pixel_size_x_um",
        "pixel_size_y_um",
        "receipt_path",
        "receipt_sha256",
    }
    if required - set(frame.columns) or frame.empty:
        raise ValueError("Incomplete or empty cycIF image handoff")
    if frame.duplicated(["assigned_tma", "confirmed_core_id", "marker"]).any():
        raise ValueError("Duplicate cycIF TMA/core/marker keys")
    verified: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        receipt_path = Path(row["receipt_path"])
        if compute_file_sha256(receipt_path) != row["receipt_sha256"]:
            raise ValueError("CycIF receipt hash mismatch")
        if str(receipt_path) not in verified:
            payload = _json(receipt_path)
            if (
                str(payload.get("assigned_tma")),
                str(payload.get("confirmed_core_id")),
            ) != (str(row["assigned_tma"]), str(row["confirmed_core_id"])):
                raise ValueError("cycIF core identity differs from its source receipt")
            for source in payload.get("sources", {}).values():
                if compute_file_sha256(source["path"]) != source["sha256"]:
                    raise ValueError("cycIF raw source changed after export")
            bindings = payload.get("artifacts", {})
            if not bindings:
                raise ValueError("CycIF receipt has no artifact bindings")
            base = Path(row["image_path"]).parent
            if not all(
                (base / name).is_file() and compute_file_sha256(base / name) == sha
                for name, sha in bindings.items()
            ):
                raise ValueError("CycIF arrays changed after source-bound export")
            verified[str(receipt_path)] = payload
        payload = verified[str(receipt_path)]
        bound_rows = [
            item
            for item in payload.get("table_rows", [])
            if item.get("marker") == row["marker"]
        ]
        if len(bound_rows) != 1:
            raise ValueError("cycIF receipt does not bind a unique biological marker")
        for column in (
            "image_path",
            "support_path",
            "tissue_mask_path",
            "reference_image_path",
            "source_path",
            "source_sha256",
        ):
            if str(bound_rows[0].get(column, "")) != str(row.get(column, "")):
                raise ValueError(
                    f"cycIF {column} differs from its receipt-bound marker"
                )
        for axis in ("x", "y"):
            pitch = float(row[f"pixel_size_{axis}_um"])
            if not math.isfinite(pitch) or pitch <= 0:
                raise ValueError("Unknown cycIF physical pixel scale")
    return frame


def prepare_native_hs_anchor(
    source: dict[str, Any],
    channel_map: pd.DataFrame,
    *,
    cache: Any,
    object_dir: str | Path,
    max_working_bytes: int,
    downsample_factor: int = 4,
) -> dict[str, Any]:
    """Prepare one reviewed native HS feature bundle independently of cycIF availability."""
    import clinical_roi_hyperspectral as hs
    from clinical_roi_cycif import extract_czi_metadata

    guard_compute()
    objects = Path(object_dir)
    objects.mkdir(parents=True, exist_ok=True)
    key = (str(source["assigned_tma"]), str(source["confirmed_core_id"]))
    core_dir = objects / f"{key[0]}_{key[1]}"
    core_dir.mkdir(exist_ok=True)
    bundle_path = core_dir / "hs_features.npz"
    geometry_path = core_dir / "hs_features_meta.json"
    prep_receipt = core_dir / "preparation_receipt.json"
    digest, signature = cache.raw_identity(Path(source["path"]))
    parameters = {
        "downsample_factor": downsample_factor,
        "max_working_bytes": max_working_bytes,
        "source_sha256": digest,
        "source_path": str(source["path"]),
        "code": {
            function.__name__: hashlib.sha256(
                inspect.getsource(function).encode()
            ).hexdigest()
            for function in (
                prepare_native_hs_anchor,
                derive_native_l1_features,
                area_average_downsample_l1,
                geometry_grid_id,
                _tissue,
                extract_czi_metadata,
                hs.load_spectral_trial_for_acquisition,
            )
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "aicspylibczi")
        },
        "support_code": compute_file_sha256(
            Path(__file__).parent / "guided_serial_registration.py"
        ),
        "channel_map": channel_map.loc[channel_map["path"].eq(source["path"])].to_json(
            orient="records"
        ),
    }
    warm = prep_receipt.is_file() and geometry_path.is_file() and bundle_path.is_file()
    if warm:
        saved = _json(prep_receipt)
        warm = saved.get("parameters") == parameters and _verify_artifacts(
            saved.get("artifacts", [])
        )
    if not warm:
        trial = hs.load_spectral_trial_for_acquisition(
            source, channel_map, cache=cache, max_working_bytes=max_working_bytes
        )
        cube = trial["cube"]
        pixels = int(np.prod(cube.shape[-2:]))
        # Bound raw cube + native float L1/total and decoder/averaging allowance.
        if cube.nbytes + pixels * (20 * 4 + 4 + 8) > max_working_bytes:
            raise MemoryError(
                "Native L1 preparation exceeds the declared memory budget; request a larger compute allocation"
            )
        l1_native, total_native = derive_native_l1_features(cube)
        l1, total, support = area_average_downsample_l1(
            l1_native, total_native, downsample_factor
        )
        from aicspylibczi import CziFile
        from clinical_roi_cycif import extract_czi_metadata

        physical = extract_czi_metadata(source["path"])
        box = CziFile(str(source["path"])).get_mosaic_bounding_box()
        if tuple(cube.shape[-2:]) != (int(box.h), int(box.w)):
            raise ValueError(
                "Decoded native HS cube differs from its explicit stage bounding box"
            )
        geometry = {
            "axes": "YXC",
            "native_shape_yx": list(cube.shape[-2:]),
            "native_origin_xy": [int(box.x), int(box.y)],
            "downsample_factor": downsample_factor,
            "block_center_offset_native_xy": [(downsample_factor - 1) / 2] * 2,
            "pixel_size_x_um": physical["pixel_size_x_um"] * downsample_factor,
            "pixel_size_y_um": physical["pixel_size_y_um"] * downsample_factor,
            "assigned_tma": key[0],
            "confirmed_core_id": key[1],
            "hs_source_sha256": digest,
            "array_shape_yx": list(total.shape),
            "decoded_native_shape_yx": list(cube.shape[-2:]),
            "source_bbox_xywh": [int(box.x), int(box.y), int(box.w), int(box.h)],
            "native_geometry_check": "Every band decoded at explicit bounding-box region; exact YX shape verified",
        }
        geometry["grid_id"] = geometry_grid_id(geometry)
        np.savez_compressed(bundle_path, l1=l1, total=total, support=support)
        save_json(geometry_path, geometry)
        cache.require_unchanged(Path(source["path"]), signature)
        save_json(
            prep_receipt,
            {
                "parameters": parameters,
                "artifacts": [_binding(bundle_path), _binding(geometry_path)],
            },
        )
        del cube, trial, l1_native, total_native, l1, total, support
    geometry = _json(geometry_path)
    cache.require_unchanged(Path(source["path"]), signature)
    print(
        f"{'CACHE_HIT' if warm else 'COMPUTED'} native HS anchor {key[0]}/{key[1]}",
        flush=True,
    )
    return {
        "hs_source_path": source["path"],
        "hs_source_sha256": digest,
        "hs_features_path": str(bundle_path.resolve()),
        "geometry_path": str(geometry_path.resolve()),
        "geometry_sha256": compute_file_sha256(geometry_path),
        "grid_id": geometry["grid_id"],
        "anchor_pixel_size_x_um": geometry["pixel_size_x_um"],
        "anchor_pixel_size_y_um": geometry["pixel_size_y_um"],
    }


def prepare_registration_inputs(
    hs_manifest: pd.DataFrame,
    channel_map: pd.DataFrame,
    cycif_manifest_path: str | Path,
    *,
    cache: Any,
    object_dir: str | Path,
    table_dir: str | Path,
    max_working_bytes: int,
    downsample_factor: int = 4,
    core_selection: str | list[tuple[str, str]] = "ALL",
) -> pd.DataFrame:
    """Prepare native-derived features and exact paired structural inputs one core at a time."""
    import clinical_roi_hyperspectral as hs

    guard_compute()
    cycif = load_cycif_handoff(cycif_manifest_path)
    final = hs._validated_spectral_manifest(hs_manifest, cache)
    if final.duplicated(["assigned_tma", "confirmed_core_id"]).any():
        raise ValueError("Ambiguous reviewed HS source")
    requested = (
        set(zip(final["assigned_tma"], final["confirmed_core_id"]))
        if core_selection == "ALL"
        else {tuple(key) for key in core_selection}
    )
    if not requested or not requested <= set(
        zip(final["assigned_tma"], final["confirmed_core_id"])
    ):
        raise ValueError("Core selection must use reviewed TMA-qualified keys")
    objects, tables = Path(object_dir), Path(table_dir)
    objects.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for source in final.to_dict("records"):
        key = (str(source["assigned_tma"]), str(source["confirmed_core_id"]))
        if key not in requested:
            continue
        markers = cycif.loc[
            cycif["assigned_tma"].eq(key[0]) & cycif["confirmed_core_id"].eq(key[1])
        ]
        dapi = markers.loc[markers["marker"].eq("DAPI")]
        if len(dapi) != 1:
            exclusions.append(
                {
                    "assigned_tma": key[0],
                    "confirmed_core_id": key[1],
                    "reason": "missing_or_ambiguous_cycif_DAPI",
                }
            )
            continue
        anchor = prepare_native_hs_anchor(
            source,
            channel_map,
            cache=cache,
            object_dir=objects,
            max_working_bytes=max_working_bytes,
            downsample_factor=downsample_factor,
        )
        rows.append({**dapi.iloc[0].to_dict(), **anchor})
    pd.DataFrame(
        exclusions, columns=["assigned_tma", "confirmed_core_id", "reason"]
    ).to_csv(tables / "pairing_exclusions.csv", index=False)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No eligible HS/cycIF pair; inspect pairing_exclusions.csv")
    result.to_csv(tables / "registration_inputs.csv", index=False)
    return result


def attach_refined_tissue_supports(
    inputs: pd.DataFrame,
    hs_mask_manifest_path: str | Path,
    cycif_mask_manifest_path: str | Path,
    *,
    hs_review_ledger_path: str | Path,
    cycif_review_ledger_path: str | Path,
    table_dir: str | Path,
) -> pd.DataFrame:
    """Attach reviewed main-core masks; each core needs an accepted, hash-bound review."""
    hs_masks = pd.read_csv(hs_mask_manifest_path, keep_default_na=False)
    cycif_masks = pd.read_csv(cycif_mask_manifest_path, keep_default_na=False)
    ledgers = (
        pd.read_csv(hs_review_ledger_path, keep_default_na=False),
        pd.read_csv(cycif_review_ledger_path, keep_default_na=False),
    )
    keys = ["assigned_tma", "confirmed_core_id"]
    for frame in (hs_masks, cycif_masks):
        if frame.duplicated(keys).any():
            raise ValueError(
                "Refined support manifest has duplicate qualified core keys"
            )
    records: list[dict[str, Any]] = []
    for source in inputs.to_dict("records"):
        selected: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        for frame, ledger in zip((hs_masks, cycif_masks), ledgers):
            match = frame.loc[
                frame[keys[0]].eq(source[keys[0]]) & frame[keys[1]].eq(source[keys[1]])
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Missing unique refined support for {source[keys[0]]}/{source[keys[1]]}"
                )
            row = match.iloc[0].to_dict()
            review = ledger.loc[
                ledger[keys[0]].eq(source[keys[0]])
                & ledger[keys[1]].eq(source[keys[1]])
            ]
            if (
                len(review) != 1
                or review.iloc[0]["review_status"] != "accepted_main_core_overview"
                or review.iloc[0]["tissue_mask_sha256"] != row["tissue_mask_sha256"]
                or review.iloc[0]["receipt_sha256"] != row["receipt_sha256"]
            ):
                raise ValueError(
                    f"No accepted source-bound mask review for {source[keys[0]]}/{source[keys[1]]}"
                )
            if compute_file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
                raise ValueError("Refined support receipt changed")
            receipt = _json(row["receipt_path"])
            if not _verify_artifacts(receipt["artifacts"]):
                raise ValueError("Refined mask, preview or feature bundle changed")
            if (
                compute_file_sha256(row["tissue_mask_path"])
                != row["tissue_mask_sha256"]
            ):
                raise ValueError("Refined tissue mask hash differs from its manifest")
            if str(Path(row["tissue_mask_path"]).resolve()) != str(
                Path(receipt["mask_path"]).resolve()
            ):
                raise ValueError("Refined mask path differs from its receipt")
            selected.append(row)
            receipts.append(receipt)
        hs_row, cycif_row = selected
        hs_receipt, cycif_receipt = receipts
        if (
            hs_row["hs_features_path"] != source["hs_features_path"]
            or hs_row["grid_id"] != source["grid_id"]
            or hs_row["geometry_sha256"] != source["geometry_sha256"]
            or compute_file_sha256(source["hs_features_path"])
            != hs_row["hs_features_sha256"]
        ):
            raise ValueError(
                "Refined HS support does not bind the prepared native feature grid"
            )
        if (
            not strict_bool(hs_receipt["spectral_arrays_exactly_preserved"])
            or hs_row["reviewed_hs_features_path"] != hs_receipt["derived_path"]
            or compute_file_sha256(hs_row["reviewed_hs_features_path"])
            != hs_row["reviewed_hs_features_sha256"]
        ):
            raise ValueError(
                "Refined HS bundle is not the verified spectral-preserving artifact"
            )
        if (
            cycif_receipt["identity"]["parent_sha256"] != source["receipt_sha256"]
            or cycif_row["reference_image_path"] != source["reference_image_path"]
            or cycif_row["reference_support_path"] != source["support_path"]
            or compute_file_sha256(source["reference_image_path"])
            != cycif_row["reference_image_sha256"]
        ):
            raise ValueError(
                "Refined cycIF support does not bind this native R1 source frame"
            )
        mask = _read_array(cycif_row["tissue_mask_path"])
        if (
            mask.shape != _read_array(source["reference_image_path"]).shape
            or mask.dtype != bool
            or not mask.any()
        ):
            raise ValueError(
                "Refined cycIF mask has invalid shape, type or empty support"
            )
        records.append(
            {
                **source,
                "original_hs_features_path": source["hs_features_path"],
                "hs_features_path": hs_row["reviewed_hs_features_path"],
                "hs_tissue_receipt_path": hs_row["receipt_path"],
                "registration_tissue_mask_path": cycif_row["tissue_mask_path"],
                "cycif_tissue_receipt_path": cycif_row["receipt_path"],
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(
        Path(table_dir) / "registration_inputs_reviewed_masks.csv", index=False
    )
    return result


def evaluate_coarse_orientations(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    moving_pixel_size_xy: tuple[float, float],
    fixed_pixel_size_xy: tuple[float, float],
    allow_reflection: bool = False,
    reflection_evidence: str = "",
    work_edge: int = 256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the existing full-angle structural screen with physical scaling and full mass."""
    import dylan_c2_image_orientation as orientation

    if allow_reflection and not reflection_evidence.strip():
        raise ValueError(
            "Reflection requires explicit acquisition/orientation evidence"
        )
    if min(*moving_pixel_size_xy, *fixed_pixel_size_xy) <= 0 or work_edge < 32:
        raise ValueError(
            "Positive physical scales and a usable orientation work grid are required"
        )
    fs = max(1, int(math.ceil(max(fixed_mask.shape) / work_edge)))
    ms = max(1, int(math.ceil(max(moving_mask.shape) / work_edge)))
    fixed = np.asarray(fixed_mask[::fs, ::fs], dtype=bool)
    moving = np.asarray(moving_mask[::ms, ::ms], dtype=bool)
    if not fixed.any() or not moving.any():
        raise ValueError(
            "Orientation needs nonempty structural tissue in both modalities"
        )
    fixed_center = np.argwhere(fixed).mean(axis=0)
    moving_center = np.argwhere(moving).mean(axis=0)
    sy = moving_pixel_size_xy[1] * ms / (fixed_pixel_size_xy[1] * fs)
    sx = moving_pixel_size_xy[0] * ms / (fixed_pixel_size_xy[0] * fs)
    physical_scale = np.diag([sy, sx, 1.0])
    candidates: list[dict[str, Any]] = []
    for reflected, _, _, angle in orientation.angular_grid():
        if reflected and not allow_reflection:
            continue
        transform = (
            orientation.similarity_matrix(
                angle, reflected, 1.0, moving_center * [sy, sx], fixed_center
            )
            @ physical_scale
        )
        score = orientation._transformed_mask_score(moving, fixed, transform, order=0)
        native = np.diag([fs, fs, 1.0]) @ transform @ np.diag([1 / ms, 1 / ms, 1.0])
        candidates.append(
            {
                "angle_degrees": angle,
                "reflected": reflected,
                **score,
                "matrix_yx_json": json.dumps(native.tolist()),
            }
        )
    table = (
        pd.DataFrame(candidates)
        .sort_values(
            ["dice", "clipped_fraction", "angle_degrees"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )
    best = table.iloc[0].to_dict()
    best["ambiguity_margin"] = (
        float(table.iloc[0]["dice"] - table.iloc[1]["dice"]) if len(table) > 1 else 0.0
    )
    separation = np.abs(
        (table["angle_degrees"] - best["angle_degrees"] + 180) % 360 - 180
    )
    distinct = table.loc[separation.ge(30)]
    best["outline_distinct_30deg_margin"] = (
        float(best["dice"] - distinct.iloc[0]["dice"]) if not distinct.empty else 0.0
    )
    best["initialization_basis"] = (
        "Tissue outline only; internal structure must be reviewed after VALIS"
    )
    best["reflection_evidence"] = reflection_evidence
    return table, best


def orientation_screen(
    inputs: pd.DataFrame,
    *,
    table_dir: str | Path,
    allow_reflection: bool = False,
    reflection_evidence: str = "",
) -> pd.DataFrame:
    directory = Path(table_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in inputs.to_dict("records"):
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed_mask = np.array(bundle["support"])
        moving = np.asarray(_read_array(row["reference_image_path"]))
        moving_mask = np.asarray(
            _read_array(
                row.get("registration_tissue_mask_path", row["tissue_mask_path"])
            ),
            dtype=bool,
        )
        if moving_mask.shape != moving.shape:
            raise ValueError("cycIF reference and tissue mask use different grids")
        candidates, best = evaluate_coarse_orientations(
            fixed_mask,
            moving_mask,
            moving_pixel_size_xy=(
                float(row["pixel_size_x_um"]),
                float(row["pixel_size_y_um"]),
            ),
            fixed_pixel_size_xy=(
                float(row["anchor_pixel_size_x_um"]),
                float(row["anchor_pixel_size_y_um"]),
            ),
            allow_reflection=allow_reflection,
            reflection_evidence=reflection_evidence,
        )
        core = f"{row['assigned_tma']}_{row['confirmed_core_id']}"
        candidates.to_csv(directory / f"{core}_orientation_candidates.csv", index=False)
        rows.append(
            {
                "assigned_tma": row["assigned_tma"],
                "confirmed_core_id": row["confirmed_core_id"],
                **best,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(directory / "coarse_orientation_ledger.csv", index=False)
    return result


def ensure_review_inputs(
    inputs: pd.DataFrame, *, table_dir: str | Path
) -> tuple[Path, Path]:
    """Create only absent human-input schemas; never erase existing measurements/reviews."""
    directory = Path(table_dir)
    directory.mkdir(parents=True, exist_ok=True)
    review_path, landmark_path = (
        directory / "registration_review.csv",
        directory / "heldout_landmarks.csv",
    )
    if not review_path.exists():
        records = []
        for row in inputs.to_dict("records"):
            records.append(
                {
                    "assigned_tma": row["assigned_tma"],
                    "confirmed_core_id": row["confirmed_core_id"],
                    "orientation_review": "PENDING",
                    "orientation_evidence": "",
                    "visual_overlay_review": "PENDING",
                    "max_median_tre_um": "",
                    "max_p95_tre_um": "",
                    "section_relationship": "unknown",
                    "section_evidence": "",
                    "group_id": "",
                    "group_evidence": "",
                }
            )
        pd.DataFrame(records).to_csv(review_path, index=False)
    if not landmark_path.exists():
        pd.DataFrame(
            columns=[
                "assigned_tma",
                "confirmed_core_id",
                "landmark_id",
                "role",
                "fixed_x",
                "fixed_y",
                "moving_x",
                "moving_y",
            ]
        ).to_csv(landmark_path, index=False)
    return review_path, landmark_path


def warp_image_yx(
    image: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
    *,
    categorical: bool = False,
) -> np.ndarray:
    """Reuse the orientation helper's forward-YX convention; categorical output stays boolean."""
    from dylan_c2_image_orientation import _warp

    warped = _warp(image, matrix, output_shape, 0 if categorical else 1)
    return warped > 0.5 if categorical else warped.astype(np.float32)


def coarse_antialias_sigma(matrix_yx: np.ndarray) -> tuple[float, float]:
    """Gaussian prefilter width for source-axis decimation, in source pixels."""
    scales = np.linalg.norm(np.asarray(matrix_yx)[:2, :2], axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid physical-coordinate sampling scale")
    return tuple(float(value) for value in np.maximum(0, (1 / scales - 1) / 2))


def warp_continuous_yx(
    image: np.ndarray,
    observed: np.ndarray,
    matrix_yx: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Antialias before decimation; retain coverage of the complete interpolation footprint."""
    values = np.asarray(image, dtype=np.float32)
    coverage = np.asarray(observed, dtype=np.float32)
    if values.shape != coverage.shape or values.ndim != 2:
        raise ValueError(
            "Continuous image and observed support must share a 2D source grid"
        )
    sigma = coarse_antialias_sigma(matrix_yx)
    if max(sigma) > 0:
        values = ndimage.gaussian_filter(
            values, sigma, mode="constant", cval=0, truncate=4
        )
        coverage = ndimage.gaussian_filter(
            coverage, sigma, mode="constant", cval=0, truncate=4
        )
    return (
        warp_image_yx(values, matrix_yx, output_shape),
        warp_image_yx(coverage, matrix_yx, output_shape),
    )


def transform_points_yx(points_xy: np.ndarray, matrix_yx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Landmarks must be finite Nx2 XY coordinates")
    yx1 = np.column_stack((points[:, ::-1], np.ones(len(points))))
    return (yx1 @ matrix_yx.T)[:, :2][:, ::-1]


def _run_isolated(
    job_path: Path, *, timeout_seconds: int | None = None
) -> dict[str, Any]:
    guard_compute()
    job = _json(job_path)
    destination = Path(job["result_path"])
    log_path = job_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                str(GUARDED_PYTHON),
                "spatial-valis",
                str(Path(__file__).resolve()),
                "--worker",
                str(job_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_seconds,
        )
    if not destination.is_file():
        raise RuntimeError(f"VALIS worker produced no result; inspect {log_path}")
    return _json(destination)


def _scale_structural_proxy(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale structural counts consistently within each modality."""
    values = np.log1p(np.maximum(np.asarray(array, dtype=np.float32), 0))
    finite = mask & np.isfinite(values)
    low, high = np.percentile(values[finite], [1, 99])
    if high <= low:
        raise ValueError("Structural image has no useful intensity range")
    return np.where(mask, np.clip((values - low) / (high - low), 0, 1), 0).astype(
        np.float32
    )


def bounded_mattes_correction(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    pixel_size_xy: tuple[float, float],
    *,
    maximum_angle_degrees: float = 5.0,
    maximum_translation_fraction: float = 0.05,
    smoothing_sigma_pixels: float = 2.0,
    fit_stride: int = 2,
) -> dict[str, Any]:
    """Fit a scale-fixed Euler correction with explicit, enforced bounds.

    Returns inverse XY in the full anchor pixel grid (fixed to moving), and
    forward YX (moving to fixed). Euler center is included in the matrices.
    Bounds constrain the search; they are not independent error estimates.
    """
    if not (fixed.shape == moving.shape == fixed_mask.shape == moving_mask.shape):
        raise ValueError("The two structural proxies and masks must share a grid")
    if fixed.ndim != 2 or fit_stride < 1:
        raise ValueError("Expected a 2D anchor and positive integer fit stride")
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    pitch = np.asarray(pixel_size_xy, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("Physical pixel spacing must be finite and positive")
    if not fixed_mask.any() or not moving_mask.any():
        raise ValueError("Both structural images require observed tissue support")
    limits = np.concatenate(
        [
            np.asarray([np.deg2rad(maximum_angle_degrees)]),
            np.asarray(fixed.shape[::-1], dtype=float)
            * pitch
            * maximum_translation_fraction,
        ]
    )
    images: list[sitk.Image] = []
    masks: list[sitk.Image] = []
    for array, mask in ((fixed, fixed_mask), (moving, moving_mask)):
        proxy = ndimage.gaussian_filter(
            _scale_structural_proxy(array, mask),
            smoothing_sigma_pixels,
            mode="constant",
        )[::fit_stride, ::fit_stride]
        image = sitk.GetImageFromArray(proxy.astype(np.float32))
        image.SetSpacing(tuple(pitch * fit_stride))
        images.append(image)
        support = sitk.GetImageFromArray(
            mask[::fit_stride, ::fit_stride].astype(np.uint8)
        )
        support.CopyInformation(image)
        masks.append(support)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricFixedMask(masks[0])
    registration.SetMetricMovingMask(masks[1])
    registration.SetInterpolator(sitk.sitkLinear)
    center = (np.asarray(fixed.shape[::-1], dtype=float) - 1) * pitch / 2
    history: list[dict[str, Any]] = []

    def transform(parameters: np.ndarray) -> sitk.Euler2DTransform:
        result = sitk.Euler2DTransform()
        result.SetCenter(tuple(center))
        result.SetParameters(tuple(np.asarray(parameters, dtype=float) * limits))
        return result

    def objective(parameters: np.ndarray) -> float:
        if np.any(np.abs(parameters) > 1 + 1e-12):
            raise ValueError("Optimizer proposed a correction outside declared bounds")
        registration.SetInitialTransform(transform(parameters))
        value = float(registration.MetricEvaluate(images[0], images[1]))
        history.append({"parameters_scaled": parameters.tolist(), "negative_mi": value})
        if len(history) % 50 == 0:
            print(f"MI evaluations={len(history)} negative_MI={value:.6f}", flush=True)
        return value

    initial = objective(np.zeros(3, dtype=float))
    result = optimize.minimize(
        objective,
        np.zeros(3, dtype=float),
        method="Powell",
        bounds=[(-1.0, 1.0)] * 3,
        options={"maxiter": 60, "maxfev": 800, "xtol": 1e-4, "ftol": 1e-5},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Bounded MI optimizer did not converge: {result.message}")
    if float(result.fun) > initial + 1e-10:
        raise RuntimeError("Bounded MI optimization worsened its initial metric")
    selected = result.x
    fitted = transform(selected)
    origin = np.asarray(fitted.TransformPoint((0.0, 0.0))) / pitch
    inverse_xy = np.eye(3)
    for axis in range(2):
        point = np.zeros(2, dtype=float)
        point[axis] = pitch[axis]
        inverse_xy[:2, axis] = (
            np.asarray(fitted.TransformPoint(tuple(point))) / pitch - origin
        )
    inverse_xy[:2, 2] = origin
    swap = np.eye(3)[[1, 0, 2]]
    forward_yx = swap @ np.linalg.inv(inverse_xy) @ swap
    corners_xy = np.asarray(
        [(x, y) for x in (0, fixed.shape[1] - 1) for y in (0, fixed.shape[0] - 1)],
        dtype=float,
    )
    mapped_matrix = np.c_[corners_xy, np.ones(len(corners_xy))] @ inverse_xy.T
    mapped_sitk = (
        np.asarray(
            [fitted.TransformPoint(tuple(point * pitch)) for point in corners_xy]
        )
        / pitch
    )
    parity_error = float(np.max(np.abs(mapped_matrix[:, :2] - mapped_sitk)))
    if parity_error > 1e-8:
        raise AssertionError("Euler center or pixel-coordinate conversion is incorrect")
    return {
        "inverse_xy": inverse_xy.tolist(),
        "forward_yx": forward_yx.tolist(),
        "angle_degrees": float(np.rad2deg(fitted.GetAngle())),
        "translation_xy_um": list(fitted.GetTranslation()),
        "center_xy_um": list(fitted.GetCenter()),
        "negative_mi_before": initial,
        "negative_mi_after": float(min(initial, result.fun)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": len(history),
        "at_search_bound": bool(np.any(np.abs(selected) >= 0.99)),
        "corner_matrix_parity_max_error_pixels": parity_error,
        "settings": {
            "maximum_angle_degrees": maximum_angle_degrees,
            "maximum_translation_fraction": maximum_translation_fraction,
            "smoothing_sigma_pixels": smoothing_sigma_pixels,
            "fit_stride": fit_stride,
            "histogram_bins": 32,
            "scale": 1.0,
            "independent_validation_n": 0,
        },
        "history": history,
    }


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    """Actual VALIS fit/warp or held-out-point operation, imported only in isolated worker."""
    guard_compute()
    from valis import registration
    from valis_supplied_transform import SuppliedTransformerValis
    try:
        from valis import feature_detectors
    except ImportError:
        feature_detectors = None
    from valis_supplied_transform import SuppliedTransformerValis
    import cv2

    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeError("The isolated VALIS backend did not load its Torch runtime")

    np.random.seed(0)
    cv2.setRNGSeed(0)
    cv2.setNumThreads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    if job["operation"] == "fit_and_warp":
        import tifffile

        folder = Path(job["core_dir"])
        fixed_name, moving_name = "fixed_hs.ome.tiff", "moving_cycif.ome.tiff"
        shape = tuple(job["anchor_shape"])
        refinement = bounded_mattes_correction(
            tifffile.imread(folder / "inputs" / fixed_name),
            tifffile.imread(folder / "inputs" / moving_name),
            np.load(job["fixed_mask_path"], allow_pickle=False),
            np.load(job["moving_mask_path"], allow_pickle=False),
            tuple(job["pixel_size_xy"]),
            **job.get("refinement_parameters", {}),
        )
        refinement_path = folder / "mi_rigid_refinement.json"
        save_json(refinement_path, refinement)
        inverse_xy = np.asarray(refinement["inverse_xy"], dtype=float)
        fixed_path = str(folder / "inputs" / fixed_name)
        moving_path = str(folder / "inputs" / moving_name)
        supplied = {
            fixed_path: {
                "M": np.eye(3),
                "transformation_src_shape_rc": shape,
                "transformation_dst_shape_rc": shape,
            },
            moving_path: {
                "M": inverse_xy,
                "transformation_src_shape_rc": shape,
                "transformation_dst_shape_rc": shape,
            },
        }
        max_dim = min(1024, max(job["anchor_shape"]))
        registrar = SuppliedTransformerValis(
            str(folder / "inputs"),
            str(folder / "valis"),
            reference_img_f=fixed_path,
            align_to_reference=True,
            do_rigid=supplied,
            imgs_ordered=True,
            crop_for_rigid_reg=False,
            check_for_reflections=False,
            image_type="fluorescence",
            non_rigid_registrar_cls=None,
            create_masks=False,
            feature_detector_cls=None,
            max_image_dim_px=max_dim,
            max_processed_image_dim_px=max_dim,
            resolution_xyu=(job["pixel_size_xy"][0], job["pixel_size_xy"][1], "µm"),
        )
                # VALIS 1.2.0 omits this attribute when nonrigid registration is disabled,
        # although cleanup() always indexes it before saving the fitted registrar.
        if not hasattr(registrar, "non_rigid_reg_kwargs"):
            registrar.non_rigid_reg_kwargs = {}
        try:
            rigid, nonrigid, errors = registrar.register()
            if (
                rigid is None
                or errors is None
                or not getattr(registrar, "reg_f", None)
                or not Path(registrar.reg_f).is_file()
            ):
                raise RuntimeError(
                    "VALIS returned failed/no persisted registration; no acceptance or identity fallback"
                )
            fixed = registrar.get_slide(fixed_name)
            moving = registrar.get_slide(moving_name)
            matrix = np.asarray(job["coarse_matrix_yx"], dtype=float)
            shape = tuple(job["anchor_shape"])
            # Verify the actual VALIS mapping against the externally estimated
            # inverse-XY transform at real-image corners and its center.
            probes = np.asarray(
                [
                    [0, 0],
                    [shape[1] - 1, 0],
                    [0, shape[0] - 1],
                    [shape[1] - 1, shape[0] - 1],
                    [(shape[1] - 1) / 2, (shape[0] - 1) / 2],
                ],
                dtype=float,
            )
            expected = (
                np.c_[probes, np.ones(len(probes))] @ np.linalg.inv(inverse_xy).T
            )[:, :2]
            actual = np.asarray(
                moving.warp_xy_from_to(
                    probes,
                    fixed,
                    src_pt_level=shape,
                    dst_slide_level=shape,
                    non_rigid=False,
                ),
                dtype=float,
            )
            parity_error = float(np.max(np.abs(actual - expected)))
            if not np.isfinite(parity_error) or parity_error > 1e-3:
                raise ValueError(
                    f"VALIS supplied-transform coordinate mismatch: {parity_error} pixels"
                )
            match_path = folder / "fitted_feature_matches.csv"
            pd.DataFrame(
                columns=[
                    "fixed_x",
                    "fixed_y",
                    "registered_moving_x",
                    "registered_moving_y",
                    "fitted_feature_residual_um",
                ]
            ).to_csv(match_path, index=False)
            unused_matches = getattr(moving, "xy_matched_to_prev", None)
            fit_quality = {
                "fitted_feature_count": 0,
                "fitted_feature_median_um": None,
                "fitted_feature_p95_um": None,
                "unused_descriptor_match_count": 0
                if unused_matches is None
                else len(unused_matches),
                "independent_validation_n": 0,
                "independent_tre_median_um": None,
                "independent_tre_p95_um": None,
                "registration_model": "Bounded SimpleITK Mattes MI rigid correction; VALIS supplied-transform persistence and warping",
                "mi_cost_before": refinement["negative_mi_before"],
                "mi_cost_after": refinement["negative_mi_after"],
                "mi_optimizer_success": refinement["optimizer_success"],
                "mi_at_search_bound": refinement["at_search_bound"],
                "fine_inverse_angle_degrees": refinement["angle_degrees"],
                "fine_inverse_translation_xy_um": json.dumps(
                    refinement["translation_xy_um"]
                ),
                "fine_max_corner_displacement_um": float(
                    np.max(
                        np.linalg.norm(
                            (expected - probes) * np.asarray(job["pixel_size_xy"]),
                            axis=1,
                        )
                    )
                ),
                "supplied_transform_parity_max_error_pixels": parity_error,
                "valis_version": importlib.metadata.version("valis-wsi"),
                "simpleitk_version": importlib.metadata.version("SimpleITK"),
                "random_seed": 0,
                "coarse_antialias_sigma_source_yx": json.dumps(
                    coarse_antialias_sigma(matrix)
                ),
                "resampling": "Gaussian prefilter for coarse decimation; bilinear coarse and VALIS warps; nearest tissue mask",
                "mi_refinement_path": str(refinement_path.resolve()),
            }
            paths: list[dict[str, Any]] = []
            for row in job["markers"]:
                print(
                    f"Warping {row['assigned_tma']} {row['confirmed_core_id']} {row['marker']}",
                    flush=True,
                )
                image = np.asarray(_read_array(row["image_path"]))
                observed = np.asarray(_read_array(row["support_path"]), dtype=bool)
                tissue = np.asarray(
                    _read_array(
                        row.get(
                            "registration_tissue_mask_path", row["tissue_mask_path"]
                        )
                    ),
                    dtype=bool,
                )
                oriented, oriented_coverage = warp_continuous_yx(
                    image, observed, matrix, shape
                )
                oriented_tissue = warp_image_yx(tissue, matrix, shape, categorical=True)
                warped = np.asarray(
                    moving.warp_img_from_to(
                        oriented,
                        fixed,
                        dst_slide_level=shape,
                        non_rigid=nonrigid is not None,
                        interp_method="bilinear",
                    ),
                    dtype=np.float32,
                )
                coverage = np.asarray(
                    moving.warp_img_from_to(
                        oriented_coverage,
                        fixed,
                        dst_slide_level=shape,
                        non_rigid=False,
                        interp_method="bilinear",
                    )
                )
                support = (
                    np.asarray(
                        moving.warp_img_from_to(
                            oriented_tissue.astype(np.uint8),
                            fixed,
                            dst_slide_level=shape,
                            non_rigid=nonrigid is not None,
                            interp_method="nearest",
                        )
                    )
                    > 0
                )
                support &= coverage >= (1 - 1e-5)
                if warped.shape != shape or support.shape != shape:
                    raise ValueError(
                        "VALIS output does not match fixed HS anchor geometry"
                    )
                marker = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in row["marker"]
                )
                image_path, support_path = (
                    folder / f"{marker}_registered.npy",
                    folder / f"{marker}_support.npy",
                )
                np.save(image_path, warped)
                np.save(support_path, support & np.isfinite(warped))
                paths.append(
                    {
                        **row,
                        "source_image_path": row["image_path"],
                        "source_support_path": row["support_path"],
                        "image_path": str(image_path.resolve()),
                        "support_path": str(support_path.resolve()),
                    }
                )
            errors.to_csv(folder / "valis_fit_diagnostics.csv", index=False)
            # Bind the registrar plus on-disk displacement data that its pickle may reference.
            transform_artifacts = [
                _binding(path)
                for path in (folder / "valis").rglob("*")
                if path.is_file()
            ]
            transform_artifacts += [
                _binding(match_path),
                _binding(refinement_path),
                _binding(folder / "valis_fit_diagnostics.csv"),
            ]
            return {
                "status": "COMPLETED",
                "rows": paths,
                "registrar_path": str(Path(registrar.reg_f).resolve()),
                "fixed_name": fixed_name,
                "moving_name": moving_name,
                "non_rigid": nonrigid is not None,
                "transform_artifacts": transform_artifacts,
                "fit_quality": fit_quality,
                "fitted_matches_path": str(match_path.resolve()),
            }
        finally:
            registration.kill_jvm()
    if job["operation"] == "points":
        try:
            registrar = registration.load_registrar(job["registrar_path"])
            moving = registrar.get_slide(job["moving_name"])
            fixed = registrar.get_slide(job["fixed_name"])
            oriented = transform_points_yx(
                np.asarray(job["moving_xy"], dtype=float),
                np.asarray(job["coarse_matrix_yx"]),
            )
            predicted = moving.warp_xy_from_to(
                oriented,
                fixed,
                src_slide_level=0,
                src_pt_level=0,
                dst_slide_level=tuple(job["anchor_shape"]),
                non_rigid=job["non_rigid"],
            )
            predicted = np.asarray(predicted, dtype=float)
            if predicted.shape != oriented.shape or not np.isfinite(predicted).all():
                raise ValueError(
                    "VALIS returned invalid transformed landmark coordinates"
                )
            return {"predicted_xy": predicted.tolist()}
        finally:
            registration.kill_jvm()
    raise ValueError(f"Unknown worker operation: {job['operation']}")


def fit_and_warp_pairs(
    inputs: pd.DataFrame,
    orientation_table: pd.DataFrame,
    cycif_manifest_path: str | Path,
    review_path: str | Path,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    allow_exploratory: bool = False,
    refinement_parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run actual shared-geometry structural fits and marker warps for reviewed orientations."""
    import tifffile

    cycif = load_cycif_handoff(cycif_manifest_path)
    reviews = pd.read_csv(review_path, keep_default_na=False)
    if reviews.duplicated(["assigned_tma", "confirmed_core_id"]).any():
        raise ValueError("Duplicate core review keys")
    records = []
    failures = []
    for row in inputs.to_dict("records"):
        tma, core = row["assigned_tma"], row["confirmed_core_id"]
        review = reviews.loc[
            reviews["assigned_tma"].eq(tma) & reviews["confirmed_core_id"].eq(core)
        ]
        choice = orientation_table.loc[
            orientation_table["assigned_tma"].eq(tma)
            & orientation_table["confirmed_core_id"].eq(core)
        ]
        if len(choice) != 1:
            raise ValueError(f"{tma}/{core}: no unique orientation candidate")
        if not allow_exploratory and (
            len(review) != 1
            or review.iloc[0]["orientation_review"] != "PASS"
            or not str(review.iloc[0]["orientation_evidence"]).strip()
        ):
            raise ValueError(
                f"{tma}/{core}: inspect structural orientation candidates and supply PASS plus orientation evidence first"
            )
        matrix = np.asarray(json.loads(choice.iloc[0]["matrix_yx_json"]), dtype=float)
        if (
            strict_bool(choice.iloc[0]["reflected"])
            and not str(choice.iloc[0].get("reflection_evidence", "")).strip()
        ):
            raise ValueError("Reflection is unsupported by acquisition evidence")
        folder = Path(object_dir) / f"{tma}_{core}"
        (folder / "inputs").mkdir(parents=True, exist_ok=True)
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed_support = np.asarray(bundle["support"], dtype=bool)
            fixed_image = np.where(fixed_support, bundle["total"], 0).astype(np.float32)
        moving_tissue_path = row.get(
            "registration_tissue_mask_path", row["tissue_mask_path"]
        )
        moving_tissue = np.asarray(_read_array(moving_tissue_path), dtype=bool)
        moving_image = np.where(
            moving_tissue,
            _read_array(row["reference_image_path"]),
            0,
        ).astype(np.float32)
        oriented, _ = warp_continuous_yx(
            moving_image, _read_array(row["support_path"]), matrix, fixed_image.shape
        )
        moving_support = warp_image_yx(
            moving_tissue, matrix, fixed_image.shape, categorical=True
        )
        # Tissue-masked structural proxies exclude neighboring fragments.
        # The separate quantitative marker arrays are never masked before export.
        geometry = _json(row["geometry_path"])
        pixel_xy = [geometry["pixel_size_x_um"], geometry["pixel_size_y_um"]]
        markers = cycif.loc[
            cycif["assigned_tma"].eq(tma) & cycif["confirmed_core_id"].eq(core)
        ].to_dict("records")
        attached = {
            key: row[key]
            for key in (
                "registration_tissue_mask_path",
                "cycif_tissue_receipt_path",
                "hs_tissue_receipt_path",
                "original_hs_features_path",
            )
            if key in row
        }
        markers = [{**marker, **attached} for marker in markers]
        base_job = {
            "operation": "fit_and_warp",
            "fixed_mask_path": str((folder / "inputs/fixed_support.npy").resolve()),
            "moving_mask_path": str((folder / "inputs/moving_support.npy").resolve()),
            "refinement_parameters": dict(refinement_parameters or {}),
            "core_dir": str(folder.resolve()),
            "coarse_matrix_yx": matrix.tolist(),
            "anchor_shape": list(fixed_image.shape),
            "pixel_size_xy": pixel_xy,
            "markers": markers,
            "result_path": str((folder / "warp_result.json").resolve()),
        }
        identity = {
            "inputs": {
                key: row[key]
                for key in ("hs_source_sha256", "geometry_sha256", "receipt_sha256")
            },
            "orientation": choice.iloc[0].to_dict(),
            "code": registration_code_bindings(),
            "worker_runtime": worker_runtime_identity(),
            "refinement_parameters": base_job["refinement_parameters"],
            "hs_features": _binding(row["hs_features_path"]),
            "moving_tissue": _binding(moving_tissue_path),
            "selected_markers": sorted(marker["marker"] for marker in markers),
            "reviewed_support_receipts": [
                _binding(row[key])
                for key in ("cycif_tissue_receipt_path", "hs_tissue_receipt_path")
                if key in row
            ],
        }
        core_receipt = folder / "warp_receipt.json"
        warm = core_receipt.is_file()
        if warm:
            saved = _json(core_receipt)
            warm = saved.get("identity") == identity and _verify_artifacts(
                saved.get("artifacts", [])
            )
        if warm:
            result = saved["result"]
        else:
            np.save(base_job["fixed_mask_path"], fixed_support)
            np.save(base_job["moving_mask_path"], moving_support)
            for name, array in (
                ("fixed_hs.ome.tiff", fixed_image),
                ("moving_cycif.ome.tiff", oriented),
            ):
                tifffile.imwrite(
                    folder / "inputs" / name,
                    array.astype(np.float32),
                    ome=True,
                    photometric="minisblack",
                    metadata={
                        "axes": "YX",
                        "PhysicalSizeX": pixel_xy[0],
                        "PhysicalSizeY": pixel_xy[1],
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeYUnit": "µm",
                    },
                )
            job_path = folder / "warp_job.json"
            save_json(job_path, base_job)
            try:
                result = _run_isolated(job_path)
            except (subprocess.CalledProcessError, RuntimeError, ValueError) as error:
                failures.append(
                    {
                        "assigned_tma": tma,
                        "confirmed_core_id": core,
                        "reason": str(error),
                        "log_path": str(job_path.with_suffix(".log")),
                    }
                )
                pd.DataFrame(failures).to_csv(
                    Path(table_dir) / "registration_fit_exclusions.csv", index=False
                )
                print(f"EXCLUDED {tma}/{core}: {error}", flush=True)
                continue
            artifacts = result["transform_artifacts"] + [
                _binding(row["geometry_path"]),
                _binding(row["hs_features_path"]),
                _binding(row["receipt_path"]),
                _binding(moving_tissue_path),
            ]
            artifacts += identity["reviewed_support_receipts"]
            artifacts += [
                _binding(item[key])
                for item in result["rows"]
                for key in ("image_path", "support_path")
            ]
            save_json(
                core_receipt,
                {"identity": identity, "result": result, "artifacts": artifacts},
            )
        for marker in result["rows"]:
            records.append(
                {
                    **marker,
                    "hs_features_path": row["hs_features_path"],
                    "hs_source_path": row["hs_source_path"],
                    "hs_source_sha256": row["hs_source_sha256"],
                    "geometry_path": row["geometry_path"],
                    "geometry_sha256": row["geometry_sha256"],
                    "grid_id": row["grid_id"],
                    "pixel_size_x_um": pixel_xy[0],
                    "pixel_size_y_um": pixel_xy[1],
                    "coarse_matrix_yx_json": json.dumps(matrix.tolist()),
                    "warp_receipt_path": str(core_receipt.resolve()),
                    "warp_receipt_sha256": compute_file_sha256(core_receipt),
                    "registration_accepted": False,
                }
            )
    result_frame = pd.DataFrame(records)
    pd.DataFrame(
        failures, columns=["assigned_tma", "confirmed_core_id", "reason", "log_path"]
    ).to_csv(Path(table_dir) / "registration_fit_exclusions.csv", index=False)
    if result_frame.empty:
        raise ValueError(
            "No VALIS fit succeeded; inspect registration_fit_exclusions.csv and worker logs"
        )
    output = Path(table_dir) / "warped_markers_inventory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)
    return result_frame


def render_registration_qc(warped: pd.DataFrame, *, plot_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    _style()
    directory = Path(plot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in warped.loc[warped["marker"].eq("DAPI")].to_dict("records"):
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed, support = np.array(bundle["total"]), np.array(bundle["support"])
        moving = np.asarray(_read_array(row["image_path"]))
        observed = np.asarray(_read_array(row["support_path"]), dtype=bool)
        valid = support & observed & np.isfinite(moving)
        if not valid.any():
            raise ValueError("Registration has no shared tissue for overlay QC")

        def normalized(image: np.ndarray) -> np.ndarray:
            low, high = np.percentile(image[valid], [1, 99])
            return (
                np.zeros_like(image, dtype=float)
                if high <= low
                else np.clip((image - low) / (high - low), 0, 1)
            )

        f, m = normalized(fixed), normalized(moving)
        rgb = np.stack((f, m, f), axis=-1)
        rgb[~valid] = 0
        yy, xx = np.indices(fixed.shape)
        tile = max(1, min(fixed.shape) // 12)
        checker = np.where(((yy // tile + xx // tile) % 2) == 0, f, m)
        checker[~valid] = 0
        stride = max(1, int(math.ceil(max(fixed.shape) / 768)))
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 76 / 25.4))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.83)
        for axis, image, label in zip(
            axes,
            (rgb, checker, valid),
            (
                "HS magenta / cycIF green",
                "Structural checkerboard",
                "Shared tissue support",
            ),
        ):
            axis.imshow(image[::stride, ::stride], cmap="gray", interpolation="nearest")
            axis.axis("off")
            fig.text(axis.get_position().x0, 0.87, label, fontsize=5)
        fig.text(
            0.02,
            0.96,
            f"{row['assigned_tma']} {row['confirmed_core_id']} — inspect internal landmarks, not outline alone",
            fontsize=5,
        )
        fig.text(
            0.02,
            0.025,
            "Generator: clinical_roi_valis.py; overlays do not establish registration acceptance",
            fontsize=5,
        )
        path = (
            directory
            / f"{row['assigned_tma']}_{row['confirmed_core_id']}_registration_qc.pdf"
        )
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def render_orientation_qc(
    inputs: pd.DataFrame, candidates: pd.DataFrame, *, plot_dir: str | Path
) -> list[Path]:
    """Show actual coarse structural overlays before the orientation review gate."""
    import matplotlib.pyplot as plt

    _style()
    directory = Path(plot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in inputs.to_dict("records"):
        matches = candidates.loc[
            candidates["assigned_tma"].eq(row["assigned_tma"])
            & candidates["confirmed_core_id"].eq(row["confirmed_core_id"])
        ]
        if len(matches) != 1:
            raise ValueError("Missing unique structural orientation candidate")
        choice = matches.iloc[0]
        matrix = np.asarray(json.loads(choice["matrix_yx_json"]))
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed, fixed_support = (
                np.array(bundle["total"]),
                np.array(bundle["support"]),
            )
        moving, _ = warp_continuous_yx(
            np.asarray(_read_array(row["reference_image_path"])),
            _read_array(row["support_path"]),
            matrix,
            fixed.shape,
        )
        moving_support = warp_image_yx(
            np.asarray(
                _read_array(
                    row.get("registration_tissue_mask_path", row["tissue_mask_path"])
                )
            ),
            matrix,
            fixed.shape,
            categorical=True,
        )
        valid = fixed_support & moving_support & np.isfinite(moving)
        if not valid.any():
            raise ValueError("Orientation candidate has no shared structural tissue")
        views: list[np.ndarray] = []
        for image in (fixed, moving):
            low, high = np.percentile(image[valid], [1, 99])
            views.append(
                np.zeros_like(image, dtype=float)
                if high <= low
                else np.clip((image - low) / (high - low), 0, 1)
            )
        rgb = np.stack((views[0], views[1], views[0]), axis=-1)
        rgb[~valid] = 0
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 76 / 25.4))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.12)
        stride = max(1, int(math.ceil(max(fixed.shape) / 768)))
        for axis, image, text in zip(
            axes,
            (*views, rgb),
            (
                "HS structural anchor",
                "Coarse oriented cycIF DAPI",
                "HS magenta / cycIF green",
            ),
        ):
            axis.imshow(image[::stride, ::stride], cmap="gray", interpolation="nearest")
            axis.axis("off")
            fig.text(axis.get_position().x0, 0.87, text, fontsize=5)
        fig.text(
            0.02,
            0.96,
            f"{row['assigned_tma']} {row['confirmed_core_id']}: angle={choice['angle_degrees']}, reflected={choice['reflected']}, structural margin={choice['ambiguity_margin']:.4g}",
            fontsize=5,
        )
        fig.text(
            0.02,
            0.025,
            "Generator: clinical_roi_valis.py; coarse orientation only, not accepted registration",
            fontsize=5,
        )
        path = (
            directory
            / f"{row['assigned_tma']}_{row['confirmed_core_id']}_coarse_orientation_qc.pdf"
        )
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def validate_landmarks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "assigned_tma",
        "confirmed_core_id",
        "landmark_id",
        "role",
        "fixed_x",
        "fixed_y",
        "moving_x",
        "moving_y",
    }
    if required - set(frame.columns):
        raise ValueError("Landmark table lacks coordinate/identity columns")
    if frame.duplicated(["assigned_tma", "confirmed_core_id", "landmark_id"]).any():
        raise ValueError(
            "Landmark IDs must be unique within a core and not shared between fit and validation"
        )
    if not frame["role"].isin(["fit", "validation"]).all():
        raise ValueError("Landmark roles must be fit or validation")
    for column in ("fixed_x", "fixed_y", "moving_x", "moving_y"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError("Landmark coordinates must be finite")
        frame = frame.assign(**{column: values})
    if frame.duplicated(
        ["assigned_tma", "confirmed_core_id", "moving_x", "moving_y"]
    ).any():
        raise ValueError(
            "Repeated landmark coordinates are not independent held-out landmarks"
        )
    if frame.duplicated(
        ["assigned_tma", "confirmed_core_id", "fixed_x", "fixed_y"]
    ).any():
        raise ValueError(
            "Repeated fixed landmarks are not independent validation points"
        )
    return frame


def landmark_tre(
    fixed_xy: np.ndarray, predicted_xy: np.ndarray, pixel_size_xy: tuple[float, float]
) -> dict[str, Any]:
    if (
        fixed_xy.shape != predicted_xy.shape
        or fixed_xy.ndim != 2
        or fixed_xy.shape[1] != 2
    ):
        raise ValueError("Landmark arrays must have matching Nx2 XY shapes")
    if (
        not np.isfinite(fixed_xy).all()
        or not np.isfinite(predicted_xy).all()
        or min(pixel_size_xy) <= 0
    ):
        raise ValueError("Require finite coordinates and positive pixel scales")
    distances = np.linalg.norm(
        (predicted_xy - fixed_xy) * np.asarray(pixel_size_xy), axis=1
    )
    return {
        "validation_n": len(distances),
        "tre_median_um": float(np.median(distances)) if len(distances) else None,
        "tre_p95_um": float(np.percentile(distances, 95)) if len(distances) else None,
        "errors_um": distances.tolist(),
    }


def evaluate_registration_decision(
    review: dict[str, Any], measured: dict[str, Any]
) -> bool:
    """User-chosen error limits and measured landmarks, never an editable accepted flag."""
    if (
        review.get("visual_overlay_review") != "PASS"
        or review.get("orientation_review") != "PASS"
    ):
        return False
    try:
        median_limit, p95_limit = (
            float(review["max_median_tre_um"]),
            float(review["max_p95_tre_um"]),
        )
    except (ValueError, TypeError, KeyError):
        return False
    if not all(
        math.isfinite(value) and value > 0 for value in (median_limit, p95_limit)
    ):
        return False
    if (
        measured.get("validation_n", 0) < 6
        or measured.get("tre_median_um") is None
        or measured.get("tre_p95_um") is None
    ):
        return False
    return bool(
        measured["tre_median_um"] <= median_limit
        and measured["tre_p95_um"] <= p95_limit
    )


def publish_registered_marker_images(
    warped: pd.DataFrame,
    review_path: str | Path,
    landmarks_path: str | Path,
    *,
    table_dir: str | Path,
    allow_exploratory: bool = False,
) -> dict[str, Any]:
    """Apply saved fine transforms to held-out coordinates and publish only measured accepted pairs."""
    review = pd.read_csv(review_path, keep_default_na=False)
    landmarks = validate_landmarks(pd.read_csv(landmarks_path, keep_default_na=False))
    if review.duplicated(["assigned_tma", "confirmed_core_id"]).any():
        raise ValueError("Duplicate registration review keys")
    directory = Path(table_dir)
    directory.mkdir(parents=True, exist_ok=True)
    accepted_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, str]] = {}
    accepted_keys: list[list[str]] = []
    for (tma, core), marker_rows in warped.groupby(
        ["assigned_tma", "confirmed_core_id"]
    ):
        matches = review.loc[
            review["assigned_tma"].eq(tma) & review["confirmed_core_id"].eq(core)
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing unique review for {tma}/{core}")
        decision = matches.iloc[0].to_dict()
        first = marker_rows.iloc[0].to_dict()
        warp_receipt_path = Path(first["warp_receipt_path"])
        if compute_file_sha256(warp_receipt_path) != first["warp_receipt_sha256"]:
            raise ValueError("Fine registration receipt changed after warping")
        warp_receipt = _json(warp_receipt_path)
        if not _verify_artifacts(warp_receipt["artifacts"]):
            raise ValueError("Registered arrays or VALIS transform artifacts changed")
        saved_matrix = json.loads(
            warp_receipt["identity"]["orientation"]["matrix_yx_json"]
        )
        if not np.allclose(json.loads(first["coarse_matrix_yx_json"]), saved_matrix):
            raise ValueError("Coarse orientation changed after marker warping")
        bindings = {item["path"]: item["sha256"] for item in warp_receipt["artifacts"]}
        for marker in marker_rows.to_dict("records"):
            original = [
                item
                for item in warp_receipt["result"]["rows"]
                if item["marker"] == marker["marker"]
            ]
            if len(original) != 1:
                raise ValueError("Warp receipt does not bind this marker")
            for field in ("image_path", "support_path"):
                if marker[field] != original[0][field] or marker[field] not in bindings:
                    raise ValueError(
                        "Warped marker path differs from its computed result"
                    )
            for field in ("hs_features_path", "geometry_path"):
                if marker[field] not in bindings:
                    raise ValueError("Unbound HS feature or physical geometry artifact")
        points = landmarks.loc[
            landmarks["assigned_tma"].eq(tma)
            & landmarks["confirmed_core_id"].eq(core)
            & landmarks["role"].eq("validation")
        ]
        measured: dict[str, Any] = {
            "validation_n": 0,
            "tre_median_um": None,
            "tre_p95_um": None,
        }
        if not points.empty:
            result = warp_receipt["result"]
            geometry = _json(first["geometry_path"])
            job_path = warp_receipt_path.parent / "landmark_job.json"
            save_json(
                job_path,
                {
                    "operation": "points",
                    "moving_xy": points[["moving_x", "moving_y"]].to_numpy().tolist(),
                    "coarse_matrix_yx": json.loads(first["coarse_matrix_yx_json"]),
                    "registrar_path": result["registrar_path"],
                    "fixed_name": result["fixed_name"],
                    "moving_name": result["moving_name"],
                    "non_rigid": result["non_rigid"],
                    "anchor_shape": geometry["array_shape_yx"],
                    "result_path": str(
                        (job_path.parent / "landmark_result.json").resolve()
                    ),
                },
            )
            predicted = np.asarray(_run_isolated(job_path)["predicted_xy"])
            measured = landmark_tre(
                points[["fixed_x", "fixed_y"]].to_numpy(dtype=float),
                predicted,
                (float(first["pixel_size_x_um"]), float(first["pixel_size_y_um"])),
            )
        accepted = evaluate_registration_decision(decision, measured)
        known_section = decision.get("section_relationship") in {
            "same_section",
            "serial_section",
        } and bool(str(decision.get("section_evidence", "")).strip())
        accepted = accepted and known_section
        if allow_exploratory:
            accepted = (
                decision.get("acceptance_basis") == "agent_reviewed_exploratory_images"
                and decision.get("visual_overlay_review") == "PASS"
                and bool(str(decision.get("visual_review_notes", "")).strip())
                and decision.get("reviewed_warp_receipt_sha256")
                == first["warp_receipt_sha256"]
                and bool(str(decision.get("validation_scope", "")).strip())
            )
            if accepted:
                for prefix in ("reviewed_pdf", "group_annotation"):
                    if (
                        compute_file_sha256(decision[f"{prefix}_path"])
                        != decision[f"{prefix}_sha256"]
                    ):
                        raise ValueError(
                            f"Changed exploratory review evidence: {prefix}"
                        )
                    artifact = _binding(decision[f"{prefix}_path"])
                    artifacts[artifact["path"]] = artifact
        quality = warp_receipt["result"].get("fit_quality", {})
        decisions.append(
            {
                "assigned_tma": tma,
                "confirmed_core_id": core,
                **{k: v for k, v in measured.items() if k != "errors_um"},
                **quality,
                "registration_accepted": accepted,
                "acceptance_basis": decision.get(
                    "acceptance_basis", "heldout_landmarks"
                ),
                "review_notes": decision.get("visual_review_notes", ""),
            }
        )
        if not accepted:
            continue
        accepted_keys.append([str(tma), str(core)])
        for item in warp_receipt["artifacts"] + [
            _binding(warp_receipt_path),
            _binding(review_path),
            _binding(landmarks_path),
        ]:
            artifacts[item["path"]] = item
        for row in marker_rows.to_dict("records"):
            accepted_rows.append(
                {
                    **row,
                    "registration_accepted": True,
                    "section_relationship": decision["section_relationship"],
                    "section_evidence": decision["section_evidence"],
                    "group_id": decision.get("group_id", ""),
                    "group_evidence": decision.get("group_evidence", ""),
                    **quality,
                    "acceptance_basis": decision.get(
                        "acceptance_basis", "heldout_landmarks"
                    ),
                    "validation_scope": decision.get(
                        "validation_scope", "independent_heldout_landmarks"
                    ),
                    "visual_review_notes": decision.get("visual_review_notes", ""),
                    "registration_driver": strict_bool(row["registration_driver"]),
                }
            )
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(directory / "registration_acceptance.csv", index=False)
    if not accepted_rows:
        raise ValueError(
            "No pair passed the selected source-bound review route; inspect registration_acceptance.csv"
        )
    receipt_path = directory / "accepted_registration_receipt.json"
    save_json(
        receipt_path,
        {
            "schema": "clinical_hs_cycif_accepted_registration_v1",
            "accepted_core_keys": accepted_keys,
            "canonical_rows": accepted_rows,
            "artifacts": list(artifacts.values()),
            "code_bindings": _code_bindings(),
            "interpretation": "Exploratory regional correspondence; MI and correction magnitude are not independent landmark TRE. Serial sections are not identical cells.",
            "allow_exploratory": allow_exploratory,
        },
    )
    receipt_digest = compute_file_sha256(receipt_path)
    published = pd.DataFrame(
        [
            {
                **row,
                "registration_receipt_path": str(receipt_path.resolve()),
                "registration_receipt_sha256": receipt_digest,
            }
            for row in accepted_rows
        ]
    )
    output = directory / "registered_marker_images.csv"
    published.to_csv(output, index=False)
    return {
        "manifest": published,
        "decisions": decision_frame,
        "manifest_path": output,
        "receipt_path": receipt_path,
    }


def measure_registration_qc(
    warped: pd.DataFrame, *, table_dir: str | Path
) -> pd.DataFrame:
    """Describe tissue coverage and fitted-feature residuals without claiming TRE."""
    rows: list[dict[str, Any]] = []
    for row in warped.loc[warped["marker"].eq("DAPI")].to_dict("records"):
        receipt = _json(row["warp_receipt_path"])
        if compute_file_sha256(row["warp_receipt_path"]) != row["warp_receipt_sha256"]:
            raise ValueError("Changed warp receipt before structural QC")
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed = np.array(bundle["support"], dtype=bool)
        moving = np.asarray(_read_array(row["support_path"]), dtype=bool)
        if fixed.shape != moving.shape:
            raise ValueError("Support grids do not match")
        shared = fixed & moving
        denominator = int(fixed.sum()) + int(moving.sum())
        quality = receipt["result"]["fit_quality"]
        rows.append(
            {
                "assigned_tma": row["assigned_tma"],
                "confirmed_core_id": row["confirmed_core_id"],
                "fixed_tissue_pixels": int(fixed.sum()),
                "moving_tissue_pixels": int(moving.sum()),
                "shared_tissue_pixels": int(shared.sum()),
                "tissue_dice": 2 * int(shared.sum()) / denominator
                if denominator
                else 0,
                "fixed_tissue_coverage": float(shared.sum() / fixed.sum()),
                "pixel_size_x_um": row["pixel_size_x_um"],
                "pixel_size_y_um": row["pixel_size_y_um"],
                **quality,
                "uncertainty_interpretation": "MI metric and fitted correction are not independent error bounds",
                "warp_receipt_path": row["warp_receipt_path"],
                "warp_receipt_sha256": row["warp_receipt_sha256"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(table_dir) / "structural_registration_qc.csv", index=False)
    return frame


def render_exploratory_registration_qc(
    warped: pd.DataFrame, *, plot_dir: str | Path
) -> list[Path]:
    """Show both modalities, coarse/fine overlays, tissue support and fitted matches."""
    import matplotlib.pyplot as plt

    _style()
    directory = Path(plot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in warped.loc[warped["marker"].eq("DAPI")].to_dict("records"):
        stem = f"{row['assigned_tma']}_{row['confirmed_core_id']}_registration_qc"
        pdf_path = directory / f"{stem}.pdf"
        receipt_path = directory / f"{stem}.json"
        identity = {
            "warp_receipt_sha256": row["warp_receipt_sha256"],
            "code_sha256": compute_file_sha256(__file__),
            "display_max_edge": 900,
            "contrast_percentiles": [1, 99],
            "canvas_mm": [180, 138],
        }
        if receipt_path.is_file():
            saved = _json(receipt_path)
            if saved.get("identity") == identity and _verify_artifacts(
                saved["artifacts"]
            ):
                paths.append(pdf_path)
                continue
        receipt = _json(row["warp_receipt_path"])
        with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
            fixed = np.array(bundle["total"])
            fixed_support = np.array(bundle["support"], dtype=bool)
        moving = np.asarray(_read_array(row["image_path"]))
        moving_support = np.asarray(_read_array(row["support_path"]), dtype=bool)
        original = np.asarray(_read_array(row["reference_image_path"]))
        coarse, _ = warp_continuous_yx(
            original,
            _read_array(row["source_support_path"]),
            np.array(json.loads(row["coarse_matrix_yx_json"])),
            fixed.shape,
        )
        valid = fixed_support & moving_support
        if not valid.any():
            raise ValueError("No shared tissue to inspect")

        def display_scale(array: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
            values = array[mask & np.isfinite(array)]
            return tuple(float(value) for value in np.percentile(values, [1, 99]))

        def normalize(array: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
            low, high = limits
            if high <= low:
                return np.zeros_like(array, dtype=np.float32)
            return np.clip((array - low) / (high - low), 0, 1)

        fixed_limits = display_scale(fixed, fixed_support)
        moving_limits = display_scale(moving, moving_support)
        fixed_view = normalize(fixed, fixed_limits)
        moving_view = normalize(moving, moving_limits)
        coarse_view = normalize(coarse, moving_limits)
        coarse_rgb = np.stack((fixed_view, coarse_view, fixed_view), axis=-1)
        fine_rgb = np.stack((fixed_view, moving_view, fixed_view), axis=-1)
        support_rgb = np.stack(
            (fixed_support, moving_support, fixed_support), axis=-1
        ).astype(float)
        stride = max(1, int(math.ceil(max(fixed.shape) / 900)))
        fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 138 / 25.4))
        fig.subplots_adjust(
            left=0.025, right=0.975, bottom=0.13, top=0.90, hspace=0.20, wspace=0.08
        )
        yy, xx = np.indices(fixed.shape)
        tiles = (
            yy // max(1, fixed.shape[0] // 12) + xx // max(1, fixed.shape[1] // 12)
        ) % 2 == 0
        checker = np.where(tiles, fixed_view, moving_view)
        views = [fixed_view, moving_view, support_rgb, coarse_rgb, fine_rgb, checker]
        titles = [
            "HS total intensity",
            "Registered cycIF R1 DAPI",
            "Tissue: HS magenta / cycIF green",
            "Before rigid MI refinement",
            "After MI refinement / VALIS warping",
            "Checkerboard: internal correspondence",
        ]
        for axis, view, title in zip(axes.flat, views, titles):
            axis.imshow(
                view[::stride, ::stride],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_title(title, fontsize=5, fontweight="normal", pad=3)
            axis.set_axis_off()
        quality = receipt["result"]["fit_quality"]
        dice = 2 * valid.sum() / (fixed_support.sum() + moving_support.sum())
        fig.text(
            0.025,
            0.96,
            f"{row['assigned_tma']} {row['confirmed_core_id']} | Native HS anchor; bounded rigid MI / VALIS",
            fontsize=5,
        )
        fig.text(
            0.025,
            0.085,
            f"Tissue Dice {dice:.3f}; HS tissue covered {valid.sum() / fixed_support.sum():.1%}. "
            f"Negative MI {quality['mi_cost_before']:.4f} → {quality['mi_cost_after']:.4f}; "
            f"inverse rotation {quality['fine_inverse_angle_degrees']:.2f}°; scale fixed at 1.",
            fontsize=5,
        )
        fig.text(
            0.025,
            0.058,
            "MI improvement and correction magnitude are not independent TRE. Inspect internal structures, rim loss and local mismatches.",
            fontsize=5,
        )
        fig.text(
            0.025,
            0.031,
            f"Display stride {stride}; quantitative arrays unchanged. Shared per-modality contrast before/after. Generator: clinical_roi_valis.py",
            fontsize=5,
        )
        fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        save_json(
            receipt_path,
            {
                "identity": identity,
                "artifacts": [_binding(pdf_path)],
                "fixed_display_limits": fixed_limits,
                "moving_display_limits": moving_limits,
            },
        )
        paths.append(pdf_path)
    return paths


def record_exploratory_reviews(
    warped: pd.DataFrame,
    decisions: list[dict[str, str]],
    *,
    review_path: str | Path,
    plot_dir: str | Path,
    group_annotation_path: str | Path,
    section_relationship: str,
    section_evidence: str,
) -> pd.DataFrame:
    """Bind actual agent image-review decisions to the exact reviewed fits and PDFs."""
    annotation = _json(group_annotation_path)
    if (
        compute_file_sha256(annotation["outcome_map"])
        != annotation["outcome_map_sha256"]
    ):
        raise ValueError(
            "Patient-independence annotation does not match its source map"
        )
    if (
        annotation["analysis_unit"]
        != "one distinct patient per non-orientation TMA-qualified core"
    ):
        raise ValueError("Unsupported patient grouping annotation")
    annotation_binding = _binding(group_annotation_path)
    records: list[dict[str, Any]] = []
    for decision in decisions:
        tma, core = decision["assigned_tma"], decision["confirmed_core_id"]
        matches = warped.loc[
            warped["assigned_tma"].eq(tma) & warped["confirmed_core_id"].eq(core)
        ]
        if (
            matches.empty
            or decision["decision"] not in {"PASS", "FAIL"}
            or not decision["notes"].strip()
        ):
            raise ValueError(
                "Review requires a fitted core, explicit decision and actual image notes"
            )
        if matches["warp_receipt_sha256"].nunique() != 1:
            raise ValueError("Mixed transforms within a reviewed core")
        pdf = Path(plot_dir) / f"{tma}_{core}_registration_qc.pdf"
        records.append(
            {
                "assigned_tma": tma,
                "confirmed_core_id": core,
                "orientation_review": decision["decision"],
                "orientation_evidence": decision["notes"],
                "visual_overlay_review": decision["decision"],
                "visual_review_notes": decision["notes"],
                "acceptance_basis": "agent_reviewed_exploratory_images",
                "validation_scope": "structural_image_and_tissue_support_qc",
                "reviewed_warp_receipt_sha256": matches.iloc[0]["warp_receipt_sha256"],
                "reviewed_pdf_path": str(pdf.resolve()),
                "reviewed_pdf_sha256": compute_file_sha256(pdf),
                "section_relationship": section_relationship,
                "section_evidence": section_evidence,
                "group_id": f"{tma}_{core}",
                "group_evidence": json.dumps(annotation_binding, sort_keys=True),
                "group_annotation_path": annotation_binding["path"],
                "group_annotation_sha256": annotation_binding["sha256"],
                "max_median_tre_um": "",
                "max_p95_tre_um": "",
            }
        )
    frame = pd.DataFrame(records)
    if frame.duplicated(["assigned_tma", "confirmed_core_id"]).any():
        raise ValueError("Duplicate exploratory review decisions")
    frame.to_csv(review_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated structural VALIS worker, never a registration acceptance shortcut"
    )
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    job = _json(args.worker)
    result = _worker(job)
    save_json(job["result_path"], result)


if __name__ == "__main__":
    main()
