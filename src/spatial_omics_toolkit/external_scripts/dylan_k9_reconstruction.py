"""Source-bound reconstruction of Dylan's eight-image PCA5/final OpenCV K9.

The legacy default returns C2; standalone reproduction can return all eight
native YX rasters. R1--R9 are reconstruction identifiers,
not recovered historical cluster identities or histological annotations. Every
cropped pixel from all eight inputs, including masked zero background, is fitted.
No outcome data, cross-modality registration, sampling, or raw-10-feature K9 is
used. The collaborator script and TIFFs are read-only inputs.

Runtime dependencies: numpy, pandas, scipy, scikit-learn, tifffile, OpenCV and
threadpoolctl. Request 32 GiB RAM and at least 6 GiB free temporary disk for the
full fit across version-dependent auto solvers; this is a conservative resource
request, not a measured peak. Scratch arrays beside the cache are removed on exit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import tifffile
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_info, threadpool_limits

CORES: tuple[str, ...] = ("A2", "B1", "B2", "C2", "A1", "B3", "C1", "C3")
SOURCE_RELATIVE = Path(
    "02_data_sources/04_collaborators/from_dylan/20260903_clustering_files"
)
OUTPUT_RELATIVE = Path(
    "03_analysis/01_exploratory_analysis/05_objects/03_for_dylan_paper/"
    "01_omni_cycif_one_core_exploratory/dylan_k9_reconstruction"
)
CROP_SIZE = 2000
PIXELS_PER_CORE = CROP_SIZE * CROP_SIZE
FEATURE_COUNT = 10
COMPONENT_COUNT = 5
CLUSTER_COUNT = 9
MASK_THRESHOLD = 40000
SCHEMA_VERSION = 2
ARTIFACTS: dict[str, str] = {
    "labels": "C2_labels_yx.npy",
    "support": "C2_support_yx.npy",
    "intensity": "C2_summed_mask_yx.npy",
    "model": "pca5_k9_model.npz",
    "state_table": "C2_state_occupancy.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_inventory(project: Path) -> pd.DataFrame:
    """List the exact eight original stacks and masks with absolute paths/hashes.

    ``project`` is the spatial-proteomics project root. No recursive search or
    selection of alternate PCA/preprocessed TIFFs is performed at runtime.
    """
    source_root = project.resolve() / SOURCE_RELATIVE
    original_source = source_root / "OpenCV_multiimage_8input.py"
    original_hash = _sha256(original_source)
    rows: list[dict[str, object]] = []
    for image_index, core in enumerate(CORES):
        for role, filename in (
            ("feature_stack", f"{core}_all2048.tif"),
            ("summed_mask_source", f"{core}Image2048.tif"),
        ):
            path = (source_root / "Original" / filename).resolve(strict=True)
            rows.append(
                {
                    "image_index": image_index,
                    "core": core,
                    "role": role,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "original_source": str(original_source),
                    "original_source_sha256": original_hash,
                }
            )
    return pd.DataFrame.from_records(rows)


def _pooled_feature_dtype(inventory: pd.DataFrame) -> np.dtype[np.generic]:
    """Match NumPy's pooled dtype after each original per-image z-score."""
    normalized_dtypes: list[np.dtype[np.generic]] = []
    for row in inventory.to_dict(orient="records"):
        if row["role"] != "feature_stack":
            continue
        with tifffile.TiffFile(Path(str(row["path"]))) as source:
            source_dtype = np.dtype(source.series[0].dtype)
        # NumPy mean/std and true division promote integer inputs to float64;
        # floating inputs retain their source precision during normalization.
        normalized_dtypes.append(
            np.dtype(np.float64) if source_dtype.kind in "biu" else source_dtype
        )
    return np.result_type(*normalized_dtypes)


def _runtime_versions() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv": cv2.__version__,
        "opencv_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "numeric_libraries": threadpool_info(),
    }
    for package in (
        "numpy", "pandas", "scipy", "scikit-learn", "tifffile", "threadpoolctl"
    ):
        versions[package] = importlib.metadata.version(package)
    return versions


def _bounds(height: int, width: int) -> tuple[int, int, int, int]:
    if min(height, width) < CROP_SIZE:
        raise ValueError(f"Expected at least {CROP_SIZE} pixels; got {(height, width)}")
    y0 = height // 2 - CROP_SIZE // 2
    x0 = width // 2 - CROP_SIZE // 2
    return y0, y0 + CROP_SIZE, x0, x0 + CROP_SIZE


def _preprocess(
    stack_path: Path, mask_path: Path, core: str
) -> tuple[NDArray[np.generic], NDArray[np.generic], dict[str, object]]:
    stack = tifffile.imread(stack_path)
    if stack.ndim != 3 or stack.shape[0] != FEATURE_COUNT:
        raise ValueError(f"{core}: expected CYX with ten channels, got {stack.shape}")
    shape_cyx = list(stack.shape)
    source_dtype = str(stack.dtype)
    clipped_counts: dict[str, int] = {}
    for channel in (3, 4):
        invalid = (stack[channel] < 0) | (stack[channel] > 10)
        clipped_counts[str(channel)] = int(np.count_nonzero(invalid))
        stack[channel][invalid] = 0
    # Deliberately reverse all axes, as np.transpose(img) in the original does.
    xyc = np.transpose(stack)
    x0, x1, y0, y1 = _bounds(xyc.shape[0], xyc.shape[1])
    flat = xyc[x0:x1, y0:y1].reshape(-1, FEATURE_COUNT)
    nonfinite_counts = np.count_nonzero(~np.isfinite(flat), axis=0).tolist()
    flat = np.nan_to_num(flat)
    # Preserve source-array NumPy reduction/arithmetic dtype before pooled PCA.
    means = flat.mean(axis=0, keepdims=True)
    standard_deviations = flat.std(axis=0, keepdims=True)
    if not np.all(np.isfinite(standard_deviations)) or np.any(standard_deviations == 0):
        raise ValueError(
            f"{core}: undefined per-feature z-score; no imputation substituted"
        )
    flat = (flat - means) / standard_deviations
    del stack, xyc

    mask_stack = tifffile.imread(mask_path)
    if mask_stack.ndim != 3 or tuple(mask_stack.shape[1:]) != tuple(shape_cyx[1:]):
        raise ValueError(
            f"{core}: mask CYX shape {mask_stack.shape} does not match stack"
        )
    mask_shape = list(mask_stack.shape)
    summed = np.sum(mask_stack, axis=0)
    del mask_stack
    my0, my1, mx0, mx1 = _bounds(summed.shape[0], summed.shape[1])
    intensity = np.ascontiguousarray(summed[my0:my1, mx0:mx1])
    # '< threshold', rather than a cluster ID, defines the separate support mask.
    excluded_xy = (intensity.T < MASK_THRESHOLD).flatten()
    flat[excluded_xy, :] = 0
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{core}: nonfinite standardized features cannot enter PCA")
    summary: dict[str, object] = {
        "core": core,
        "stack_shape_cyx": shape_cyx,
        "mask_shape_cyx": mask_shape,
        "source_feature_dtype": source_dtype,
        "standardized_feature_dtype": str(flat.dtype),
        "lifetime_clipped_full_source_counts": clipped_counts,
        "nonfinite_cropped_feature_counts_before_nan_to_num": nonfinite_counts,
        "per_feature_mean_before_mask": means.ravel().tolist(),
        "per_feature_std_ddof0_before_mask": standard_deviations.ravel().tolist(),
        "crop_bounds_native_yx": [y0, y1, x0, x1],
        "mask_crop_bounds_native_yx": [my0, my1, mx0, mx1],
        "crop_origin_native_yx": [y0, x0],
        "supported_pixels": int(np.count_nonzero(~excluded_xy)),
        "masked_zero_pixels": int(np.count_nonzero(excluded_xy)),
    }
    return flat, intensity, summary


def _state_table(
    labels: NDArray[np.uint8], support: NDArray[np.bool_], core: str = "C2",
) -> pd.DataFrame:
    all_counts = np.bincount(labels.ravel(), minlength=CLUSTER_COUNT)
    supported_counts = np.bincount(labels[support], minlength=CLUSTER_COUNT)
    support_total = int(support.sum())
    return pd.DataFrame(
        {
            "core": [core] * CLUSTER_COUNT,
            "original_k9_label": np.arange(CLUSTER_COUNT),
            "reconstruction_id": [f"R{label + 1}" for label in range(CLUSTER_COUNT)],
            "all_crop_pixels": all_counts,
            "supported_pixels": supported_counts,
            "outside_support_pixels": all_counts - supported_counts,
            "fraction_all_crop": all_counts / labels.size,
            "fraction_supported_pixels": (
                supported_counts / support_total
                if support_total else np.full(CLUSTER_COUNT, np.nan)
            ),
            "support_fraction_within_state": np.divide(
                supported_counts,
                all_counts,
                out=np.full(CLUSTER_COUNT, np.nan),
                where=all_counts != 0,
            ),
        }
    )


def _core_artifacts(core: str) -> dict[str, str]:
    """Name native YX output arrays without assigning historical cluster IDs."""
    return {
        "labels": f"{core}_labels_yx.npy",
        "support": f"{core}_support_yx.npy",
        "intensity": f"{core}_summed_mask_yx.npy",
        "state_table": f"{core}_state_occupancy.csv",
    }


def _artifact_names(returned_cores: tuple[str, ...]) -> dict[str, str]:
    """Retain legacy C2 aliases and add requested maps from the same shared fit."""
    names = dict(ARTIFACTS)
    for core in returned_cores:
        for kind, filename in _core_artifacts(core).items():
            if core != "C2":
                names[f"{core}_{kind}"] = filename
    return names


def _result(cache_dir: Path, receipt: dict[str, object]) -> dict[str, object]:
    parameters = cast(dict[str, Any], cast(dict[str, Any], receipt["identity"])["parameters"])
    returned_cores = tuple(parameters.get("returned_cores", ["C2"]))
    paths = {name: cache_dir / filename for name, filename in _artifact_names(returned_cores).items()}
    per_core: dict[str, dict[str, Any]] = {}
    for core in returned_cores:
        core_paths = {kind: cache_dir / filename for kind, filename in _core_artifacts(core).items()}
        per_core[core] = {
            kind: np.load(core_paths[kind], mmap_mode="r", allow_pickle=False)
            for kind in ("labels", "support", "intensity")
        }
        per_core[core]["paths"] = core_paths
        per_core[core]["state_table"] = pd.read_csv(core_paths["state_table"])
    paths["receipt"] = cache_dir / "receipt.json"
    return {
        "labels": np.load(paths["labels"], mmap_mode="r", allow_pickle=False),
        "support": np.load(paths["support"], mmap_mode="r", allow_pickle=False),
        "intensity": np.load(paths["intensity"], mmap_mode="r", allow_pickle=False),
        "receipt": receipt,
        "state_table": pd.read_csv(paths["state_table"]),
        "paths": paths,
        "per_core": per_core,
    }



def _fit_code_sha256() -> str:
    """Bind computation and I/O dependencies while allowing independent display edits."""
    import inspect

    functions = (
        _sha256, _canonical_json, source_inventory, _pooled_feature_dtype, _runtime_versions,
        _bounds, _preprocess, _state_table, _core_artifacts, _artifact_names,
        _result, _fit_code_sha256, reconstruct,
    )
    constants: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "cores": CORES,
        "source_relative": str(SOURCE_RELATIVE), "output_relative": str(OUTPUT_RELATIVE),
        "crop_size": CROP_SIZE, "pixels_per_core": PIXELS_PER_CORE,
        "feature_count": FEATURE_COUNT, "component_count": COMPONENT_COUNT,
        "cluster_count": CLUSTER_COUNT, "mask_threshold": MASK_THRESHOLD,
        "artifacts": ARTIFACTS,
    }
    payload = _canonical_json(constants) + "\n" + "\n".join(
        inspect.getsource(function) for function in functions
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reconstruct(
    project: Path, seed: int = 20261022, force: bool = False,
    *, returned_cores: tuple[str, ...] = ("C2",), output_root: Path | None = None,
) -> dict[str, object]:
    """Load or fit shared PCA5/K9; preserve C2 aliases and optionally export all cores.

    Cache identity binds all source hashes, original/computation hashes, parameters,
    Python/package/native-library versions, and OpenCV build. Cache artifacts have
    independent SHA-256 checksums. ``force=True`` repeats the deterministic fit.
    PCA retains the original pooled feature dtype and default auto solver policy,
    with an explicit reconstructed random_state. The installed scikit-learn
    version determines auto's chosen solver. BLAS/OpenMP and OpenCV fit threads
    are limited to one; OpenCV's global RNG is explicitly seeded. Historical
    stochastic labels cannot be recovered exactly.
    """
    project = project.resolve()
    if (
        not returned_cores or len(set(returned_cores)) != len(returned_cores)
        or not set(returned_cores).issubset(CORES) or "C2" not in returned_cores
    ):
        raise ValueError("returned_cores must be unique source cores and include C2")
    artifact_names = _artifact_names(returned_cores)
    inventory = source_inventory(project)
    pooled_dtype = _pooled_feature_dtype(inventory)
    effective_seed = ((int(seed) + 2**31) % 2**32) - 2**31
    # Capture the numerical runtime under the same thread limit used for fitting.
    with threadpool_limits(limits=1):
        versions = _runtime_versions()
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "fit_code_sha256": _fit_code_sha256(),
        "sources": inventory.to_dict(orient="records"),
        "versions": versions,
        "parameters": {
            "input_order": list(CORES),
            "returned_core": "C2",
            "returned_cores": list(returned_cores),
            "crop_size": CROP_SIZE,
            "feature_count": FEATURE_COUNT,
            "lifetime_channel_indices": [3, 4],
            "lifetime_valid_range_closed": [0, 10],
            "mask_threshold": MASK_THRESHOLD,
            "zscore_scope": "per-image, per-feature, all cropped pixels before masking",
            "zscore_ddof": 0,
            "nan_to_num": "NumPy defaults in source dtype before z-scoring",
            "pca_components": COMPONENT_COUNT,
            "pca_solver": "auto",
            "pca_random_state": int(seed),
            "pca_dtype": str(pooled_dtype),
            "pca_dtype_policy": (
                "NumPy result_type of original per-image z-scored arrays"
            ),
            "pca_copy": True,
            "pca_whiten": False,
            "fit_population": (
                "all eight crops including zero-masked pixels; no sampling"
            ),
            "kmeans_dtype": "float32",
            "kmeans_k": CLUSTER_COUNT,
            "kmeans_criteria": "EPS + MAX_ITER",
            "kmeans_max_iter": 5,
            "kmeans_epsilon": 0.1,
            "kmeans_attempts": 5,
            "kmeans_initialization": "KMEANS_PP_CENTERS",
            "requested_seed": int(seed),
            "opencv_signed_int32_seed": effective_seed,
            "numeric_threads": 1,
            "opencv_threads": 1,
        },
    }
    cache_id = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    cache_dir = (project / OUTPUT_RELATIVE if output_root is None else output_root.resolve()) / cache_id
    receipt_path = cache_dir / "receipt.json"
    if not force and receipt_path.is_file():
        receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
        saved_hashes = cast(dict[str, str], receipt.get("artifact_sha256", {}))
        if (
            receipt.get("cache_id") == cache_id
            and receipt.get("identity") == identity
            and all(
                (cache_dir / filename).is_file()
                and saved_hashes.get(name) == _sha256(cache_dir / filename)
                for name, filename in artifact_names.items()
            )
        ):
            print(f"LOADED SAVED RESULT: Dylan PCA5/K9 {list(returned_cores)} ({cache_id})")
            return _result(cache_dir, receipt)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Removing the completion marker prevents partially overwritten forced fits
    # from being accepted. The receipt is written last, only after all artifacts.
    receipt_path.unlink(missing_ok=True)
    source_paths = {
        (str(row["core"]), str(row["role"])): Path(str(row["path"]))
        for row in inventory.to_dict(orient="records")
    }
    summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="scratch_", dir=cache_dir) as temporary:
        scratch = Path(temporary)
        features = np.lib.format.open_memmap(
            scratch / "all_features_xy.npy", mode="w+", dtype=pooled_dtype,
            shape=(len(CORES) * PIXELS_PER_CORE, FEATURE_COUNT),
        )
        for image_index, core in enumerate(CORES):
            flat, intensity, summary = _preprocess(
                source_paths[(core, "feature_stack")],
                source_paths[(core, "summed_mask_source")], core,
            )
            start = image_index * PIXELS_PER_CORE
            features[start:start + PIXELS_PER_CORE] = flat
            summaries.append(summary)
            if core in returned_cores:
                core_names = _core_artifacts(core)
                np.save(cache_dir / core_names["intensity"], intensity, allow_pickle=False)
                np.save(
                    cache_dir / core_names["support"],
                    ~(intensity < MASK_THRESHOLD), allow_pickle=False,
                )
            del flat, intensity
        features.flush()
        scores = np.lib.format.open_memmap(
            scratch / "all_pc_scores_yx.npy", mode="w+", dtype=np.float32,
            shape=(len(CORES) * PIXELS_PER_CORE, COMPONENT_COUNT),
        )
        previous_cv_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(1)
            with threadpool_limits(limits=1):
                pca = PCA(
                    n_components=COMPONENT_COUNT, svd_solver="auto",
                    random_state=seed,
                )
                pca.fit(features)
                # Keep the original pooled features intact for pca.transform;
                # auto may select different solver implementations by version.
                # Recreate original PC XY reshape -> transpose to YXC -> flatten.
                for image_index in range(len(CORES)):
                    start = image_index * PIXELS_PER_CORE
                    feature_xyc = features[start:start + PIXELS_PER_CORE].reshape(
                        CROP_SIZE, CROP_SIZE, FEATURE_COUNT
                    )
                    native_yxc = scores[start:start + PIXELS_PER_CORE].reshape(
                        CROP_SIZE, CROP_SIZE, COMPONENT_COUNT
                    )
                    for x0 in range(0, CROP_SIZE, 64):
                        x1 = min(x0 + 64, CROP_SIZE)
                        projected = pca.transform(
                            feature_xyc[x0:x1].reshape(-1, FEATURE_COUNT)
                        )
                        native_yxc[:, x0:x1, :] = projected.reshape(
                            x1 - x0, CROP_SIZE, COMPONENT_COUNT
                        ).transpose(1, 0, 2)
                    del feature_xyc, native_yxc, projected
                scores.flush()
                del features
                cv2.setRNGSeed(effective_seed)
                compactness, fit_labels, centers = cv2.kmeans(
                    scores, CLUSTER_COUNT, None,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.1),
                    5, cv2.KMEANS_PP_CENTERS,
                )
                del scores
        finally:
            cv2.setNumThreads(previous_cv_threads)
        global_counts = np.bincount(fit_labels.ravel(), minlength=CLUSTER_COUNT)
        for core in returned_cores:
            start = CORES.index(core) * PIXELS_PER_CORE
            labels = fit_labels[start:start + PIXELS_PER_CORE].reshape(
                CROP_SIZE, CROP_SIZE
            ).astype(np.uint8)
            core_names = _core_artifacts(core)
            np.save(cache_dir / core_names["labels"], labels, allow_pickle=False)
            support = np.load(cache_dir / core_names["support"], allow_pickle=False)
            _state_table(labels, support, core=core).to_csv(cache_dir / core_names["state_table"], index=False)
        del fit_labels
        np.savez(
            cache_dir / ARTIFACTS["model"],
            pca_mean=pca.mean_, pca_components=pca.components_,
            pca_loadings=pca.components_.T * np.sqrt(pca.explained_variance_),
            pca_explained_variance=pca.explained_variance_,
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
            pca_singular_values=pca.singular_values_,
            pca_noise_variance=np.asarray(pca.noise_variance_),
            pca_n_samples=np.asarray(pca.n_samples_),
            kmeans_centers=centers,
            kmeans_compactness=np.asarray(compactness),
            all_fit_cluster_counts=global_counts,
        )
    if not source_inventory(project).equals(inventory):
        raise ValueError("Delivered source content changed during the reconstruction")
    receipt = {
        "cache_id": cache_id,
        "identity": identity,
        "generating_helper_path": str(Path(__file__).resolve()),
        "generating_helper_full_sha256": _sha256(Path(__file__).resolve()),
        "preprocessing_by_core": summaries,
        "pca_resolved_fit_solver": getattr(pca, "_fit_svd_solver", None),
        "pca_fitted_components_dtype": str(pca.components_.dtype),
        "axis_provenance": {
            "input": "TIFF CYX",
            "normalization_and_pca": (
                "np.transpose(CYX) -> XYC; crop; C-order flatten XY"
            ),
            "final_kmeans": (
                "PC XY reshape -> transpose YXC -> C-order flatten YX"
            ),
            "returned_rasters": (
                "native cropped YX, not registered to any other modality"
            ),
        },
        "label_semantics": (
            "raw fitted labels 0..8 retained; R1..R9 = label+1; "
            "no cluster removed"
        ),
        "historical_identity_warning": (
            "Reconstructed labels are not authenticated historical cluster identities. "
            "The original PCA solver/version, seed and fit state were not recorded."
        ),
        "implementation_deviations": [
            (
                "Explicit reconstructed PCA random_state replaces the original "
                "unrecorded RNG state; original default auto solver policy and "
                "pooled source dtype are retained."
            ),
            (
                "Seeded OpenCV RNG and single-thread OpenCV/BLAS replace "
                "unrecorded stochastic/runtime state."
            ),
            (
                "Disk-backed arrays and bounded pca.transform calls replace "
                "redundant full-array copies; no pixels sampled."
            ),
            (
                "Plotting, metrics-loop epsilon 0.01 and subsequent unrelated "
                "exploratory fits are not executed."
            ),
        ],
        "kmeans_compactness": float(compactness),
        "all_fit_cluster_counts": global_counts.tolist(),
        "artifact_sha256": {
            name: _sha256(cache_dir / filename) for name, filename in artifact_names.items()
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"RECOMPUTED: Dylan PCA5/K9 {list(returned_cores)} ({cache_id})")
    return _result(cache_dir, receipt)



def feature_schema(project: Path) -> pd.DataFrame:
    """Read the received producer's stack declaration without executing its code."""
    import ast

    producer = project.resolve() / SOURCE_RELATIVE / "processing_source/PhasorPy_dh_3_received_2026-09-10.py"
    tree = ast.parse(producer.read_text(encoding="utf-8-sig"))
    matches: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "output_data1" for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "stack" and value.args
            and isinstance(value.args[0], (ast.List, ast.Tuple))
        ):
            matches.append([cast(ast.Name, item).id for item in value.args[0].elts])
    if len(matches) != 1 or len(matches[0]) != FEATURE_COUNT:
        raise ValueError("Received producer must declare one ten-feature output_data1 stack")
    return pd.DataFrame({
        "feature_index": np.arange(FEATURE_COUNT),
        "producer_variable": matches[0],
        "modality": ["FLIM"] * 5 + ["hyperspectral"] * 5,
        "producer_path": str(producer),
        "producer_sha256": _sha256(producer),
        "evidence_boundary": "Code-declared feature order; raw-to-delivered equivalence audited separately",
    })


def delivered_input_metadata(project: Path) -> pd.DataFrame:
    """Audit exact delivered stack metadata and hashes, retaining all source rows."""
    records: list[dict[str, Any]] = []
    for row in source_inventory(project).to_dict(orient="records"):
        path = Path(str(row["path"]))
        with tifffile.TiffFile(path) as source:
            series = source.series[0]
            shape, axes, dtype = tuple(series.shape), series.axes, str(series.dtype)
        if len(shape) != 3 or min(shape[1:]) < CROP_SIZE:
            raise ValueError(f"Unexpected delivered shape: {path}: {shape}")
        if row["role"] == "feature_stack" and shape[0] != FEATURE_COUNT:
            raise ValueError(f"Expected ten features: {path}: {shape}")
        records.append(dict(row, shape_cyx=json.dumps(shape), stored_axes=axes,
                            source_dtype=dtype, size_bytes=path.stat().st_size))
    return pd.DataFrame(records)


def reproduction_tables(
    result: dict[str, Any], project: Path, table_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Expose each recorded preprocessing/PCA/K9 stage from the single shared fit."""
    table_dir.mkdir(parents=True, exist_ok=True)
    schema = feature_schema(project)
    receipt = result["receipt"]
    preprocessing: list[dict[str, Any]] = []
    for item in receipt["preprocessing_by_core"]:
        for index, variable in enumerate(schema["producer_variable"]):
            preprocessing.append({
                "core": item["core"], "feature_index": index,
                "producer_variable": variable,
                "mean_before_mask": item["per_feature_mean_before_mask"][index],
                "std_ddof0_before_mask": item["per_feature_std_ddof0_before_mask"][index],
                "nonfinite_before_nan_to_num": item["nonfinite_cropped_feature_counts_before_nan_to_num"][index],
                "supported_pixels": item["supported_pixels"],
                "masked_zero_pixels": item["masked_zero_pixels"],
                "source_dtype": item["source_feature_dtype"],
            })
    with np.load(result["paths"]["model"], allow_pickle=False) as model:
        variance = pd.DataFrame({
            "component": np.arange(1, len(model["pca_explained_variance_ratio"]) + 1),
            "explained_variance": model["pca_explained_variance"],
            "explained_variance_ratio": model["pca_explained_variance_ratio"],
            "cumulative_variance_ratio": np.cumsum(model["pca_explained_variance_ratio"]),
        })
        loadings = pd.DataFrame(model["pca_components"].T,
                                columns=[f"PC{i + 1}" for i in range(COMPONENT_COUNT)])
        loadings.insert(0, "producer_variable", schema["producer_variable"])
        centers = pd.DataFrame(model["kmeans_centers"],
                               columns=[f"PC{i + 1}" for i in range(COMPONENT_COUNT)])
        centers.insert(0, "reconstruction_id", [f"R{i + 1}" for i in range(CLUSTER_COUNT)])
        centers["all_fit_pixels"] = model["all_fit_cluster_counts"]
    occupancy = pd.concat([
        item["state_table"].assign(core=core)
        for core, item in result["per_core"].items()
    ], ignore_index=True)
    tables = {
        "feature_schema": schema, "preprocessing": pd.DataFrame(preprocessing),
        "pca_variance": variance, "pca_components": loadings,
        "k9_centers": centers, "occupancy": occupancy,
    }
    for name, table in tables.items():
        table.to_csv(table_dir / f"dylan_hs_flim_{name}.csv", index=False)
    return tables


def _historical_display(
    labels_yx: NDArray[np.uint8], intensity_yx: NDArray[np.generic],
    *, source_plot_transpose: bool = False, palette_hex: tuple[str, ...] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], dict[str, float]]:
    """Apply the source alpha overlay on native YX; optionally replay its transpose."""
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if labels_yx.shape != intensity_yx.shape or not np.isfinite(intensity_yx).all():
        raise ValueError("Labels and finite intensity must share native YX geometry")
    source_map = mpl.colormaps["nipy_spectral"]
    custom = LinearSegmentedColormap.from_list(
        "dylan_source_spectral", source_map(np.linspace(0.03, 0.97, 256)),
    )
    # Shared 1..9 normalization preserves the same fitted cluster color per core.
    if palette_hex is None:
        flat = custom(labels_yx.astype(np.float32) / (CLUSTER_COUNT - 1))[..., :3]
    else:
        if len(palette_hex) != CLUSTER_COUNT:
            raise ValueError("Display palette must contain one color per fitted cluster")
        palette_rgb = np.asarray([mpl.colors.to_rgb(color) for color in palette_hex], dtype=np.float32)
        flat = palette_rgb[labels_yx]
    low, high = np.percentile(intensity_yx, [1, 96])
    clipped = np.clip(intensity_yx.astype(np.float64), low, high)
    if high <= low or high <= 0:
        raise ValueError("Historical overlay has undefined intensity range")
    powered = clipped ** 0.7
    alpha = 1.0 - powered / powered.max()
    if alpha.max() <= 0:
        raise ValueError("Historical overlay has undefined opacity range")
    alpha /= alpha.max()
    grayscale = mpl.colormaps["Greys_r"]((clipped - low) / (high - low))[..., :3]
    textured = alpha[..., None] * grayscale + (1.0 - alpha[..., None]) * flat
    metrics = {"intensity_p01": float(low), "intensity_p96": float(high),
               "gamma": 0.7, "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max())}
    if source_plot_transpose:
        flat, grayscale, textured = (array.transpose(1, 0, 2) for array in (flat, grayscale, textured))
    return (flat.astype(np.float32), grayscale.astype(np.float32),
            textured.astype(np.float32), metrics)


def render_reproduction(
    result: dict[str, Any], tables: dict[str, pd.DataFrame], plot_dir: Path,
    *, force: bool = False,
) -> dict[str, Any]:
    """Export source-style all-input/focal galleries and shared PCA diagnostics."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import LinearSegmentedColormap
    from clinical_roi_hyperspectral import _register_project_arial, _pseudocolor_style

    plot_dir.mkdir(parents=True, exist_ok=True)
    _register_project_arial()
    outputs = {
        "all_inputs": plot_dir / "dylan_hs_flim_all_eight_textured_maps.pdf",
        "focal_comparison": plot_dir / "dylan_hs_flim_C2_A2_B2_intensity_flat_textured.pdf",
        "pca": plot_dir / "dylan_hs_flim_shared_pca_diagnostics.pdf",
        "focal_muted": plot_dir / "dylan_hs_flim_C2_A2_B2_muted_textured_maps.pdf",
    }
    muted_palette = (
        "#4E79A7", "#76B7B2", "#59A14F", "#9C755F", "#F28E2B",
        "#D783A5", "#B07AA1", "#BAB0AC", "#E4B85B",
    )
    receipt_path = plot_dir / "dylan_hs_flim_display_receipt.json"
    identity = {
        "schema": "dylan_hs_flim_display_v1", "fit_cache_id": result["receipt"]["cache_id"],
        "fit_receipt_sha256": _sha256(result["paths"]["receipt"]),
        "helper_sha256": _sha256(Path(__file__).resolve()),
        "matplotlib": mpl.__version__,
        "style": {key: repr(value) for key, value in _pseudocolor_style().items()},
        "display": {
            "percentiles": [1, 96], "gamma": 0.7,
            "recipe": "Greys_r clipped HS sum over shared nipy_spectral colors with source alpha formula",
            "orientation": "native label YX and intensity YX, matching supplied panel G morphology; source plotting transpose not applied",
            "normalization": "shared raw label 0..8 / display R1..R9 across all cores",
            "modification": "No clustering, masking, smoothing or label changes during rendering",
            "additional_focal_palette": {f"R{index + 1}": color for index, color in enumerate(muted_palette)},
            "additional_focal_palette_semantics": "Fixed muted display palette; not a mapping to authenticated historical labels",
            "additional_focal_order": ["C2", "A2", "B2"],
        },
    }
    if not force and receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved.get("identity") == identity and all(
            path.is_file() and saved.get("artifact_sha256", {}).get(name) == _sha256(path)
            for name, path in outputs.items()
        ):
            print("LOADED SAVED RESULT: combined HS + FLIM figures")
            return {"pdfs": outputs, "receipt": saved, "receipt_path": receipt_path}
    per_core = result["per_core"]
    if tuple(per_core) != CORES:
        raise ValueError("All-eight figure requires all source cores in the recorded source order")
    metrics: list[dict[str, Any]] = []
    source_map = mpl.colormaps["nipy_spectral"]
    custom = LinearSegmentedColormap.from_list(
        "dylan_source_spectral", source_map(np.linspace(0.03, 0.97, 256)),
    )
    legend = [Patch(color=custom(index / (CLUSTER_COUNT - 1)), linewidth=0.5, label=f"R{index + 1}")
              for index in range(CLUSTER_COUNT)]
    with mpl.rc_context(_pseudocolor_style()):
        fig, axes = plt.subplots(2, 4, figsize=(240 / 25.4, 135 / 25.4))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.105, top=0.97, hspace=0.09, wspace=0.04)
        for ax, core in zip(axes.ravel(), CORES):
            data = per_core[core]
            before = hashlib.sha256(np.ascontiguousarray(data["labels"]).tobytes()).hexdigest()
            flat, gray, textured, summary = _historical_display(data["labels"], data["intensity"])
            ax.imshow(textured, interpolation="none")
            ax.set_axis_off()
            bounds = ax.get_position()
            fig.text(bounds.x0, bounds.y1 + 0.005, core, fontsize=5)
            after = hashlib.sha256(np.ascontiguousarray(data["labels"]).tobytes()).hexdigest()
            if before != after:
                raise AssertionError("Display changed a fitted label map")
            metrics.append(dict(core=core, **summary, label_sha256_before=before,
                                label_sha256_after=after, labels_unchanged=True))
            del flat, gray, textured
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=CLUSTER_COUNT, frameon=False)
        fig.text(0.02, 0.025, "Shared PCA5 / K9; R identifiers belong to this reconstruction. Per-core brightness stretch; native YX orientation.", fontsize=5)
        fig.text(0.02, 0.01, f"Generated by {Path(__file__).name}; delivered-feature reproduction; exploratory, no patient-level inference.", fontsize=5)
        fig.savefig(outputs["all_inputs"], format="pdf")
        plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(190 / 25.4, 205 / 25.4))
        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.075, top=0.96, hspace=0.07, wspace=0.04)
        for row, core in enumerate(("C2", "A2", "B2")):
            flat, gray, textured, _ = _historical_display(per_core[core]["labels"], per_core[core]["intensity"])
            for column, (label, rgb) in enumerate(zip(("Summed HS", "Shared K9 labels", "Source grayscale overlay"), (gray, flat, textured))):
                ax = axes[row, column]
                ax.imshow(rgb, interpolation="none")
                ax.set_axis_off()
                bounds = ax.get_position()
                fig.text(bounds.x0, bounds.y1 + 0.004, f"{core} | {label}", fontsize=5)
            del flat, gray, textured
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=CLUSTER_COUNT, frameon=False)
        fig.text(0.04, 0.025, "Flat and textured views contain identical labels. Brightness is display-scaled within each core; labels include masked-zero fit pixels.", fontsize=5)
        fig.text(0.04, 0.012, f"Generated by {Path(__file__).name}; source-prepared HS + FLIM; original historical IDs are unavailable.", fontsize=5)
        fig.savefig(outputs["focal_comparison"], format="pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 75 / 25.4))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.17, top=0.96, wspace=0.035)
        for ax, core in zip(axes, ("C2", "A2", "B2")):
            data = per_core[core]
            before = hashlib.sha256(np.ascontiguousarray(data["labels"]).tobytes()).hexdigest()
            flat, gray, textured, _ = _historical_display(
                data["labels"], data["intensity"], palette_hex=muted_palette,
            )
            ax.imshow(textured, interpolation="none")
            ax.set_axis_off()
            bounds = ax.get_position()
            fig.text(bounds.x0, bounds.y1 + 0.009, core, fontsize=5)
            after = hashlib.sha256(np.ascontiguousarray(data["labels"]).tobytes()).hexdigest()
            if before != after:
                raise AssertionError("Muted display changed a fitted label map")
            for summary in metrics:
                if summary["core"] == core:
                    summary["muted_labels_unchanged"] = True
            del flat, gray, textured
        muted_legend = [Patch(color=color, linewidth=0.5, label=f"R{index + 1}")
                        for index, color in enumerate(muted_palette)]
        fig.legend(handles=muted_legend, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=CLUSTER_COUNT, frameon=False)
        fig.text(0.02, 0.048, "Same shared K9 labels and source intensity overlay; fixed muted colors. R IDs belong to this reconstruction.", fontsize=5)
        fig.text(0.02, 0.022, f"Generated by {Path(__file__).name}; native YX; per-core brightness stretch; no historical ID correspondence asserted.", fontsize=5)
        fig.savefig(outputs["focal_muted"], format="pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 75 / 25.4))
        fig.subplots_adjust(left=0.07, right=0.92, bottom=0.32, top=0.95, wspace=0.3)
        variance = tables["pca_variance"]
        axes[0].bar(variance["component"], variance["explained_variance_ratio"], color="#0072B2", label="Individual")
        axes[0].plot(variance["component"], variance["cumulative_variance_ratio"], marker="o", markersize=2, markeredgewidth=0.5, color="#D55E00", label="Cumulative")
        axes[0].set(xlabel="Shared principal component", ylabel="Explained variance fraction", ylim=(0, 1.02))
        axes[0].legend(frameon=False)
        weights = tables["pca_components"].iloc[:, 1:].to_numpy().T
        limit = float(np.max(np.abs(weights)))
        im = axes[1].imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="none")
        axes[1].set_yticks(np.arange(COMPONENT_COUNT), labels=[f"PC{i + 1}" for i in range(COMPONENT_COUNT)])
        axes[1].set_xticks(np.arange(FEATURE_COUNT), labels=tables["feature_schema"]["producer_variable"], rotation=60, ha="right")
        fig.colorbar(im, ax=axes[1], label="Component coefficient", fraction=0.05)
        axes[1].grid(False)
        axes[0].set_axisbelow(True)
        axes[0].grid(True, color="#E5E7EB", linewidth=0.5)
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(width=0.5)
        fig.text(0.04, 0.03, f"Generated by {Path(__file__).name}; all source crop pixels including zero-masked features; descriptive fit diagnostics.", fontsize=5)
        fig.savefig(outputs["pca"], format="pdf")
        plt.close(fig)
    receipt = {"identity": identity, "per_core_display": metrics,
               "artifact_sha256": {name: _sha256(path) for name, path in outputs.items()},
               "artifacts": {name: str(path) for name, path in outputs.items()},
               "claim_label": "Exploratory"}
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("RECOMPUTED: combined HS + FLIM figures")
    return {"pdfs": outputs, "receipt": receipt, "receipt_path": receipt_path}


def audit_reproduction(result: dict[str, Any], project: Path, table_dir: Path) -> pd.DataFrame:
    """Check saved artifacts, all-pixel counts, support and current source identity."""
    receipt = result["receipt"]
    rows: list[dict[str, Any]] = []
    current = source_inventory(project).to_dict(orient="records")
    rows.append({"check": "current_delivered_sources_equal_fit_sources", "passed": current == receipt["identity"]["sources"]})
    for name, expected in receipt["artifact_sha256"].items():
        rows.append({"check": f"artifact_sha256:{name}", "passed": _sha256(result["paths"][name]) == expected})
    counts = np.zeros(CLUSTER_COUNT, dtype=np.int64)
    for core, item in result["per_core"].items():
        labels, support, intensity = item["labels"], item["support"], item["intensity"]
        checks = {
            "native_geometry": labels.shape == support.shape == intensity.shape == (CROP_SIZE, CROP_SIZE),
            "label_range": labels.dtype == np.uint8 and int(labels.min()) >= 0 and int(labels.max()) < CLUSTER_COUNT,
            "support_equals_source_threshold": np.array_equal(support, intensity >= MASK_THRESHOLD),
            "finite_intensity": bool(np.isfinite(intensity).all()),
            "state_table_core_identity": item["state_table"]["core"].eq(core).all(),
        }
        counts += np.bincount(labels.ravel(), minlength=CLUSTER_COUNT)
        for name, passed in checks.items():
            rows.append({"check": f"{core}:{name}", "passed": bool(passed)})
    if tuple(result["per_core"]) == CORES:
        rows.append({"check": "all_returned_maps_conserve_all_fit_cluster_counts",
                     "passed": np.array_equal(counts, receipt["all_fit_cluster_counts"])})
        rows.append({"check": "all_source_pixels_returned",
                     "passed": int(counts.sum()) == len(CORES) * PIXELS_PER_CORE})
    table = pd.DataFrame(rows)
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "dylan_hs_flim_reproduction_checks.csv", index=False)
    if not bool(table["passed"].all()):
        raise AssertionError(table.loc[~table["passed"]].to_string(index=False))
    print(f"OK: {len(table)} source/artifact/geometry checks")
    return table
