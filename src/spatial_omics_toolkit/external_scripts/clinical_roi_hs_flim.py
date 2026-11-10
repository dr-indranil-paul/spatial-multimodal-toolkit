"""Staged, source-bound raw HS / exported FLIM integration on explicit paired grids.

The original raw-pair entrypoints use within-acquisition translation. The separate
serial-section entrypoints below consume immutable accepted joint maps and their
prepared registered-HS sums, then fit new structural cycIF transforms. They never
reinterpret the accepted crop as an unregistered raw-HS grid or refit its labels.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, resize, warp
from threadpoolctl import threadpool_limits

FEATURE_NAMES = (
    'flim_intensity', 'flim_g1', 'flim_s1', 'apparent_phase_lifetime_ns',
    'apparent_modulation_lifetime_ns', 'hs_intensity', 'spectral_g1',
    'spectral_s1', 'spectral_g2', 'spectral_s2',
)
FLIM_KEYS = ('intensity', 'real', 'imag', 'phase_lifetime_ns', 'modulation_lifetime_ns')


def digest(path: Path) -> str:
    """Hash complete source bytes with bounded memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def _identity(stage: str, inputs: Any, parameters: dict[str, Any], functions: tuple[Callable[..., Any], ...]) -> dict[str, Any]:
    """Bind stage mathematics separately from display-only source changes."""
    code = '\n'.join(inspect.getsource(function) for function in functions)
    return {'schema': 'clinical_hs_flim_v1', 'stage': stage, 'inputs': inputs,
            'parameters': parameters, 'feature_names': list(FEATURE_NAMES),
            'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'versions': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'scipy', 'scikit-image', 'scikit-learn', 'phasorpy', 'aicspylibczi')}}


def _cache_paths(root: Path, identity: dict[str, Any]) -> tuple[Path, Path]:
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    directory = Path(root) / identity['stage'] / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory, directory / 'receipt.json'


def _load_cache(receipt_path: Path, identity: dict[str, Any], force: bool) -> dict[str, Any] | None:
    if not force and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('identity') == identity and all(
            Path(item['path']).is_file() and digest(Path(item['path'])) == item['sha256']
            for item in receipt.get('artifacts', [])
        ):
            print(f"CACHE_HIT {identity['stage']}: {receipt_path.parent.name[:12]}", flush=True)
            return receipt
    print(f"RECOMPUTE {identity['stage']}: {receipt_path.parent.name[:12]}", flush=True)
    return None


def _save_receipt(path: Path, identity: dict[str, Any], artifacts: list[Path], **details: Any) -> dict[str, Any]:
    result = {'identity': identity, 'claim_label': 'Exploratory', 'generating_script': str(Path(__file__).resolve()),
              'artifacts': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in artifacts], **details}
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


def validate_pair_manifest(pairs: pd.DataFrame, flim_exports: pd.DataFrame, *, table_dir: Path) -> pd.DataFrame:
    """Resolve exact raw SDT path + detector block into source-bound notebook05 exports."""
    required = {'pair_id', 'hs_path', 'flim_source_path', 'flim_block_index',
                'hs_to_flim_scale_y', 'hs_to_flim_scale_x', 'pairing_evidence', 'geometry_basis'}
    if not required.issubset(pairs) or pairs.empty or pairs.pair_id.duplicated().any():
        raise ValueError(f'Unique explicit pair rows required: {sorted(required)}')
    export_required = {'source_path', 'source_sha256', 'block_index', 'feature_path', 'feature_sha256',
                       'receipt_path', 'receipt_sha256', 'shape_y', 'shape_x', 'calibration_status',
                       'minimum_total_counts', 'frequency_mhz', 'frequency_provenance', 'lifetime_status'}
    if not export_required.issubset(flim_exports):
        raise ValueError('Notebook05 export manifest lacks the required source/array/receipt contract')
    if flim_exports.duplicated(['source_path', 'block_index']).any():
        raise ValueError('FLIM source path + detector block must be unique')
    import re
    if not pairs.pair_id.map(lambda value: isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+', value) is not None).all():
        raise ValueError('Pair IDs must be nonempty path-safe letters, digits, underscore or hyphen')
    rows: list[dict[str, Any]] = []
    for row in pairs.to_dict('records'):
        hs = Path(row['hs_path']).resolve(strict=True)
        raw_flim = Path(row['flim_source_path']).resolve(strict=True)
        if hs.suffix.lower() != '.czi' or raw_flim.suffix.lower() != '.sdt':
            raise ValueError('Pairs require native HS CZI and FLIM SDT paths')
        match = flim_exports.loc[flim_exports.source_path.map(lambda p: str(Path(p).resolve())).eq(str(raw_flim))
                                 & flim_exports.block_index.eq(int(row['flim_block_index']))]
        if len(match) != 1:
            raise ValueError(f"{row['pair_id']}: require one exact raw path/detector export")
        exported = match.iloc[0].to_dict()
        for field, hash_field in (('feature_path', 'feature_sha256'), ('receipt_path', 'receipt_sha256'),
                                  ('source_path', 'source_sha256')):
            if digest(Path(exported[field])) != exported[hash_field]:
                raise ValueError(f"{row['pair_id']}: changed FLIM {field}")
        scales = np.asarray([row['hs_to_flim_scale_y'], row['hs_to_flim_scale_x']], dtype=float)
        if not np.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError('Explicit positive HS-to-FLIM pixel scales are required')
        if not str(row['pairing_evidence']).strip() or not str(row['geometry_basis']).strip():
            raise ValueError('Pairing and common-grid evidence must be declared, not inferred from size')
        with np.load(exported['feature_path'], allow_pickle=False) as data:
            if not set((*FLIM_KEYS, 'valid')).issubset(data.files):
                raise ValueError('FLIM feature NPZ lacks native intensity, phasors, lifetimes or validity')
            shape = tuple(data['intensity'].shape)
            if len(shape) != 2 or shape != (int(exported['shape_y']), int(exported['shape_x'])):
                raise ValueError('FLIM exported geometry mismatch')
            if any(data[key].shape != shape for key in (*FLIM_KEYS, 'valid')):
                raise ValueError('All FLIM feature planes must share native YX geometry')
        actual_hs_sha = digest(hs)
        if row.get('hs_sha256_expected') and row['hs_sha256_expected'] != actual_hs_sha:
            raise ValueError('HS source differs from the explicitly bound pairing evidence')
        if row.get('flim_sha256_expected') and row['flim_sha256_expected'] != exported['source_sha256']:
            raise ValueError('FLIM source differs from the explicitly bound pairing evidence')
        rows.append(dict(row, hs_path=str(hs), hs_sha256=actual_hs_sha,
                         flim_source_path=str(raw_flim), flim_source_sha256=exported['source_sha256'],
                         flim_feature_path=exported['feature_path'], flim_feature_sha256=exported['feature_sha256'],
                         flim_receipt_path=exported['receipt_path'], flim_receipt_sha256=exported['receipt_sha256'],
                         shape_y=shape[0], shape_x=shape[1], calibration_status=exported['calibration_status'],
                         flim_minimum_total_counts=exported['minimum_total_counts'],
                         flim_frequency_mhz=exported['frequency_mhz'],
                         flim_frequency_provenance=exported['frequency_provenance'],
                         flim_lifetime_status=exported['lifetime_status']))
    result = pd.DataFrame(rows)
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(table_dir) / 'validated_hs_flim_pairs.csv', index=False)
    return result


def read_native_hs_moments(path: Path, *, max_working_bytes: int = 4 * 1024**3) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Reuse HS metadata/layout checks; read native channels once and retain only spectral sums."""
    from aicspylibczi import CziFile
    from phasorpy.phasor import phasor_from_signal
    from clinical_roi_hyperspectral import PRIMARY_ROLE, _layout, inspect_raw_metadata, _spectral_channel_grid

    row = {'path': str(path), 'processing_role': PRIMARY_ROLE,
           'classification': 'raw_acquisition', 'modality': 'hyperspectral',
           'assigned_tma': '', 'collaborator': '', 'candidate_core_ids': '[]',
           'source_set': 'explicit_pair_manifest', 'source_scope': 'EXPLICIT_PAIR',
           'size_bytes': Path(path).stat().st_size}
    reader = CziFile(str(path))
    dimensions = _layout(reader)
    metadata, channel_rows = inspect_raw_metadata(row, reader=reader)
    channels = _spectral_channel_grid(channel_rows, expected_count=dimensions['C'][1])
    n_channels = len(channels)
    if n_channels < 5:
        raise ValueError('Two spectral harmonics require at least five uniformly spaced bands')
    selectors = {axis: span[0] for axis, span in dimensions.items() if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}}
    if 'M' in dimensions:
        box = reader.get_mosaic_bounding_box()
        region = tuple(int(v) for v in (box.x, box.y, box.w, box.h))
        height, width = region[3], region[2]
    else:
        region = None
        height, width = dimensions['Y'][1] - dimensions['Y'][0], dimensions['X'][1] - dimensions['X'][0]
    bound = height * width * (n_channels * 2 + 8 * 18)
    if bound > max_working_bytes:
        raise MemoryError(f'Native HS working bound {bound} exceeds declared limit {max_working_bytes}')
    cube = np.empty((n_channels, height, width), dtype=np.uint16)
    for channel in range(n_channels):
        if region is None:
            plane, _ = reader.read_image(C=channel, **selectors)
        else:
            plane = reader.read_mosaic(C=channel, region=region, scale_factor=1.0,
                                       background_color=(0., 0., 0.), **selectors)
        plane = np.squeeze(plane)
        if plane.shape != (height, width) or plane.dtype != np.uint16:
            raise ValueError('Unsupported native HS channel dtype/shape; no implicit conversion')
        cube[channel] = plane
    total = cube.sum(axis=0, dtype=np.uint64).astype(np.float64)
    _, real, imag = phasor_from_signal(cube, axis=0, harmonic=[1, 2], use_fft=True, num_threads=1)
    if any((~np.isfinite(values[:, total > 0])).any() for values in (real, imag)):
        raise ValueError('Undefined spectral phasors at positive-signal pixels')
    moments = np.stack([total, *(np.where(total > 0, values * total, 0.0)
                                for values in (real[0], imag[0], real[1], imag[1]))])
    return moments, channels, {'native_shape_yx': [height, width], 'channel_count': n_channels,
                               'working_bound_bytes': bound, 'metadata_xml_sha256': metadata['metadata_xml_sha256'],
                               'native_region_xywh': region, 'spectral_phase_convention': 'PhasorPy harmonics1,2 in recorded contiguous equal-width C order'}


def prepare_hs_inputs(pairs: pd.DataFrame, *, object_dir: Path, table_dir: Path, force: bool = False) -> dict[str, Any]:
    """Cache five native spectral moment sums per exact raw HS acquisition, never full raw cubes."""
    outputs: dict[str, Any] = {}
    for row in pairs.to_dict('records'):
        identity = _identity('hs_moments', {'path': row['hs_path'], 'sha256': row['hs_sha256']},
                             {'harmonics': [1, 2], 'band_scaling': None, 'smoothing': None,
                              'metadata_validator_sha256': digest(Path(__file__).with_name('clinical_roi_hyperspectral.py'))},
                             (read_native_hs_moments, prepare_hs_inputs))
        directory, receipt_path = _cache_paths(object_dir, identity)
        receipt = _load_cache(receipt_path, identity, force)
        if receipt is None:
            path = Path(row['hs_path'])
            if digest(path) != row['hs_sha256']:
                raise ValueError('HS source changed after pair validation')
            before = (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
            moments, channels, metadata = read_native_hs_moments(path)
            if before != (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns):
                raise ValueError('HS source changed while decoding')
            array_path, band_path = directory / 'spectral_moments.npy', directory / 'bands.csv'
            np.save(array_path, moments, allow_pickle=False)
            channels.to_csv(band_path, index=False)
            receipt = _save_receipt(receipt_path, identity, [array_path, band_path],
                                    array_path=str(array_path.resolve()), bands_path=str(band_path.resolve()), metadata=metadata)
        outputs[row['pair_id']] = {'receipt_path': str(receipt_path.resolve()), **receipt}
    return outputs


def _scaled_moments(moments: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    target = (int(row['shape_y']), int(row['shape_x']))
    scale = (float(row['hs_to_flim_scale_y']), float(row['hs_to_flim_scale_x']))
    expected = tuple(int(round(size * factor)) for size, factor in zip(moments.shape[1:], scale))
    if expected != target:
        raise ValueError('Explicit scale does not establish the FLIM field of view; supply correct ROI/geometry')
    if moments.shape[1:] == target and scale == (1.0, 1.0):
        return moments
    return np.stack([resize(np.array(plane, copy=True, order='C'), target, order=1, preserve_range=True, anti_aliasing=False) for plane in moments])


def _log_correlation(fixed: np.ndarray, moving: np.ndarray, support: np.ndarray) -> float | None:
    usable = support & np.isfinite(fixed) & np.isfinite(moving) & (fixed > 0) & (moving > 0)
    if usable.sum() < 3:
        return None
    a, b = np.log1p(fixed[usable]), np.log1p(moving[usable])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def register_pairs(pairs: pd.DataFrame, hs_inputs: dict[str, Any], *, object_dir: Path,
                   table_dir: Path, upsample_factor: int = 30, force: bool = False) -> dict[str, Any]:
    """Estimate within-acquisition translation and warp spectral numerators before division."""
    outputs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for row in pairs.to_dict('records'):
        hs = hs_inputs[row['pair_id']]
        identity = _identity('registration', {'pair': row, 'hs_receipt_sha256': digest(Path(hs['receipt_path']))},
                             {'model': 'translation', 'upsample_factor': upsample_factor, 'normalization': 'phase',
                              'interpolation_order': 1, 'quantization': None},
                             (_scaled_moments, _log_correlation, register_pairs))
        directory, receipt_path = _cache_paths(object_dir, identity)
        receipt = _load_cache(receipt_path, identity, force)
        if receipt is None:
            moments = _scaled_moments(np.load(hs['array_path'], mmap_mode='r'), row)
            with np.load(row['flim_feature_path'], allow_pickle=False) as flim:
                fixed = flim['intensity']
            with threadpool_limits(limits=1):
                shift, error, phase = phase_cross_correlation(fixed, moments[0], upsample_factor=upsample_factor, normalization='phase')
            transform = SimilarityTransform(translation=(-float(shift[1]), -float(shift[0])))
            registered = np.stack([warp(np.array(plane, copy=True, order='C'), transform, preserve_range=True, order=1) for plane in moments])
            overlap = warp(np.ones(fixed.shape, dtype=np.uint8), transform, order=0, preserve_range=True) > 0
            array_path = directory / 'registered_moments.npz'
            np.savez_compressed(array_path, moments=registered, overlap=overlap)
            summary = {'pair_id': row['pair_id'], 'shift_y': float(shift[0]), 'shift_x': float(shift[1]),
                       'phase_error': float(error) if np.isfinite(error) else None,
                       'overlap_pixels': int(overlap.sum()), 'total_pixels': int(overlap.size),
                       'log_correlation_before': _log_correlation(fixed, moments[0], overlap),
                       'log_correlation_after': _log_correlation(fixed, registered[0], overlap),
                       'qc_status': 'EXPLORATORY_TRANSLATION_REQUIRES_IMAGE_REVIEW', 'claim_label': 'Exploratory'}
            receipt = _save_receipt(receipt_path, identity, [array_path], array_path=str(array_path.resolve()), summary=summary)
        outputs[row['pair_id']] = {'receipt_path': str(receipt_path.resolve()), **receipt}
        rows.append(receipt['summary'])
    pd.DataFrame(rows).to_csv(Path(table_dir) / 'hs_flim_registration_qc.csv', index=False)
    return outputs


def derive_joint_features(pairs: pd.DataFrame, registrations: dict[str, Any], *, object_dir: Path,
                          table_dir: Path, minimum_flim_counts: float = 0,
                          minimum_hs_sum: float = 0, force: bool = False) -> dict[str, Any]:
    """Build ten raw-derived maps on FLIM YX; preserve undefined values and explicit support."""
    outputs: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for row in pairs.to_dict('records'):
        registered = registrations[row['pair_id']]
        identity = _identity('joint_features', {'pair': row, 'registration_receipt_sha256': digest(Path(registered['receipt_path']))},
                             {'minimum_flim_counts_exclusive': minimum_flim_counts, 'minimum_hs_sum_exclusive': minimum_hs_sum,
                              'support': 'geometric overlap AND exported FLIM validity AND all10 finite AND both positive-signal cutoffs',
                              'lifetime_clipping': None, 'smoothing': None}, (derive_joint_features,))
        directory, receipt_path = _cache_paths(object_dir, identity)
        receipt = _load_cache(receipt_path, identity, force)
        if receipt is None:
            with np.load(registered['array_path'], allow_pickle=False) as reg:
                moments, overlap = reg['moments'], reg['overlap']
            total = moments[0]
            ratios = np.full(moments[1:].shape, np.nan, dtype=np.float64)
            np.divide(moments[1:], total[None], out=ratios, where=total[None] > 0)
            with np.load(row['flim_feature_path'], allow_pickle=False) as flim:
                features = np.stack([*(flim[key] for key in FLIM_KEYS), total, *ratios]).astype(np.float32)
                support = overlap & flim['valid'].astype(bool)
            support &= np.isfinite(features).all(axis=0) & (features[0] > minimum_flim_counts) & (total > minimum_hs_sum)
            if not support.any():
                raise ValueError(f"{row['pair_id']}: no jointly valid pixels for the declared ten-feature family")
            array_path = directory / 'joint_features.npz'
            np.savez_compressed(array_path, features=features, support=support)
            summary = {'pair_id': row['pair_id'], 'shape_y': features.shape[1], 'shape_x': features.shape[2],
                       'joint_valid_pixels': int(support.sum()), 'all_pixels': int(support.size),
                       'calibration_status': row['calibration_status'], 'claim_label': 'Exploratory'}
            receipt = _save_receipt(receipt_path, identity, [array_path], array_path=str(array_path.resolve()), summary=summary)
        outputs[row['pair_id']] = {'receipt_path': str(receipt_path.resolve()), **receipt}
        summaries.append(receipt['summary'])
    pd.DataFrame(summaries).to_csv(Path(table_dir) / 'joint_feature_support.csv', index=False)
    return outputs


def fit_shared_pca(features: dict[str, Any], *, object_dir: Path, table_dir: Path,
                   transform: str = 'per_pair_zscore', samples_per_pair: int = 20000,
                   components: int = 5, seed: int = 20261024, force: bool = False) -> dict[str, Any]:
    """Fit one shared PCA on deterministic balanced raw-derived pixel samples."""
    if transform not in {'per_pair_zscore', 'shared_zscore'} or samples_per_pair < components:
        raise ValueError('Declare per_pair_zscore/shared_zscore and a sufficient positive sample cap')
    identity = _identity('shared_pca', {key: digest(Path(value['receipt_path'])) for key, value in features.items()},
                         {'transform': transform, 'samples_per_pair': samples_per_pair, 'components': components, 'seed': seed, 'pair_order': list(features),
                          'sampling': 'uniform without replacement from joint valid pixels; independent seed per declared pair order'},
                         (fit_shared_pca,))
    directory, receipt_path = _cache_paths(object_dir, identity)
    receipt = _load_cache(receipt_path, identity, force)
    if receipt is None:
        blocks: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        statistics: dict[str, dict[str, list[float]]] = {}
        for index, (pair, item) in enumerate(features.items()):
            with np.load(item['array_path'], allow_pickle=False) as data:
                values = data['features'][:, data['support']].T
            means, scales = values.mean(axis=0, dtype=np.float64), values.std(axis=0, dtype=np.float64)
            if not np.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError(f'{pair}: undefined feature standardization')
            rng = np.random.default_rng(seed + index)
            ranks = rng.choice(len(values), size=min(samples_per_pair, len(values)), replace=False)
            sampled = values[ranks].astype(np.float64)
            if transform == 'per_pair_zscore':
                sampled = (sampled - means) / scales
            blocks.append(sampled)
            statistics[pair] = {'mean': means.tolist(), 'scale': scales.tolist()}
            rows.append({'pair_id': pair, 'joint_valid_pixels': len(values), 'sampled_pixels': len(sampled)})
        pooled = np.vstack(blocks)
        if transform == 'shared_zscore':
            shared_mean, shared_scale = pooled.mean(axis=0), pooled.std(axis=0)
            if (shared_scale <= 0).any():
                raise ValueError('Undefined pooled shared feature standardization')
            pooled = (pooled - shared_mean) / shared_scale
            statistics = {pair: {'mean': shared_mean.tolist(), 'scale': shared_scale.tolist()} for pair in features}
        with threadpool_limits(limits=1):
            pca = PCA(n_components=components, svd_solver='full', random_state=seed).fit(pooled)
            scores = pca.transform(pooled)
        model_path = directory / 'shared_pca.npz'
        np.savez_compressed(model_path, pca_mean=pca.mean_, components=pca.components_,
                            explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_,
                            pooled_scores=scores)
        sampling_path = directory / 'sampling.csv'
        pd.DataFrame(rows).to_csv(sampling_path, index=False)
        receipt = _save_receipt(receipt_path, identity, [model_path, sampling_path], model_path=str(model_path.resolve()),
                                statistics=statistics, sampling_path=str(sampling_path.resolve()))
    with np.load(receipt['model_path'], allow_pickle=False) as model:
        pd.DataFrame({'component': np.arange(1, components + 1),
                      'explained_variance_ratio': model['explained_variance_ratio'],
                      'cumulative_variance_ratio': np.cumsum(model['explained_variance_ratio'])}).to_csv(Path(table_dir) / 'shared_raw_joint_pca_variance.csv', index=False)
        pd.DataFrame(model['components'].T, index=FEATURE_NAMES).to_csv(Path(table_dir) / 'shared_raw_joint_pca_components.csv')
    return {'receipt_path': str(receipt_path.resolve()), **receipt}


def cluster_shared_features(features: dict[str, Any], pca: dict[str, Any], *, object_dir: Path,
                            table_dir: Path, k: int = 9, seed: int = 20261024,
                            force: bool = False) -> dict[str, Any]:
    """Fit one exploratory K-means model in the shared PCA space and predict every valid native pixel."""
    identity = _identity('shared_clusters', {'pca_receipt_sha256': digest(Path(pca['receipt_path']))},
                         {'k': k, 'seed': seed, 'method': 'MiniBatchKMeans', 'n_init': 10, 'batch_size': 4096,
                          'label_zero': 'outside declared joint valid support'}, (cluster_shared_features,))
    directory, receipt_path = _cache_paths(object_dir, identity)
    receipt = _load_cache(receipt_path, identity, force)
    if receipt is None:
        with np.load(pca['model_path'], allow_pickle=False) as model:
            mean, components, pooled_scores = model['pca_mean'], model['components'], model['pooled_scores']
        if k < 2 or k > len(pooled_scores):
            raise ValueError('Cluster count outside supported sample bounds')
        with threadpool_limits(limits=1):
            estimator = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=4096).fit(pooled_scores)
        model_path = directory / 'shared_kmeans_centers.npy'
        np.save(model_path, estimator.cluster_centers_, allow_pickle=False)
        maps: dict[str, str] = {}
        artifacts: list[Path] = [model_path]
        counts: list[dict[str, Any]] = []
        with threadpool_limits(limits=1):
            for pair, item in features.items():
                with np.load(item['array_path'], allow_pickle=False) as data:
                    cube, support = data['features'], data['support']
                labels = np.zeros(support.shape, dtype=np.uint16)
                ranks = np.flatnonzero(support)
                stats = pca['statistics'][pair]
                local_mean, scale = np.asarray(stats['mean']), np.asarray(stats['scale'])
                flat = cube.reshape(len(FEATURE_NAMES), -1)
                for start in range(0, len(ranks), 65536):
                    chosen = ranks[start:start + 65536]
                    values = (flat[:, chosen].T - local_mean) / scale
                    projected = (values - mean) @ components.T
                    labels.ravel()[chosen] = estimator.predict(projected) + 1
                path = directory / f'{pair}_labels_yx.npy'
                np.save(path, labels, allow_pickle=False)
                maps[pair] = str(path.resolve())
                artifacts.append(path)
                for cluster, count in enumerate(np.bincount(labels[support], minlength=k + 1)[1:], 1):
                    counts.append({'pair_id': pair, 'cluster_id': cluster, 'pixels': int(count),
                                   'joint_valid_pixels': int(support.sum()), 'fraction': float(count / support.sum()),
                                   'claim_label': 'Exploratory'})
        count_path = directory / 'occupancy.csv'
        pd.DataFrame(counts).to_csv(count_path, index=False)
        artifacts.append(count_path)
        receipt = _save_receipt(receipt_path, identity, artifacts, label_paths=maps,
                                centers_path=str(model_path.resolve()), occupancy_path=str(count_path.resolve()))
    pd.read_csv(receipt['occupancy_path']).to_csv(Path(table_dir) / 'raw_joint_cluster_occupancy.csv', index=False)
    return {'receipt_path': str(receipt_path.resolve()), **receipt}


def _display_limits(values: np.ndarray) -> tuple[float, float]:
    """Return finite-value display bounds without changing quantitative values."""
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0, 0.0
    low, high = np.percentile(values[finite], [1, 99])
    return float(low), float(high)


def _display_scale(values: np.ndarray, *, limits: tuple[float, float] | None = None) -> np.ndarray:
    """Display-only robust scaling; quantitative maps are never altered."""
    finite = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.float32)
    if finite.any():
        low, high = _display_limits(values) if limits is None else limits
        if high > low:
            output[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
    return output


def render_registration_qc(pairs: pd.DataFrame, hs_inputs: dict[str, Any], registrations: dict[str, Any],
                           *, plot_dir: Path, display_stride: int = 4, force: bool = False) -> Path:
    """Show every explicit pair before/after translation; FLIM red, HS cyan."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from clinical_roi_hyperspectral import _register_project_arial, _pseudocolor_style

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    output = Path(plot_dir) / 'raw_hs_flim_registration_before_after.pdf'
    sources = {'registrations': {key: digest(Path(value['receipt_path'])) for key, value in registrations.items()},
               'native_hs_moments': {key: digest(Path(value['array_path'])) for key, value in hs_inputs.items()},
               'fixed_flim_features': {row['pair_id']: digest(Path(row['flim_feature_path'])) for row in pairs.to_dict('records')}}
    identity = _identity('registration_qc_display', sources,
                         {'stride': display_stride, 'stretch': 'log1p percentiles1,99; native HS bounds shared before/after; fixed FLIM bounds',
                          'color': 'FLIM red; HS cyan'}, (_display_limits, _display_scale, render_registration_qc))
    receipt_path = Path(plot_dir) / 'registration_display_receipt.json'
    if _load_cache(receipt_path, identity, force) is not None:
        return output
    _register_project_arial()
    display_limits: dict[str, Any] = {}
    with mpl.rc_context(_pseudocolor_style()), PdfPages(output) as pdf:
        for row in pairs.to_dict('records'):
            pair = row['pair_id']
            native = _scaled_moments(np.load(hs_inputs[pair]['array_path'], mmap_mode='r'), row)[0]
            with np.load(row['flim_feature_path'], allow_pickle=False) as data:
                fixed = data['intensity']
            with np.load(registrations[pair]['array_path'], allow_pickle=False) as data:
                moving = data['moments'][0]
            fixed_log, native_log, moving_log = (np.log1p(np.maximum(values, 0)) for values in (fixed, native, moving))
            flim_limits, hs_limits = _display_limits(fixed_log), _display_limits(native_log)
            display_limits[pair] = {'flim_log1p': list(flim_limits), 'hs_log1p_shared_before_after': list(hs_limits)}
            a = _display_scale(fixed_log, limits=flim_limits)[::display_stride, ::display_stride]
            b = _display_scale(native_log, limits=hs_limits)[::display_stride, ::display_stride]
            c = _display_scale(moving_log, limits=hs_limits)[::display_stride, ::display_stride]
            before, after = np.stack([a, b, b], axis=-1), np.stack([a, c, c], axis=-1)
            fig, axes = plt.subplots(1, 4, figsize=(220 / 25.4, 65 / 25.4))
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.91, wspace=0.04)
            for ax, values, label in zip(axes, (a, b, before, after),
                                         ('FLIM summed counts', 'HS sum on declared scale', 'Before: red FLIM / cyan HS', 'After translation')):
                ax.imshow(values, cmap='gray', interpolation='none', vmin=0, vmax=1)
                ax.set_axis_off()
                bounds = ax.get_position()
                fig.text(bounds.x0, bounds.y1 + 0.02, label, fontsize=5)
            summary = registrations[pair]['summary']
            fig.text(0.02, 0.07, f"{pair}; shift Y={summary['shift_y']:.4f}, X={summary['shift_x']:.4f} pixels. Exploratory within-acquisition alignment; inspect internal structure.", fontsize=5)
            fig.text(0.02, 0.025, f"Generator: {Path(__file__).name}; stride {display_stride}; HS native-based stretch shared before/after; fixed FLIM stretch; no physical scale calibration assumed.", fontsize=5)
            pdf.savefig(fig)
            plt.close(fig)
    _save_receipt(receipt_path, identity, [output], display_limits=display_limits)
    return output


def render_joint_feature_qc(features: dict[str, Any], *, plot_dir: Path,
                            display_stride: int = 4, force: bool = False) -> Path:
    """Inspect all ten raw-derived features separately from accepted benchmark labels."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from clinical_roi_hyperspectral import _register_project_arial, _pseudocolor_style

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    output = Path(plot_dir) / 'raw_hs_flim_ten_feature_maps.pdf'
    identity = _identity('joint_feature_display', {key: digest(Path(value['receipt_path'])) for key, value in features.items()},
                         {'stride': display_stride, 'stretch': 'independent finite1,99 percentiles',
                          'mask_for_display': 'nonfinite values explicitly masked', 'undefined_color': '#D9D9D9'},
                         (_display_limits, _display_scale, render_joint_feature_qc))
    receipt_path = Path(plot_dir) / 'joint_feature_display_receipt.json'
    if _load_cache(receipt_path, identity, force) is not None:
        return output
    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()), PdfPages(output) as pdf:
        for pair, item in features.items():
            with np.load(item['array_path'], allow_pickle=False) as data:
                cube, support = data['features'], data['support']
            fig, axes = plt.subplots(2, 5, figsize=(240 / 25.4, 115 / 25.4))
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.105, top=0.94, wspace=0.04, hspace=0.12)
            feature_cmap = mpl.colormaps['viridis'].copy()
            feature_cmap.set_bad('#D9D9D9')
            for index, ax in enumerate(axes.ravel()):
                scaled = _display_scale(cube[index])
                values = np.ma.array(scaled, mask=~np.isfinite(cube[index]))[::display_stride, ::display_stride]
                ax.imshow(values, cmap=feature_cmap, vmin=0, vmax=1, interpolation='none')
                ax.set_axis_off()
                bounds = ax.get_position()
                fig.text(bounds.x0, bounds.y1 + 0.01, FEATURE_NAMES[index], fontsize=5)
            fig.text(0.02, 0.055, f"{pair}: {int(support.sum())} / {support.size} pixels satisfy the declared joint feature support. Lifetimes are apparent unless calibration is documented.", fontsize=5)
            fig.text(0.02, 0.02, f"Generator: {Path(__file__).name}; raw-derived integration maps; undefined values = gray; finite-feature display stretches; stride {display_stride}; source benchmark separate.", fontsize=5)
            pdf.savefig(fig)
            plt.close(fig)
    _save_receipt(receipt_path, identity, [output])
    return output


def load_accepted_benchmark(notebook_receipt_path: Path) -> dict[str, Any]:
    """Read and verify an accepted completed reference without rerendering or refitting it."""
    notebook_receipt = json.loads(Path(notebook_receipt_path).read_text())
    for field in ('fit_receipt', 'display_receipt'):
        if digest(Path(notebook_receipt[field])) != notebook_receipt[f'{field}_sha256']:
            raise ValueError(f'Accepted benchmark {field} no longer matches its completion witness')
    display = json.loads(Path(notebook_receipt['display_receipt']).read_text())
    for name, path in display['artifacts'].items():
        if digest(Path(path)) != display['artifact_sha256'][name]:
            raise ValueError(f'Accepted benchmark PDF changed: {path}')
    fit_path = Path(notebook_receipt['fit_receipt'])
    fit = json.loads(fit_path.read_text())
    from dylan_k9_reconstruction import _artifact_names
    for name, filename in _artifact_names(tuple(fit['identity']['parameters']['returned_cores'])).items():
        if digest(fit_path.parent / filename) != fit['artifact_sha256'][name]:
            raise ValueError(f'Accepted benchmark model/map artifact changed: {filename}')
    print('VERIFIED_ACCEPTED_REFERENCE: saved maps, model and PDFs reused without writes')
    return {'fit_receipt': fit, 'display_receipt': display, 'pdfs': display['artifacts'],
            'notebook_receipt': notebook_receipt, 'witness_path': str(Path(notebook_receipt_path).resolve())}


def audit_integration(pairs: pd.DataFrame, hs_inputs: dict[str, Any], registrations: dict[str, Any],
                      features: dict[str, Any], *, table_dir: Path,
                      pca: dict[str, Any] | None = None, clusters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Verify real source bindings, native geometry, ratios and optional label/support conservation."""
    rows: list[dict[str, Any]] = []
    for name, stage in (('hs', hs_inputs), ('registration', registrations), ('joint_features', features)):
        rows.append({'check': f'{name}:pair_membership', 'passed': set(stage) == set(pairs.pair_id)})
        for pair, item in stage.items():
            saved = json.loads(Path(item['receipt_path']).read_text())
            for artifact in saved['artifacts']:
                rows.append({'check': f'{pair}:{name}:artifact:{Path(artifact["path"]).name}',
                             'passed': digest(Path(artifact['path'])) == artifact['sha256']})
    for row in pairs.to_dict('records'):
        pair = row['pair_id']
        with np.load(features[pair]['array_path'], allow_pickle=False) as data:
            cube, support = data['features'], data['support']
        with np.load(registrations[pair]['array_path'], allow_pickle=False) as data:
            moments, overlap = data['moments'], data['overlap']
        with np.load(row['flim_feature_path'], allow_pickle=False) as data:
            raw_flim_intensity = data['intensity']
        checks = {
            'declared_shape': cube.shape == (len(FEATURE_NAMES), int(row['shape_y']), int(row['shape_x'])),
            'joint_support_finite': bool(np.isfinite(cube[:, support]).all()),
            'support_inside_registration': bool((~support | overlap).all()),
            'flim_intensity_preserved': np.array_equal(cube[0], raw_flim_intensity.astype(np.float32)),
            'registered_hs_sum_preserved': np.array_equal(cube[5], moments[0].astype(np.float32)),
            'spectral_ratios_on_support': bool(np.allclose(cube[6:, support],
                                                         moments[1:, support] / moments[0, support], rtol=2e-7, atol=1e-7)),
        }
        if clusters is not None:
            labels = np.load(clusters['label_paths'][pair], mmap_mode='r')
            checks['label_support_exact'] = np.array_equal(labels > 0, support)
            checks['label_geometry'] = labels.shape == support.shape
            checks['label_range'] = int(labels.max()) <= clusters['identity']['parameters']['k']
        for check, passed in checks.items():
            rows.append({'check': f'{pair}:{check}', 'passed': bool(passed)})
    for name, model in (('pca', pca), ('clusters', clusters)):
        if model is not None:
            for artifact in model['artifacts']:
                rows.append({'check': f'{name}:artifact:{Path(artifact["path"]).name}',
                             'passed': digest(Path(artifact['path'])) == artifact['sha256']})
    result = pd.DataFrame(rows)
    result.to_csv(Path(table_dir) / 'raw_hs_flim_integration_checks.csv', index=False)
    if not bool(result.passed.all()):
        raise AssertionError(result.loc[~result.passed].to_string(index=False))
    print(f'OK: {len(result)} source/geometry/value/support checks')
    return result


SERIAL_SECTION_EVIDENCE = "wiki sources/SRC-2026-10-25-008: user confirmed Serial sections, matched cores"
SERIAL_ROTATION_EVIDENCE = "wiki sources/SRC-2026-10-25-009: user observed relative rotation"


def _serial_binding(path: str | Path) -> dict[str, str]:
    import clinical_roi_valis as valis
    return {"path": str(Path(path).resolve()), "sha256": valis.compute_file_sha256(path)}


def prepare_serial_section_inputs(
    benchmark_receipt_path: str | Path,
    raw_manifest_path: str | Path,
    registration_checks_path: str | Path,
    cycif_manifest_path: str | Path,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    manifest_dir: str | Path,
) -> dict[str, Any]:
    """Bind every accepted joint map to its exact prepared crop and new cycIF pair.

    The anchor NPZ deliberately contains only total/support, not invented L1
    spectra. Raw fitted IDs 0..K-1 become display domain IDs 1..K; ID0 denotes
    unsupported pixels only. The original labels and benchmark stay untouched.
    """
    import clinical_roi_valis as valis
    from clinical_roi_cycif import extract_czi_metadata
    from clinical_roi_tissue import derive_tissue_envelope
    from clinical_roi_hyperspectral import _layout
    from dylan_k9_reconstruction import _artifact_names, _core_artifacts
    from aicspylibczi import CziFile
    import tifffile
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    valis.guard_compute()
    objects, tables, manifests = Path(object_dir), Path(table_dir), Path(manifest_dir)
    for directory in (objects, tables, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(benchmark_receipt_path)
    accepted = json.loads(receipt_path.read_text())
    raw = pd.read_csv(raw_manifest_path, keep_default_na=False)
    checks = pd.read_csv(registration_checks_path, keep_default_na=False)
    cycif = valis.load_cycif_handoff(cycif_manifest_path)
    cores = accepted["identity"]["parameters"]["returned_cores"]
    if len(cores) != 8 or len(set(cores)) != 8:
        raise ValueError("The accepted eight-map benchmark is required without substitution")
    producer_names = _artifact_names(tuple(cores))
    artifact_hashes = {
        filename: accepted["artifact_sha256"][alias]
        for alias, filename in producer_names.items()
    }
    k = int(accepted["identity"]["parameters"]["kmeans_k"])
    provenance = {item["core"]: item for item in accepted["preprocessing_by_core"]}
    source_map = {(item["core"], item["role"]): item for item in accepted["identity"]["sources"]}
    source_bindings = [_serial_binding(path) for path in
                       (receipt_path, raw_manifest_path, registration_checks_path, cycif_manifest_path)]
    identity = {
        "sources": source_bindings, "section_evidence": SERIAL_SECTION_EVIDENCE,
        "rotation_evidence": SERIAL_ROTATION_EVIDENCE,
        "accepted_sources": [_serial_binding(item["path"]) for item in accepted["identity"]["sources"]],
        "native_candidates": [_serial_binding(path) for path in checks["native_hs_path"].unique()],
        "code": {function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                 for function in (prepare_serial_section_inputs, derive_tissue_envelope,
                                  extract_czi_metadata, _artifact_names, _core_artifacts)},
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "scikit-image", "opencv-python-headless",
                                 "tifffile", "aicspylibczi", "matplotlib")},
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = objects / "preparation" / cache_key
    folder.mkdir(parents=True, exist_ok=True)
    preparation_receipt = folder / "receipt.json"
    if preparation_receipt.is_file():
        previous = json.loads(preparation_receipt.read_text())
        if previous["identity"] == identity and valis._verify_artifacts(previous["artifacts"]):
            return {"inputs": pd.read_csv(previous["inputs_path"], keep_default_na=False),
                    "labels_manifest": pd.read_csv(previous["labels_manifest_path"], keep_default_na=False),
                    "receipt_path": str(preparation_receipt), "cache_status": "CACHE_HIT"}
    color_map = LinearSegmentedColormap.from_list(
        "accepted_joint_k9", mpl.colormaps["nipy_spectral"](np.linspace(0.03, 0.97, 256)))
    palette = np.zeros((k + 1, 3), dtype=np.float32)
    palette[1:] = color_map(np.arange(k) / (k - 1))[:, :3]
    palette_path = folder / "joint_domain_palette.npy"
    np.save(palette_path, palette)
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    artifacts = [palette_path]
    for core in cores:
        core_dir = folder / core
        core_dir.mkdir(exist_ok=True)
        core_checks = checks.loc[checks["core"].eq(core)]
        exact = core_checks.loc[core_checks["registered_all_channels_exact"].map(valis.strict_bool)]
        if len(exact) != 1:
            raise ValueError(f"{core}: raw HS identity is ambiguous or not exact")
        mapping = exact.iloc[0].to_dict()
        native_path = Path(mapping["native_hs_path"])
        raw_match = raw.loc[raw["core"].eq(core) & raw["path"].eq(str(native_path))]
        if len(raw_match) != 1 or valis.compute_file_sha256(native_path) != raw_match.iloc[0]["sha256"]:
            raise ValueError(f"{core}: exact June CZI identity is not verified")
        markers = cycif.loc[cycif["assigned_tma"].eq("TMA1") & cycif["confirmed_core_id"].eq(core)]
        dapi = markers.loc[markers["marker"].eq("DAPI")]
        if len(dapi) != 1 or markers["marker"].duplicated().any() or len(markers) != 21:
            raise ValueError(f"{core}: require one DAPI driver and all 20 unique antibodies")
        for role in ("feature_stack", "summed_mask_source"):
            source = source_map[(core, role)]
            if valis.compute_file_sha256(source["path"]) != source["sha256"]:
                raise ValueError(f"{core}: accepted source changed: {role}")
        if Path(mapping["supplied_registered_cube_path"]).resolve() != Path(source_map[(core, "summed_mask_source")]["path"]).resolve():
            raise ValueError(f"{core}: registered HS provenance does not name the accepted sum source")
        originals: dict[str, Path] = {
            role: receipt_path.parent / filename
            for role, filename in _core_artifacts(core).items()
            if role in {"labels", "support", "intensity"}
        }
        for role, path in originals.items():
            if valis.compute_file_sha256(path) != artifact_hashes[path.name]:
                raise ValueError(f"{core}: accepted {role} artifact changed")
        labels = np.load(originals["labels"], allow_pickle=False)
        support = np.load(originals["support"], allow_pickle=False)
        total = np.load(originals["intensity"], allow_pickle=False)
        prep = provenance[core]
        y0, y1, x0, x1 = map(int, prep["crop_bounds_native_yx"])
        my0, my1, mx0, mx1 = map(int, prep["mask_crop_bounds_native_yx"])
        if (y0, y1, x0, x1) != (my0, my1, mx0, mx1):
            raise ValueError(f"{core}: feature/sum crop mismatch")
        registered = tifffile.imread(source_map[(core, "summed_mask_source")]["path"])
        if list(registered.shape) != prep["mask_shape_cyx"]:
            raise ValueError(f"{core}: registered HS shape changed")
        summed = registered.sum(axis=0)[y0:y1, x0:x1]
        if not np.array_equal(summed, total) or total.shape != (y1 - y0, x1 - x0):
            raise ValueError(f"{core}: cropped HS sum does not exactly reproduce accepted anchor")
        del registered, summed
        if (labels.shape != total.shape or support.shape != total.shape or support.dtype != bool
                or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= k):
            raise ValueError(f"{core}: invalid accepted label/support geometry")
        physical = extract_czi_metadata(native_path)
        layout = _layout(CziFile(str(native_path)))
        native_shape = [layout[axis][1] - layout[axis][0] for axis in ("Y", "X")]
        decimation = int(mapping["raw_hs_decimation"])
        prepared_shape = list(prep["stack_shape_cyx"][1:])
        if native_shape != [value * decimation for value in prepared_shape]:
            raise ValueError(f"{core}: recorded CZI layout/decimation does not reproduce prepared grid")
        pitch_x = float(physical["pixel_size_x_um"]) * decimation
        pitch_y = float(physical["pixel_size_y_um"]) * decimation
        geometry: dict[str, Any] = {
            "axes": "YX", "array_shape_yx": list(total.shape),
            "native_shape_yx": prepared_shape, "native_origin_xy": [x0, y0],
            "downsample_factor": 1, "block_center_offset_native_xy": [0, 0],
            "pixel_size_x_um": pitch_x, "pixel_size_y_um": pitch_y,
            "assigned_tma": "TMA1", "confirmed_core_id": core,
            "hs_source_sha256": str(raw_match.iloc[0]["sha256"]),
            "frame": "accepted cropped prepared registered-HS / FLIM YX",
            "raw_czi_shape_yx": native_shape, "raw_hs_decimation": decimation,
            "crop_bounds_prepared_yx": [y0, y1, x0, x1],
            "registered_hs_shift_yx": [float(mapping["shift_y"]), float(mapping["shift_x"])],
            "axis_provenance": accepted["axis_provenance"],
            "axis_interpretation": "CYX to XYC for fit, then YXC; returned rasters already YX; no display transpose applied",
            "source_mapping": mapping, "benchmark_receipt": source_bindings[0],
            "label_interpolation": "none; frozen labels retained on original grid; categorical supports nearest-neighbor",
            "continuous_interpolation": "VALIS bilinear after physical decimation Gaussian antialias; full footprint support",
        }
        geometry["grid_id"] = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
        refined, _, tissue_summary = derive_tissue_envelope(
            total, np.isfinite(total), pixel_size_x_um=pitch_x, pixel_size_y_um=pitch_y)
        fixed_support = support & refined & np.isfinite(total) & (total > 0)
        moving = valis._read_array(dapi.iloc[0]["reference_image_path"])
        observed = np.asarray(valis._read_array(dapi.iloc[0]["support_path"]), dtype=bool)
        moving_mask, _, moving_summary = derive_tissue_envelope(
            moving, observed, pixel_size_x_um=float(dapi.iloc[0]["pixel_size_x_um"]),
            pixel_size_y_um=float(dapi.iloc[0]["pixel_size_y_um"]))
        anchor_path, geometry_path = core_dir / "joint_sum_anchor.npz", core_dir / "joint_geometry.json"
        labels_path, moving_mask_path = core_dir / "joint_domains.npy", core_dir / "cycif_refined_tissue.npy"
        np.savez_compressed(anchor_path, total=total, support=fixed_support)
        np.save(labels_path, np.where(support, labels.astype(np.uint16) + 1, 0).astype(np.uint8))
        np.save(moving_mask_path, moving_mask)
        valis.save_json(geometry_path, geometry)
        row = {
            **dapi.iloc[0].to_dict(), "hs_features_path": str(anchor_path.resolve()),
            "hs_source_path": str(native_path), "hs_source_sha256": geometry["hs_source_sha256"],
            "geometry_path": str(geometry_path.resolve()),
            "geometry_sha256": valis.compute_file_sha256(geometry_path), "grid_id": geometry["grid_id"],
            "anchor_pixel_size_x_um": pitch_x, "anchor_pixel_size_y_um": pitch_y,
            "registration_tissue_mask_path": str(moving_mask_path.resolve()),
            "domain_anchor_kind": "joint_sum_hs", "domain_label": "Joint HS+FLIM domains",
            "section_relationship": "serial_sections_matched_cores", "section_evidence": SERIAL_SECTION_EVIDENCE,
        }
        rows.append(row)
        label_rows.append({
            "assigned_tma": "TMA1", "confirmed_core_id": core, "labels_path": str(labels_path.resolve()),
            "labels_sha256": valis.compute_file_sha256(labels_path),
            "label_geometry_path": str(geometry_path.resolve()), "geometry_sha256": row["geometry_sha256"],
            "palette_path": str(palette_path.resolve()), "palette_sha256": valis.compute_file_sha256(palette_path),
            "grid_id": geometry["grid_id"], "cluster_count": k, "method": "accepted_joint_HS_FLIM_PCA5_K9",
            "original_labels_path": str(originals["labels"]),
            "original_labels_sha256": artifact_hashes[originals["labels"].name],
            "id_mapping": "domain_id = raw fitted ID + 1 on benchmark support; ID0 outside support",
        })
        audit.append({"core": core, "passed": True, "raw_czi_path": str(native_path),
                      "grid_id": geometry["grid_id"], "pitch_x_um": pitch_x, "pitch_y_um": pitch_y,
                      "crop_bounds_yx": json.dumps([y0, y1, x0, x1]), "axis_provenance": json.dumps(accepted["axis_provenance"]),
                      "fixed_tissue_summary": json.dumps(tissue_summary), "moving_tissue_summary": json.dumps(moving_summary),
                      "section_relationship": "serial_sections_matched_cores", "section_evidence": SERIAL_SECTION_EVIDENCE,
                      "rotation_evidence": SERIAL_ROTATION_EVIDENCE, "claim_label": "Exploratory"})
        artifacts.extend([anchor_path, geometry_path, labels_path, moving_mask_path, *originals.values()])
    inputs, labels_manifest = pd.DataFrame(rows), pd.DataFrame(label_rows)
    inputs_path, labels_manifest_path = tables / "joint_registration_inputs.csv", tables / "joint_labels_manifest.csv"
    inputs.to_csv(inputs_path, index=False)
    labels_manifest.to_csv(labels_manifest_path, index=False)
    audit_path = manifests / "serial_pairing_geometry_audit.csv"
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    artifacts.extend([inputs_path, labels_manifest_path, audit_path])
    valis.save_json(preparation_receipt, {
        "identity": identity, "inputs_path": str(inputs_path.resolve()),
        "labels_manifest_path": str(labels_manifest_path.resolve()),
        "artifacts": [_serial_binding(path) for path in artifacts],
        "analysis_scope": "regional serial-section correspondence only; no identical-cell claims or P values",
    })
    return {"inputs": inputs, "labels_manifest": labels_manifest,
            "receipt_path": str(preparation_receipt), "cache_status": "COMPUTED"}


def _serial_structural_orientation(
    row: dict[str, Any], candidates: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank full-angle physical initializations by HS-sum/DAPI structure, not markers."""
    import clinical_roi_valis as valis
    with np.load(row["hs_features_path"], allow_pickle=False) as bundle:
        total = np.asarray(bundle["total"])
        fixed_mask = np.asarray(bundle["support"], dtype=bool)
    moving = np.asarray(valis._read_array(row["reference_image_path"]))
    moving_mask = np.asarray(valis._read_array(row["registration_tissue_mask_path"]), dtype=bool)
    fs = max(1, int(np.ceil(max(total.shape) / 256)))
    ms = max(1, int(np.ceil(max(moving.shape) / 256)))
    fixed = np.log1p(total[::fs, ::fs])
    fixed_mask = fixed_mask[::fs, ::fs]
    source = np.log1p(moving[::ms, ::ms])
    source_mask = moving_mask[::ms, ::ms]
    fixed_values, source_values = fixed[fixed_mask], source[source_mask]
    if min(fixed_values.size, source_values.size) < 64:
        raise ValueError("Insufficient structural tissue for angular screening")
    fixed_range = tuple(np.percentile(fixed_values, [1, 99]))
    moving_range = tuple(np.percentile(source_values, [1, 99]))
    if fixed_range[1] <= fixed_range[0] or moving_range[1] <= moving_range[0]:
        raise ValueError("Constant structural intensity cannot establish orientation")
    records: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        matrix = np.asarray(json.loads(candidate["matrix_yx_json"]))
        small = np.diag([1 / fs, 1 / fs, 1]) @ matrix @ np.diag([ms, ms, 1])
        warped = valis.warp_image_yx(source, small, fixed.shape)
        warped_mask = valis.warp_image_yx(source_mask, small, fixed.shape, categorical=True)
        valid = fixed_mask & warped_mask & np.isfinite(warped)
        mi = float("-inf")
        if valid.sum() >= 64:
            histogram = np.histogram2d(
                np.clip(fixed[valid], *fixed_range),
                np.clip(warped[valid], *moving_range), bins=32,
                range=[fixed_range, moving_range])[0]
            probability = histogram / histogram.sum()
            expected = probability.sum(axis=1)[:, None] * probability.sum(axis=0)[None, :]
            positive = probability > 0
            mi = float(np.sum(probability[positive] * np.log(probability[positive] / expected[positive])))
        records.append({**candidate, "structural_mutual_information": mi,
                        "structural_overlap_samples": int(valid.sum())})
    ranked = pd.DataFrame(records)
    # Outline limits prevent a tiny coincident fragment from winning the MI screen.
    ranked["outline_eligible"] = ranked["dice"] >= ranked["dice"].max() - 0.15
    eligible = ranked.loc[ranked["outline_eligible"] & np.isfinite(ranked["structural_mutual_information"])]
    if eligible.empty:
        raise ValueError("No usable full-angle structural candidate")
    eligible = eligible.sort_values(["structural_mutual_information", "dice"], ascending=False)
    best = eligible.iloc[0].to_dict()
    separation = np.abs((eligible["angle_degrees"] - best["angle_degrees"] + 180) % 360 - 180)
    distinct = eligible.loc[separation.ge(30)]
    best.update({
        "ambiguity_margin": float(eligible.iloc[0]["structural_mutual_information"] - eligible.iloc[1]["structural_mutual_information"])
        if len(eligible) > 1 else 0.0,
        "initialization_basis": "Full angular physical outline search; internal HS-sum/DAPI MI among outline-eligible candidates",
        "structural_distinct_30deg_margin": float(best["structural_mutual_information"] - distinct.iloc[0]["structural_mutual_information"])
        if not distinct.empty else 0.0,
        "fixed_center_yx_json": json.dumps(np.argwhere(fixed_mask).mean(axis=0).tolist()),
        "moving_center_yx_json": json.dumps(np.argwhere(source_mask).mean(axis=0).tolist()),
        "center_units": f"screen pixels; fixed stride={fs}, moving stride={ms}",
        "direction": "forward moving cycIF YX -> fixed cropped joint-map YX; positive angle follows existing orientation helper",
        "rotation_evidence": SERIAL_ROTATION_EVIDENCE,
        "reflection_evidence": "Reflection disabled; fit-axis transpose is documented and undone in accepted returned YX rasters",
    })
    return ranked, best


def fit_serial_section_registration(
    prepared: dict[str, Any],
    cycif_manifest_path: str | Path,
    *,
    object_dir: str | Path,
    table_dir: str | Path,
    plot_dir: str | Path,
    retry_failed: bool = True,
) -> dict[str, Any]:
    """Fit new full-angle structural initializations plus isolated bounded Mattes/VALIS.

    This phase never accepts a registration. Every success requires subsequent
    source-bound visual review; failed fits remain explicit exclusions.
    Explicit registration phases retry failed cores by default. Review-bound
    analysis uses retry_failed=False to replay a verified complete or partial
    receipt without rewriting its orientations, QC or mandatory review digest.
    Missing or changed replay inputs require an explicit registration refresh.
    """
    import clinical_roi_valis as valis
    valis.guard_compute()
    inputs = prepared["inputs"]
    tables, plots = Path(table_dir), Path(plot_dir)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    preparation = json.loads(Path(prepared["receipt_path"]).read_text())
    if not valis._verify_artifacts(preparation["artifacts"]):
        raise ValueError("Prepared serial-section artifacts changed")
    identity = {
        "preparation": _serial_binding(prepared["receipt_path"]),
        "cycif": _serial_binding(cycif_manifest_path),
        "structural_code": hashlib.sha256(inspect.getsource(_serial_structural_orientation).encode()).hexdigest(),
        "fit_code": hashlib.sha256(inspect.getsource(fit_serial_section_registration).encode()).hexdigest(),
        "registration_code": valis.registration_code_bindings(),
        "worker_runtime": valis.worker_runtime_identity(),
        "supplied_transform_adapter": _serial_binding(Path(valis.__file__).with_name("valis_supplied_transform.py")),
    }
    fit_receipt = tables / "joint_registration_receipt.json"
    warped_path = tables / "joint_warped_markers.csv"
    orientations_path = tables / "joint_structural_orientations.csv"
    exclusions_path = tables / "registration_fit_exclusions.csv"
    requested_cores = sorted(inputs["confirmed_core_id"].astype(str).unique())
    if fit_receipt.is_file():
        saved = json.loads(fit_receipt.read_text())
        if (saved["identity"] == identity
                and (saved.get("fit_complete") is True or not retry_failed)
                and saved.get("requested_cores") == requested_cores
                and valis._verify_artifacts(saved["artifacts"])):
            return {"warped": pd.read_csv(warped_path, keep_default_na=False),
                    "orientation_table": pd.read_csv(orientations_path, keep_default_na=False),
                    "receipt_path": str(fit_receipt), "qc_paths": saved["qc_paths"],
                    "cache_status": "CACHE_HIT" if saved.get("fit_complete") is True else "PARTIAL_CACHE_REPLAY"}
    if not retry_failed:
        raise RuntimeError(
            "Review-bound registration replay has no current source- and artifact-verified receipt; "
            "run the explicit registration phase and renew structural review before analysis"
        )
    valis.orientation_screen(inputs, table_dir=tables, allow_reflection=False)
    choices: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for row in inputs.to_dict("records"):
        path = tables / f"{row['assigned_tma']}_{row['confirmed_core_id']}_orientation_candidates.csv"
        ranked, best = _serial_structural_orientation(row, pd.read_csv(path))
        ranked.to_csv(path, index=False)
        candidate_paths.append(path)
        choices.append({"assigned_tma": row["assigned_tma"], "confirmed_core_id": row["confirmed_core_id"], **best})
    orientations = pd.DataFrame(choices)
    orientations.to_csv(orientations_path, index=False)
    review_path, _ = valis.ensure_review_inputs(inputs, table_dir=tables)
    orientation_qc = valis.render_orientation_qc(inputs, orientations, plot_dir=plots / "orientation")
    warped = valis.fit_and_warp_pairs(
        inputs, orientations, cycif_manifest_path, review_path,
        object_dir=Path(object_dir) / "registration", table_dir=tables, allow_exploratory=True)
    # Carry the descriptive-only adapter contract through the generic VALIS exporter.
    for key, value in {
        "domain_anchor_kind": "joint_sum_hs", "domain_label": "Joint HS+FLIM domains",
        "section_relationship": "serial_sections_matched_cores", "section_evidence": SERIAL_SECTION_EVIDENCE,
    }.items():
        warped[key] = value
    warped.to_csv(warped_path, index=False)
    qc_paths = [*orientation_qc, *valis.render_exploratory_registration_qc(warped, plot_dir=plots)]
    qc_table = valis.measure_registration_qc(warped, table_dir=tables)
    qc_table_path = tables / "joint_registration_metrics.csv"
    qc_table.to_csv(qc_table_path, index=False)
    failures = pd.read_csv(exclusions_path, keep_default_na=False)
    fitted_cores = sorted(warped["confirmed_core_id"].astype(str).unique())
    fit_complete = failures.empty and fitted_cores == requested_cores
    artifacts = [warped_path, orientations_path, qc_table_path, exclusions_path,
                 *candidate_paths, *qc_paths]
    artifacts.extend(Path(path) for path in failures["log_path"] if Path(path).is_file())
    for path in warped["warp_receipt_path"].unique():
        core_receipt = json.loads(Path(path).read_text())
        artifacts.extend(Path(item["path"]) for item in core_receipt["artifacts"])
        artifacts.append(Path(path))
    valis.save_json(fit_receipt, {
        "identity": identity, "artifacts": [_serial_binding(path) for path in artifacts],
        "fit_complete": fit_complete, "requested_cores": requested_cores,
        "fitted_cores": fitted_cores, "fit_failures": failures.to_dict("records"),
        "incomplete_cache_policy": "Retry failures on the next fit phase; reuse verified per-core successes",
        "qc_paths": [str(Path(path).resolve()) for path in qc_paths],
        "review_required": "Inspect anatomy, rim loss, full-angle alternatives and checkerboard before PASS; ambiguity/poor anatomy is FAIL",
        "registration_accepted": False,
    })
    return {"warped": warped, "orientation_table": orientations,
            "receipt_path": str(fit_receipt), "qc_paths": qc_paths,
            "cache_status": "COMPUTED" if fit_complete else "PARTIAL_RETRY_REQUIRED"}


def publish_serial_section_correspondence(
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    decisions: dict[str, Any],
    *,
    table_dir: str | Path,
    plot_dir: str | Path,
) -> dict[str, Any]:
    """Publish only visually accepted structural fits, then quantify every antibody.

    Decisions are a JSON-compatible mapping containing registration_receipt_sha256
    and decisions={core: {decision: PASS|FAIL, reason: actual image observations}}.
    No marker endpoint influences orientation, fitting or acceptance.
    """
    import clinical_roi_valis as valis
    import clinical_roi_pattern_recovery as recovery
    valis.guard_compute()
    receipt = json.loads(Path(fitted["receipt_path"]).read_text())
    if not valis._verify_artifacts(receipt["artifacts"]):
        raise ValueError("Registration/QC artifacts changed after fitting")
    if decisions.get("registration_receipt_sha256") != valis.compute_file_sha256(fitted["receipt_path"]):
        raise ValueError("Visual decisions are not bound to the current fitted registration receipt")
    review = decisions.get("decisions", {})
    warped = fitted["warped"]
    fitted_cores = set(warped["confirmed_core_id"])
    if set(review) != fitted_cores:
        raise ValueError("Supply exactly one explicit PASS/FAIL review for every fitted core")
    accepted_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for core in sorted(fitted_cores):
        decision = review[core]
        if decision.get("decision") not in {"PASS", "FAIL"} or not str(decision.get("reason", "")).strip():
            raise ValueError(f"{core}: actual structural image observations and PASS/FAIL required")
        subset = warped.loc[warped["confirmed_core_id"].eq(core)]
        dapi = subset.loc[subset["marker"].eq("DAPI")]
        if len(dapi) != 1 or subset["warp_receipt_sha256"].nunique() != 1:
            raise ValueError(f"{core}: driver/transform identity is ambiguous")
        review_rows.append({
            "assigned_tma": "TMA1", "confirmed_core_id": core, **decision,
            "registration_receipt_sha256": decisions["registration_receipt_sha256"],
            "warp_receipt_sha256": subset.iloc[0]["warp_receipt_sha256"],
            "section_evidence": SERIAL_SECTION_EVIDENCE,
            "acceptance_scope": "visual regional structural anatomy only; no independent TRE or cell identity",
        })
        if decision["decision"] == "FAIL":
            continue
        antibodies = subset.loc[~subset["marker"].eq("DAPI")]
        if len(antibodies) != 20 or antibodies["marker"].duplicated().any():
            raise ValueError(f"{core}: all 20 antibody endpoints are required")
        label = prepared["labels_manifest"].loc[prepared["labels_manifest"]["confirmed_core_id"].eq(core)]
        if len(label) != 1:
            raise ValueError(f"{core}: missing exact accepted joint labels")
        for row in antibodies.to_dict("records"):
            if row["grid_id"] != label.iloc[0]["grid_id"]:
                raise ValueError(f"{core}: label and quantitative marker grid IDs disagree")
            recovery.load_domain_marker_pair(row)
            accepted_rows.append({
                **row, "registration_accepted": True, "orientation_review": "PASS",
                "visual_review_notes": decision["reason"], "group_id": f"TMA1_{core}",
                "claim_label": "Exploratory", "registration_driver": False,
                "image_sha256": valis.compute_file_sha256(row["image_path"]),
                "support_sha256": valis.compute_file_sha256(row["support_path"]),
            })
    tables = Path(table_dir)
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(tables / "joint_structural_visual_reviews.csv", index=False)
    if not accepted_rows:
        raise ValueError("No structural registration passed review; no marker analysis was substituted")
    manifest = pd.DataFrame(accepted_rows)
    manifest_path = tables / "joint_registered_antibody_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    labels = prepared["labels_manifest"].loc[prepared["labels_manifest"]["confirmed_core_id"].isin(manifest["confirmed_core_id"])]
    atlas = recovery.run_frozen_domain_atlas(
        manifest, labels, table_dir=tables / "correspondence", plot_dir=Path(plot_dir) / "correspondence")
    atlas["summary"].attrs["source_receipts"] = {
        "preparation": _serial_binding(prepared["receipt_path"]),
        "registration": _serial_binding(fitted["receipt_path"]),
        "manifest": _serial_binding(manifest_path),
        "visual_review": _serial_binding(tables / "joint_structural_visual_reviews.csv"),
    }
    return {"manifest": manifest, "atlas": atlas, "manifest_path": manifest_path}
