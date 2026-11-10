"""General raw-SDT QC, physical-time phasors, explicit masks and lifetime stages.

SDT block index is not a biological identity or a guaranteed physical detector.
All stages bind raw files, effective parameters, code, dependencies and exports.
No input shape, channel count, file naming convention or laser frequency is assumed.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, BinaryIO
import zipfile

import numpy as np
import pandas as pd
from phasorpy.filter import phasor_filter_median
from phasorpy.lifetime import phasor_calibrate, phasor_to_apparent_lifetime
from phasorpy.phasor import phasor_from_signal
from threadpoolctl import threadpool_limits


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024**2):
            digest.update(data)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sdt() -> Any:
    try:
        import sdtfile.sdtfile as module
    except ImportError:
        workspace = next((p for p in Path(__file__).resolve().parents if (p / '00_environments').is_dir()), None)
        if workspace is None:
            raise RuntimeError('Install sdtfile in the active analysis environment')
        dependency = workspace / '00_environments/python-packages/sdtfile-2024.5.24'
        if not dependency.is_dir():
            raise RuntimeError(f'The SDT decoder dependency is unavailable: {dependency}')
        sys.path.insert(0, str(dependency))
        import sdtfile.sdtfile as module
    return module


def supported_flim_formats() -> dict[str, str]:
    return {'.sdt': 'SUPPORTED: SDT decay/image blocks with audited TAC timing',
            '.ptu': 'NOT_IMPLEMENTED in this workflow; no implicit conversion',
            '.fbd': 'NOT_IMPLEMENTED in this workflow; no implicit conversion',
            '.flim': 'NOT_IMPLEMENTED in this workflow; no implicit conversion'}


def _versions() -> dict[str, str]:
    result = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'phasorpy', 'scipy', 'threadpoolctl')}
    result['sdtfile'] = _sdt().__version__
    result['python'] = sys.version
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


class _ByteRange(io.RawIOBase):
    """Seekable view of one on-disk SDT ZIP block, without buffering the archive."""

    def __init__(self, path: Path, offset: int, length: int) -> None:
        super().__init__()
        self._stream: BinaryIO = path.open('rb')
        self._offset = offset
        self._length = length
        self._position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError('Invalid SDT block seek origin')
        target = offset if whence == io.SEEK_SET else (
            self._position + offset if whence == io.SEEK_CUR else self._length + offset)
        if target < 0:
            raise ValueError('Negative SDT block seek')
        self._position = target
        return target

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self._length - self._position)
        count = remaining if size < 0 else min(size, remaining)
        self._stream.seek(self._offset + self._position)
        result = self._stream.read(count)
        self._position += len(result)
        return result

    def close(self) -> None:
        self._stream.close()
        super().close()


def _inspect_sdt(path: Path, digest: str) -> list[dict[str, Any]]:
    """Interpret the official sdtfile header structures; read no decay cubes."""
    sdt = _sdt()
    acquisition_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    records: list[dict[str, Any]] = []
    with path.open('rb') as stream:
        header = np.rec.fromfile(stream, dtype=sdt.FILE_HEADER, shape=1, byteorder='<')[0]
        if header.chksum != 0x55AA and header.header_valid != 0x5555:
            raise ValueError(f'Invalid SDT header: {path}')
        count = int(header.reserved1 if header.no_of_data_blocks == 0x7FFF else header.no_of_data_blocks)
        if count <= 0:
            raise ValueError(f'No SDT data blocks: {path}')
        measure_type = sdt.struct_dtype(sdt.MEASURE_INFO, int(header.meas_desc_block_length))
        measures = []
        for index in range(int(header.no_of_meas_desc_blocks)):
            stream.seek(int(header.meas_desc_block_offs) + index * int(header.meas_desc_block_length))
            measures.append(np.rec.fromfile(stream, dtype=measure_type, shape=1, byteorder='<')[0])
        block_type = sdt.BLOCK_HEADER if sdt.FileRevision(header.revision).revision >= 15 else sdt.BLOCK_HEADER_OLD
        offset = int(header.data_block_offs)
        for index in range(count):
            stream.seek(offset)
            block = np.rec.fromfile(stream, dtype=block_type, shape=1, byteorder='<')[0]
            measure = measures[int(block.meas_desc_block_no)]
            layout = sdt.BlockType(block.block_type)
            dtype = np.dtype(layout.dtype)
            if dtype.kind not in 'uif':
                raise ValueError(f'{path} block{index}: unsupported histogram dtype {dtype}')
            if layout.contents == 'MCS_BLOCK':
                raise ValueError(f'{path} block{index}: MCS counts are not TCSPC decay histograms')
            bins = int(measure.adc_re) or 65536
            values = int(block.block_length) // dtype.itemsize
            fields = set(measure.dtype.names or ())
            scan_x = int(measure.scan_x) if 'scan_x' in fields else 0
            scan_y = int(measure.scan_y) if 'scan_y' in fields else 0
            image_x = int(measure.image_x) if 'image_x' in fields else 0
            image_y = int(measure.image_y) if 'image_y' in fields else 0
            if scan_x > 0 and scan_y > 0 and values == scan_x * scan_y * bins:
                shape = [scan_y, scan_x]
                kind = 'IMAGE_YX'
            elif image_x > 0 and image_y > 0 and values == image_x * image_y * bins:
                shape = [image_y, image_x]
                kind = 'IMAGE_YX'
            elif bins > 1 and values % bins == 0:
                shape = [values // bins]
                kind = 'POINT_OR_LINE; spatial axes not established'
            else:
                raise ValueError(f'{path} block{index}: counts do not match a supported decay layout')
            if bins < 3 or int(measure.tac_g) <= 0:
                raise ValueError(f'{path} block{index}: invalid TCSPC timing metadata')
            step = float(measure.tac_r) / (int(measure.tac_g) * bins)
            if not np.isfinite(step) or step <= 0:
                raise ValueError(f'{path} block{index}: nonpositive time step')
            next_offset = int(block.next_block_offs)
            data_offset = int(block.data_offs)
            span = (next_offset if next_offset > data_offset else path.stat().st_size) - data_offset
            stop_info = measure.StopInfo if 'StopInfo' in fields else None
            record = {
                'acquisition_id': acquisition_id, 'detector_key': f'{acquisition_id}_d{index}',
                'path': str(path), 'source_sha256': digest, 'size_bytes': path.stat().st_size,
                'file_name': path.name, 'detector_index': index, 'block_index': index,
                'selection_kind': 'SDT data block; physical detector identity requires metadata/review',
                'native_detector_code': int(measure.det) if 'det' in fields else None,
                'data_block_count': count, 'shape_json': json.dumps(shape), 'layout_kind': kind,
                'pixel_count': math.prod(shape), 'time_bins': bins, 'dtype': dtype.str,
                'time_step_seconds': step, 'window_seconds': step * bins,
                'window_frequency_mhz': 1e-6 / (step * bins),
                'time_origin': 'sdtfile native timing starts at zero; instrument phase offset uncalibrated',
                'tac_range_seconds': float(measure.tac_r), 'tac_gain': int(measure.tac_g),
                'tac_offset_native': _finite(measure.tac_of) if 'tac_of' in fields else None,
                'recorded_min_sync_rate_hz': _finite(stop_info.min_sync_rate) if stop_info is not None else None,
                'recorded_max_sync_rate_hz': _finite(stop_info.max_sync_rate) if stop_info is not None else None,
                'configured_excitation_frequency_mhz': None, 'calibration_status': 'UNCALIBRATED',
                'irf_reference_status': 'NOT_CONFIGURED', 'compressed': bool(layout.compress),
                'pixel_size_y_um': None, 'pixel_size_x_um': None,
                'physical_scale_status': 'NOT_ESTABLISHED_FROM_SDT_TIMING_METADATA',
                'data_offset': data_offset, 'compressed_span_bytes': span,
                'uncompressed_bytes': int(block.block_length), 'block_type': int(block.block_type),
                'claim_label': 'Exploratory',
            }
            records.append(record)
            offset = next_offset
    return records


def iter_flim_histogram_chunks(
        record: dict[str, Any], *, chunk_pixels: int = 8192,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield flattened pixel-by-time chunks for any audited SDT image/point block."""
    if not isinstance(chunk_pixels, int) or chunk_pixels <= 0:
        raise ValueError('chunk_pixels must be a positive integer')
    path = Path(record['path'])
    bins, pixels = int(record['time_bins']), int(record['pixel_count'])
    dtype = np.dtype(record['dtype'])
    # Bound the histogram chunk independently of the requested spatial chunk size.
    if chunk_pixels * bins * dtype.itemsize > 256 * 1024**2:
        raise MemoryError('Requested TCSPC chunk exceeds256MiB; lower chunk_pixels')
    if record['compressed']:
        with _ByteRange(path, int(record['data_offset']), int(record['compressed_span_bytes'])) as block:
            with zipfile.ZipFile(block) as archive:
                if len(archive.filelist) != 1 or archive.filelist[0].file_size != record['uncompressed_bytes']:
                    raise ValueError('SDT compressed block differs from audited histogram extent')
                with archive.open(archive.filelist[0]) as stream:
                    for start in range(0, pixels, chunk_pixels):
                        count = min(chunk_pixels, pixels - start)
                        raw = stream.read(count * bins * dtype.itemsize)
                        if len(raw) != count * bins * dtype.itemsize:
                            raise ValueError('Truncated SDT histogram block')
                        yield start, np.frombuffer(raw, dtype=dtype).reshape(count, bins)
                    if stream.read(1):
                        raise ValueError('Unexpected data after audited SDT block')
    else:
        with path.open('rb') as stream:
            stream.seek(int(record['data_offset']))
            for start in range(0, pixels, chunk_pixels):
                count = min(chunk_pixels, pixels - start)
                array = np.fromfile(stream, dtype=dtype, count=count * bins)
                if array.size != count * bins:
                    raise ValueError('Truncated SDT histogram block')
                yield start, array.reshape(count, bins)


def _verify_sources(records: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    unique = {record['path']: record['source_sha256'] for record in records}
    signatures = {}
    for name, expected in unique.items():
        path = Path(name)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if _sha(path) != expected:
            raise ValueError(f'Raw source changed; rerun inventory: {path}')
        signatures[name] = signature
    _require_unchanged(signatures)
    return signatures


def _require_unchanged(signatures: dict[str, tuple[int, int, int]]) -> None:
    for name, expected in signatures.items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != expected:
            raise ValueError(f'Raw source changed during stage: {name}')


def _cache_start(
        stage: str, identity: dict[str, Any], object_dir: Path, force: bool,
) -> tuple[Path, Path, dict[str, Any] | None]:
    identity = identity | {'stage': stage, 'code_sha256': _sha(Path(__file__)), 'versions': _versions()}
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    root = Path(object_dir) / stage / key
    receipt_path = root / 'receipt.json'
    if not force and receipt_path.is_file():
        try:
            saved = json.loads(receipt_path.read_text())
            if saved['identity'] == identity and all(
                    Path(a['path']).is_file() and _sha(Path(a['path'])) == a['sha256'] for a in saved['artifacts']):
                print(f'CACHE_HIT {stage}: {key[:12]}', flush=True)
                return root, receipt_path, _result(saved, receipt_path, 'CACHE_HIT')
        except (OSError, ValueError, KeyError):
            pass
    print(f'{"FORCE_RECOMPUTE" if force else "CACHE_MISS_RECOMPUTED"} {stage}: {key[:12]}', flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'identity.json').write_text(_json(identity))
    return root, receipt_path, None


def _finish(
        root: Path, receipt_path: Path, summary: list[dict[str, Any]], detectors: dict[str, dict[str, Any]],
        *, table_dir: Path, stage: str, force: bool = False,
) -> dict[str, Any]:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    # Content-addressed tables prevent one configuration from invalidating another cache.
    table = table_dir / f'{stage}_{root.name[:12]}.csv'
    pd.DataFrame(summary).to_csv(table, index=False)
    artifacts = [table, *[Path(item['object_path']) for item in detectors.values()]]
    receipt = {'identity': json.loads((root / 'identity.json').read_text()), 'stage': stage,
               'summary_path': str(table), 'detectors': detectors, 'claim_label': 'Exploratory',
               'generating_script': str(Path(__file__).resolve()),
               'artifacts': [{'path': str(p), 'sha256': _sha(p), 'size_bytes': p.stat().st_size} for p in artifacts]}
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return _result(receipt, receipt_path, 'FORCE_RECOMPUTE' if force else 'CACHE_MISS_RECOMPUTED')


def _result(receipt: dict[str, Any], path: Path, status: str) -> dict[str, Any]:
    return {'summary': pd.read_csv(receipt['summary_path']), 'detectors': receipt['detectors'],
            'receipt_path': path, 'cache_status': status, 'stage': receipt['stage']}


def _parent_identity(result: dict[str, Any]) -> dict[str, Any]:
    path = Path(result['receipt_path'])
    receipt = json.loads(path.read_text())
    for artifact in receipt['artifacts']:
        if _sha(Path(artifact['path'])) != artifact['sha256']:
            raise ValueError(f'Parent stage artifact changed: {artifact["path"]}')
    return {'parent_receipt': str(path), 'parent_sha256': _sha(path)}


def load_flim_detector(result: dict[str, Any], detector_key: str) -> dict[str, np.ndarray]:
    with np.load(result['detectors'][detector_key]['object_path'], allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def _clear_lifetimes(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    """Changed phasors require a fresh lifetime-conversion stage."""
    for field in ('phase_lifetime_ns', 'modulation_lifetime_ns'):
        arrays.pop(field, None)
    return {key: value for key, value in metadata.items()
            if key not in {'lifetime_status', 'lifetime_interpretation', 'period_error_bins'}}


def inspect_flim_inputs(
        paths: Sequence[str | Path], *, cache_dir: str | Path,
        detector_indices: Sequence[int] | None = None, force: bool = False,
) -> dict[str, Any]:
    """List every block, and explicitly mark requested blocks; no image data are read."""
    sources = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Supply a nonempty unique explicit list of raw SDT paths')
    if any(path.suffix.lower() != '.sdt' for path in sources):
        raise ValueError(f'Unsupported raw FLIM format. Workflow support: {supported_flim_formats()}')
    if detector_indices is not None and any(not isinstance(i, int) or isinstance(i, bool) for i in detector_indices):
        raise ValueError('SDT block indices must be integers')
    selections = None if detector_indices is None else tuple(detector_indices)
    if selections is not None and (not selections or len(set(selections)) != len(selections) or min(selections) < 0):
        raise ValueError('Use explicit unique nonnegative SDT block indices')
    hashes = [{'path': str(path), 'sha256': _sha(path)} for path in sources]
    identity = {'sources': hashes, 'selected_blocks': list(selections) if selections is not None else 'ALL'}
    root, receipt, cached = _cache_start('flim_inventory', identity, Path(cache_dir), force)
    if cached is None:
        records = []
        for source in hashes:
            rows = _inspect_sdt(Path(source['path']), source['sha256'])
            available = {row['block_index'] for row in rows}
            if selections is not None and not set(selections).issubset(available):
                raise ValueError(f'{source["path"]}: selected block absent; available {sorted(available)}')
            for row in rows:
                row['selected'] = selections is None or row['block_index'] in selections
            records.extend(rows)
        manifest_path = root / 'inventory_records.json'
        manifest_path.write_text(json.dumps(records, indent=2, allow_nan=False))
        result = _finish(root, receipt, records, {}, table_dir=root, stage='flim_inventory', force=force)
        # Bind the exact structured metadata used downstream, not CSV dtype inference.
        saved = json.loads(receipt.read_text())
        saved['records_path'] = str(manifest_path)
        saved['artifacts'].append({'path': str(manifest_path), 'sha256': _sha(manifest_path), 'size_bytes': manifest_path.stat().st_size})
        receipt.write_text(json.dumps(saved, indent=2, allow_nan=False))
    else:
        result = cached
    saved = json.loads(receipt.read_text())
    result['records'] = json.loads(Path(saved['records_path']).read_text())
    result['manifest'] = pd.DataFrame(result['records'])
    result['supported_formats'] = supported_flim_formats()
    return result


def _selected(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records = [dict(row) for row in inventory['records'] if row['selected']]
    if not records:
        raise ValueError('No selected SDT blocks')
    _parent_identity(inventory)
    return records


def select_flim_blocks(inventory: dict[str, Any], block_indices: Sequence[int]) -> dict[str, Any]:
    """Declare blocks after inspecting the complete inventory; preserve its source receipt."""
    if any(not isinstance(i, int) or isinstance(i, bool) for i in block_indices):
        raise ValueError('SDT block indices must be integers')
    selected = tuple(block_indices)
    if not selected or min(selected) < 0 or len(set(selected)) != len(selected):
        raise ValueError('Declare unique nonnegative SDT block indices')
    for path in {row['path'] for row in inventory['records']}:
        available = {row['block_index'] for row in inventory['records'] if row['path'] == path}
        if not set(selected).issubset(available):
            raise ValueError(f'{path}: selected block absent; available {sorted(available)}')
    records = [row | {'selected': row['block_index'] in selected} for row in inventory['records']]
    return inventory | {'records': records, 'manifest': pd.DataFrame(records)}


def _raw_stage(
        inventory: dict[str, Any], *, object_dir: Path, table_dir: Path, chunk_pixels: int,
        frequency_mhz: float | None, frequency_provenance: str | None, force: bool,
        max_working_bytes: int,
) -> dict[str, Any]:
    if not isinstance(max_working_bytes, int) or max_working_bytes <= 0:
        raise ValueError('max_working_bytes must be a positive integer')
    records = _selected(inventory)
    signatures = _verify_sources(records)
    stage = 'flim_phasors' if frequency_mhz is not None else 'flim_qc'
    identity = _parent_identity(inventory) | {'records': records, 'chunk_pixels': chunk_pixels,
                'max_working_bytes': max_working_bytes,
                'frequency_mhz': frequency_mhz, 'frequency_provenance': frequency_provenance,
                'phasor_time_basis': 'native_bin_times_at_configured_frequency' if frequency_mhz else None}
    root, receipt, cached = _cache_start(stage, identity, object_dir, force)
    if cached is not None:
        _require_unchanged(signatures)
        return cached
    summaries, detectors = [], {}
    for record in records:
        key = record['detector_key']
        print(f'{stage}: {record["file_name"]} block{record["block_index"]}', flush=True)
        pixels, bins = int(record['pixel_count']), int(record['time_bins'])
        bound = pixels * 64 + chunk_pixels * bins * 32 + 64 * 1024**2
        if bound > max_working_bytes:
            raise MemoryError(f'{key}: map/chunk working bound {bound} exceeds {max_working_bytes}; '
                              'reduce chunk_pixels or explicitly request an adequate allocated-memory budget')
        shape = tuple(json.loads(record['shape_json']))
        intensity = np.empty(pixels, dtype=np.float64)
        decay = np.zeros(bins, dtype=np.float64)
        time = np.arange(bins, dtype=np.float64) * record['time_step_seconds']
        real = np.full(pixels, np.nan, dtype=np.float64) if frequency_mhz else None
        imag = np.full(pixels, np.nan, dtype=np.float64) if frequency_mhz else None
        phase = 2 * np.pi * frequency_mhz * 1e6 * time if frequency_mhz else None
        maximum_bin = 0.
        with threadpool_limits(limits=1):
            for start, data in iter_flim_histogram_chunks(record, chunk_pixels=chunk_pixels):
                if not np.isfinite(data).all() or (data < 0).any():
                    raise ValueError('Raw decay histograms must contain finite nonnegative counts')
                counts = data.sum(axis=-1, dtype=np.float64)
                intensity[start:start + len(data)] = counts
                decay += data.sum(axis=0, dtype=np.float64)
                maximum_bin = max(maximum_bin, float(data.max()))
                if frequency_mhz:
                    _, g, s = phasor_from_signal(data, axis=-1, harmonic=1, sample_phase=phase,
                                                 use_fft=False, num_threads=1)
                    assert real is not None and imag is not None
                    real[start:start + len(data)], imag[start:start + len(data)] = g, s
        arrays = {'intensity': intensity.reshape(shape), 'global_decay': decay, 'time_seconds': time}
        metadata = record | {'frequency_mhz': frequency_mhz, 'frequency_provenance': frequency_provenance,
                              'calibration_status': 'UNCALIBRATED', 'minimum_total_counts': 0.,
                              'phasor_time_basis': 'native_bin_times_at_configured_frequency' if frequency_mhz else None}
        if real is not None and imag is not None:
            valid = (intensity > 0) & np.isfinite(real) & np.isfinite(imag)
            real[~valid], imag[~valid] = np.nan, np.nan
            arrays.update(real=real.reshape(shape), imag=imag.reshape(shape), valid=valid.reshape(shape))
        output = root / f'{key}.npz'
        np.savez_compressed(output, **arrays)
        detectors[key] = metadata | {'object_path': str(output)}
        summaries.append(metadata | {'total_counts': float(intensity.sum()), 'zero_pixels': int((intensity == 0).sum()),
                                     'positive_pixels': int((intensity > 0).sum()), 'mean_counts': float(intensity.mean()),
                                     'maximum_counts': float(intensity.max()), 'maximum_time_bin_count': maximum_bin,
                                     'decay_peak_bin': int(decay.argmax()),
                                     'valid_pixels': int(arrays['valid'].sum()) if 'valid' in arrays else int((intensity > 0).sum())})
    _require_unchanged(signatures)
    return _finish(root, receipt, summaries, detectors, table_dir=table_dir, stage=stage, force=force)


def summarize_flim_qc(
        inventory: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        chunk_pixels: int = 8192, max_working_bytes: int = 4 * 1024**3, force: bool = False,
) -> dict[str, Any]:
    return _raw_stage(inventory, object_dir=Path(object_dir), table_dir=Path(table_dir), chunk_pixels=chunk_pixels,
                      frequency_mhz=None, frequency_provenance=None, force=force, max_working_bytes=max_working_bytes)


def compute_flim_phasors(
        inventory: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        frequency_mhz: float, frequency_provenance: str, chunk_pixels: int = 8192, force: bool = False,
        max_working_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    if not np.isfinite(frequency_mhz) or frequency_mhz <= 0 or not frequency_provenance.strip():
        raise ValueError('Supply a positive excitation frequency and its explicit provenance')
    return _raw_stage(inventory, object_dir=Path(object_dir), table_dir=Path(table_dir), chunk_pixels=chunk_pixels,
                      frequency_mhz=float(frequency_mhz), frequency_provenance=frequency_provenance, force=force,
                      max_working_bytes=max_working_bytes)


def _derived_start(
        result: dict[str, Any], stage: str, parameters: dict[str, Any], object_dir: Path, force: bool,
) -> tuple[Path, Path, dict[str, Any] | None]:
    _verify_sources(list(result['detectors'].values()))
    return _cache_start(stage, _parent_identity(result) | parameters, object_dir, force)


def mask_flim_phasors(
        result: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        minimum_total_counts: float, force: bool = False,
) -> dict[str, Any]:
    if not np.isfinite(minimum_total_counts) or minimum_total_counts < 0:
        raise ValueError('The count threshold must be finite and nonnegative')
    root, receipt, cached = _derived_start(result, 'flim_count_mask', {'minimum_total_counts': float(minimum_total_counts)}, Path(object_dir), force)
    if cached is not None:
        return cached
    summaries, detectors = [], {}
    for key, metadata in result['detectors'].items():
        arrays = load_flim_detector(result, key)
        if minimum_total_counts < metadata['minimum_total_counts']:
            raise ValueError('To relax a count threshold, restart from the raw phasor result')
        metadata = _clear_lifetimes(arrays, metadata)
        valid = arrays['valid'] & (arrays['intensity'] >= minimum_total_counts) & (arrays['intensity'] > 0)
        arrays['real'][~valid], arrays['imag'][~valid] = np.nan, np.nan
        arrays['valid'] = valid
        output = root / f'{key}.npz'
        np.savez_compressed(output, **arrays)
        updated = metadata | {'object_path': str(output), 'minimum_total_counts': float(minimum_total_counts)}
        detectors[key] = updated
        summaries.append(updated | {'valid_pixels': int(valid.sum()), 'excluded_pixels': int(valid.size - valid.sum())})
    return _finish(root, receipt, summaries, detectors, table_dir=Path(table_dir), stage='flim_count_mask', force=force)


def filter_flim_phasors(
        result: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        size: int = 3, repeat: int = 1, force: bool = False,
) -> dict[str, Any]:
    if not isinstance(size, int) or size < 1 or size % 2 != 1 or not isinstance(repeat, int) or repeat < 1:
        raise ValueError('Median size must be a positive odd integer and repeat a positive integer')
    root, receipt, cached = _derived_start(result, 'flim_median_filter', {'size': size, 'repeat': repeat}, Path(object_dir), force)
    if cached is not None:
        return cached
    summaries, detectors = [], {}
    for key, metadata in result['detectors'].items():
        arrays = load_flim_detector(result, key)
        metadata = _clear_lifetimes(arrays, metadata)
        if arrays['real'].ndim != 2:
            raise ValueError('Spatial median filtering requires an identified YX image, not point/line data')
        original_valid = arrays['valid'].copy()
        _, real, imag = phasor_filter_median(arrays['intensity'], arrays['real'], arrays['imag'],
                                            size=size, repeat=repeat, use_scipy=False, num_threads=1)
        valid = original_valid & np.isfinite(real) & np.isfinite(imag)
        real[~valid], imag[~valid] = np.nan, np.nan
        arrays.update(real=real, imag=imag, valid=valid)
        output = root / f'{key}.npz'
        np.savez_compressed(output, **arrays)
        updated = metadata | {'object_path': str(output), 'filter': 'median_phasor_coordinates',
                               'filter_size': size, 'filter_repeat': repeat}
        detectors[key] = updated
        summaries.append(updated | {'valid_pixels': int(valid.sum()),
                                     'original_mask_preserved': bool(np.array_equal(valid, original_valid))})
    return _finish(root, receipt, summaries, detectors, table_dir=Path(table_dir), stage='flim_median_filter', force=force)


def _period_check(metadata: dict[str, Any], tolerance_bins: float) -> tuple[bool, float]:
    if metadata.get('frequency_mhz') is None:
        raise ValueError('Compute phasors with an explicit configured frequency before lifetime or calibration stages')
    period = 1e-6 / float(metadata['frequency_mhz'])
    error_bins = abs(float(metadata['window_seconds']) - period) / float(metadata['time_step_seconds'])
    return error_bins <= tolerance_bins, error_bins


def calibrate_flim_phasors(
        result: dict[str, Any], reference_result: dict[str, Any], *,
        object_dir: str | Path, table_dir: str | Path, reference_detector_key: str,
        known_lifetime_ns: float, calibration_provenance: str, detector_keys: Sequence[str],
        instrument_settings_match_confirmed: bool = False, force: bool = False,
) -> dict[str, Any]:
    if not np.isfinite(known_lifetime_ns) or known_lifetime_ns < 0 or not calibration_provenance.strip():
        raise ValueError('A measured reference, nonnegative known lifetime and calibration provenance are required')
    if not instrument_settings_match_confirmed:
        raise ValueError('Explicitly confirm that reference and targets share the relevant detector, timing and optical settings')
    targets = list(detector_keys)
    if not targets or len(set(targets)) != len(targets) or not set(targets).issubset(result['detectors']):
        raise ValueError('Declare the unique target blocks to which this reference applies')
    reference_metadata = reference_result['detectors'][reference_detector_key]
    _verify_sources([reference_metadata])
    if reference_metadata['calibration_status'] != 'UNCALIBRATED':
        raise ValueError('Reference calibration requires uncalibrated reference phasors')
    if not _period_check(reference_metadata, 1.)[0]:
        raise ValueError('Reference TCSPC window does not span one configured excitation period')
    reference_identity = _parent_identity(reference_result)
    parameters = {'reference': reference_identity, 'reference_detector_key': reference_detector_key,
                  'known_lifetime_ns': float(known_lifetime_ns), 'calibration_provenance': calibration_provenance,
                  'target_detector_keys': targets, 'instrument_settings_match_confirmed': True,
                  'reference_aggregation': 'count-weighted mean of valid reference pixels'}
    root, receipt, cached = _derived_start(result, 'flim_reference_calibration', parameters, Path(object_dir), force)
    if cached is not None:
        return cached
    ref = load_flim_detector(reference_result, reference_detector_key)
    eligible = ref['valid'] & np.isfinite(ref['real']) & np.isfinite(ref['imag']) & (ref['intensity'] > 0)
    if not eligible.any():
        raise ValueError('Reference contains no valid positive-count pixels')
    weights = ref['intensity'][eligible]
    ref_g, ref_s = float(np.average(ref['real'][eligible], weights=weights)), float(np.average(ref['imag'][eligible], weights=weights))
    if np.hypot(ref_g, ref_s) <= np.sqrt(np.finfo(float).eps):
        raise ValueError('The pooled reference phasor is too near zero for stable calibration')
    summaries, detectors = [], {}
    for key, metadata in result['detectors'].items():
        arrays = load_flim_detector(result, key)
        metadata = _clear_lifetimes(arrays, metadata)
        updated = dict(metadata)
        if key in targets:
            if metadata['calibration_status'] != 'UNCALIBRATED':
                raise ValueError('Target is already calibrated; start from the uncalibrated stage')
            if not math.isclose(metadata['frequency_mhz'], reference_metadata['frequency_mhz'], rel_tol=1e-12):
                raise ValueError('Reference and target excitation frequencies differ')
            if not _period_check(metadata, 1.)[0]:
                raise ValueError('Target TCSPC window does not span one configured excitation period')
            real, imag = phasor_calibrate(arrays['real'], arrays['imag'], float(weights.sum()), ref_g, ref_s,
                                           frequency=metadata['frequency_mhz'], lifetime=known_lifetime_ns,
                                           unit_conversion=1e-3)
            valid = arrays['valid'] & np.isfinite(real) & np.isfinite(imag)
            real[~valid], imag[~valid] = np.nan, np.nan
            arrays.update(real=real, imag=imag, valid=valid)
            updated.update(calibration_status='REFERENCE_CALIBRATED', calibration_provenance=calibration_provenance,
                           reference_detector_key=reference_detector_key, reference_source_sha256=reference_metadata['source_sha256'],
                           reference_known_lifetime_ns=float(known_lifetime_ns))
        output = root / f'{key}.npz'
        np.savez_compressed(output, **arrays)
        updated['object_path'] = str(output)
        detectors[key] = updated
        summaries.append(updated | {'valid_pixels': int(arrays['valid'].sum())})
    return _finish(root, receipt, summaries, detectors, table_dir=Path(table_dir), stage='flim_reference_calibration', force=force)


def compute_flim_lifetimes(
        result: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        period_tolerance_bins: float = 1., allow_period_mismatch: bool = False, force: bool = False,
) -> dict[str, Any]:
    if not np.isfinite(period_tolerance_bins) or period_tolerance_bins < 0:
        raise ValueError('Timing coverage tolerance must be finite and nonnegative')
    for key, metadata in result['detectors'].items():
        passed, error = _period_check(metadata, period_tolerance_bins)
        if not passed and not allow_period_mismatch:
            raise ValueError(f'{key}: TCSPC window differs from one configured period by {error:.4g} bins. '
                             'No lifetime conversion; do not infer laser frequency from window length. '
                             'Explicit allow_period_mismatch=True retains phasors but exports NaN lifetimes.')
    parameters = {'period_tolerance_bins': float(period_tolerance_bins), 'allow_period_mismatch': bool(allow_period_mismatch)}
    root, receipt, cached = _derived_start(result, 'flim_apparent_lifetimes', parameters, Path(object_dir), force)
    if cached is not None:
        return cached
    summaries, detectors = [], {}
    for key, metadata in result['detectors'].items():
        arrays = load_flim_detector(result, key)
        passed, error = _period_check(metadata, period_tolerance_bins)
        if passed:
            phase, modulation = phasor_to_apparent_lifetime(arrays['real'], arrays['imag'],
                                                            metadata['frequency_mhz'], unit_conversion=1e-3)
            phase[~arrays['valid']], modulation[~arrays['valid']] = np.nan, np.nan
        else:
            phase = np.full_like(arrays['real'], np.nan)
            modulation = np.full_like(arrays['real'], np.nan)
        arrays.update(phase_lifetime_ns=phase, modulation_lifetime_ns=modulation)
        output = root / f'{key}.npz'
        np.savez_compressed(output, **arrays)
        updated = metadata | {'object_path': str(output), 'period_error_bins': error,
                               'lifetime_status': 'APPARENT_LIFETIMES' if passed else 'WINDOW_PERIOD_MISMATCH',
                               'lifetime_interpretation': 'apparent; instrument phase and modulation uncorrected'
                               if metadata['calibration_status'] == 'UNCALIBRATED' else 'reference-calibrated apparent values'}
        detectors[key] = updated
        summaries.append(updated | {'finite_phase_lifetime_pixels': int(np.isfinite(phase).sum()),
                                     'finite_modulation_lifetime_pixels': int(np.isfinite(modulation).sum()),
                                     'negative_phase_lifetime_pixels': int((phase < 0).sum())})
    return _finish(root, receipt, summaries, detectors, table_dir=Path(table_dir), stage='flim_apparent_lifetimes', force=force)


def export_flim_results(
        result: dict[str, Any], *, destination: str | Path, force: bool = False,
) -> dict[str, Any]:
    """Publish a flat, hash-bound feature manifest for explicit HS/FLIM pairing."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    signatures = _verify_sources(list(result['detectors'].values()))
    parent = _parent_identity(result)
    identity = parent | {'stage': 'flim_feature_export', 'code_sha256': _sha(Path(__file__)),
                         'destination': str(destination), 'schema_version': 'flim_feature_manifest_v1'}
    receipt_path = destination.with_suffix('.receipt.json')
    if not force and destination.is_file() and receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved['identity'] == identity and _sha(destination) == saved['manifest_sha256']:
            _require_unchanged(signatures)
            print('CACHE_HIT flim feature export', flush=True)
            return {'manifest': pd.read_csv(destination), 'manifest_path': destination,
                    'receipt_path': receipt_path, 'cache_status': 'CACHE_HIT'}
    rows = []
    for key, metadata in result['detectors'].items():
        feature_path = Path(metadata['object_path'])
        with np.load(feature_path, allow_pickle=False) as saved:
            required = {'intensity', 'real', 'imag', 'valid', 'phase_lifetime_ns', 'modulation_lifetime_ns'}
            if not required.issubset(saved.files):
                raise ValueError('Publish final apparent-lifetime results after the explicit lifetime stage')
            shape = saved['intensity'].shape
        rows.append({
            'detector_key': key, 'source_path': metadata['path'], 'source_sha256': metadata['source_sha256'],
            'block_index': metadata['block_index'], 'detector_index': metadata['detector_index'],
            'selection_kind': metadata['selection_kind'], 'feature_path': str(feature_path),
            'feature_sha256': _sha(feature_path), 'receipt_path': parent['parent_receipt'],
            'receipt_sha256': parent['parent_sha256'], 'shape_json': json.dumps(shape),
            'shape_y': shape[0] if len(shape) == 2 else None, 'shape_x': shape[1] if len(shape) == 2 else None,
            'frequency_mhz': metadata['frequency_mhz'], 'frequency_provenance': metadata['frequency_provenance'],
            'calibration_status': metadata['calibration_status'], 'lifetime_status': metadata['lifetime_status'],
            'minimum_total_counts': metadata['minimum_total_counts'], 'feature_stage': result['stage'],
            'pixel_size_y_um': metadata.get('pixel_size_y_um'), 'pixel_size_x_um': metadata.get('pixel_size_x_um'),
            'physical_scale_status': metadata.get('physical_scale_status'), 'claim_label': 'Exploratory',
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(destination, index=False)
    _require_unchanged(signatures)
    receipt_path.write_text(json.dumps({'identity': identity, 'manifest_sha256': _sha(destination),
                                       'generating_script': str(Path(__file__).resolve()),
                                       'claim_label': 'Exploratory', 'feature_fields': sorted(required),
                                       'independent_units': 'Not established by image or pixel counts; no inference'}, indent=2))
    status = 'FORCE_RECOMPUTE' if force else 'CACHE_MISS_RECOMPUTED'
    print(f'{status} flim feature export: {destination}', flush=True)
    return {'manifest': manifest, 'manifest_path': destination, 'receipt_path': receipt_path, 'cache_status': status}
