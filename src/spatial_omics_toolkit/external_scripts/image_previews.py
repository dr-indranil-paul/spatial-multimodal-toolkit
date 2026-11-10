"""Display-only CZI, raster and batched SDT previews; source files stay unchanged.

Readers return bounded JPEG bytes and diagnostics, not analysis-ready pixels.
CZI intensity callers may override the preferred metadata channel name or supply
an explicit channel index; defaults prefer metadata name '469', then C=0.
Selected indices are checked against channel bounds before pixel decoding.
SDT preload blocks are bounded after opening metadata using long arithmetic;
decay bins remain separate until float64 integration for each detector channel.
Callers own biological grouping, provenance, review status and fallback policy.
"""

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image


@dataclass
class SDTPreview:
    channels: list
    details: str
    error: str = ''


def jpeg_thumbnail(image, *, max_dimension=850, quality=90):
    """Encode a bounded RGB JPEG in memory; never write a permanent preview."""
    with image.convert('RGB') as thumbnail:
        thumbnail.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        with BytesIO() as buffer:
            thumbnail.save(buffer, format='JPEG', quality=quality)
            return buffer.getvalue()


def intensity_thumbnail(pixels):
    """Apply a display-only 1–99 percentile stretch and return JPEG bytes."""
    pixels = np.asarray(pixels, dtype=np.float64)
    finite = pixels[np.isfinite(pixels)]
    if pixels.ndim != 2 or not finite.size:
        raise ValueError(f'Expected a finite 2D intensity plane, got {pixels.shape}')
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        raise ValueError('No displayable intensity range (1st/99th percentiles coincide)')
    scaled = np.clip(np.nan_to_num((pixels - low) / (high - low)), 0, 1)
    with Image.fromarray((scaled * 255).astype(np.uint8)) as image:
        return jpeg_thumbnail(image)


def raster_preview(path):
    """Return bounded JPEG bytes and exact reader/scaling diagnostics.

    Small TIFF pages are inspected with tifffile before Pillow can discard
    native precision. High-bit RGB(A) uses one global RGB percentile stretch;
    alpha is explicitly discarded, not composited or used for scaling.
    """
    path = Path(path)
    tiff_page_count = None
    if path.suffix.lower() in {'.tif', '.tiff'}:
        # Permit moderate multipage files for metadata inspection, but decode
        # only page 0 after bounding its uncompressed pixel allocation.
        file_limit = 1024 * 1024 * 1024
        page_limit = 32 * 1024 * 1024
        file_bytes = path.stat().st_size
        if file_bytes > file_limit:
            raise ValueError(
                f'TIFF preview file-size guard: source_path={path}; '
                f'file_bytes={file_bytes}; file_limit_bytes={file_limit}; refusing TIFF metadata parse'
            )
        import tifffile

        with tifffile.TiffFile(path) as reader:
            tiff_page_count = len(reader.pages)
            page = reader.pages[0]
            dtype = np.dtype(page.dtype)
            mode = page.photometric.name
            if mode == 'RGB' and page.axes == 'YXS' and page.shape[-1] == 4:
                mode = 'RGBA'
            details = (
                f'reader=tifffile.TiffFile.pages[0].asarray; page=0; page_count={tiff_page_count}; '
                f'selection=page 0 of {tiff_page_count}; '
                f'source_shape={page.shape}; source_axes={page.axes}; '
                f'source_photometric={page.photometric.name}; source_mode={mode}; '
                f'decoded_dtype={dtype}; bits_per_sample={page.bitspersample}; '
                f'file_bytes={file_bytes}; file_limit_bytes={file_limit}; '
                f'decoded_page_bytes={page.nbytes}; decoded_page_limit_bytes={page_limit}'
            )
            if page.nbytes > page_limit:
                raise ValueError(f'TIFF preview decoded-size guard: {details}; refusing full raster decode')
            if dtype.kind in 'uif' and dtype != np.dtype('uint8'):
                if len(page.shape) == 2 and page.axes == 'YX':
                    pixels = page.asarray()
                    return intensity_thumbnail(pixels), details + '; display-only 1–99 percentile grayscale stretch'
                if (
                    len(page.shape) != 3 or page.axes != 'YXS'
                    or page.shape[-1] not in (3, 4) or page.photometric.name != 'RGB'
                ):
                    raise ValueError(f'Unsupported high-bit-depth TIFF layout: {details}')
                pixels = page.asarray()
                rgb = pixels[..., :3].astype(np.float64)
                finite = rgb[np.isfinite(rgb)]
                if not finite.size:
                    raise ValueError(f'No finite RGB intensity values: {details}')
                low, high = np.percentile(finite, [1, 99])
                if high <= low:
                    raise ValueError(f'No displayable global RGB percentile range: {details}; low={low}; high={high}')
                scaled = np.clip(
                    np.nan_to_num((rgb - low) / (high - low), nan=0, posinf=1, neginf=0), 0, 1,
                )
                alpha = (
                    f'discarded, no compositing or unpremultiplication; extrasamples={page.extrasamples}'
                    if pixels.shape[-1] == 4 else 'absent'
                )
                details += (
                    f'; display-only one global 1–99 percentile stretch across RGB channels; '
                    f'low={low}; high={high}; alpha_treatment={alpha}; '
                    'NaN/-inf map to black, +inf to white; RGB uint8 JPEG'
                )
                with Image.fromarray((scaled * 255).astype(np.uint8)) as image:
                    return jpeg_thumbnail(image), details
    with Image.open(path) as image:
        pixels = np.asarray(image)
        details = f'reader=PIL.Image.open; source_mode={image.mode}; decoded_dtype={pixels.dtype}'
        if tiff_page_count is not None:
            details += (
                f'; page=0; page_count={tiff_page_count}; selection=page 0 of {tiff_page_count}; '
                f'file_bytes={file_bytes}; file_limit_bytes={file_limit}; '
                f'decoded_page_limit_bytes={page_limit}; first-page-only preview'
            )
        bits = getattr(image, 'tag_v2', {}).get(258, ())
        bits = (bits,) if isinstance(bits, int) else bits
        high_bit_metadata = any(bit > 8 for bit in bits)
        high_bit_pixels = pixels.dtype.kind in 'uif' and pixels.dtype != np.dtype('uint8')
        if high_bit_pixels:
            if pixels.ndim != 2:
                raise ValueError(f'Unsupported high-bit-depth raster layout: {details}; shape={pixels.shape}')
            return (
                intensity_thumbnail(pixels),
                details + '; display-only 1–99 percentile grayscale stretch before RGB/JPEG conversion',
            )
        if high_bit_metadata:
            raise ValueError(
                f'High-bit-depth raster was not exposed at native precision: {details}; '
                f'bits_per_sample={bits}; refusing a clipped display'
            )
        return jpeg_thumbnail(image), details + '; existing 8-bit/color rendering; no intensity stretch'


def czi_intensity_preview(path, *, channel_name='469', channel_index=None):
    """Return a labeled single-channel CZI preview, without guessing scenes.

    An explicit channel_index overrides channel_name. Otherwise prefer the named
    metadata channel, falling back explicitly to C=0 if absent or None. Defaults
    preserve the gallery's metadata-name '469' then C=0 selection.
    """
    from aicspylibczi import CziFile

    reader = CziFile(str(path))
    nodes = reader.meta.findall('.//Information/Image/Dimensions/Channels/Channel')
    names = [node.findtext('Name', default=node.get('Name', '')) for node in nodes]
    if channel_index is not None:
        from operator import index

        if isinstance(channel_index, (bool, np.bool_)):
            raise TypeError('channel_index must be an integer channel index, not a boolean')
        channel = index(channel_index)
        metadata_name = names[channel] if 0 <= channel < len(names) else 'unavailable'
        choice = f'explicit channel C={channel} (metadata name: {metadata_name})'
    elif channel_name is not None and channel_name in names:
        channel = names.index(channel_name)
        choice = f'metadata channel {channel_name}'
    else:
        channel = 0
        choice = f'deterministic fallback C=0 (metadata name: {names[0] if names else "unavailable"})'
    dimensions = reader.get_dims_shape()
    if len(dimensions) != 1:
        raise ValueError(f'Multiple CZI dimension blocks require manual scene choice: {dimensions}')
    dims = dimensions[0]
    channel_start, channel_stop = dims.get('C', (0, 1))
    if not channel_start <= channel < channel_stop:
        raise ValueError(
            f'Selected C={channel} outside available channel range '
            f'[{channel_start}, {channel_stop}); selection={choice}'
        )
    selectors = {'C': channel}
    notes = [f'selected_C={channel} of [{channel_start}, {channel_stop})']
    for axis in ('S', 'T', 'Z'):
        if axis in dims:
            start, stop = dims[axis]
            selectors[axis] = start
            notes.append(f'{axis}={start} of [{start}, {stop})')
    if reader.is_mosaic():
        if 'S' in dims and dims['S'][1] - dims['S'][0] > 1:
            raise ValueError(f'Multiple mosaic scenes require manual choice: {dimensions}')
        selectors.pop('S', None)
        pixels = reader.read_mosaic(scale_factor=0.25, **selectors)
    else:
        pixels, axes = reader.read_image(**selectors)
        notes.append(f'read_image axes={axes}')
    pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(f'Expected one grayscale spectral plane, got {pixels.shape}')
    details = f'{choice}; {"; ".join(notes)}; pixel_type={reader.pixel_type}; display-only 1–99 percentile stretch'
    return intensity_thumbnail(pixels), details


def czi_rgb_preview(path, *, scale_factor=0.12):
    """Read a BGR24 mosaic at C=0 and return a bounded RGB JPEG."""
    from aicspylibczi import CziFile

    reader = CziFile(str(path))
    pixels = np.squeeze(reader.read_mosaic(
        C=0, scale_factor=scale_factor, background_color=(1., 1., 1.),
    ))
    if reader.pixel_type != 'bgr24' or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f'Unexpected RGB encoding: {reader.pixel_type}, {pixels.shape}')
    with Image.fromarray(np.ascontiguousarray(pixels[..., ::-1])) as image:
        return jpeg_thumbnail(image)


_SDT_SCRIPT = r'''
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import groovy.transform.CompileStatic
import loci.formats.FormatTools
import loci.formats.in.SDTReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

@CompileStatic
class IntensityPlanes {
    static void add(double[] sum, byte[] bytes, int type, boolean little) {
        ByteBuffer plane = ByteBuffer.wrap(bytes).order(little ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN)
        switch (type) {
            case FormatTools.UINT8:
                for (int i = 0; i < sum.length; i++) sum[i] += bytes[i] & 0xff
                break
            case FormatTools.INT8:
                for (int i = 0; i < sum.length; i++) sum[i] += bytes[i]
                break
            case FormatTools.UINT16:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getShort(i * 2) & 0xffff
                break
            case FormatTools.INT16:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getShort(i * 2)
                break
            case FormatTools.UINT32:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getInt(i * 4) & 0xffffffffL
                break
            case FormatTools.INT32:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getInt(i * 4)
                break
            case FormatTools.FLOAT:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getFloat(i * 4)
                break
            case FormatTools.DOUBLE:
                for (int i = 0; i < sum.length; i++) sum[i] += plane.getDouble(i * 8)
                break
            default:
                throw new IllegalArgumentException('Unsupported pixel type: ' + type)
        }
    }
}

@CompileStatic
class BoundedSDTReader extends SDTReader {
    Map configurePreviewPreload() {
        // Installed Bio-Formats 8.1.1 SDTReader.initFile uses int multiplication for its
        // 512^3 pixel-bin threshold; openBytes then allocates binSize*blockLength.
        // At 2048^2 with 256 uint16 bins that byte count wraps to -2^31.
        // Source: https://github.com/ome/bioformats/blob/2a6f596fc917dd979e1edcf3a8ab82fbdeb1ab62/components/formats-gpl/src/loci/formats/in/SDTReader.java
        // Keep its block cache and decoding; only reduce the protected block size.
        long threshold = 512L * 512L * 512L
        long paddedWidth = (long) getSizeX() + ((4 - (getSizeX() % 4)) % 4)
        long planePixels = paddedWidth * (long) getSizeY()
        int bytesPerPixel = FormatTools.getBytesPerPixel(getPixelType())
        long maxBlockLength = Math.min(Math.floorDiv(threshold, planePixels),
            Math.floorDiv(Math.floorDiv(Integer.MAX_VALUE - 8L, (long) bytesPerPixel), planePixels))
        int originalBlockLength = blockLength
        if (originalBlockLength < 1 || maxBlockLength < 1) {
            throw new IllegalArgumentException('SDT preload cannot fit a single time-bin plane safely')
        }
        int safeBlockLength = originalBlockLength
        while ((long) safeBlockLength > maxBlockLength) safeBlockLength = safeBlockLength >> 1
        blockLength = safeBlockLength
        // setId may close/reset the reader, so configure only after setId and
        // before the first openBytes. Do not use intensity mode's short sum.
        setPreLoad(true)
        return [
            preload_enabled: true,
            preload_workaround_applied: safeBlockLength != originalBlockLength,
            preload_original_block_length: originalBlockLength,
            preload_block_length: safeBlockLength,
            preload_padded_width_bound: paddedWidth,
            preload_plane_bytes_bound: planePixels * bytesPerPixel,
            preload_block_bytes_bound: planePixels * bytesPerPixel * safeBlockLength,
            preload_pixel_bin_threshold: threshold,
            preload_policy: 'Bio-Formats block cache; long-arithmetic 512^3 pixel-bin bound and Java byte-array limit'
        ]
    }
}

File outputDirectory = new File(System.getenv('IMAGE_PREVIEW_DIRECTORY'))
List groups = (List) new JsonSlurper().parse(new File(System.getenv('IMAGE_PREVIEW_GROUPS')))
for (List group : groups) {
    for (Map candidate : group) {
        String source = (String) candidate.path
        String key = candidate.key.toString()
        Map metadata = [source_path: source, status: 'processing']
        File metadataFile = new File(outputDirectory, key + '.json')
        List<File> pixelFiles = []
        try {
            BoundedSDTReader reader = new BoundedSDTReader()
            try {
                reader.setIntensity(false)
                reader.setId(source)
                reader.setSeries(0)
                int type = reader.getPixelType()
                metadata.putAll(reader.configurePreviewPreload())
                metadata.putAll([
                    reader: reader.getClass().getName(), series: 0, series_count: reader.getSeriesCount(),
                    dimension_order: reader.getDimensionOrder(), X: reader.getSizeX(), Y: reader.getSizeY(),
                    C: reader.getSizeC(), Z: reader.getSizeZ(), T: reader.getSizeT(),
                    time_bins: reader.getTimeBinCount(), detector_channels: reader.getChannelCount(),
                    pixel_type: FormatTools.getPixelTypeString(type), little_endian: reader.isLittleEndian(),
                    rgb: reader.isRGB(), samples_per_pixel: reader.getRGBChannelCount(),
                    selected_channels: (0..<reader.getSizeC()).toList(), selected_z: 0,
                    preview: 'time-integrated intensity; all decay bins; separate detector channels; not lifetime/phasor'
                ])
                metadataFile.text = JsonOutput.toJson(metadata)
                println('IMAGE_PREVIEW_AXES ' + JsonOutput.toJson(metadata))
                if (reader.isRGB() || reader.getRGBChannelCount() != 1 || reader.getSizeZ() != 1 ||
                    reader.getSizeC() < 1 || reader.getSizeT() <= 1 ||
                    reader.getSizeT() != reader.getTimeBinCount()) {
                    throw new IllegalArgumentException('Unsupported SDT axes; refusing to guess decay/time layout')
                }
                for (int channel = 0; channel < reader.getSizeC(); channel++) {
                    double[] sum = new double[reader.getSizeX() * reader.getSizeY()]
                    int expected = sum.length * FormatTools.getBytesPerPixel(type)
                    for (int t = 0; t < reader.getSizeT(); t++) {
                        byte[] plane = reader.openBytes(reader.getIndex(0, channel, t))
                        if (plane.length != expected) throw new IllegalArgumentException('Unexpected plane byte count')
                        IntensityPlanes.add(sum, plane, type, reader.isLittleEndian())
                    }
                    File pixels = new File(outputDirectory, key + '.c' + channel + '.float64')
                    pixelFiles.add(pixels)
                    new DataOutputStream(new BufferedOutputStream(new FileOutputStream(pixels))).withCloseable { out ->
                        for (double value : sum) out.writeDouble(value)
                    }
                }
            } finally {
                reader.close()
            }
            metadata.status = 'ok'
        } catch (Exception error) {
            metadata.status = 'error'
            metadata.error = error.getClass().getName() + ': ' + error.getMessage()
            pixelFiles.each { it.delete() }
            System.err.println('IMAGE_PREVIEW_ERROR source_path=' + source)
            error.printStackTrace(System.err)
        }
        metadataFile.text = JsonOutput.toJson(metadata)
        new File(outputDirectory, key + '.ready').text = 'ready'
        // Python must accept the JPEG thumbnails before skipping any alternatives.
        File acknowledgement = new File(outputDirectory, key + '.ack')
        while (!acknowledgement.exists()) Thread.sleep(50)
        boolean accepted = acknowledgement.text.trim() == 'accepted'
        pixelFiles.each { it.delete() }
        if (accepted) break
    }
}
'''


def _stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def prepare_sdt_previews(groups, *, fiji, temp_parent, timeout_per_source=600):
    """Try ordered raw paths per group in one Fiji process, returning attempted paths.

    A group stops only after Python accepts at least one displayable detector JPEG.
    Every attempted failure retains its exact path and available Bio-Formats metadata.
    Unused alternatives have no result unless the whole batch becomes unavailable.
    Bio-Formats preload blocks use a metadata-derived long-arithmetic bound;
    diagnostics record the original/effective block lengths and byte bounds.
    All scripts, pixels and logs are temporary, including on interruption.
    """
    groups = [[str(path) for path in group] for group in groups if group]
    results = {}
    if not groups:
        return results
    if not os.environ.get('SLURM_JOB_ID'):
        error = 'Raw FLIM preview requires a verified SLURM allocation; Fiji launcher enforces the allocation guard'
        return {path: SDTPreview([], f'source_path={path}; {error}', error) for group in groups for path in group}
    planned = []
    for group in groups:
        planned.append([{'path': path, 'key': f'{len(planned)}_{index}'} for index, path in enumerate(group)])
    process = None
    current = None
    try:
        with tempfile.TemporaryDirectory(prefix='.image-preview-', dir=temp_parent) as directory:
            directory = Path(directory)
            script = directory / 'integrate_sdt.groovy'
            manifest = directory / 'groups.json'
            script.write_text(_SDT_SCRIPT)
            manifest.write_text(json.dumps(planned))
            process_env = dict(os.environ, IMAGE_PREVIEW_GROUPS=str(manifest), IMAGE_PREVIEW_DIRECTORY=str(directory))
            with (directory / 'fiji.log').open('w+b') as log:
                try:
                    process = subprocess.Popen(
                        [str(fiji), '--headless', '--console', '--run', str(script)],
                        env=process_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    for group in planned:
                        for candidate in group:
                            current = candidate['path']
                            key = candidate['key']
                            ready = directory / f'{key}.ready'
                            deadline = time.monotonic() + timeout_per_source
                            while not ready.exists():
                                if process.poll() is not None:
                                    raise RuntimeError(f'Fiji exited with status {process.returncode} before source completed')
                                if time.monotonic() >= deadline:
                                    raise TimeoutError(f'Fiji source exceeded {timeout_per_source}s')
                                time.sleep(0.05)
                            metadata = {'source_path': current}
                            try:
                                metadata = json.loads((directory / f'{key}.json').read_text())
                                if metadata.get('source_path') != current:
                                    raise ValueError('Fiji metadata source-path mismatch')
                                if metadata.get('status') != 'ok':
                                    raise RuntimeError(metadata.get('error', 'Fiji did not complete this source'))
                                channels = []
                                channel_errors = {}
                                for channel in metadata['selected_channels']:
                                    pixels_path = directory / f'{key}.c{channel}.float64'
                                    pixels = np.fromfile(pixels_path, dtype='>f8').reshape(metadata['Y'], metadata['X'])
                                    try:
                                        payload = intensity_thumbnail(pixels)
                                    except ValueError as error:
                                        payload = None
                                        channel_errors[str(channel)] = str(error)
                                    finally:
                                        del pixels
                                        pixels_path.unlink(missing_ok=True)
                                    channels.append((channel, payload))
                                metadata['display_channel_errors'] = channel_errors
                                if not any(payload is not None for _, payload in channels):
                                    raise ValueError(f'No detector channel has displayable intensity: {channel_errors}')
                                results[current] = SDTPreview(channels, json.dumps(metadata) + '; display-only per-channel 1–99 percentile stretch')
                            except Exception as error:
                                details = f'source_path={current}; {type(error).__name__}: {error}; metadata={json.dumps(metadata)}'
                                results[current] = SDTPreview([], details, str(error))
                            accepted = not results[current].error
                            acknowledgement = directory / f'{key}.ack'
                            pending_acknowledgement = directory / f'{key}.ack.pending'
                            pending_acknowledgement.write_text('accepted' if accepted else 'rejected')
                            pending_acknowledgement.replace(acknowledgement)
                            if accepted:
                                break
                    current = None
                    process.wait(timeout=30)
                    if process.returncode:
                        raise RuntimeError(f'Fiji exited with status {process.returncode}')
                except Exception as error:
                    if process is not None:
                        _stop_process(process)
                    log.flush()
                    log.seek(0, os.SEEK_END)
                    size = log.tell()
                    log.seek(max(0, size - 6000))
                    diagnostic = log.read().decode('utf-8', errors='replace')
                    details = f'{type(error).__name__}: {error}; Fiji diagnostics={diagnostic}'
                    if current is not None and current not in results:
                        metadata_path = directory / f'{key}.json'
                        metadata_text = metadata_path.read_text() if metadata_path.exists() else 'unavailable'
                        results[current] = SDTPreview(
                            [], f'source_path={current}; metadata={metadata_text}; {details}', str(error),
                        )
                    # Completed raw previews survive a later process failure. Record the
                    # current source and unavailable remaining candidates, never drop paths.
                    for group in groups:
                        for path in group:
                            results.setdefault(path, SDTPreview([], f'source_path={path}; {details}', str(error)))
                finally:
                    if process is not None:
                        _stop_process(process)
    except Exception as error:
        for group in groups:
            for path in group:
                results.setdefault(path, SDTPreview([], f'source_path={path}; {type(error).__name__}: {error}', str(error)))
    return results
