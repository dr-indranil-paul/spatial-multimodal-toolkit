"""Clinical HS inventory, exact metadata, native QC and exploratory spectral phasors.

No FLIM processing, correction, registration, inference or acquisition ranking.
The reviewed manifest is an independent decision ledger, never a generated queue.
"""

from collections.abc import Callable
from typing import Any
from hashlib import sha256
from datetime import datetime, timezone
from math import prod
from pathlib import Path
import json
import base64
import importlib.metadata
import tempfile
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from image_previews import czi_intensity_preview

_STYLE_UTILS = Path(__file__).resolve().parents[1] / '01_utils'
if str(_STYLE_UTILS) not in sys.path:
    sys.path.insert(0, str(_STYLE_UTILS))

import plotnine_style_helper as plot_style  # noqa: E402


SOURCE_COLUMNS = (
    'collaborator', 'dated_set', 'path', 'filename', 'classification', 'source_scope',
    'modality', 'block_provenance', 'size_bytes', 'partial_sha256',
    'candidate_core_ids', 'core_match_evidence', 'core_match_status', 'source_set',
)
MANIFEST_NAME = 'hyperspectral_assignment_and_processing_manifest.csv'
PRIMARY_ROLE = 'RAW_HS_QC_INPUT'
AUXILIARY_ROLE = 'RAW_HS_AUXILIARY_REVIEW_ONLY'
SELECTION_LEDGER_NAME = 'hyperspectral_tma1_manual_core_selection.csv'
FINAL_ANALYSIS_MANIFEST_NAME = 'hyperspectral_final_analysis_manifest.csv'
SETTINGS_MANIFEST_NAME = 'hyperspectral_settings_provisional_working_set.csv'
TMA1_CORES = tuple(f'{row}{column}' for row in 'ABC' for column in '123')
SELECTION_SOURCE_COLUMNS = SOURCE_COLUMNS + (
    'assigned_tma',
    'assignment_basis',
    'processing_role',
    'analysis_input',
    'core_identity_status',
    'acquisition_selection',
    'decision_authority',
    'decision_scope',
    'decision_date',
    'source_queue_sha256',
)
SELECTION_MANUAL_DEFAULTS = {
    'selection_status': 'PENDING',
    'confirmed_core_id': '',
    'core_identity_review': 'PENDING',
    'core_identity_notes': '',
    'he_alignment_review': 'PENDING',
    'he_reference_path': '',
    'he_alignment_notes': '',
    'core_coverage_review': 'PENDING',
    'focus_review': 'PENDING',
    'artifacts_review': 'PENDING',
    'tile_completeness_review': 'PENDING',
    'acquisition_settings_review': 'PENDING',
    'selection_reason': '',
    'reviewer': '',
    'review_date': '',
}
PIXEL_DTYPES = {'gray8': 'uint8', 'gray16': 'uint16', 'gray32': 'uint32',
                'gray32float': 'float32', 'gray64float': 'float64'}


def _file_signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _full_file_hash(path):
    """Hash every byte with bounded working memory, never the queue's partial hash."""
    digest = sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_style_sha256() -> str:
    """Hash the shared plot style without coupling it to data-cache keys."""
    style_path = (
        Path(__file__).resolve().parents[1]
        / '01_utils' / 'plotnine_style_helper.py'
    )
    return _full_file_hash(style_path)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


class HSCache:
    """Content-addressed small-output cache; construction never creates directories.

    Metadata reuse still reads the actual XML/layout. Pixel inputs are fully hashed
    on first access in this object; repeat access reuses that digest only while the
    current stat signature matches. Stat identities are never persisted cache keys.
    Native pixels are never serialized. Each JSON payload has an integrity digest.
    """

    def __init__(self, root, *, manifest_path, queue_path, force=False):
        self.root = Path(root)
        self.force = bool(force)
        self.manifest_path = Path(manifest_path)
        self.queue_path = Path(queue_path)
        self.manifest_sha256 = _full_file_hash(self.manifest_path)
        self.queue_sha256 = _full_file_hash(self.queue_path)
        scripts = Path(__file__).resolve().parent
        self.code_sha256 = {
            'clinical_roi_hyperspectral.py': _full_file_hash(
                scripts / 'clinical_roi_hyperspectral.py',
            ),
            'image_previews.py': _full_file_hash(scripts / 'image_previews.py'),
        }
        # Style changes affect rendered figures, not cached scientific summaries.
        self.plot_style_sha256 = _plot_style_sha256()
        self.package_versions = {'python': sys.version}
        for package in (
                'aicspylibczi', 'numpy', 'pandas', 'Pillow', 'phasorpy',
                'matplotlib', 'scipy', 'scikit-learn', 'threadpoolctl'):
            try:
                self.package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.package_versions[package] = 'NOT_INSTALLED'
        self.events = []
        self.raw_file_full_hashes = {}
        self._raw_identities = {}

    def _context(self):
        # A long-lived kernel must not silently use a replaced decision ledger.
        if (_full_file_hash(self.manifest_path) != self.manifest_sha256
                or _full_file_hash(self.queue_path) != self.queue_sha256):
            raise ValueError(
                'Assignment manifest or queue changed; reload the reviewed '
                'inventory and cache context'
            )
        return {
            'schema': 'clinical_hs_small_outputs_v1',
            'manifest_sha256': self.manifest_sha256,
            'queue_sha256': self.queue_sha256,
            'code_sha256': self.code_sha256,
            'package_versions': self.package_versions,
        }

    def raw_identity(self, path):
        path = Path(path)
        signature = _file_signature(path)
        prior = self._raw_identities.get(str(path))
        if prior is not None and prior[1] == signature:
            return prior
        digest = _full_file_hash(path)
        self.require_unchanged(path, signature)
        self.raw_file_full_hashes[str(path)] = digest
        self._raw_identities[str(path)] = (digest, signature)
        return digest, signature

    @staticmethod
    def require_unchanged(path, signature):
        if _file_signature(path) != signature:
            raise ValueError(f'Source changed during hashing/decoding/cache access: {path}')

    def result(
            self, stage: str, digest: str, parameters: dict[str, Any],
            compute: Callable[[], Any], *,
            source_kind: str = 'full_file_sha256',
            report: Callable[[str], Any] = print,
            validate: Callable[[Any], None] | None = None,
    ) -> tuple[Any, str, str]:
        if stage not in {
                'metadata', 'preview', 'native_qc', 'spectral_phasor',
                'phasor_gmm', 'pseudocolor', 'centroid_summary',
                'cohort_prep', 'cohort_clusters'}:
            raise ValueError(f'Unsupported HS cache stage: {stage}')
        identity = self._context() | {
            'stage': stage, 'source_kind': source_kind,
            'input_sha256': digest, 'parameters': parameters,
        }
        key = sha256(_json_bytes(identity)).hexdigest()
        destination = self.root / stage / f'{key}.json'
        status = 'FORCE_RECOMPUTE' if self.force else 'CACHE_MISS_RECOMPUTED'
        reason = 'explicit force' if self.force else 'no output for current inputs/code/parameters'
        value = None
        if not self.force and destination.is_file():
            try:
                envelope = json.loads(destination.read_text(encoding='utf-8'))
                value = envelope['result']
                if (envelope['identity'] != json.loads(_json_bytes(identity))
                        or envelope['result_sha256'] != sha256(_json_bytes(value)).hexdigest()):
                    raise ValueError('cached identity or payload digest mismatch')
                if validate is not None:
                    validate(value)
                status, reason = 'CACHE_HIT', 'validated saved output'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status, reason = (
                    'CACHE_MISS_RECOMPUTED',
                    f'invalid saved output: {type(exc).__name__}: {exc}',
                )
        row = parameters.get('row', {})
        path = parameters.get('path', row.get('path', ''))
        report(f'{status} {stage}: {path} [{key[:12]}]; {reason}')
        if status != 'CACHE_HIT':
            value = compute()
            envelope = {
                'identity': identity, 'result': value,
                'result_sha256': sha256(_json_bytes(value)).hexdigest(),
            }
            payload = _json_bytes(envelope)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp',
                                                 delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.events.append({
            'stage': stage, 'status': status, 'reason': reason, 'key': key, 'path': str(path),
            'source_kind': source_kind, 'input_sha256': digest,
            'parameters': parameters, 'cache_path': str(destination),
        })
        return value, status, key

    def receipt(self):
        return {
            'cache_root': str(self.root), 'force': self.force,
            'manifest_path': str(self.manifest_path), 'manifest_sha256': self.manifest_sha256,
            'queue_path': str(self.queue_path), 'queue_sha256': self.queue_sha256,
            'code_sha256': dict(self.code_sha256),
            'plot_style_sha256': self.plot_style_sha256,
            'package_versions': dict(self.package_versions),
            'raw_file_full_hashes': dict(self.raw_file_full_hashes),
            'raw_identity_policy': (
                'FULL_SHA256_ON_FIRST_ACCESS_PER_CACHE_OBJECT; '
                'IN_MEMORY_REUSE_ONLY_WHILE_STAT_SIGNATURE_UNCHANGED'
            ),
            'events': list(self.events),
        }


def load_reviewed_inventory(manifest_path, queue_path):
    """Require a complete, unchanged queue snapshot; never invent new decisions."""
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    queue = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
    required = set(SOURCE_COLUMNS) | {
        'assigned_tma', 'assignment_basis', 'processing_role', 'analysis_input',
        'core_identity_status',
        'acquisition_selection',
        'decision_authority',
        'source_queue_sha256',
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f'Manual manifest lacks columns: {sorted(missing)}')
    queue_fingerprints = manifest['source_queue_sha256'].unique()
    if len(queue_fingerprints) != 1 or queue_fingerprints[0] != _full_file_hash(queue_path):
        raise ValueError(
            'Reviewed manifest source_queue_sha256 does not uniquely match '
            'the current queue; reconcile the decision ledger before processing'
        )
    if manifest['path'].duplicated().any() or queue['path'].duplicated().any():
        raise ValueError('Duplicate exact paths in queue or reviewed manifest')
    old = manifest.set_index('path')[list(SOURCE_COLUMNS[:2]) + list(SOURCE_COLUMNS[3:])]
    new = queue.set_index('path')[old.columns]
    if set(old.index) != set(new.index) or not old.sort_index().equals(new.sort_index()):
        raise ValueError(
            'Source queue changed: reconcile the separate human decision '
            'ledger before processing'
        )
    if not manifest['analysis_input'].isin(['true', 'false']).all():
        raise ValueError('analysis_input must be explicit true/false')
    inputs = manifest[manifest['analysis_input'].eq('true')].copy()
    valid = (
        inputs['processing_role'].eq(PRIMARY_ROLE)
        & inputs['classification'].eq('raw_acquisition')
        & inputs['modality'].eq('hyperspectral')
        & inputs['source_scope'].eq('PRIMARY_CANDIDATE')
        & inputs['core_match_status'].eq('CORE_CANDIDATE')
        & inputs['path'].map(lambda value: Path(value).suffix.lower() == '.czi')
        & ((inputs['collaborator'].eq('dylan') & inputs['assigned_tma'].eq('TMA1'))
           | (inputs['collaborator'].eq('sean') & inputs['assigned_tma'].eq('TMA2')))
    )
    if inputs.empty or not valid.all():
        raise ValueError(
            'Analysis inputs violate the reviewed raw-HS/source/core-candidate '
            'contract'
        )
    for labels in manifest['candidate_core_ids']:
        if not isinstance(json.loads(labels), list):
            raise ValueError('candidate_core_ids must remain a JSON evidence list')
    return (
        manifest,
        inputs.sort_values(
            ['assigned_tma', 'source_set', 'path']
        ).reset_index(drop=True),
    )


def load_or_initialize_core_selection(raw_inputs, ledger_path):
    """Snapshot eligible Dylan/TMA1 evidence once; never rewrite human decisions."""
    missing = set(SELECTION_SOURCE_COLUMNS) - set(raw_inputs.columns)
    if missing:
        raise ValueError(f'Raw inputs lack selection evidence: {sorted(missing)}')
    eligible = raw_inputs.loc[
        raw_inputs['collaborator'].eq('dylan')
        & raw_inputs['assigned_tma'].eq('TMA1')
    ].copy()
    valid = (
        eligible['analysis_input'].eq('true')
        & eligible['processing_role'].eq(PRIMARY_ROLE)
        & eligible['classification'].eq('raw_acquisition')
        & eligible['modality'].eq('hyperspectral')
        & eligible['source_scope'].eq('PRIMARY_CANDIDATE')
        & eligible['core_match_status'].eq('CORE_CANDIDATE')
        & eligible['path'].map(lambda value: Path(value).suffix.lower() == '.czi')
    )
    if eligible.empty or not valid.all() or eligible['path'].duplicated().any():
        raise ValueError('Selection requires unique, eligible Dylan/TMA1 raw CZI inputs')
    evidence = eligible[list(SELECTION_SOURCE_COLUMNS)].fillna('').astype(str)
    evidence = evidence.sort_values('path').reset_index(drop=True)
    for labels in evidence['candidate_core_ids']:
        candidates = json.loads(labels)
        if (not isinstance(candidates, list) or not candidates
                or any(core not in TMA1_CORES for core in candidates)):
            raise ValueError('Selection candidate evidence must list TMA1 cores A1-C3')
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        initial = evidence.assign(**SELECTION_MANUAL_DEFAULTS)
        # Exclusive creation protects any ledger created concurrently by a reviewer.
        try:
            with ledger_path.open('x', encoding='utf-8', newline='') as handle:
                initial.to_csv(handle, index=False)
        except FileExistsError:
            pass
    ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    required = set(SELECTION_SOURCE_COLUMNS) | set(SELECTION_MANUAL_DEFAULTS)
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f'Selection ledger lacks columns: {sorted(missing)}')
    if ledger['path'].duplicated().any():
        raise ValueError('Selection ledger has duplicate raw paths')
    recorded = ledger[list(SELECTION_SOURCE_COLUMNS)].sort_values('path')
    recorded = recorded.reset_index(drop=True)
    if not recorded.equals(evidence):
        raise ValueError(
            'Selection candidate paths/source evidence changed; reconcile the '
            'separate ledger manually. Existing choices were not overwritten.'
        )
    return ledger


def validate_core_selection(ledger):
    """Require human alignment, identity and technical review before any selection."""
    allowed = {
        'selection_status': {'PENDING', 'SELECTED', 'NOT_SELECTED', 'UNSUITABLE'},
        'core_identity_review': {'PENDING', 'CONFIRMED', 'UNRESOLVED'},
        'he_alignment_review': {'PENDING', 'REVIEWED', 'UNRESOLVED'},
    }
    technical_fields = (
        'core_coverage_review', 'focus_review', 'artifacts_review',
        'tile_completeness_review', 'acquisition_settings_review',
    )
    allowed.update(
        {
            field: {'PENDING', 'PASS', 'CONCERN', 'FAIL'}
            for field in technical_fields
        }
    )
    for field, values in allowed.items():
        if not ledger[field].isin(values).all():
            raise ValueError(f'{field} must use only {sorted(values)}')
    confirmed = ledger['core_identity_review'].eq('CONFIRMED')
    named = ledger['confirmed_core_id'].ne('')
    if not named.eq(confirmed).all():
        raise ValueError('Only CONFIRMED identity rows may name a confirmed core')
    if not ledger.loc[confirmed, 'confirmed_core_id'].isin(TMA1_CORES).all():
        raise ValueError('Confirmed TMA1 core IDs must be A1-C3')
    if confirmed.any():
        identity_ready = (
            ledger.loc[confirmed, 'he_alignment_review'].eq('REVIEWED')
            & ledger.loc[confirmed, 'core_identity_notes'].str.strip().ne('')
            & ledger.loc[confirmed, 'he_reference_path'].str.strip().ne('')
            & ledger.loc[confirmed, 'he_alignment_notes'].str.strip().ne('')
        )
        if not identity_ready.all():
            raise ValueError(
                'Core confirmation requires documented H&E alignment review, '
                'an H&E reference path and core-identity notes'
            )
    selected = ledger.loc[ledger['selection_status'].eq('SELECTED')]
    ready = (
        selected['core_identity_review'].eq('CONFIRMED')
        & selected['he_alignment_review'].eq('REVIEWED')
    )
    for field in technical_fields:
        ready &= selected[field].isin({'PASS', 'CONCERN'})
    for field in ('selection_reason', 'reviewer', 'review_date'):
        ready &= selected[field].str.strip().ne('')
    if not ready.all():
        raise ValueError(
            'SELECTED requires confirmed identity, reviewed H&E alignment, '
            'all technical reviews completed with no FAIL, a reason, '
            'reviewer and review date'
        )
    if selected['confirmed_core_id'].duplicated().any():
        raise ValueError('At most one SELECTED raw CZI path is allowed per confirmed core')
    return selected.copy()


def _build_final_analysis_manifest(*, ledger_path, assignment_manifest_path,
                                   queue_path, settings_manifest_path):
    """Build a read-only, evidence-bound handoff; never initialize a ledger."""
    sources = {
        'ledger': Path(ledger_path).resolve(),
        'assignment_manifest': Path(assignment_manifest_path).resolve(),
        'queue': Path(queue_path).resolve(),
        'settings_manifest': Path(settings_manifest_path).resolve(),
    }
    hashes = {name: _full_file_hash(path) for name, path in sources.items()}
    _, inputs = load_reviewed_inventory(sources['assignment_manifest'], sources['queue'])
    ledger = pd.read_csv(sources['ledger'], dtype=str, keep_default_na=False)
    required = set(SELECTION_SOURCE_COLUMNS) | set(SELECTION_MANUAL_DEFAULTS)
    if required - set(ledger.columns):
        raise ValueError('Selection ledger lacks required source/manual evidence')
    eligible = inputs.loc[
        inputs['collaborator'].eq('dylan') & inputs['assigned_tma'].eq('TMA1'),
        list(SELECTION_SOURCE_COLUMNS),
    ].sort_values('path').reset_index(drop=True)
    recorded = ledger[list(SELECTION_SOURCE_COLUMNS)].sort_values('path').reset_index(drop=True)
    if ledger['path'].duplicated().any() or not recorded.equals(eligible):
        raise ValueError('Selection ledger paths/source evidence changed')
    selected = validate_core_selection(
        ledger.loc[ledger['selection_status'].eq('SELECTED')].copy()
    )
    if len(selected) != 9 or set(selected['confirmed_core_id']) != set(TMA1_CORES):
        raise ValueError('Final Dylan selection requires exactly one selected A1-C3')
    settings = pd.read_csv(sources['settings_manifest'], dtype=str, keep_default_na=False)
    required_settings = {
        'assigned_tma', 'collaborator', 'candidate_core_label_evidence', 'path',
        'working_set_role', 'selection_status',
    }
    if required_settings - set(settings.columns):
        raise ValueError('Settings manifest lacks final-slot evidence')
    if (len(settings) != 18 or not settings['working_set_role'].isin({
            'DYLAN_PROVISIONAL_SETTINGS_CANDIDATE', 'SEAN_TMA2_REFERENCE',
    }).all()):
        raise ValueError('Settings manifest must contain only the 18 prescribed final slots')
    rows = []
    for tma, collaborator, role, status in (
        ('TMA1', 'dylan', 'DYLAN_PROVISIONAL_SETTINGS_CANDIDATE', 'PROVISIONAL_SETTINGS_ONLY'),
        ('TMA2', 'sean', 'SEAN_TMA2_REFERENCE', 'USER_CONFIRMED_TMA2_REFERENCE'),
    ):
        slots = settings.loc[settings['working_set_role'].eq(role)].copy()
        if (len(slots) != 9
                or slots['candidate_core_label_evidence'].duplicated().any()
                or set(slots['candidate_core_label_evidence']) != set(TMA1_CORES)
                or not slots['assigned_tma'].eq(tma).all()
                or not slots['collaborator'].eq(collaborator).all()
                or not slots['selection_status'].eq(status).all()
                or slots['path'].eq('').any()
                or slots['path'].duplicated().any()):
            raise ValueError(f'{tma} settings slots must resolve exactly one eligible A1-C3')
        for slot in slots.to_dict('records'):
            core, path = slot['candidate_core_label_evidence'], slot['path']
            source = inputs.loc[
                inputs['path'].eq(path) & inputs['assigned_tma'].eq(tma)
                & inputs['collaborator'].eq(collaborator)
            ]
            if len(source) != 1 or core not in json.loads(source.iloc[0]['candidate_core_ids']):
                raise ValueError(f'{tma} settings path/label lacks exact eligible source evidence')
            if not Path(path).is_file():
                raise ValueError(f'Final selected source does not exist: {path}')
            if tma == 'TMA1':
                match = selected.loc[selected['confirmed_core_id'].eq(core)]
                if match.iloc[0]['path'] != path:
                    raise ValueError(
                        'Selected Dylan paths/labels differ from resolved '
                        'settings slots'
                    )
                row = match.iloc[0].to_dict()
                row['core_identity_authority'] = 'HUMAN_LEDGER_HE_TMA_CORRESPONDENCE'
            else:
                row = {**source.iloc[0].to_dict(),
                       **{field: '' for field in SELECTION_MANUAL_DEFAULTS}}
                row.update(
                    confirmed_core_id=core, selection_status='SELECTED',
                    core_identity_review='CONFIRMED', he_alignment_review='REVIEWED',
                    core_identity_authority='USER_CONFIRMED_TMA2_REFERENCE_CORRESPONDENCE',
                    reviewer='user',
                    selection_reason=(
                        'User explicitly confirmed this exact Sean/TMA2 reference path '
                        'and core label against H&E/TMA correspondence; no independent '
                        'technical PASS reviews are asserted.'
                    ),
                )
            rows.append(row)
    final = pd.DataFrame(rows).fillna('').astype(str)
    if final['path'].duplicated().any():
        raise ValueError('Final analysis manifest contains duplicate selected paths')
    for name, path in sources.items():
        if _full_file_hash(path) != hashes[name]:
            raise ValueError(f'{name} changed while constructing final handoff')
        final[f'{name}_path'] = str(path)
        final[f'{name}_sha256'] = hashes[name]
    return final.sort_values(['assigned_tma', 'confirmed_core_id']).reset_index(drop=True)


def write_final_analysis_manifest(final_path, *, ledger_path, assignment_manifest_path,
                                  queue_path, settings_manifest_path):
    """Atomically publish only the fully validated final 18-acquisition handoff."""
    final = _build_final_analysis_manifest(
        ledger_path=ledger_path, assignment_manifest_path=assignment_manifest_path,
        queue_path=queue_path, settings_manifest_path=settings_manifest_path,
    )
    destination = Path(final_path)
    source_paths = final.loc[0, [
        'ledger_path', 'assignment_manifest_path', 'queue_path', 'settings_manifest_path',
    ]]
    if str(destination.resolve()) in set(source_paths):
        raise ValueError('Final output must not overwrite its source evidence')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=destination.parent, suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            final.to_csv(handle, index=False)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return final


def load_final_analysis_manifest(final_path, *, ledger_path, assignment_manifest_path,
                                 queue_path, settings_manifest_path):
    """Read only; reject any deviation from the current validated source handoff."""
    expected = _build_final_analysis_manifest(
        ledger_path=ledger_path, assignment_manifest_path=assignment_manifest_path,
        queue_path=queue_path, settings_manifest_path=settings_manifest_path,
    )
    recorded = pd.read_csv(final_path, dtype=str, keep_default_na=False)
    if not recorded.equals(expected):
        raise ValueError('Final analysis manifest differs from validated source handoff')
    return recorded


def core_selection_counts(ledger):
    """Count evidence and manual choices without mapping candidate labels to identity."""
    rows = []
    candidate_lists = ledger['candidate_core_ids'].map(json.loads)
    for core in TMA1_CORES:
        confirmed = ledger['confirmed_core_id'].eq(core)
        rows.append({
            'core_id': core,
            'candidate_path_count': candidate_lists.map(lambda ids: core in ids).sum(),
            'confirmed_path_count': confirmed.sum(),
            'selected_path_count': (
                confirmed & ledger['selection_status'].eq('SELECTED')
            ).sum(),
        })
    return pd.DataFrame(rows)


def _require_raw_hs(row, *, allow_auxiliary=False):
    roles = {PRIMARY_ROLE, AUXILIARY_ROLE} if allow_auxiliary else {PRIMARY_ROLE}
    if (row['processing_role'] not in roles or row['classification'] != 'raw_acquisition'
            or row['modality'] != 'hyperspectral' or Path(row['path']).suffix.lower() != '.czi'):
        raise ValueError(
            'Only explicitly reviewed raw HS CZI rows may be decoded; '
            'FLIM and derivatives are excluded'
        )
    return Path(row['path'])


def _layout(reader):
    blocks = reader.get_dims_shape()
    if len(blocks) != 1:
        raise ValueError(f'Unsupported multiple dimension blocks/scenes: {blocks}')
    dims = blocks[0]
    if not {'C', 'Y', 'X'} <= set(dims):
        raise ValueError(f'Explicit C/Y/X axes required: {dims}')
    if dims['C'][0] != 0:
        raise ValueError(f'Nonzero C-axis origin requires manual channel mapping: {dims}')
    for axis, (start, stop) in dims.items():
        if stop <= start:
            raise ValueError(f'Invalid {axis} extent: {(start, stop)}')
        if axis not in {'C', 'M', 'Y', 'X'} and stop - start != 1:
            raise ValueError(f'Unsupported non-singleton {axis}; no Z/T/scene guessing: {dims}')
    if reader.pixel_type not in PIXEL_DTYPES:
        raise ValueError(f'Unsupported non-grayscale/native pixel type: {reader.pixel_type}')
    return dims


def _leaf_values(element):
    """Preserve child-element text, attributes and relative XML paths, not guessed units."""
    values = []
    def visit(node, path):
        if node.text and node.text.strip() or node.attrib:
            values.append({'xml_path': path, 'text': (node.text or '').strip(),
                           'attributes': dict(node.attrib)})
        for index, child in enumerate(node):
            visit(child, f'{path}/{child.tag}[{index}]')
    visit(element, element.tag)
    return json.dumps(values, ensure_ascii=False)


def _acquisition_fields(metadata, *, tokens=None):
    """Retain exact XML paths/text for acquisition settings outside channel nodes too."""
    selected = []
    if tokens is None:
        tokens = ('wavelength', 'gain', 'pixeltime', 'dwell', 'exposure', 'detector',
                  'acquisitionmode', 'laser', 'filter', 'binning', 'pixeltype',
                  'pinhole', 'objective', 'scaling', 'distance')
    def visit(node, path):
        if any(token in node.tag.lower() for token in tokens):
            selected.append({'xml_path': path, 'text': (node.text or '').strip(),
                             'attributes': dict(node.attrib),
                             'descendant_fields_json': _leaf_values(node)})
            return
        for index, child in enumerate(node):
            visit(child, f'{path}/{child.tag}[{index}]')
    visit(metadata, metadata.tag)
    return json.dumps(selected, ensure_ascii=False)


def inspect_raw_metadata(row, *, reader=None):
    """Read CZI metadata only; channel order is XML order checked against the C axis."""
    if reader is None:
        from aicspylibczi import CziFile
        reader = CziFile(str(_require_raw_hs(row, allow_auxiliary=True)))
    path = _require_raw_hs(row, allow_auxiliary=True)
    metadata = reader.meta
    dimensions = reader.get_dims_shape()
    nodes = metadata.findall('.//Information/Image/Dimensions/Channels/Channel')
    count = (
        dimensions[0].get('C', (0, 0))[1]
        - dimensions[0].get('C', (0, 0))[0]
        if len(dimensions) == 1
        else None
    )
    channel_rows = []
    for index, node in enumerate(nodes):
        channel_rows.append({
            'path': str(path),
            'channel_index': index,
            'channel_id': node.get('Id', ''),
            'recorded_channel_name': node.findtext(
                'Name',
                default=node.get('Name', ''),
            ),
            'excitation_wavelength_recorded': node.findtext(
                'ExcitationWavelength',
                default='',
            ),
            'emission_wavelength_recorded': node.findtext(
                'EmissionWavelength',
                default='',
            ),
            'detection_wavelength_recorded': node.findtext(
                'DetectionWavelength',
                default='',
            ),
            'detection_wavelength_range_recorded': ' | '.join(
                (element.text or '').strip()
                for element in node.findall('DetectionWavelength/Ranges')
            ),
            'wavelength_start_recorded': node.findtext(
                './/WavelengthStart',
                default='',
            ),
            'wavelength_end_recorded': node.findtext(
                './/WavelengthEnd',
                default='',
            ),
            'acquisition_mode_recorded': node.findtext(
                'AcquisitionMode',
                default='',
            ),
            'detector_recorded_json': _acquisition_fields(
                node,
                tokens=('detector',),
            ),
            'gain_recorded_json': _acquisition_fields(
                node,
                tokens=('gain',),
            ),
            'pixel_dwell_exposure_recorded_json': _acquisition_fields(
                node,
                tokens=('pixeltime', 'dwell', 'exposure'),
            ),
            'binning_recorded_json': _acquisition_fields(
                node,
                tokens=('binning',),
            ),
            'laser_filter_recorded_json': _acquisition_fields(
                node,
                tokens=('laser', 'filter', 'lightsource'),
            ),
            'pixel_type': reader.pixel_type,
            'cross_file_raw_intensity_comparability':
                'BLOCKED_NOT_ESTABLISHED_SETTINGS_AND_REFERENCE_REVIEW_REQUIRED',
            'channel_xml_fields_json': _leaf_values(node),
        })
    result = {
        'path': str(path),
        'assigned_tma': row['assigned_tma'],
        'collaborator': row['collaborator'],
        'candidate_core_ids': row['candidate_core_ids'],
        'source_set': row['source_set'],
        'source_scope': row['source_scope'],
        'processing_role': row['processing_role'],
        'source_size_bytes': path.stat().st_size,
        'indexed_size_bytes': int(row['size_bytes']),
        'size_matches_index': path.stat().st_size == int(row['size_bytes']),
        'dimensions_json': json.dumps(dimensions),
        'pixel_type': reader.pixel_type,
        'dimension_geometry': (
            'native CZI tile/layout extents; not assembled mosaic shape'
        ),
        'assembled_mosaic_shape': 'NOT_DECODED_METADATA_ONLY',
        'is_mosaic': reader.is_mosaic(),
        'metadata_channel_count': len(nodes),
        'c_axis_channel_count': count,
        'channel_names_json': json.dumps(
            [
                node.findtext('Name', default=node.get('Name', ''))
                for node in nodes
            ]
        ),
        'metadata_xml_sha256': sha256(ET.tostring(metadata)).hexdigest(),
        'acquisition_xml_fields_json': _acquisition_fields(metadata),
        'metadata_provenance': (
            'aicspylibczi.meta; Information/Image/Dimensions/Channels/Channel; '
            'child elements, not attributes'
        ),
        'spectral_calibration_status':
            'RECORDED_METADATA_NOT_INDEPENDENTLY_CALIBRATED',
        'cross_file_raw_intensity_comparability':
            'BLOCKED_NOT_ESTABLISHED_SETTINGS_AND_REFERENCE_REVIEW_REQUIRED',
        'acquisition_consistency_gate': (
            'Different or missing gain/dwell/binning/detector/laser/filter/grid '
            'settings forbid pooled raw-intensity comparisons'
        ),
    }
    try:
        _layout(reader)
        if count != len(nodes) or not nodes:
            raise ValueError(f'Channel XML/C-axis mismatch: XML={len(nodes)}, C={count}')
        if not result['size_matches_index']:
            raise ValueError('Source size differs from reviewed index snapshot')
        result.update(qc_status='METADATA_READ_LAYOUT_SUPPORTED', error='')
    except ValueError as exc:
        result.update(qc_status='UNSUPPORTED_OR_CHANGED_SOURCE', error=str(exc))
    return result, channel_rows


def audit_metadata(
        inputs: pd.DataFrame, *, cache: HSCache,
        report: Callable[[str], object] = print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache metadata with stable columns across cold and sorted-JSON cache reads."""
    from aicspylibczi import CziFile

    records, channels = [], []
    for row in inputs.to_dict('records'):
        try:
            path = _require_raw_hs(row)
            signature = _file_signature(path)
            reader = CziFile(str(path))
            xml_digest = sha256(ET.tostring(reader.meta)).hexdigest()
            parameters = {
                'row': row, 'dimensions': reader.get_dims_shape(),
                'pixel_type': reader.pixel_type, 'is_mosaic': reader.is_mosaic(),
                'source_size_bytes': path.stat().st_size,
            }
            def compute() -> tuple[dict[str, object], list[dict[str, object]]]:
                value = inspect_raw_metadata(row, reader=reader)
                cache.require_unchanged(path, signature)
                return value
            result, status, key = cache.result(
                'metadata', xml_digest, parameters,
                compute,
                source_kind='exact_czi_xml_sha256', report=report,
            )
            cache.require_unchanged(path, signature)
            record, channel_records = result
            record = dict(record, cache_status=status, cache_key=key)
            records.append(record)
            channels.extend(
                dict(item, cache_status=status, cache_key=key)
                for item in channel_records
            )
        except Exception as exc:
            report(
                f"ERROR_NOT_CACHED metadata: {row['path']}; "
                f"{type(exc).__name__}: {exc}"
            )
            source_columns = (
                'path',
                'assigned_tma',
                'collaborator',
                'candidate_core_ids',
                'source_set',
                'source_scope',
                'processing_role',
            )
            source_record = {
                column: row[column] for column in source_columns
            }
            records.append(
                source_record
                | {
                    'qc_status': 'METADATA_ERROR',
                    'error': f'{type(exc).__name__}: {exc}',
                    'cache_status': 'ERROR_NOT_CACHED',
                }
            )
    # Persisted JSON sorts dictionary keys; cold results retain XML-helper order.
    # Canonicalize presentation only, preserving every source field and its dtype.
    return (
        pd.DataFrame(records).sort_index(axis=1),
        pd.DataFrame(channels).sort_index(axis=1),
    )


def preview_raw_hs(row, *, cache, channel_name='469', channel_index=None):
    """Cache the existing bounded decoder by full raw-file content and effective parameters."""
    from aicspylibczi import CziFile

    path = _require_raw_hs(row)
    digest, signature = cache.raw_identity(path)
    parameters = {
        'path': str(path), 'channel_name': channel_name, 'channel_index': channel_index,
        'mosaic_scale_factor': 0.25, 'stretch_percentiles': [1, 99],
        'jpeg_max_dimension': 850, 'jpeg_quality': 90,
        'layout_policy': 'reject_non_singleton_Z_T_scene_or_unknown_axes',
    }
    def compute():
        reader = CziFile(str(path))
        _layout(reader)
        del reader
        payload, details = czi_intensity_preview(
            path,
            channel_name=channel_name,
            channel_index=channel_index,
        )
        cache.require_unchanged(path, signature)
        return {
            'jpeg_base64': base64.b64encode(payload).decode('ascii'),
            'details': details,
        }
    result, status, key = cache.result('preview', digest, parameters, compute)
    cache.require_unchanged(path, signature)
    return (
        base64.b64decode(result['jpeg_base64'], validate=True),
        result['details'],
        status,
        key,
        digest,
    )


def display_preview_page(
    inputs,
    metadata_qc,
    *,
    cache,
    page=0,
    per_source=2,
    channel_name='469',
    channel_index=None,
):
    """Bound notebook payload by per-source pages; lexicographic order is not image ranking."""
    from IPython.display import Image, Markdown, display

    if page < 0 or not 1 <= per_source <= 6:
        raise ValueError('page must be nonnegative and per_source between 1 and 6')
    statuses = metadata_qc.set_index('path')['qc_status'].to_dict()
    records = []
    for (tma, collaborator), group in inputs.groupby(['assigned_tma', 'collaborator'], sort=True):
        group = group.sort_values(['source_set', 'path'])
        start = page * per_source
        page_label = (
            f'**{tma} / {collaborator}: acquisition page {page}; '
            f'rows {start}:{start + per_source} of {len(group)}. '
            'No best-image selection.**'
        )
        display(Markdown(page_label))
        for row in group.iloc[start:start + per_source].to_dict('records'):
            record = {
                'path': row['path'],
                'assigned_tma': tma,
                'candidate_core_ids': row['candidate_core_ids'],
                'cache_status': 'ERROR_NOT_CACHED',
                'preview_status': 'ERROR',
                'details': '',
            }
            evidence_label = (
                f"Candidate evidence `{row['candidate_core_ids']}`; "
                f"source `{row['source_set']}`. Full path:"
            )
            display(Markdown(evidence_label))
            print(row['path'])
            try:
                if statuses.get(row['path']) != 'METADATA_READ_LAYOUT_SUPPORTED':
                    raise ValueError(
                        'Metadata QC blocks preview: '
                        f"{statuses.get(row['path'], 'NOT_AUDITED')}"
                    )
                payload, details, status, key, digest = preview_raw_hs(
                    row,
                    cache=cache,
                    channel_name=channel_name,
                    channel_index=channel_index,
                )
                record.update(
                    preview_status='DISPLAY_ONLY_DECODED',
                    details=details,
                    cache_status=status,
                    cache_key=key,
                    full_file_sha256=digest,
                )
                print(status + '; ' + details)
                display(Image(data=payload, format='jpeg'))
            except Exception as exc:
                record['details'] = f'{type(exc).__name__}: {exc}'
                print('PREVIEW ERROR: ' + record['details'])
            records.append(record)
    return pd.DataFrame(records)


def iter_native_channel_qc(row, *, max_plane_bytes=256 * 1024**2, report=print):
    """Yield QC in recorded channel order, decoding only one native C plane at a time.

    Each C plane retains all unstitched M tiles and singleton axes. No whole cube,
    read_mosaic, downsampling, rotation, correction or cross-section transform.
    """
    from aicspylibczi import CziFile

    path = _require_raw_hs(row)
    reader = CziFile(str(path))
    dims = _layout(reader)
    nodes = reader.meta.findall('.//Information/Image/Dimensions/Channels/Channel')
    labels = tuple(node.findtext('Name', default=node.get('Name', '')) for node in nodes)
    if reader.is_mosaic() and 'M' not in dims:
        raise ValueError('Mosaic tile extent unavailable: cannot bound native C planes')
    if len(labels) != dims['C'][1] - dims['C'][0]:
        raise ValueError('Cannot preserve channel mapping: C-axis/XML count mismatch')
    dtype = np.dtype(PIXEL_DTYPES[reader.pixel_type])
    plane_pixels = prod(
        stop - start
        for axis, (start, stop) in dims.items()
        if axis != 'C'
    )
    plane_bytes = plane_pixels * dtype.itemsize
    if plane_bytes > max_plane_bytes:
        raise MemoryError(
            f'One native C plane requires {plane_bytes} bytes; '
            f'limit={max_plane_bytes}. No downsampling fallback.'
        )
    expected_axes = None
    for channel, label in enumerate(labels):
        report(
            f"Native raw QC C={channel}, recorded name={label!r}: {path}"
        )
        pixels, axes = reader.read_image(C=channel)
        axes = tuple((axis, int(size)) for axis, size in axes)
        if (tuple(size for _, size in axes) != pixels.shape or pixels.dtype != dtype
                or dict(axes).get('C') != 1 or pixels.nbytes > max_plane_bytes):
            raise ValueError(
                f'Native C-plane dtype/axis/memory mismatch: C={channel}; '
                f'axes={axes}; dtype={pixels.dtype}; bytes={pixels.nbytes}'
            )
        decoded = dict(axes)
        if any(
            decoded.get(axis) != stop - start
            for axis, (start, stop) in dims.items()
            if axis != 'C'
        ):
            raise ValueError(
                'Native C-plane axis extents differ from recorded layout: '
                f'axes={axes}; dimensions={dims}'
            )
        if expected_axes is not None and axes != expected_axes:
            raise ValueError(
                f'Inconsistent native tile axes at C={channel}: '
                f'{axes} != {expected_axes}'
            )
        expected_axes = axes
        pixels.flags.writeable = False
        finite = pixels if np.issubdtype(pixels.dtype, np.integer) else pixels[np.isfinite(pixels)]
        ceiling = np.iinfo(pixels.dtype).max if np.issubdtype(pixels.dtype, np.integer) else None
        zero_count = int(np.count_nonzero(finite == 0))
        ceiling_count = int(np.count_nonzero(finite == ceiling)) if ceiling is not None else None
        node = nodes[channel]
        record = {
            'path': str(path),
            'assigned_tma': row['assigned_tma'],
            'candidate_core_ids': row['candidate_core_ids'],
            'channel_index': channel,
            'recorded_channel_name': label,
            'native_dtype': str(pixels.dtype),
            'emission_wavelength_recorded': node.findtext(
                'EmissionWavelength',
                default='',
            ),
            'excitation_wavelength_recorded': node.findtext(
                'ExcitationWavelength',
                default='',
            ),
            'wavelength_start_recorded': node.findtext(
                './/WavelengthStart',
                default='',
            ),
            'wavelength_end_recorded': node.findtext(
                './/WavelengthEnd',
                default='',
            ),
            'pixel_count_including_unstitched_tiles': pixels.size,
            'native_axes_json': json.dumps(axes),
            'decoded_plane_bytes': pixels.nbytes,
            'native_tile_count': decoded.get('M', 1),
            'native_tile_shape_yx_json': json.dumps(
                [decoded['Y'], decoded['X']]
            ),
            'summary_granularity': (
                'ONE_ROW_PER_CHANNEL_AGGREGATING_ALL_NATIVE_M_'
                'PIXELS_NOT_PER_TILE_STATISTICS'
            ),
            'assembled_mosaic_shape': 'NOT_ASSEMBLED_NATIVE_TILES_PRESERVED',
            'nonfinite_count': int(pixels.size - finite.size),
            'finite_count': int(finite.size),
            'zero_count': zero_count,
            'dtype_ceiling_count': ceiling_count,
            'raw_min': finite.min().item() if finite.size else None,
            'raw_max': finite.max().item() if finite.size else None,
            'raw_mean': (
                float(finite.mean(dtype=np.float64))
                if finite.size
                else None
            ),
            'zero_fraction': (
                float(zero_count / finite.size) if finite.size else None
            ),
            'dtype_ceiling_fraction': (
                float(ceiling_count / finite.size)
                if finite.size and ceiling_count is not None
                else None
            ),
            'clipping_status': (
                'DTYPE_CEILING_IS_POTENTIAL_CLIPPING_NOT_CONFIRMED_'
                'DETECTOR_SATURATION'
            ),
            'spatial_support': (
                'all native M-tile pixels aggregated per channel; '
                'overlaps not deduplicated'
            ),
            'cross_file_raw_intensity_comparability':
                'BLOCKED_NOT_ESTABLISHED_SETTINGS_AND_REFERENCE_REVIEW_REQUIRED',
        }
        del finite, pixels
        yield record


def cached_native_channel_qc(row, *, cache, max_plane_bytes=256 * 1024**2, report=print):
    """Cache only the small native-QC rows, not native pixels or spectral cubes."""
    path = _require_raw_hs(row)
    digest, signature = cache.raw_identity(path)
    parameters = {
        'path': str(path), 'assigned_tma': row['assigned_tma'],
        'candidate_core_ids': row['candidate_core_ids'], 'max_plane_bytes': max_plane_bytes,
        'reader': 'read_image(C=channel)', 'mosaic_policy': 'retain_unstitched_M',
        'qc': ['native_dtype', 'axes', 'min', 'max', 'mean', 'nonfinite', 'zero', 'dtype_ceiling'],
    }
    def compute():
        result = list(iter_native_channel_qc(row, max_plane_bytes=max_plane_bytes, report=report))
        cache.require_unchanged(path, signature)
        return result
    records, status, key = cache.result('native_qc', digest, parameters, compute, report=report)
    cache.require_unchanged(path, signature)
    return (
        pd.DataFrame(records)
        .assign(
            cache_status=status,
            cache_key=key,
            full_file_sha256=digest,
        )
    )


def _parse_detection_range(channel_xml_fields_json):
    """Read one nested detection range; rounded channel/emission labels are not bounds."""
    fields = json.loads(channel_xml_fields_json)
    range_fields = [
        field for field in fields
        if re.fullmatch(
            r'Channel/DetectionWavelength\[\d+\]/Ranges\[\d+\]',
            field.get('xml_path', ''),
        )
    ]
    if (len(range_fields) != 1
            or not range_fields[0]['xml_path'].endswith('/Ranges[0]')):
        raise ValueError('Exactly one nested DetectionWavelength/Ranges[0] is required')
    ranges = [range_fields[0]['text'].strip()]
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    match = re.fullmatch(rf'\s*({number})\s*-\s*({number})\s*', ranges[0])
    if match is None:
        raise ValueError(f'Invalid detection wavelength range: {ranges[0]!r}')
    start, end = map(float, match.groups())
    if not np.isfinite([start, end]).all() or not 0 < start < end:
        raise ValueError('Detection band bounds must be finite, positive and increasing')
    return ranges[0], start, end


def _spectral_channel_grid(channel_rows, *, expected_count=20):
    """Validate recorded C order and exact, contiguous equal-width wavelength bands."""
    channels = pd.DataFrame(channel_rows).copy()
    if (len(channels) != expected_count
            or not np.array_equal(channels['channel_index'], np.arange(expected_count))):
        raise ValueError('Spectral channel count/order does not match native C order')
    if (channels['channel_id'].eq('').any()
            or channels['channel_id'].duplicated().any()):
        raise ValueError('Unique recorded channel IDs are required')
    ranges = [_parse_detection_range(value)
              for value in channels['channel_xml_fields_json']]
    starts = np.asarray([item[1] for item in ranges])
    ends = np.asarray([item[2] for item in ranges])
    centers = (starts + ends) / 2
    widths = ends - starts
    # Absolute nm tolerance accommodates decimal XML precision, not rounded labels.
    tolerance_nm = 1e-6
    if not (np.diff(starts) > 0).all():
        raise ValueError('Detection band starts must increase in native channel order')
    if not np.allclose(widths, widths[0], rtol=0, atol=tolerance_nm):
        raise ValueError('Detection bands must have equal widths')
    if not np.allclose(starts[1:], ends[:-1], rtol=0, atol=tolerance_nm):
        raise ValueError('Detection bands must be contiguous')
    steps = np.diff(centers)
    if not (steps > 0).all() or not np.allclose(
            steps, steps[0], rtol=0, atol=tolerance_nm):
        raise ValueError('Detection band centers must be uniformly spaced')
    return channels.assign(
        detection_wavelength_range_recorded=[item[0] for item in ranges],
        detection_band_start_nm=starts,
        detection_band_end_nm=ends,
        detection_band_center_nm=centers,
        detection_band_width_nm=widths,
    )


def _validated_spectral_manifest(final_manifest, cache):
    """Require the exact current source-bound final handoff, not provisional inputs."""
    if len(final_manifest) != 18 or final_manifest['path'].duplicated().any():
        raise ValueError('Spectral analysis requires the complete final 18-path manifest')
    source_names = ('ledger', 'assignment_manifest', 'queue', 'settings_manifest')
    sources = {}
    for name in source_names:
        column = f'{name}_path'
        if column not in final_manifest or final_manifest[column].nunique() != 1:
            raise ValueError(f'Final manifest lacks unique {column} source binding')
        sources[column] = final_manifest.iloc[0][column]
    recorded = load_final_analysis_manifest(cache.manifest_path, **sources)
    if not recorded.equals(final_manifest):
        raise ValueError('Supplied rows differ from the validated final manifest')
    cache._context()
    return recorded


def _spectral_layout(metadata):
    if metadata['qc_status'] != 'METADATA_READ_LAYOUT_SUPPORTED':
        raise ValueError(f"Unsupported spectral source metadata: {metadata.get('error')}")
    dimensions = json.loads(metadata['dimensions_json'])[0]
    if (metadata['metadata_channel_count'] != 20
            or metadata['c_axis_channel_count'] != 20
            or metadata['pixel_type'] != 'gray16'
            or dimensions.get('M', [0, 0])[1] - dimensions.get('M', [0, 0])[0] != 4):
        raise ValueError('Reviewed spectral inputs require 20 gray16 C channels and four M tiles')
    return dimensions


def audit_spectral_channel_maps(
        final_manifest: pd.DataFrame, *, cache: HSCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit exact metadata with cache-independent columns, without decoding pixels."""
    inputs = _validated_spectral_manifest(final_manifest, cache)
    metadata, channels = audit_metadata(inputs, cache=cache)
    maps, audits, tma_grids = [], [], {}
    for row in inputs.to_dict('records'):
        record = metadata.loc[metadata['path'].eq(row['path'])].iloc[0]
        dimensions = _spectral_layout(record)
        current = _spectral_channel_grid(
            channels.loc[channels['path'].eq(row['path'])]
            .sort_values('channel_index').to_dict('records')
        )
        bounds = current[['detection_band_start_nm', 'detection_band_end_nm']].to_numpy()
        prior = tma_grids.setdefault(row['assigned_tma'], bounds)
        if not np.allclose(bounds, prior, rtol=0, atol=1e-6):
            raise ValueError(f"Detection grids differ within {row['assigned_tma']}")
        current = current.assign(
            assigned_tma=row['assigned_tma'],
            confirmed_core_id=row['confirmed_core_id'],
            metadata_xml_sha256=record['metadata_xml_sha256'],
            dimensions_json=record['dimensions_json'],
        )
        maps.append(current)
        audits.append({
            'assigned_tma': row['assigned_tma'],
            'confirmed_core_id': row['confirmed_core_id'],
            'path': row['path'],
            'channel_count': len(current),
            'native_tile_count': dimensions['M'][1] - dimensions['M'][0],
            'first_band_center_nm': float(current['detection_band_center_nm'].iloc[0]),
            'last_band_center_nm': float(current['detection_band_center_nm'].iloc[-1]),
            'band_width_nm': float(current['detection_band_width_nm'].iloc[0]),
            'center_step_nm': float(np.diff(current['detection_band_center_nm'])[0]),
            'grid_status': 'PASS_EQUAL_WIDTH_CONTIGUOUS_UNIFORM_CENTERS',
            'wavelength_source': 'exact DetectionWavelength/Ranges midpoint; not rounded labels',
            'metadata_xml_sha256': record['metadata_xml_sha256'],
            'cache_status': record['cache_status'],
        })
    if set(tma_grids) != {'TMA1', 'TMA2'}:
        raise ValueError('Both separately reviewed TMAs must be present')
    differ = not np.allclose(tma_grids['TMA1'], tma_grids['TMA2'], rtol=0, atol=1e-6)
    grid_audit = pd.DataFrame(audits).assign(
        tma_grids_differ=differ,
        comparison_policy='Separate TMAs; no cross-TMA intensity or biological comparison',
    )
    return pd.concat(maps, ignore_index=True), grid_audit


def _spectral_channel_maps_equal(
        current: pd.DataFrame, supplied: pd.DataFrame,
) -> bool:
    """Compare all source fields, dtypes and C-row order, not cache/presentation order."""
    transient = ['cache_status', 'cache_key']
    return current.drop(columns=transient, errors='ignore').sort_index(axis=1).equals(
        supplied.drop(columns=transient, errors='ignore').sort_index(axis=1)
    )


def _spectral_phasor_parameters(row, channel_map, *, max_working_bytes, num_threads):
    """Bind exact bands, metadata and every computation/display parameter to cache keys."""
    return {
        'row': row,
        'channel_map': channel_map[
            ['channel_index', 'channel_id', 'recorded_channel_name',
             'detection_wavelength_range_recorded', 'detection_band_start_nm',
             'detection_band_end_nm', 'detection_band_center_nm',
             'detection_band_width_nm']
        ].to_dict('records'),
        'metadata_xml_sha256': channel_map['metadata_xml_sha256'].iloc[0],
        'dimensions_json': channel_map['dimensions_json'].iloc[0],
        'transform': 'phasorpy.phasor.phasor_from_signal',
        'axis': 0, 'harmonic': 1, 'normalize': True, 'use_fft': False,
        'dtype': 'float64', 'num_threads': num_threads,
        'max_working_bytes': max_working_bytes,
        'histogram_bins': 128, 'histogram_range': [-1.000001, 1.000001],
        'histogram_chunk_pixels': 65536,
        'spatial_filters': {
            'median': {
                'function': 'phasorpy.filter.phasor_filter_median',
                'size': 3, 'repeat': 3, 'use_scipy': False,
                'num_threads': num_threads,
            },
            'gaussian': {
                'function': 'phasorpy.filter.phasor_filter_gaussian',
                'size': 3, 'repeat': 3, 'sigma': None, 'mode': 'nearest',
                'weighted': True, 'nan_safe': True,
            },
        },
        'filter_scope': 'independent native Y/X planes; never across M or other axes',
        'sample_max_observations': 50000,
        'sample_method': 'floor(j * finite_count / sample_count), native flat order',
        'wavelength_phase_convention': (
            'PhasorPy default uniform first-harmonic channel phases, in native C order, '
            'after exact detection-band grid validation; no custom sample_phase'
        ),
        'reader': 'read_image(C=channel_index, cores=num_threads); all native M tiles',
        'mosaic_policy': 'unstitched native tiles; overlapping observations not deduplicated',
        'threshold': None, 'smoothing': None, 'correction': None,
    }


def _compute_spectral_phasor(row, channels, parameters, phasor_from_signal):
    """Read native C planes with their complete unstitched M axis, never guessed tile indices."""
    from aicspylibczi import CziFile

    reader = CziFile(str(_require_raw_hs(row)))
    metadata, current_rows = inspect_raw_metadata(row, reader=reader)
    dimensions = _spectral_layout(metadata)
    current = _spectral_channel_grid(current_rows)
    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm',
    ]
    if (metadata['metadata_xml_sha256'] != parameters['metadata_xml_sha256']
            or not current[columns].equals(channels[columns].reset_index(drop=True))
            or metadata['dimensions_json'] != parameters['dimensions_json']):
        raise ValueError('Current raw metadata differs from the audited spectral channel map')
    dtype = np.dtype(PIXEL_DTYPES[reader.pixel_type])
    plane_pixels = prod(stop - start for axis, (start, stop) in dimensions.items()
                        if axis != 'C')
    channel_count = len(current)
    bins = parameters['histogram_bins']
    chunk_pixels = min(plane_pixels, parameters['histogram_chunk_pixels'])
    # Reserve the native cube and decoder planes, raw phasors, worst-case
    # tile filter copies/workspaces, chunk temporaries and JSON histograms.
    # Filtering is sequential by method and native Y/X plane.
    working_bytes = (
        plane_pixels * (channel_count * dtype.itemsize + 2 * dtype.itemsize + 256)
        + chunk_pixels * 128 + 27 * bins * bins * 128
        + parameters['sample_max_observations'] * 2048
    )
    if working_bytes > parameters['max_working_bytes']:
        raise MemoryError(
            f'Native spectral working allocation requires at least {working_bytes} bytes; '
            f"limit={parameters['max_working_bytes']}. No downsampling fallback."
        )
    histogram = np.zeros((bins, bins), dtype=np.int64)
    channel_sums = np.zeros(channel_count, dtype=np.float64)
    total_pixels = plane_pixels
    finite_pixels = nonfinite_pixels = 0
    real_sum = imag_sum = 0.0
    lower, upper = parameters['histogram_range']
    signal = np.empty((channel_count, plane_pixels), dtype=dtype)
    expected_axes = None
    for channel in range(channel_count):
        pixels, axes = reader.read_image(C=channel, cores=parameters['num_threads'])
        axes = tuple((axis, int(size)) for axis, size in axes)
        decoded = dict(axes)
        if (pixels.dtype != dtype or pixels.size != plane_pixels
                or tuple(size for _, size in axes) != pixels.shape
                or decoded.get('C') != 1
                or any(decoded.get(axis) != stop - start
                       for axis, (start, stop) in dimensions.items() if axis != 'C')
                or (expected_axes is not None and axes != expected_axes)):
            raise ValueError(f'Native spectral C-plane layout mismatch: {axes}')
        expected_axes = axes
        signal[channel] = pixels.reshape(-1)
        channel_sums[channel] += pixels.sum(dtype=np.float64)
        del pixels
    mean, real, imag = phasor_from_signal(
        signal, axis=0, harmonic=1, use_fft=False, normalize=True,
        dtype=np.float64, num_threads=parameters['num_threads'],
    )
    del signal
    if any(array.shape != (plane_pixels,) for array in (mean, real, imag)):
        raise ValueError('Unexpected PhasorPy output shape')
    zero_pixels = int(np.count_nonzero(mean == 0))
    for start in range(0, plane_pixels, parameters['histogram_chunk_pixels']):
        stop = min(start + parameters['histogram_chunk_pixels'], plane_pixels)
        g, s = real[start:stop], imag[start:stop]
        valid = np.isfinite(g) & np.isfinite(s)
        count = int(np.count_nonzero(valid))
        finite_pixels += count
        nonfinite_pixels += valid.size - count
        if not count:
            continue
        g, s = g[valid], s[valid]
        if (g.min() < lower or g.max() > upper
                or s.min() < lower or s.max() > upper):
            raise ValueError('Phasor coordinate outside nonnegative-spectrum bounds')
        real_sum += float(g.sum(dtype=np.float64))
        imag_sum += float(s.sum(dtype=np.float64))
        counts, _, _ = np.histogram2d(
            g, s, bins=bins, range=((lower, upper), (lower, upper)),
        )
        histogram += counts.astype(np.int64)
    filters = _cached_spatial_filter_distributions(
        mean, real, imag, expected_axes, parameters,
    )
    sample = _systematic_phasor_sample(
        real, imag, finite_pixels,
        max_samples=parameters['sample_max_observations'],
        chunk_pixels=parameters['histogram_chunk_pixels'],
    )
    del mean, real, imag
    if int(histogram.sum()) != finite_pixels or nonfinite_pixels != zero_pixels:
        raise ValueError('Phasor finite/undefined counts do not match zero-signal observations')
    return {
        'summary': {
            'assigned_tma': row['assigned_tma'],
            'confirmed_core_id': row['confirmed_core_id'],
            'path': row['path'],
            'channel_count': channel_count,
            'native_tile_count': dimensions['M'][1] - dimensions['M'][0],
            'within_image_pixel_observations': total_pixels,
            'finite_phasor_pixel_observations': finite_pixels,
            'undefined_zero_signal_pixel_observations': zero_pixels,
            'mean_real_within_image': real_sum / finite_pixels if finite_pixels else None,
            'mean_imag_within_image': imag_sum / finite_pixels if finite_pixels else None,
            'native_axes_json': json.dumps(expected_axes),
            'working_allocation_bound_bytes': working_bytes,
            'observation_unit': 'native tile pixels within one acquisition; not biological n',
            'mosaic_overlap_policy': 'unstitched M tiles; overlaps counted, not deduplicated',
            'claim_label': 'Exploratory',
            'intensity_comparison': 'not established; no cross-TMA comparison',
        },
        'channel_mean_raw_intensity': (channel_sums / total_pixels).tolist(),
        'phasor_histogram': histogram.tolist(),
        'filter_distributions': filters,
        'systematic_raw_phasor_sample': sample,
    }


def _register_project_arial() -> Path:
    """Register and verify the project font through the shared style helper."""
    return plot_style.register_project_arial()


def _format_phasor_count(value: float, position: int | None = None) -> str:
    """Plain numeric log-count labels; never mathtext or reduced-size exponents."""
    return f'{value:,.0f}' if value >= 1 else f'{value:g}'


def _spectral_phasor_figures(results, channels, assigned_tma, plot_dir):
    """Save separate acquisition spectra/phasor densities and return their editable PDF paths."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from matplotlib.colors import LogNorm

    _register_project_arial()
    style = _pseudocolor_style()
    with mpl.rc_context(style):
        spectrum, spectral_axes = plt.subplots(
            3, 3, figsize=(180 / 25.4, 160 / 25.4), squeeze=False,
        )
        phasor, phasor_axes = plt.subplots(
            3, 3, figsize=(180 / 25.4, 160 / 25.4), squeeze=False,
        )
        for result, spec_ax, phase_ax in zip(
                results, spectral_axes.flat, phasor_axes.flat):
            summary = result['summary']
            core, path = summary['confirmed_core_id'], summary['path']
            wavelengths = channels.loc[
                channels['path'].eq(path), 'detection_band_center_nm'
            ].to_numpy()
            spec_ax.plot(wavelengths, result['channel_mean_raw_intensity'], color='#0072B2')
            spec_ax.set_xlabel(f'{core}: exact detection-band midpoint (nm)')
            spec_ax.set_ylabel('Mean raw intensity (native units)')
            density = np.asarray(result['phasor_histogram'], dtype=np.int64)
            image = phase_ax.imshow(
                np.ma.masked_equal(density.T, 0), origin='lower',
                extent=(-1.000001, 1.000001, -1.000001, 1.000001),
                interpolation='nearest', cmap='cividis', aspect='equal',
                norm=LogNorm(vmin=1, vmax=max(2, int(density.max()))),
            )
            colorbar = phasor.colorbar(
                image, ax=phase_ax, fraction=.046, pad=.04,
                format=FuncFormatter(_format_phasor_count),
            )
            colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
            colorbar.ax.tick_params(
                labelsize=5,
                width=.5,
                which='both',
            )
            for text in colorbar.ax.get_yticklabels():
                text.set_fontfamily('Arial')
                text.set_fontsize(5)
            colorbar.set_label('Within-image pixel observations')
            colorbar.outline.set_linewidth(.5)
            phase_ax.set_xlabel(f'{core}: normalized first-harmonic G')
            phase_ax.set_ylabel('Normalized first-harmonic S')
            for axis in (spec_ax, phase_ax):
                axis.set_axisbelow(True)
                axis.grid(color='#E5E7EB', linewidth=.5)
                axis.tick_params(length=2, pad=1, width=.5)
        for figure in (spectrum, phasor):
            figure.subplots_adjust(left=.09, right=.95, bottom=.14, top=.96,
                                   hspace=.55, wspace=.75)
            figure.text(
                .09, .06,
                f'{assigned_tma} only; native M overlaps retained; pixels are not biological n.',
            )
            figure.text(
                .09, .035, 'Internal review only; not Cell manuscript figures. '
                'Generator: clinical_roi_hyperspectral.py',
            )
        spectrum.text(.09, .085, 'Independent intensity axes; no intensity comparability established.')
        spectrum_path = plot_dir / f'{assigned_tma.lower()}_raw_spectral_signatures.pdf'
        phasor_path = plot_dir / f'{assigned_tma.lower()}_spectral_phasor_histograms.pdf'
        spectrum.savefig(spectrum_path)
        phasor.savefig(phasor_path)
        # The notebook previews these PDFs in memory, not Matplotlib's PNG renderer.
        plt.close(spectrum)
        plt.close(phasor)
    return spectrum_path, phasor_path


def _write_spectral_analysis_receipt(
        selected, summary, parameters_by_path, *, cache, cache_events, artifacts,
        destination):
    """Atomically bind one TMA's exploratory artifacts to exact sources and computation."""
    first = selected.iloc[0]
    source_bindings = {
        name: {
            'path': first[f'{name}_path'],
            'sha256': first[f'{name}_sha256'],
        }
        for name in ('ledger', 'assignment_manifest', 'queue', 'settings_manifest')
    }
    receipt = {
        'schema': 'clinical_hs_spectral_phasor_analysis_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'claim_label': 'Exploratory',
        'assigned_tma': first['assigned_tma'],
        'final_manifest': {
            'path': str(cache.manifest_path.resolve()),
            'sha256': cache.manifest_sha256,
        },
        'source_bindings': source_bindings,
        'selected_acquisitions': (
            summary.astype(object).where(pd.notna(summary), None).to_dict('records')
        ),
        'raw_czi_full_sha256': dict(zip(summary['path'], summary['full_file_sha256'])),
        'parameters_and_exact_band_maps': parameters_by_path,
        'helper_code_sha256': dict(cache.code_sha256),
        'plot_style_sha256': cache.plot_style_sha256,
        'package_versions': dict(cache.package_versions),
        'cache_root': str(cache.root.resolve()),
        'cache_events': cache_events,
        'output_artifacts': {
            name: {'path': str(Path(path).resolve()), 'sha256': _full_file_hash(path)}
            for name, path in artifacts.items()
        },
        'limitations': [
            'Native unstitched M-tile observations include background and overlaps.',
            'No tissue mask, deduplication, intensity threshold or correction applied.',
            'Pixels are within-image observations, not independent biological n.',
            'No cross-TMA intensity comparison, component attribution or biological inference.',
        ],
    }
    cache._context()
    for binding in source_bindings.values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Source binding changed before spectral analysis receipt publication')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(receipt))
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def analyze_tma_spectral_phasors(
        final_manifest, channel_map, *, assigned_tma, cache, table_dir, plot_dir,
        max_working_bytes, num_threads):
    """Return one TMA's first-harmonic acquisition summary and two editable PDF paths."""
    if assigned_tma not in {'TMA1', 'TMA2'}:
        raise ValueError('assigned_tma must be TMA1 or TMA2; pooling TMAs is prohibited')
    if (not isinstance(max_working_bytes, int) or max_working_bytes <= 0
            or not isinstance(num_threads, int) or num_threads <= 0):
        raise ValueError('Positive integer working-memory and thread limits are required')
    version = importlib.metadata.version('phasorpy')
    if version.split('.')[:2] != ['0', '12']:
        raise RuntimeError(f'PhasorPy 0.12 is required; found {version}')
    from phasorpy.phasor import phasor_from_signal

    inputs = _validated_spectral_manifest(final_manifest, cache)
    selected = inputs.loc[inputs['assigned_tma'].eq(assigned_tma)]
    if (len(selected) != 9
            or set(selected['confirmed_core_id']) != set(TMA1_CORES)
            or set(channel_map['path']) != set(inputs['path'])):
        raise ValueError('A complete final-manifest channel audit and nine A1-C3 rows are required')
    results, summaries, parameters_by_path, raw_signatures = [], [], {}, {}
    event_start = len(cache.events)
    for row in selected.to_dict('records'):
        channels = channel_map.loc[channel_map['path'].eq(row['path'])].reset_index(drop=True)
        validated = _spectral_channel_grid(channels.to_dict('records'))
        for column in ('detection_band_start_nm', 'detection_band_end_nm',
                       'detection_band_center_nm', 'detection_band_width_nm'):
            if not np.array_equal(channels[column], validated[column]):
                raise ValueError('Supplied detection-band coordinates differ from exact XML')
        parameters = _spectral_phasor_parameters(
            row, channels, max_working_bytes=max_working_bytes, num_threads=num_threads,
        )
        path = _require_raw_hs(row)
        digest, signature = cache.raw_identity(path)
        parameters_by_path[str(path)] = parameters
        raw_signatures[str(path)] = signature
        def compute():
            result = _compute_spectral_phasor(row, channels, parameters, phasor_from_signal)
            cache.require_unchanged(path, signature)
            return result
        result, status, key = cache.result('spectral_phasor', digest, parameters, compute)
        cache.require_unchanged(path, signature)
        results.append(result)
        summaries.append(result['summary'] | {
            'full_file_sha256': digest, 'metadata_xml_sha256': parameters['metadata_xml_sha256'],
            'phasorpy_version': version, 'cache_status': status, 'cache_key': key,
        })
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary_path = table_dir / f'{assigned_tma.lower()}_spectral_phasor_acquisitions.csv'
    summary.to_csv(summary_path, index=False)
    spectrum_path, phasor_path = _spectral_phasor_figures(
        results, channel_map, assigned_tma, plot_dir,
    )
    for path, signature in raw_signatures.items():
        cache.require_unchanged(path, signature)
    receipt_path = table_dir / f'{assigned_tma.lower()}_spectral_phasor_receipt.json'
    _write_spectral_analysis_receipt(
        selected, summary, parameters_by_path,
        cache=cache, cache_events=cache.events[event_start:],
        artifacts={
            'acquisition_summary_csv': summary_path,
            'raw_spectral_signatures_pdf': spectrum_path,
            'spectral_phasor_histograms_pdf': phasor_path,
        },
        destination=receipt_path,
    )
    summary.attrs['analysis_receipt_path'] = str(receipt_path)
    summary.attrs['spectral_phasor_results'] = results
    summary.attrs['spectral_phasor_context'] = {
        'receipt_sha256': _full_file_hash(receipt_path),
        'result_sha256': [sha256(_json_bytes(value)).hexdigest() for value in results],
        'parameters_by_path': parameters_by_path,
        'raw_signatures': raw_signatures,
    }
    return summary, spectrum_path, phasor_path


def _systematic_phasor_sample(
        real, imag, finite_count, *, max_samples=50000, chunk_pixels=65536):
    """Sample finite ranks exactly, without a full-size index/mask allocation."""
    count = min(finite_count, max_samples)
    ranks = np.arange(count, dtype=np.int64) * finite_count // max(1, count)
    indices, coordinates = [], []
    seen = 0
    for start in range(0, real.size, chunk_pixels):
        stop = min(start + chunk_pixels, real.size)
        g, s = real[start:stop], imag[start:stop]
        valid = np.flatnonzero(np.isfinite(g) & np.isfinite(s))
        left = int(np.searchsorted(ranks, seen))
        right = int(np.searchsorted(ranks, seen + valid.size))
        selected = valid[ranks[left:right] - seen]
        indices.extend((start + selected).tolist())
        coordinates.extend(np.column_stack((g[selected], s[selected])).tolist())
        seen += valid.size
    if seen != finite_count or len(indices) != count:
        raise ValueError('Finite observation count changed while sampling raw phasors')
    return {
        'finite_population_count': finite_count,
        'sample_count': count, 'flat_indices': indices, 'coordinates_g_s': coordinates,
        'method': 'systematic finite ranks in native decoder C-plane flat order',
    }


def _cached_spatial_filter_distributions(mean, real, imag, axes, parameters):
    """Apply phasor-coordinate filters to one native Y/X plane at a time."""
    from phasorpy.filter import phasor_filter_gaussian, phasor_filter_median

    shape = tuple(size for _, size in axes)
    names = tuple(axis for axis, _ in axes)
    if (names.count('Y') != 1 or names.count('X') != 1
            or len(set(names)) != len(names)):
        raise ValueError('Unique native Y and X axes are required for spatial filtering')
    spatial = (names.index('Y'), names.index('X'))
    others = tuple(index for index in range(len(shape)) if index not in spatial)
    order = others + spatial
    views = [
        value.reshape(shape).transpose(order) for value in (mean, real, imag)
    ]
    output = {}
    functions = {
        'median': phasor_filter_median, 'gaussian': phasor_filter_gaussian,
    }
    bins = parameters['histogram_bins']
    bounds = parameters['histogram_range']
    for method, function in functions.items():
        options = dict(parameters['spatial_filters'][method])
        options.pop('function')
        histogram = np.zeros((bins, bins), dtype=np.int64)
        finite_count = paired_count = undefined_count = 0
        displacement_sum = squared_displacement_sum = 0.0
        for index in np.ndindex(views[0].shape[:-2]):
            native_mean, native_g, native_s = (view[index] for view in views)
            filtered_mean, filtered_g, filtered_s = function(
                native_mean, native_g, native_s, **options,
            )
            # Keep only one filtered plane and chunk-sized comparison buffers.
            g_flat, s_flat = filtered_g.reshape(-1), filtered_s.reshape(-1)
            raw_g, raw_s = native_g.reshape(-1), native_s.reshape(-1)
            for start in range(0, g_flat.size, parameters['histogram_chunk_pixels']):
                stop = start + parameters['histogram_chunk_pixels']
                g, s = g_flat[start:stop], s_flat[start:stop]
                rg, rs = raw_g[start:stop], raw_s[start:stop]
                valid = np.isfinite(g) & np.isfinite(s)
                raw_valid = np.isfinite(rg) & np.isfinite(rs)
                if not np.array_equal(valid, raw_valid):
                    raise ValueError('Spatial filter changed undefined zero-signal support')
                count = int(valid.sum())
                finite_count += count
                undefined_count += valid.size - count
                if not count:
                    continue
                g, s = g[valid], s[valid]
                if (g.min() < bounds[0] or g.max() > bounds[1]
                        or s.min() < bounds[0] or s.max() > bounds[1]):
                    raise ValueError('Filtered phasor coordinate outside histogram bounds')
                histogram += np.histogram2d(
                    g, s, bins=bins, range=(bounds, bounds),
                )[0].astype(np.int64)
                displacement = np.hypot(g - rg[valid], s - rs[valid])
                paired_count += count
                displacement_sum += float(displacement.sum(dtype=np.float64))
                squared_displacement_sum += float(
                    np.square(displacement).sum(dtype=np.float64)
                )
            del filtered_mean, filtered_g, filtered_s, g_flat, s_flat
        output[method] = {
            'phasor_histogram': histogram.tolist(),
            'finite_phasor_pixel_observations': finite_count,
            'undefined_zero_signal_pixel_observations': undefined_count,
            'paired_raw_filtered_observations': paired_count,
            'mean_paired_g_s_displacement': (
                displacement_sum / paired_count if paired_count else None
            ),
            'rms_paired_g_s_displacement': (
                (squared_displacement_sum / paired_count) ** .5
                if paired_count else None
            ),
        }
    return output


def _phasor_sensitivity_inputs(summary, *, assigned_tma, cache):
    """Require intact base results and receipts, without reopening raw CZI bytes."""
    if assigned_tma not in {'TMA1', 'TMA2'}:
        raise ValueError('assigned_tma must be TMA1 or TMA2; pooling TMAs is prohibited')
    required = {'path', 'assigned_tma', 'confirmed_core_id', 'full_file_sha256',
                'cache_key'}
    if (not required.issubset(summary.columns) or len(summary) != 9
            or not summary['assigned_tma'].eq(assigned_tma).all()
            or set(summary['confirmed_core_id']) != set(TMA1_CORES)
            or summary['path'].duplicated().any()):
        raise ValueError('Require the complete nine-core base summary for this TMA')
    cache._context()
    context = summary.attrs.get('spectral_phasor_context')
    results = summary.attrs.get('spectral_phasor_results')
    receipt_path = summary.attrs.get('analysis_receipt_path')
    if not context or not results or not receipt_path:
        raise ValueError('Base summary lacks source-bound cached phasor inputs')
    expected_receipt_sha256 = context['receipt_sha256']
    actual_receipt_sha256 = _full_file_hash(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise ValueError(
            f'Base analysis receipt changed: path={receipt_path}; '
            f'expected={expected_receipt_sha256}; current={actual_receipt_sha256}'
        )
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    if (receipt['helper_code_sha256'] != cache.code_sha256
            or receipt['package_versions'] != cache.package_versions
            or receipt['final_manifest']['sha256'] != cache.manifest_sha256
            or receipt['parameters_and_exact_band_maps']
            != context['parameters_by_path']):
        raise ValueError('Base analysis method, dependencies or manifest changed')
    recorded = pd.DataFrame(receipt['selected_acquisitions'])
    normalized = summary.astype(object).where(pd.notna(summary), None)
    if normalized.to_dict('records') != recorded.to_dict('records'):
        raise ValueError('Base summary differs from its source-bound receipt')
    final = pd.read_csv(cache.manifest_path, dtype=str, keep_default_na=False)
    final = _validated_spectral_manifest(final, cache)
    selected = final.loc[final['assigned_tma'].eq(assigned_tma)]
    if set(selected['path']) != set(summary['path']):
        raise ValueError('Base acquisitions differ from the final reviewed handoff')
    hashes = [sha256(_json_bytes(value)).hexdigest() for value in results]
    if hashes != context['result_sha256'] or len(results) != len(summary):
        raise ValueError('Cached phasor sensitivity inputs changed')
    events = {event['key']: event for event in receipt['cache_events']}
    for row, result, digest in zip(summary.to_dict('records'), results, hashes):
        path = row['path']
        cache.require_unchanged(path, context['raw_signatures'][path])
        event = events.get(row['cache_key'])
        if event is None or event['input_sha256'] != row['full_file_sha256']:
            raise ValueError('Missing source-bound base cache event')
        envelope = json.loads(Path(event['cache_path']).read_text(encoding='utf-8'))
        if (envelope['result_sha256'] != digest
                or sha256(_json_bytes(envelope['result'])).hexdigest() != digest
                or result['summary']['path'] != path):
            raise ValueError('Base cache payload differs from sensitivity inputs')
    return selected, results, context


def _write_phasor_sensitivity_outputs(
        output, selected, base_summary, parameters, *, cache, event_start,
        table_dir, stem, pdf_paths):
    table_path = table_dir / f'{stem}.csv'
    output.to_csv(table_path, index=False)
    receipt_path = table_dir / f'{stem}_receipt.json'
    context = base_summary.attrs['spectral_phasor_context']
    for path, signature in context['raw_signatures'].items():
        cache.require_unchanged(path, signature)
    if (_full_file_hash(base_summary.attrs['analysis_receipt_path'])
            != context['receipt_sha256']):
        raise ValueError('Base analysis receipt changed before sensitivity publication')
    _write_spectral_analysis_receipt(
        selected, output, parameters, cache=cache,
        cache_events=cache.events[event_start:],
        artifacts={
            'summary_csv': table_path,
            'base_analysis_receipt': base_summary.attrs['analysis_receipt_path'],
            **{f'pdf_{index}': path for index, path in enumerate(pdf_paths)},
        },
        destination=receipt_path,
    )
    output.attrs['analysis_receipt_path'] = str(receipt_path)


def _phasor_sensitivity_pdf(
        results: list[dict[str, Any]],
        rows: pd.DataFrame,
        *,
        assigned_tma: str,
        destination: Path,
        k: int | None = None,
) -> None:
    """Render density and model summaries without selecting a method or class."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import FuncFormatter, NullFormatter

    _register_project_arial()
    style = _pseudocolor_style()
    filter_view = k is None
    nrows = 9 if filter_view else 3
    height = 280 if filter_view else 160
    samples: list[np.ndarray] = []
    density_norm = None
    density_mesh = None
    component_colors = tuple(plt.get_cmap('tab20').colors)
    component_ids: list[int] = []

    if not filter_view:
        samples = [
            np.asarray(
                result['systematic_raw_phasor_sample']['coordinates_g_s'],
                dtype=np.float64,
            ).reshape(-1, 2)
            for result in results
        ]
        density_counts = [
            np.histogram2d(
                sample[:, 0],
                sample[:, 1],
                bins=128,
                range=((-1, 1), (-1, 1)),
            )[0]
            for sample in samples
        ]
        maximum_count = max(
            2.0,
            max(
                (float(counts.max()) for counts in density_counts),
                default=0.0,
            ),
        )
        density_norm = LogNorm(vmin=1, vmax=maximum_count)
        component_ids = sorted(
            int(value)
            for value in rows.loc[
                rows['K'].eq(k) & rows['cluster_id'].notna(),
                'cluster_id',
            ].unique()
        )

    with mpl.rc_context(style):
        figure, axes = plt.subplots(
            nrows, 3, figsize=(180 / 25.4, height / 25.4), squeeze=False,
        )
        for index, result in enumerate(results):
            source = result['summary']
            core = source['confirmed_core_id']
            if filter_view:
                histograms = [
                    result['phasor_histogram'],
                    result['filter_distributions']['median']['phasor_histogram'],
                    result['filter_distributions']['gaussian']['phasor_histogram'],
                ]
                vmax = max(2, max(np.max(value) for value in histograms))
                panels = zip(axes[index], ('raw', 'median', 'gaussian'), histograms)
                for axis, method, histogram in panels:
                    axis.imshow(
                        np.ma.masked_equal(np.asarray(histogram).T, 0),
                        origin='lower', extent=(-1, 1, -1, 1), aspect='equal',
                        cmap='cividis', interpolation='nearest',
                        norm=LogNorm(vmin=1, vmax=vmax),
                    )
                    axis.set_xlabel(f'{core}, {method}: G')
                    axis.set_ylabel('S')
            else:
                axis = axes.flat[index]
                sample = samples[index]
                density_mesh = axis.hist2d(
                    sample[:, 0],
                    sample[:, 1],
                    bins=128,
                    range=((-1, 1), (-1, 1)),
                    cmap='cividis',
                    norm=density_norm,
                    cmin=1,
                )[3]
                density_mesh.set_rasterized(True)

                fitted = rows.loc[
                    rows['path'].eq(source['path']) & rows['K'].eq(k)
                ]
                for row in fitted.to_dict('records'):
                    if row['fit_status'] not in {'FITTED', 'CONVERGENCE_WARNING'}:
                        continue
                    cluster_id = int(row['cluster_id'])
                    component_color = component_colors[
                        (cluster_id - 1) % len(component_colors)
                    ]
                    axis.add_patch(Ellipse(
                        (row['center_real'], row['center_imag']),
                        width=2 * row['radius_major'],
                        height=2 * row['radius_minor'],
                        angle=np.degrees(row['angle_radians']),
                        fill=False, linewidth=.5, alpha=.95,
                        edgecolor=component_color,
                        linestyle=(
                            '--'
                            if row['fit_status'] == 'CONVERGENCE_WARNING'
                            else '-'
                        ),
                    ))
                    axis.scatter(
                        [row['center_real']],
                        [row['center_imag']],
                        s=3.14, marker='x', color=[component_color],
                        linewidths=.5, zorder=4,
                    )
                statuses = sorted(set(fitted['fit_status']) - {'FITTED'})
                if statuses:
                    axis.text(
                        .02, .98, '\n'.join(statuses),
                        transform=axis.transAxes, va='top',
                        fontsize=5, color='#D55E00',
                    )
                axis.set_xlabel(
                    f'{core}, K={k}, n={len(sample)} fit-sample points: G'
                )
                axis.set_ylabel('S')
                axis.set_xlim(-1, 1)
                axis.set_ylim(-1, 1)
                axis.set_aspect('equal')

        for axis in axes.flat:
            axis.set_axisbelow(True)
            axis.grid(color='#E5E7EB', linewidth=.5)
            axis.tick_params(length=2, pad=1, width=.5)

        if filter_view:
            figure.subplots_adjust(
                left=.08, right=.98, bottom=.11, top=.98,
                hspace=.8, wspace=.65,
            )
            note = (
                'Raw / median / Gaussian; common log-count scale within each core; '
                'no winner selected.'
            )
            figure.text(.08, .065, f'{assigned_tma} only. {note}', fontsize=5)
            figure.text(
                .08, .045,
                'Native M overlaps retained; pixels are not biological n.',
            )
            figure.text(
                .08, .025,
                'Internal review only; not Cell manuscript figures. '
                'Generator: clinical_roi_hyperspectral.py',
            )
        else:
            figure.subplots_adjust(
                left=.08, right=.86, bottom=.21, top=.98,
                hspace=.8, wspace=.55,
            )
            legend_handles = [
                Line2D(
                    [], [], marker='x',
                    color=component_colors[
                        (cluster_id - 1) % len(component_colors)
                    ],
                    linestyle='None', markersize=3.14, markeredgewidth=.5,
                    label=str(cluster_id),
                )
                for cluster_id in component_ids
            ]
            if legend_handles:
                figure.legend(
                    handles=legend_handles,
                    title='GMM component ID (fit-local)',
                    loc='lower center', bbox_to_anchor=(.43, .105),
                    ncol=min(8, len(legend_handles)),
                    frameon=False, fontsize=5, title_fontsize=5,
                    handlelength=.6, handletextpad=.25,
                    columnspacing=.55, borderaxespad=0,
                )
            if density_mesh is not None:
                colorbar_axis = figure.add_axes([.89, .25, .018, .62])
                colorbar = figure.colorbar(
                    density_mesh, cax=colorbar_axis,
                    format=FuncFormatter(_format_phasor_count),
                )
                colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
                colorbar.set_label(
                    'Fit-sample counts/bin (log scale)', fontsize=5,
                )
                colorbar.ax.tick_params(
                    which='both', labelsize=5, width=.5, length=1.5,
                )
                colorbar.outline.set_linewidth(.5)
            figure.text(
                .08, .08,
                f'{assigned_tma} only. Shading shows log-count density '
                'of the deterministic fit sample.',
                fontsize=5,
            )
            figure.text(
                .08, .06,
                'Colored 2σ ellipses/crosses are fit-local covariance '
                'summaries, not pixel assignments or identities.',
                fontsize=5,
            )
            figure.text(
                .08, .04,
                'Overlapping ellipses indicate weak separation in this '
                '2D H1 view.',
                fontsize=5,
            )
            figure.text(
                .08, .02,
                'Native M overlaps retained; pixels are not biological n. '
                'Internal review only; not Cell manuscript figures. '
                'Generator: clinical_roi_hyperspectral.py',
                fontsize=5,
            )
        figure.savefig(destination)
        plt.close(figure)

def analyze_tma_phasor_filter_sensitivity(
        summary, *, assigned_tma, cache, table_dir, plot_dir):
    """Compare raw and independent tile-filtered phasors, without choosing a winner."""
    selected, results, context = _phasor_sensitivity_inputs(
        summary, assigned_tma=assigned_tma, cache=cache,
    )
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for base, result in zip(summary.to_dict('records'), results):
        raw = np.asarray(result['phasor_histogram'], dtype=np.float64)
        raw_count = int(raw.sum())
        for method in ('raw', 'median', 'gaussian'):
            if method == 'raw':
                values = {
                    'finite_phasor_pixel_observations': raw_count,
                    'undefined_zero_signal_pixel_observations':
                        base['undefined_zero_signal_pixel_observations'],
                    'paired_raw_filtered_observations': raw_count,
                    'mean_paired_g_s_displacement': 0.0 if raw_count else None,
                    'rms_paired_g_s_displacement': 0.0 if raw_count else None,
                }
                histogram = raw
            else:
                values = dict(result['filter_distributions'][method])
                histogram = np.asarray(values.pop('phasor_histogram'), dtype=np.float64)
            count = int(histogram.sum())
            total_variation = (
                float(np.abs(histogram / count - raw / raw_count).sum() / 2)
                if count and raw_count else None
            )
            rows.append({
                'assigned_tma': assigned_tma, 'confirmed_core_id': base['confirmed_core_id'],
                'path': base['path'], 'full_file_sha256': base['full_file_sha256'],
                'method': method, 'size': 0 if method == 'raw' else 3,
                'repeat': 0 if method == 'raw' else 3,
                **values, 'histogram_total_variation_vs_raw': total_variation,
                'claim_label': 'Exploratory',
                'interpretation': 'spatial-filter sensitivity only; no winner selected',
                'observation_unit': 'within-image pixel observations; not biological n',
            })
    output = pd.DataFrame(rows)
    stem = f'{assigned_tma.lower()}_phasor_filter_sensitivity'
    pdf_path = plot_dir / f'{stem}.pdf'
    _phasor_sensitivity_pdf(
        results, output, assigned_tma=assigned_tma, destination=pdf_path,
    )
    parameters = {
        'analysis': 'PhasorPy spatial phasor-coordinate filter sensitivity',
        'base_receipt_sha256': context['receipt_sha256'],
        'base_result_sha256': context['result_sha256'],
        'base_parameters_and_exact_band_maps': context['parameters_by_path'],
        'change_metrics': {
            'histogram_total_variation_vs_raw': '0.5 * sum(abs(p_filtered - p_raw))',
            'mean_paired_g_s_displacement': 'mean Euclidean G/S displacement vs raw',
            'rms_paired_g_s_displacement': 'root mean square G/S displacement vs raw',
        },
        'limitations': [
            'Median is nonlinear and not an intensity-weighted region summary.',
            'Gaussian uses intensity-weighted phasor coordinates.',
            'No raw-signal filtering; no cross-tile filtering; no winner selection.',
        ],
    }
    _write_phasor_sensitivity_outputs(
        output, selected, summary, parameters, cache=cache, event_start=len(cache.events),
        table_dir=table_dir, stem=stem, pdf_paths=[pdf_path],
    )
    return output, pdf_path


def analyze_tma_phasor_gmm_sensitivity(
        summary, *, assigned_tma, cache, table_dir, plot_dir,
        cluster_counts=(2, 3, 4, 6, 8, 12, 16), max_samples=50000, random_state=0):
    """Fit each requested K per acquisition to cached systematic raw G/S samples."""
    from phasorpy.cluster import phasor_cluster_gmm
    from threadpoolctl import threadpool_limits
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    cluster_counts = tuple(cluster_counts)
    if (not cluster_counts or any(type(k) is not int or k < 2 for k in cluster_counts)
            or len(set(cluster_counts)) != len(cluster_counts)):
        raise ValueError('cluster_counts must contain distinct integers >= 2')
    if type(max_samples) is not int or not 1 <= max_samples <= 50000:
        raise ValueError('max_samples must be an integer from 1 to 50000')
    if type(random_state) is not int or not 0 <= random_state < 2**32:
        raise ValueError('random_state must be a nonnegative 32-bit integer')
    selected, results, context = _phasor_sensitivity_inputs(
        summary, assigned_tma=assigned_tma, cache=cache,
    )
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    method = {
        'function': 'phasorpy.cluster.phasor_cluster_gmm',
        'cluster_counts': list(cluster_counts), 'max_samples': max_samples,
        'random_state': random_state, 'sigma': 2.0, 'sort': 'phasor',
        'covariance_type': 'full', 'tol': .001, 'reg_covar': 1e-6,
        'max_iter': 100, 'n_init': 1, 'init_params': 'kmeans',
        'weights_init': None, 'means_init': None, 'precisions_init': None,
        'warm_start': False, 'verbose': 0, 'verbose_interval': 10,
        'threadpool_limits': 1,
        'sampling': 'systematic cached finite ranks; subset if a smaller cap is requested',
        'interpretation': 'unlabeled exploratory modes, not molecules or abundance',
    }
    event_start = len(cache.events)
    rows = []
    for base, result, result_hash in zip(
            summary.to_dict('records'), results, context['result_sha256']):
        sample = result['systematic_raw_phasor_sample']
        coordinates = np.asarray(sample['coordinates_g_s'], dtype=np.float64).reshape(-1, 2)
        count = min(len(coordinates), max_samples)
        ranks = np.arange(count, dtype=np.int64) * len(coordinates) // max(1, count)
        coordinates = coordinates[ranks]
        if not np.isfinite(coordinates).all():
            raise ValueError('Cached GMM sample contains undefined phasor coordinates')
        per_source = method | {
            'path': base['path'], 'base_result_sha256': result_hash,
            'base_cache_key': base['cache_key'],
            'sample_flat_indices': [sample['flat_indices'][index] for index in ranks],
        }
        def compute():
            fitted_rows = []
            options = {
                key: value for key, value in method.items()
                if key in {
                    'sigma', 'sort', 'random_state', 'covariance_type', 'tol',
                    'reg_covar', 'max_iter', 'n_init', 'init_params',
                    'weights_init', 'means_init', 'precisions_init', 'warm_start',
                    'verbose', 'verbose_interval',
                }
            }
            for k in cluster_counts:
                common = {
                    'K': k, 'sample_count': count,
                    'distinct_sample_count': len(np.unique(coordinates, axis=0)),
                    'warning_message': '',
                    'finite_population_count': sample['finite_population_count'],
                }
                insufficient = (
                    'INSUFFICIENT_FINITE_SAMPLE' if count < k else
                    'INSUFFICIENT_DISTINCT_SAMPLE'
                    if common['distinct_sample_count'] < k else None
                )
                if insufficient:
                    fitted_rows.append(common | {
                        'cluster_id': None, 'fit_status': insufficient,
                        'center_real': None, 'center_imag': None,
                        'radius_major': None, 'radius_minor': None,
                        'angle_radians': None,
                    })
                    continue
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always', ConvergenceWarning)
                    with threadpool_limits(limits=1):
                        ellipses = phasor_cluster_gmm(
                            coordinates[:, 0], coordinates[:, 1], clusters=k, **options,
                        )
                convergence_messages = [
                    str(item.message) for item in captured
                    if issubclass(item.category, ConvergenceWarning)
                ]
                # Persist warnings and also re-emit them: capture is not suppression.
                for item in captured:
                    warnings.warn_explicit(item.message, item.category,
                                           item.filename, item.lineno)
                common['warning_message'] = '\n'.join(convergence_messages)
                if (len(ellipses) != 5 or any(len(values) != k for values in ellipses)
                        or not np.isfinite(np.asarray(ellipses)).all()):
                    raise ValueError('PhasorPy GMM returned invalid ellipse parameters')
                for cluster, values in enumerate(zip(*ellipses), start=1):
                    fitted_rows.append(common | {
                        'cluster_id': cluster,
                        'fit_status': ('CONVERGENCE_WARNING' if convergence_messages
                                       else 'FITTED'),
                        **dict(zip(('center_real', 'center_imag', 'radius_major',
                                    'radius_minor', 'angle_radians'), values)),
                    })
            return fitted_rows
        fitted, status, key = cache.result(
            'phasor_gmm', base['full_file_sha256'], per_source, compute,
        )
        for fitted_row in fitted:
            rows.append({
                'assigned_tma': assigned_tma, 'confirmed_core_id': base['confirmed_core_id'],
                'path': base['path'], 'full_file_sha256': base['full_file_sha256'],
                **fitted_row, 'ellipse_sigma': 2.0, 'random_state': random_state,
                'cache_status': status, 'cache_key': key,
                'claim_label': 'Exploratory',
                'interpretation': 'unlabeled modes; not molecular identity or abundance',
                'observation_unit': 'within-image pixel observations; not biological n',
            })
    output = pd.DataFrame(rows)
    stem = f'{assigned_tma.lower()}_phasor_gmm_sensitivity'
    # Plot exactly the same sample used for fitting when a smaller cap is requested.
    plot_results = []
    for result in results:
        sample = result['systematic_raw_phasor_sample']
        count = min(sample['sample_count'], max_samples)
        ranks = np.arange(count) * sample['sample_count'] // max(1, count)
        plot_results.append(result | {'systematic_raw_phasor_sample': sample | {
            'coordinates_g_s': [sample['coordinates_g_s'][index] for index in ranks],
        }})
    pdf_paths = []
    for k in cluster_counts:
        destination = plot_dir / f'{stem}_k{k}.pdf'
        _phasor_sensitivity_pdf(
            plot_results, output, assigned_tma=assigned_tma, destination=destination, k=k,
        )
        pdf_paths.append(destination)
    parameters = {
        'method': method, 'base_receipt_sha256': context['receipt_sha256'],
        'base_result_sha256': context['result_sha256'],
        'base_parameters_and_exact_band_maps': context['parameters_by_path'],
    }
    _write_phasor_sensitivity_outputs(
        output, selected, summary, parameters, cache=cache, event_start=event_start,
        table_dir=table_dir, stem=stem, pdf_paths=pdf_paths,
    )
    return output, pdf_paths


def _join_hyperspectral_progression(final_manifest, confirmation_path):
    """Join outcomes on both TMA and core, backed by the PI's hashed confirmation."""
    confirmation_path = Path(confirmation_path).resolve()
    confirmation_hash = _full_file_hash(confirmation_path)
    confirmation = json.loads(confirmation_path.read_text(encoding='utf-8'))
    if (confirmation.get('schema_version') != 'tma-patient-independence-v2'
            or confirmation.get('analysis_unit')
            != 'one distinct patient per non-orientation TMA-qualified core'
            or not confirmation.get('confirmed_by')
            or not confirmation.get('confirmation_date')):
        raise ValueError('Require explicit distinct-patient confirmation per qualified core')
    map_path = Path(confirmation['outcome_map'])
    if not map_path.is_absolute():
        map_path = confirmation_path.parent / map_path
    map_path = map_path.resolve()
    map_hash = _full_file_hash(map_path)
    if map_hash != confirmation.get('outcome_map_sha256'):
        raise ValueError('Progression map SHA-256 differs from patient confirmation')
    outcome = pd.read_csv(map_path, dtype=str, keep_default_na=False)
    required = ['tma_id', 'core_id', 'progression_group']
    if not set(required).issubset(outcome.columns):
        raise ValueError('Progression map lacks TMA-qualified core outcome columns')
    if outcome.duplicated(['tma_id', 'core_id']).any():
        raise ValueError('Duplicate TMA-qualified outcome-map keys')
    groups = {'progressor', 'non_progressor', 'orientation_core'}
    if not outcome['progression_group'].isin(groups).all():
        raise ValueError('Unknown progression group in source map')
    joined = final_manifest.copy()
    identity = {'TMA1': 'hBE-TMA1-1', 'TMA2': 'hBE-TMA2-1'}
    if (len(joined) != 18
            or joined.duplicated(['assigned_tma', 'confirmed_core_id']).any()
            or set(joined['assigned_tma']) != set(identity)):
        raise ValueError('Require eighteen unique final TMA-qualified cores')
    for tma in identity:
        cores = joined.loc[joined['assigned_tma'].eq(tma), 'confirmed_core_id']
        if len(cores) != 9 or set(cores) != set(TMA1_CORES):
            raise ValueError('Require exactly one A1-C3 per TMA in final manifest')
    joined['tma_id'] = joined['assigned_tma'].map(identity)
    joined = joined.merge(
        outcome[required], how='left', validate='one_to_one',
        left_on=['tma_id', 'confirmed_core_id'],
        right_on=['tma_id', 'core_id'], indicator=True,
    )
    if (not joined['_merge'].eq('both').all()
            or not joined['progression_group'].isin(groups - {'orientation_core'}).all()):
        raise ValueError('Missing or non-patient outcome for a final qualified core')
    joined = joined.drop(columns='_merge')
    joined['analysis_core_id'] = (
        joined['assigned_tma'] + '_' + joined['confirmed_core_id']
    )
    joined['grid_row'] = joined['confirmed_core_id'].str[0]
    joined['grid_column'] = joined['confirmed_core_id'].str[1].astype(int)
    joined['analysis_unit'] = confirmation['analysis_unit']
    joined['claim_label'] = 'Exploratory'
    joined['position_interpretation'] = (
        'Same A1-C3 labels across TMAs are grid positions, not matched patients'
    )
    bindings = {
        'patient_confirmation': {
            'path': str(confirmation_path), 'sha256': confirmation_hash,
        },
        'progression_map': {'path': str(map_path), 'sha256': map_hash},
    }
    for binding in bindings.values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Outcome source changed during validated join')
    joined.attrs['source_bindings'] = bindings
    return joined, bindings


def _write_hs_exploration_receipt(
        selected, *, cache, parameters, artifacts, destination, event_start=0,
        raw_hashes=None, source_bindings=None):
    """Publish a compact source-bound receipt for descriptive HS extensions."""
    first = selected.iloc[0]
    bindings = {
        name: {'path': first[f'{name}_path'], 'sha256': first[f'{name}_sha256']}
        for name in ('ledger', 'assignment_manifest', 'queue', 'settings_manifest')
    }
    bindings.update(source_bindings or {})
    cache._context()
    for binding in bindings.values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Exploratory source binding changed before publication')
    tmas = sorted(selected['assigned_tma'].unique().tolist())
    receipt = {
        'schema': 'clinical_hs_descriptive_exploration_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'claim_label': 'Exploratory',
        'assigned_tma': tmas[0] if len(tmas) == 1 else tmas,
        'final_manifest': {
            'path': str(cache.manifest_path.resolve()),
            'sha256': cache.manifest_sha256,
        },
        'source_bindings': bindings,
        'selected_acquisitions': selected.to_dict('records'),
        'raw_czi_full_sha256': raw_hashes or {},
        'parameters': parameters,
        'helper_code_sha256': dict(cache.code_sha256),
        'plot_style_sha256': cache.plot_style_sha256,
        'package_versions': dict(cache.package_versions),
        'cache_root': str(cache.root.resolve()),
        'cache_events': cache.events[event_start:],
        'output_artifacts': {
            name: {'path': str(Path(path).resolve()), 'sha256': _full_file_hash(path)}
            for name, path in artifacts.items()
        },
        'figure_policy': 'Internal review only; not Cell manuscript figures',
        'limitations': [
            'Exploratory description only: no P values, paired tests or prediction.',
            'Matching core labels across TMAs are positions, not matched patients.',
            'Binary progression map has no follow-up interval or event definition.',
            'No validated HS tissue mask, anatomical STUs or pathology registration.',
        ],
    }
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(receipt))
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt


def load_hyperspectral_progression_audit(
        final_manifest, *, cache, confirmation_path, table_dir, plot_dir):
    """Export the validated patient-core join and its visible position confounding."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    selected = _validated_spectral_manifest(final_manifest, cache)
    joined, bindings = _join_hyperspectral_progression(selected, confirmation_path)
    counts = joined.groupby(
        ['assigned_tma', 'grid_row', 'grid_column', 'progression_group'],
        observed=True,
    ).size().rename('independent_patient_cores').reset_index()
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    join_path = table_dir / 'hyperspectral_progression_core_audit.csv'
    count_path = table_dir / 'hyperspectral_progression_position_counts.csv'
    joined.to_csv(join_path, index=False)
    counts.to_csv(count_path, index=False)
    pdf_path = plot_dir / 'hyperspectral_progression_position_audit.pdf'
    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()):
        figure, axes = plt.subplots(
            1, 2, figsize=(160 / 25.4, 92 / 25.4), squeeze=False,
        )
        for axis, tma in zip(axes.flat, ('TMA1', 'TMA2')):
            rows = joined.loc[joined['assigned_tma'].eq(tma)]
            for row in rows.to_dict('records'):
                label = 'P' if row['progression_group'] == 'progressor' else 'NP'
                color = '#d95f02' if label == 'P' else '#1b9e77'
                y = 'ABC'.index(row['grid_row'])
                axis.scatter(row['grid_column'], y, s=3.14, color=color)
                axis.text(
                    row['grid_column'], y + .19,
                    f"{row['analysis_core_id']}: {label}", ha='center', va='top',
                )
            axis.set(
                xlim=(.5, 3.5), ylim=(2.6, -.5),
                xticks=(1, 2, 3), yticks=(0, 1, 2), yticklabels=tuple('ABC'),
                xlabel=f'{tma} grid column', ylabel='Grid row',
            )
            axis.tick_params(width=.5)
        total = joined['analysis_core_id'].nunique()
        group_counts = joined.groupby('progression_group').size().to_dict()
        figure.text(
            .06, .13,
            f"Exploratory; independent patient cores n={total}; {group_counts}.",
            fontsize=5,
        )
        figure.text(
            .06, .085,
            'Same A1-C3 across TMAs = positions, NOT matched patients. No paired tests.',
            fontsize=5,
        )
        figure.text(
            .06, .04,
            'Internal review only, not Cell manuscript figures. '
            'Generator: clinical_roi_hyperspectral.py',
            fontsize=5,
        )
        figure.subplots_adjust(bottom=.29, top=.97, wspace=.3)
        figure.savefig(pdf_path)
        plt.close(figure)
    receipt_path = table_dir / 'hyperspectral_progression_audit_receipt.json'
    parameters = {
        'analysis': 'Source-validated descriptive outcome and grid-position audit',
        'join_keys': ['assigned_tma mapped to tma_id', 'confirmed_core_id to core_id'],
        'clinical_endpoint': 'Binary source annotation only; timing/event unspecified',
        'comparison_policy': 'No outcome tests, classifier or biological separation claim',
        'independent_patient_cores': joined['analysis_core_id'].nunique(),
        'group_counts': joined.groupby('progression_group').size().to_dict(),
        'position_column_counts': joined.groupby(
            ['grid_column', 'progression_group'],
        ).size().reset_index(name='n').to_dict('records'),
    }
    _write_hs_exploration_receipt(
        selected, cache=cache, parameters=parameters,
        artifacts={'core_join': join_path, 'position_counts': count_path,
                   'position_audit_pdf': pdf_path},
        destination=receipt_path, source_bindings=bindings,
        event_start=len(cache.events),
    )
    joined.attrs['analysis_receipt_path'] = str(receipt_path)
    joined.attrs['analysis_receipt_sha256'] = _full_file_hash(receipt_path)
    return joined, counts, pdf_path


def _balanced_phasor_samples(results, max_samples):
    """Give every core equal fit mass while retaining its exact evaluation sample."""
    if type(max_samples) is not int or not 1 <= max_samples <= 50000:
        raise ValueError('max_samples must be an integer from 1 to 50000')
    coordinates = []
    for result in results:
        sample = result['systematic_raw_phasor_sample']
        values = np.asarray(sample['coordinates_g_s'], dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 2
                or len(values) != sample['sample_count']
                or not np.isfinite(values).all()):
            raise ValueError('Invalid or nonfinite cached phasor sample')
        if not len(values):
            raise ValueError('Cannot fit a core-balanced model with an empty core')
        count = min(len(values), max_samples)
        ranks = np.arange(count, dtype=np.int64) * len(values) // count
        coordinates.append(values[ranks])
    if not coordinates:
        raise ValueError('No core samples supplied for balanced model')
    fit_count = min(map(len, coordinates))
    pooled = np.concatenate([
        values[np.arange(fit_count, dtype=np.int64) * len(values) // fit_count]
        for values in coordinates
    ])
    return coordinates, pooled, fit_count


def _fit_pooled_phasor_gmm(coordinates_by_core, pooled, *, k, random_states):
    """Record every deterministic fit; retain the first seed, never a label winner."""
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    import warnings
    if len(pooled) < k or len(np.unique(pooled, axis=0)) < k:
        raise ValueError(f'K={k} requires at least K distinct balanced sample points')
    diagnostics, assignments, fitted = [], [], []
    for seed in random_states:
        model = GaussianMixture(
            n_components=k, covariance_type='full', tol=.001, reg_covar=1e-6,
            max_iter=300, n_init=1, init_params='kmeans', random_state=seed,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            with threadpool_limits(limits=1):
                model.fit(pooled)
                labels = model.predict(pooled)
        messages = '\n'.join(str(item.message) for item in captured)
        for item in captured:
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno,
            )
        assignments.append(labels)
        fitted.append(model)
        occupancy = np.bincount(labels, minlength=k)
        for component in range(k):
            diagnostics.append({
                'record_type': 'component', 'K': k, 'seed': seed,
                'cluster_id': component + 1, 'converged': bool(model.converged_),
                'n_iter': int(model.n_iter_), 'lower_bound': float(model.lower_bound_),
                'fit_sample_count': len(pooled), 'weight': float(model.weights_[component]),
                'assigned_sample_count': int(occupancy[component]),
                'assigned_sample_fraction': float(occupancy[component] / len(pooled)),
                'warning_message': messages,
            })
    for left in range(len(assignments)):
        for right in range(left + 1, len(assignments)):
            diagnostics.append({
                'record_type': 'stability_pair', 'K': k,
                'seed': random_states[left], 'comparison_seed': random_states[right],
                'adjusted_rand_index': float(adjusted_rand_score(
                    assignments[left], assignments[right],
                )),
                'fit_sample_count': len(pooled),
            })
    reference = fitted[0]
    composition = []
    with threadpool_limits(limits=1):
        for index, values in enumerate(coordinates_by_core):
            counts = np.bincount(reference.predict(values), minlength=k)
            for component in range(k):
                composition.append({
                    'core_index': index, 'K': k, 'cluster_id': component + 1,
                    'sample_count': len(values),
                    'cluster_sample_count': int(counts[component]),
                    'cluster_proportion': float(counts[component] / len(values)),
                })
    model_parameters = {
        'weights': reference.weights_.tolist(),
        'means': reference.means_.tolist(),
        'covariances': reference.covariances_.tolist(),
        'precisions_cholesky': reference.precisions_cholesky_.tolist(),
        'converged': bool(reference.converged_), 'seed': random_states[0],
        'n_iter': int(reference.n_iter_), 'lower_bound': float(reference.lower_bound_),
    }
    return {
        'model': model_parameters, 'composition': composition,
        'diagnostics': diagnostics,
    }


def _pooled_phasor_composition_pdf(composition, *, assigned_tma, k, destination):
    """Show every patient's sample composition and grid position without inference."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    _register_project_arial()
    data = composition.loc[composition['K'].eq(k)]
    with mpl.rc_context(_pseudocolor_style()):
        figure, axes = plt.subplots(
            2, 1, figsize=(180 / 25.4, 130 / 25.4),
            gridspec_kw={'height_ratios': (1, 1)},
        )
        order = sorted(data['analysis_core_id'].unique())
        bottom = np.zeros(len(order))
        palette = plt.get_cmap('tab20')
        for component in range(1, k + 1):
            rows = data.loc[data['cluster_id'].eq(component)].set_index('analysis_core_id')
            values = rows.loc[order, 'cluster_proportion'].to_numpy()
            axes[0].bar(
                np.arange(len(order)), values, bottom=bottom,
                color=palette((component - 1) % 20), width=.8, linewidth=0,
                label=f'{assigned_tma} mode {component}',
            )
            bottom += values
        labels = []
        for core in order:
            row = data.loc[data['analysis_core_id'].eq(core)].iloc[0]
            group = 'P' if row['progression_group'] == 'progressor' else 'NP'
            labels.append(f'{core}\n{group}; {row["grid_row"]}{row["grid_column"]}')
        axes[0].set(
            xticks=np.arange(len(order)), xticklabels=labels,
            ylabel='Sample cluster proportion', ylim=(0, 1),
        )
        axes[0].legend(
            ncol=min(8, k), frameon=False, loc='upper center',
            bbox_to_anchor=(.5, 1.23), fontsize=5,
        )
        for row in data.to_dict('records'):
            color = '#d95f02' if row['progression_group'] == 'progressor' else '#1b9e77'
            marker = {'A': 'o', 'B': 's', 'C': '^'}[row['grid_row']]
            offset = (int(row['grid_column']) - 2) * .16
            axes[1].scatter(
                row['cluster_id'] + offset, row['cluster_proportion'],
                color=color, marker=marker, s=3.14,
            )
        axes[1].set(
            xticks=range(1, k + 1), xlabel=f'{assigned_tma}-specific spectral mode',
            ylabel='One point per patient core', ylim=(0, 1),
        )
        for axis in axes:
            axis.tick_params(width=.5, length=2)
        figure.text(
            .08, .095,
            f'Exploratory {assigned_tma}, K={k}; first seed reference, no winner selected. '
            'Orange P; green NP. Shapes A/B/C; offsets columns 1/2/3.',
            fontsize=5,
        )
        positions = data[
            ['analysis_core_id', 'grid_column', 'progression_group']
        ].drop_duplicates()
        column_group_counts = positions.groupby('grid_column')[
            'progression_group'
        ].nunique()
        single_group_columns = column_group_counts.loc[
            column_group_counts.eq(1)
        ].index.astype(int).tolist()
        position_note = (
            f'Outcome/column confounding in columns {single_group_columns}.'
            if single_group_columns else
            'Both outcome groups occur in every column.'
        )
        figure.text(
            .08, .06,
            f'{position_note} Same TMA labels are positions, not paired patients.\n'
            'No P values or paired tests; counts are not biological n.',
            fontsize=5,
        )
        figure.text(
            .08, .025,
            'Internal review only, not Cell manuscript figures. '
            'Generator: clinical_roi_hyperspectral.py',
            fontsize=5,
        )
        figure.subplots_adjust(
            left=.09, right=.98, bottom=.21, top=.88, hspace=.5,
        )
        figure.savefig(destination)
        plt.close(figure)


def analyze_tma_pooled_phasor_clusters(
        summary, core_outcomes, *, assigned_tma, cache, table_dir, plot_dir,
        cluster_counts=(2, 3, 4, 6, 8, 12, 16), max_samples=50000,
        random_states=(0, 1, 2)):
    """Fit outcome-blind core-balanced models and describe one composition per core."""
    cluster_counts, random_states = tuple(cluster_counts), tuple(random_states)
    if (not cluster_counts or len(set(cluster_counts)) != len(cluster_counts)
            or any(type(k) is not int or k < 2 for k in cluster_counts)):
        raise ValueError('cluster_counts must contain distinct integers >= 2')
    if (len(random_states) < 2 or len(set(random_states)) != len(random_states)
            or any(type(seed) is not int or not 0 <= seed < 2**32
                   for seed in random_states)):
        raise ValueError('Require multiple distinct deterministic 32-bit seeds')
    selected, results, context = _phasor_sensitivity_inputs(
        summary, assigned_tma=assigned_tma, cache=cache,
    )
    bindings = core_outcomes.attrs.get('source_bindings')
    if not bindings or 'patient_confirmation' not in bindings:
        raise ValueError('Outcome table lacks validated source bindings')
    final = pd.read_csv(cache.manifest_path, dtype=str, keep_default_na=False)
    final = _validated_spectral_manifest(final, cache)
    audited, current_bindings = _join_hyperspectral_progression(
        final, bindings['patient_confirmation']['path'],
    )
    columns = [
        'path', 'assigned_tma', 'confirmed_core_id', 'analysis_core_id',
        'progression_group', 'grid_row', 'grid_column',
    ]
    expected = audited[columns].sort_values('path').reset_index(drop=True)
    supplied = core_outcomes[columns].sort_values('path').reset_index(drop=True)
    if current_bindings != bindings or not supplied.equals(expected):
        raise ValueError('Supplied outcome join differs from current source-bound audit')
    for base, result in zip(summary.to_dict('records'), results):
        source_parameters = context['parameters_by_path'][base['path']]
        sample = result['systematic_raw_phasor_sample']
        if (source_parameters['sample_max_observations'] != 50000
                or sample['sample_count']
                != min(50000, sample['finite_population_count'])):
            raise ValueError('Recompute base phasors with the 50000-point sampling cap')
    coordinates, pooled, fit_count = _balanced_phasor_samples(results, max_samples)
    method = {
        'analysis': 'Outcome-blind TMA-specific core-balanced first-harmonic GMM',
        'function': 'sklearn.mixture.GaussianMixture',
        'cluster_counts': list(cluster_counts), 'max_samples': max_samples,
        'random_states': list(random_states), 'reference_seed': random_states[0],
        'reference_policy': 'First prespecified seed, not likelihood or group separation',
        'covariance_type': 'full', 'tol': .001, 'reg_covar': 1e-6,
        'max_iter': 300, 'n_init': 1, 'init_params': 'kmeans',
        'threadpool_limits': 1, 'fit_count_per_core': fit_count,
        'fit_sampling': 'Equal finite ranks per core; every core has equal fit mass',
        'evaluation_sampling': 'Each core exact cached sample, systematic capped subset',
        'stability': 'Pairwise adjusted Rand index on identical balanced fit points',
        'component_scope': 'TMA and K and seed specific; never align IDs across TMAs',
        'rare_feature_limit': (
            '50000 samples improve sample resolution, not a rare-feature guarantee; '
            'high-K components may describe noise, background or tile overlaps'
        ),
        'comparison_policy': 'Descriptive only; no tests, paired comparisons or prediction',
        'base_receipt_sha256': context['receipt_sha256'],
        'base_result_sha256': context['result_sha256'],
        'base_parameters_and_exact_band_maps': context['parameters_by_path'],
    }
    digest = sha256(_json_bytes({
        'raw_hashes': summary['full_file_sha256'].tolist(),
        'base_result_sha256': context['result_sha256'],
    })).hexdigest()
    event_start = len(cache.events)
    compositions, diagnostics, models = [], [], {}
    base_rows = summary.to_dict('records')
    outcomes = audited.set_index('path')
    for k in cluster_counts:
        parameters = method | {'K': k}
        value, status, key = cache.result(
            'phasor_gmm', digest, parameters,
            lambda: _fit_pooled_phasor_gmm(
                coordinates, pooled, k=k, random_states=random_states,
            ),
            source_kind='sha256_of_ordered_raw_and_base_result_hashes',
        )
        models[k] = value['model']
        for row in value['composition']:
            base = base_rows[row['core_index']]
            outcome = outcomes.loc[base['path']]
            compositions.append({
                **{column: outcome[column] for column in columns if column != 'path'},
                **row, 'path': base['path'], 'full_file_sha256': base['full_file_sha256'],
                'fit_sample_count_per_core': fit_count,
                'finite_population_count': results[row['core_index']][
                    'systematic_raw_phasor_sample']['finite_population_count'],
                'reference_seed': random_states[0], 'cache_status': status, 'cache_key': key,
                'independent_patient_cores_in_tma': len(summary),
                'claim_label': 'Exploratory',
            })
        diagnostics.extend([
            row | {'assigned_tma': assigned_tma, 'cache_status': status,
                   'cache_key': key, 'claim_label': 'Exploratory'}
            for row in value['diagnostics']
        ])
    composition, diagnostic = pd.DataFrame(compositions), pd.DataFrame(diagnostics)
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{assigned_tma.lower()}_pooled_phasor'
    composition_path = table_dir / f'{stem}_core_composition.csv'
    diagnostic_path = table_dir / f'{stem}_fit_diagnostics.csv'
    composition.to_csv(composition_path, index=False)
    diagnostic.to_csv(diagnostic_path, index=False)
    pdfs = []
    for k in cluster_counts:
        path = plot_dir / f'{stem}_core_composition_k{k}.pdf'
        _pooled_phasor_composition_pdf(
            composition, assigned_tma=assigned_tma, k=k, destination=path,
        )
        pdfs.append(path)
    parameters = method | {'models': {str(k): value for k, value in models.items()}}
    for path, signature in context['raw_signatures'].items():
        cache.require_unchanged(path, signature)
    if (_full_file_hash(summary.attrs['analysis_receipt_path'])
            != context['receipt_sha256']):
        raise ValueError('Base receipt changed before pooled-model publication')
    receipt_path = table_dir / f'{stem}_receipt.json'
    _write_hs_exploration_receipt(
        selected, cache=cache, parameters=parameters, destination=receipt_path,
        event_start=event_start, source_bindings=bindings,
        raw_hashes=dict(zip(summary['path'], summary['full_file_sha256'])),
        artifacts={
            'core_composition': composition_path, 'fit_diagnostics': diagnostic_path,
            'base_receipt': summary.attrs['analysis_receipt_path'],
            **{f'composition_k{k}': path for k, path in zip(cluster_counts, pdfs)},
        },
    )
    composition.attrs.update(
        pooled_models=models, cluster_parameters=parameters, sample_context=context,
        analysis_receipt_path=str(receipt_path),
        analysis_receipt_sha256=_full_file_hash(receipt_path),
    )
    diagnostic.attrs['analysis_receipt_path'] = str(receipt_path)
    return composition, diagnostic, pdfs


def _centroid_pseudocolor(
        total: np.ndarray, weighted_nm: np.ndarray,
        wavelength_range: list[float] | tuple[float, float], *,
        chunk_pixels: int = 65536,
        hue_range: list[float] | tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode native centroid hue and independent all-pixel log brightness.

    The optional contrast hue limits are shared across cores, not per-core.
    Percentiles include zero-signal gaps/background. Constant positive
    brightness maps to one; zeros remain black. Equal hue limits yield one
    midpoint hue, not artificial variation. Clipping counts exclude zeros
    and count strictly outside the limits; endpoints themselves are not clipped.
    """
    from matplotlib.colors import hsv_to_rgb

    if (total.ndim != 2 or weighted_nm.shape != total.shape
            or total.size == 0 or not isinstance(chunk_pixels, int)
            or chunk_pixels <= 0):
        raise ValueError('Nonempty matching Y/X planes and chunk limit required')
    lower, upper = map(float, wavelength_range)
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise ValueError('Finite increasing positive wavelength range required')
    hue_lower, hue_upper = (lower, upper) if hue_range is None else hue_range
    if (not np.isfinite([hue_lower, hue_upper]).all()
            or not lower <= hue_lower <= hue_upper <= upper):
        raise ValueError('Hue limits must be ordered within the audited range')
    flat_total, flat_weighted = total.reshape(-1), weighted_nm.reshape(-1)
    logged = np.empty(total.size, dtype=np.float64)
    zero_count, below, above = 0, 0, 0
    for start in range(0, total.size, chunk_pixels):
        stop = min(start + chunk_pixels, total.size)
        signal, weighted = flat_total[start:stop], flat_weighted[start:stop]
        if (not np.isfinite(signal).all() or not np.isfinite(weighted).all()
                or (signal < 0).any() or (weighted < 0).any()
                or ((signal == 0) & (weighted != 0)).any()):
            raise ValueError('Signal and weighted signal must be finite nonnegative')
        positive = signal > 0
        centroid = np.divide(
            weighted, signal, out=np.full(signal.shape, lower), where=positive,
        )
        if ((centroid < lower - 1e-6) | (centroid > upper + 1e-6)).any():
            raise ValueError('Spectral centroid outside shared wavelength range')
        if hue_range is not None:
            np.clip(centroid, lower, upper, out=centroid)
            below += int(np.count_nonzero(positive & (centroid < hue_lower)))
            above += int(np.count_nonzero(positive & (centroid > hue_upper)))
        zero_count += int(np.count_nonzero(~positive))
        np.log1p(signal, out=logged[start:stop])
    low, high = np.percentile(logged, [1, 99], method='linear')
    rgb = np.empty((total.size, 3), dtype=np.uint8)
    for start in range(0, total.size, chunk_pixels):
        stop = min(start + chunk_pixels, total.size)
        signal, weighted = flat_total[start:stop], flat_weighted[start:stop]
        positive = signal > 0
        centroid = np.divide(
            weighted, signal, out=np.full(signal.shape, lower), where=positive,
        )
        if hue_upper > hue_lower:
            hue = .72 * (1 - np.clip(
                (centroid - hue_lower) / (hue_upper - hue_lower), 0, 1,
            ))
        else:
            hue = np.full(centroid.shape, .36, dtype=np.float64)
        value = logged[start:stop].copy()
        if high > low:
            value -= low
            value /= high - low
            np.clip(value, 0, 1, out=value)
        else:
            value[:] = positive
        value[~positive] = 0
        hsv = np.column_stack((hue, np.ones(hue.size), value))
        rgb[start:stop] = np.rint(hsv_to_rgb(hsv) * 255).astype(np.uint8)
    summary = {
        'log_signal_p01': float(low), 'log_signal_p99': float(high),
        'zero_signal_pixels': zero_count,
    }
    if hue_range is not None:
        summary.update({
            'centroid_below_hue_limit_pixels': below,
            'centroid_above_hue_limit_pixels': above,
        })
    return rgb.reshape((*total.shape, 3)), summary


def _pseudocolor_frame_extent(width, height):
    """Center an unstretched image in a square frame in display coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError('Positive image width and height required')
    scale = max(width, height)
    dx, dy = width / scale / 2, height / scale / 2
    return .5 - dx, .5 + dx, .5 - dy, .5 + dy


def _mosaic_channel_sums(
        row: dict[str, Any], channels: pd.DataFrame,
        parameters: dict[str, Any], *, reader: Any = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Accumulate native channels once, shared by both centroid displays."""
    if reader is None:
        from aicspylibczi import CziFile
        reader = CziFile(str(_require_raw_hs(row)))
    metadata, current_rows = inspect_raw_metadata(row, reader=reader)
    dimensions = _spectral_layout(metadata)
    current = _spectral_channel_grid(current_rows)
    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm',
        'detection_band_center_nm',
    ]
    if (metadata['metadata_xml_sha256'] != parameters['metadata_xml_sha256']
            or metadata['dimensions_json'] != parameters['dimensions_json']
            or not current[columns].equals(channels[columns].reset_index(drop=True))):
        raise ValueError('Current CZI metadata differs from audited channel map')
    box = reader.get_mosaic_bounding_box()
    region = tuple(int(value) for value in (box.x, box.y, box.w, box.h))
    x, y, width, height = region
    if width <= 0 or height <= 0:
        raise ValueError('CZI mosaic geometry must have positive extent')
    pixels = width * height
    native_pixels = prod(
        stop - start for axis, (start, stop) in dimensions.items() if axis != 'C'
    )
    # Conservative managed-array/decoder/PNG/PDF allowance, not process RSS:
    # float64 sums/weighted sums, percentile copy, RGB, decoder and rendering.
    # One core is processed at a time; no C20 cube is allocated.
    # Include all 18 bounded overview RGB previews and Matplotlib copies.
    working_bound = (
        pixels * 192 + native_pixels * 16 + min(pixels, 65536) * 256
        + 18 * 768**2 * 16
    )
    if 'centroid_histogram_bins' in parameters:
        # Count-list objects, JSON serialization, and NumPy histogram buffers.
        working_bound += 18 * parameters['centroid_histogram_bins'] * 64
    if working_bound > parameters['max_working_bytes']:
        raise MemoryError(
            f'Full-resolution mosaic needs {working_bound} bounded bytes; '
            f"limit={parameters['max_working_bytes']}; no downsampling fallback"
        )
    selectors = {
        axis: start for axis, (start, stop) in dimensions.items()
        if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}
    }
    # Single-scene metadata was required above; never guess a scene selector.
    # Its full stage bounding box defines this one acquisition's frame.
    total = np.zeros((height, width), dtype=np.float64)
    weighted = np.zeros_like(total)
    for channel in current.to_dict('records'):
        plane = reader.read_mosaic(
            C=int(channel['channel_index']), scale_factor=1.0,
            region=region, background_color=(0., 0., 0.), **selectors,
        )
        if (plane.dtype != np.dtype('uint16')
                or plane.shape[-2:] != (height, width)
                or plane.size != pixels):
            raise ValueError(
                f'Full-scale CZI mosaic shape/dtype mismatch: '
                f'{plane.shape}, {plane.dtype}'
            )
        flat_plane = plane.reshape(-1)
        for start in range(0, pixels, 65536):
            stop = min(start + 65536, pixels)
            signal = flat_plane[start:stop].astype(np.float64)
            total.reshape(-1)[start:stop] += signal
            signal *= channel['detection_band_center_nm']
            weighted.reshape(-1)[start:stop] += signal
        del plane, flat_plane
    return total, weighted, {
        'assigned_tma': row['assigned_tma'],
        'confirmed_core_id': row['confirmed_core_id'], 'path': row['path'],
        'mosaic_x': x, 'mosaic_y': y,
        'mosaic_width_pixels': width, 'mosaic_height_pixels': height,
        'channel_count': len(current), 'scale_factor': 1.0,
        'working_allocation_bound_bytes': working_bound,
        'display_policy': 'pseudocolor only; no molecular or biological inference',
        'overlap_policy': 'Zeiss/libCZI read_mosaic stage-geometry composition',
    }


def _mosaic_pseudocolor(
        row: dict[str, Any], channels: pd.DataFrame,
        parameters: dict[str, Any], *, reader: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a native mosaic from the shared channel accumulation helper."""
    total, weighted, summary = _mosaic_channel_sums(
        row, channels, parameters, reader=reader,
    )
    rgb, scaling = _centroid_pseudocolor(
        total, weighted, parameters['wavelength_range_nm'],
        hue_range=parameters.get('centroid_hue_range_nm'),
    )
    return rgb, summary | scaling


def _validate_pseudocolor_cache(value):
    """Reject absent, replaced, truncated or wrong-shape display cache images."""
    from PIL import Image

    path = Path(value['rgb_png_path'])
    if _full_file_hash(path) != value['rgb_png_sha256']:
        raise ValueError('Pseudocolor RGB cache integrity mismatch')
    summary = value['summary']
    with Image.open(path) as image:
        if (image.mode != 'RGB'
                or image.size != (summary['mosaic_width_pixels'],
                                  summary['mosaic_height_pixels'])):
            raise ValueError('Pseudocolor RGB cache geometry mismatch')
        image.verify()


def _pseudocolor_style() -> dict[str, object]:
    """Return the shared project Matplotlib style for every generated figure."""
    return plot_style.matplotlib_rc_params()


def _draw_pseudocolor_frame(figure, bounds, rgb):
    axis = figure.add_axes(bounds, facecolor='black')
    height, width = rgb.shape[:2]
    axis.imshow(
        rgb, extent=_pseudocolor_frame_extent(width, height),
        origin='upper', interpolation='none', resample=False, aspect='equal',
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_linewidth(.5)
        spine.set_edgecolor('#444444')
    return axis


def _pseudocolor_legend(
        figure: Any, wavelength_range: list[float] | tuple[float, float], *,
        bottom: float, compact: bool = False, contrast: bool = False,
) -> None:
    from matplotlib.colors import hsv_to_rgb

    lower, upper = wavelength_range
    position = np.linspace(0, 1, 512)
    hue = .72 * (1 - position)
    extent = (lower, upper, 0, 1)
    ticks, labels = [lower, (lower + upper) / 2, upper], None
    if lower == upper:
        hue[:] = .36
        extent, ticks, labels = (0, 1, 0, 1), [.5], [f'{lower:g}']
    hsv = np.column_stack((hue, np.ones(position.size), np.ones(position.size)))
    bounds = [.10, bottom, .80, .025] if compact else [.13, bottom, .46, .035]
    axis = figure.add_axes(bounds)
    axis.imshow(
        hsv_to_rgb(hsv)[None, ...], extent=extent,
        interpolation='none', resample=False, aspect='auto',
    )
    axis.set(yticks=[], xticks=ticks)
    if labels is not None:
        axis.set_xticklabels(labels)
    axis.tick_params(width=.5, length=2, pad=1)
    label = 'Hue: intensity-weighted spectral centroid (nm)'
    if contrast:
        label = 'Shared centroid p1–p99 (nm); tails clipped'
        if lower == upper:
            label = 'Constant centroid (nm); one shared hue'
    axis.set_xlabel(label, labelpad=1)
    for spine in axis.spines.values():
        spine.set_linewidth(.5)
    if compact:
        figure.text(
            .10, .065,
            'Brightness: log(1 + sum); per-core 1st–99th percentile\n'
            'Zero: black; brightness not comparable across cores/TMAs',
            va='center', linespacing=1.3,
        )
    else:
        figure.text(
            .63, bottom + .015,
            'Brightness: log(1 + summed signal)\n'
            'Independent per-core 1st–99th percentile\n'
            'Zero signal: black; not comparable across cores',
            va='center', linespacing=1.3,
        )


def _verify_pseudocolor_pdf(path, canvas_mm, *, native_size=None):
    """Check generated PDF geometry and native pixel embedding at runtime."""
    import pymupdf

    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError(f'Missing or empty pseudocolor PDF: {path}')
    with pymupdf.open(path) as document:
        if len(document) != 1:
            raise ValueError('Pseudocolor figure must have exactly one page')
        page = document[0]
        expected = np.asarray(canvas_mm) / 25.4 * 72
        if not np.allclose(
                [page.rect.width, page.rect.height], expected, atol=.02, rtol=0):
            raise ValueError('Pseudocolor PDF canvas changed')
        if native_size is not None:
            sizes = {(image[2], image[3]) for image in page.get_images()}
            if tuple(native_size) not in sizes:
                raise ValueError('Per-core PDF does not embed native-resolution RGB')
        for drawing in page.get_drawings():
            if drawing.get('color') is not None and not np.isclose(
                    drawing['width'], .5, atol=1e-6):
                raise ValueError('Pseudocolor PDF contains a non-0.5-pt stroke')
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if ('Arial' not in span['font']
                            or not np.isclose(span['size'], 5, atol=.01)):
                        raise ValueError('Pseudocolor PDF text is not Arial 5 pt')
                    x0, y0, x1, y1 = span['bbox']
                    if (min(x0, y0) < -.02
                            or x1 > page.rect.width + .02
                            or y1 > page.rect.height + .02):
                        raise ValueError('Pseudocolor PDF text extends off canvas')


def _pseudocolor_figures(
        results: list[dict[str, Any]],
        wavelength_range: list[float] | tuple[float, float], plot_dir: Path, *,
        contrast: bool = False,
) -> dict[str, Path]:
    """Embed native per-core pixels; reduce only the separate overview grid."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from PIL import Image

    _register_project_arial()
    paths = {}
    suffix = '_shared_contrast' if contrast else ''
    with mpl.rc_context(_pseudocolor_style()):
        overview = plt.figure(figsize=(180 / 25.4, 112 / 25.4))
        try:
            for result in results:
                summary = result['summary']
                tma, core = summary['assigned_tma'], summary['confirmed_core_id']
                width = summary['mosaic_width_pixels']
                height = summary['mosaic_height_pixels']
                with Image.open(result['rgb_png_path']) as image:
                    rgb = np.asarray(image)
                    detail = plt.figure(figsize=(90 / 25.4, 100 / 25.4))
                    try:
                        _draw_pseudocolor_frame(
                            detail, [.10, .23, .80, .72], rgb,
                        )
                        detail.text(.10, .98, f'{tma} · {core}', va='top')
                        _pseudocolor_legend(
                            detail, wavelength_range, bottom=.145,
                            compact=True, contrast=contrast,
                        )
                        detail.text(
                            .10, .015,
                            'Display pseudocolor; no H&E registration/correction.\n'
                            'Internal review only; not Cell manuscript figures.\n'
                            'Generator: clinical_roi_hyperspectral.py',
                        )
                        destination = plot_dir / (
                            f'{tma.lower()}_{core.lower()}'
                            f'_hyperspectral_pseudocolor{suffix}.pdf'
                        )
                        detail.savefig(destination)
                    finally:
                        detail.clear()
                        plt.close(detail)
                    del rgb
                    _verify_pseudocolor_pdf(
                        destination, (90, 100), native_size=(width, height),
                    )
                    paths[f'{tma}_{core}_pdf'] = destination
                    # Nearest-neighbor only; never soften edges or stretch aspect.
                    image.thumbnail((768, 768), resample=Image.Resampling.NEAREST)
                    preview = np.asarray(image).copy()
                block = 0 if tma == 'TMA1' else 1
                column, row = int(core[1]) - 1, 'ABC'.index(core[0])
                x_mm = 5 + block * 87 + column * 27
                y_mm = 79 - row * 28
                _draw_pseudocolor_frame(
                    overview, [x_mm / 180, y_mm / 112, 25 / 180, 25 / 112],
                    preview,
                )
                overview.text(
                    x_mm / 180, (y_mm + 25.8) / 112, core, va='bottom',
                )
            overview.text(5 / 180, 109 / 112, 'TMA1', va='top')
            overview.text(92 / 180, 109 / 112, 'TMA2', va='top')
            _pseudocolor_legend(
                overview, wavelength_range, bottom=.13, contrast=contrast,
            )
            overview.text(
                .035, .028,
                'Display pseudocolor; no molecular assignments or H&E registration.\n'
                'Matching A1-C3 = grid positions, NOT matched patients. '
                'Each TMA-qualified core is a distinct patient.\n'
                'Internal review only; not Cell manuscript figures. '
                'Generator: clinical_roi_hyperspectral.py',
            )
            combined = plot_dir / (
                f'hyperspectral_pseudocolor{suffix}_tma1_tma2.pdf'
            )
            overview.savefig(combined)
        finally:
            overview.clear()
            plt.close(overview)
    _verify_pseudocolor_pdf(combined, (180, 112))
    paths['combined_grid_pdf'] = combined
    return paths


def render_hyperspectral_pseudocolor(
        final_manifest: pd.DataFrame, channel_map: pd.DataFrame, *, cache: HSCache,
        table_dir: str | Path, plot_dir: str | Path,
        max_working_bytes: int, num_threads: int = 1,
) -> tuple[pd.DataFrame, Path]:
    """Create source-bound display-only full-scale mosaics and an equal-frame grid."""
    from PIL import Image

    if (not isinstance(max_working_bytes, int) or max_working_bytes <= 0
            or not isinstance(num_threads, int) or num_threads <= 0):
        raise ValueError('Positive integer memory and thread limits required')
    inputs = _validated_spectral_manifest(final_manifest, cache)
    audited, _ = audit_spectral_channel_maps(inputs, cache=cache)
    if not _spectral_channel_maps_equal(audited, channel_map):
        raise ValueError('Supplied map differs from current exact spectral audit')
    for tma in ('TMA1', 'TMA2'):
        selected = inputs.loc[inputs['assigned_tma'].eq(tma)]
        if (len(selected) != 9
                or set(selected['confirmed_core_id']) != set(TMA1_CORES)):
            raise ValueError('Both TMAs require exactly one selected A1-C3 core')
    wavelength_range = [
        float(audited['detection_band_center_nm'].min()),
        float(audited['detection_band_center_nm'].max()),
    ]
    results, summaries, parameters_by_path, signatures = [], [], {}, {}
    event_start = len(cache.events)
    for row in inputs.to_dict('records'):
        channels = audited.loc[audited['path'].eq(row['path'])].reset_index(drop=True)
        base = _spectral_phasor_parameters(
            row, channels, max_working_bytes=max_working_bytes,
            num_threads=num_threads,
        )
        parameters = {
            key: base[key] for key in (
                'row', 'channel_map', 'metadata_xml_sha256', 'dimensions_json',
                'max_working_bytes', 'num_threads',
            )
        } | {
            'schema': 'clinical_hs_centroid_pseudocolor_v1',
            'wavelength_range_nm': wavelength_range,
            'reader': 'CziFile.read_mosaic; scale_factor=1.0; full stage bounding box',
            'selectors': 'recorded singleton axes; no S or M selector',
            'decoder_threads': (
                'read_mosaic has no per-call thread control; num_threads '
                'records requested allocation only'
            ),
            'overlap_policy': 'Zeiss/libCZI stage-geometry composition',
            'hue': 'HSV hue 0.72*(1-(centroid-min)/(max-min)); saturation=1',
            'brightness': 'log1p(sum C); all-pixel linear percentiles [1,99]',
            'constant_brightness': 'positive pixels=1; zero pixels=0',
            'zero_signal': 'black', 'rgb': 'round to uint8; no spatial filtering',
            'threshold': None, 'mask': None, 'correction': None,
            'registration': None, 'rotation': None, 'scale_factor': 1.0,
            'overview_max_edge_pixels': 768,
            'overview_resampling': 'nearest neighbor only',
            'core_pdf_canvas_mm': [90, 100], 'grid_canvas_mm': [180, 112],
        }
        path = _require_raw_hs(row)
        digest, signature = cache.raw_identity(path)
        signatures[str(path)] = signature
        parameters_by_path[str(path)] = parameters
        artifact_key = sha256(_json_bytes({
            'context': cache._context(), 'source_sha256': digest,
            'parameters': parameters,
        })).hexdigest()
        image_path = cache.root / 'pseudocolor_rgb' / f'{artifact_key}.png'

        def compute() -> dict[str, object]:
            rgb, summary = _mosaic_pseudocolor(row, channels, parameters)
            cache.require_unchanged(path, signature)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                        dir=image_path.parent, suffix='.png', delete=False) as handle:
                    temporary = Path(handle.name)
                image = Image.fromarray(rgb)
                try:
                    image.save(temporary, format='PNG')
                finally:
                    image.close()
                temporary.replace(image_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return {
                'summary': summary, 'rgb_png_path': str(image_path.resolve()),
                'rgb_png_sha256': _full_file_hash(image_path),
            }

        result, status, key = cache.result(
            'pseudocolor', digest, parameters, compute,
            validate=_validate_pseudocolor_cache,
        )
        cache.require_unchanged(path, signature)
        _validate_pseudocolor_cache(result)
        results.append(result)
        summaries.append(result['summary'] | {
            'full_file_sha256': digest,
            'metadata_xml_sha256': parameters['metadata_xml_sha256'],
            'wavelength_min_nm': wavelength_range[0],
            'wavelength_max_nm': wavelength_range[1],
            'rgb_png_path': result['rgb_png_path'],
            'rgb_png_sha256': result['rgb_png_sha256'],
            'cache_status': status, 'cache_key': key,
        })
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _pseudocolor_figures(results, wavelength_range, plot_dir)
    summary = pd.DataFrame(summaries)
    summary['full_resolution_pdf'] = [
        str(artifacts[f"{row['assigned_tma']}_{row['confirmed_core_id']}_pdf"])
        for row in summaries
    ]
    summary_path = table_dir / 'hyperspectral_pseudocolor_acquisitions.csv'
    summary.to_csv(summary_path, index=False)
    artifacts['acquisition_summary_csv'] = summary_path
    first = inputs.iloc[0]
    bindings = {
        name: {'path': first[f'{name}_path'], 'sha256': first[f'{name}_sha256']}
        for name in ('ledger', 'assignment_manifest', 'queue', 'settings_manifest')
    }
    cache._context()
    for binding in bindings.values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Source binding changed before pseudocolor publication')
    for path, signature in signatures.items():
        cache.require_unchanged(path, signature)
    receipt = {
        'schema': 'clinical_hs_centroid_pseudocolor_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': (
            'Display-only spectral-centroid pseudocolor; '
            'no biological inference'
        ),
        'final_manifest': {
            'path': str(cache.manifest_path.resolve()),
            'sha256': cache.manifest_sha256,
        },
        'source_bindings': bindings,
        'raw_czi_full_sha256': dict(zip(
            summary['path'], summary['full_file_sha256'],
        )),
        'parameters_and_exact_band_maps': parameters_by_path,
        'selected_acquisitions': summary.to_dict('records'),
        'helper_code_sha256': dict(cache.code_sha256),
        'package_versions': dict(cache.package_versions),
        'cache_events': cache.events[event_start:],
        'output_artifacts': {
            name: {'path': str(path.resolve()), 'sha256': _full_file_hash(path)}
            for name, path in artifacts.items()
        },
        'limitations': [
            'Pseudocolor is not calibrated true color or a molecular assignment.',
            'Brightness is independently scaled, not comparable across cores/TMAs.',
            'No tissue mask, correction, threshold, smoothing or inference.',
            'Within-core stage assembly only; no rotation or H&E registration.',
            'Zeiss/libCZI overlap composition is not quantitative deduplication.',
            'Memory bound excludes external-library process RSS.',
            'Native RGB in per-core PDFs; overview only reduced nearest-neighbor.',
        ],
    }
    receipt_path = table_dir / 'hyperspectral_pseudocolor_receipt.json'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=table_dir, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(receipt))
        temporary.replace(receipt_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary.attrs['analysis_receipt_path'] = str(receipt_path)
    summary.attrs['output_artifacts'] = {
        key: str(path) for key, path in artifacts.items()
    }
    return summary, artifacts['combined_grid_pdf']


def _candidate_domain_json(value):
    """Canonicalize in-memory model arrays and source-context tuples."""
    if isinstance(value, np.ndarray):
        return _candidate_domain_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _candidate_domain_json(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_candidate_domain_json(item) for item in value]
    return value


def _candidate_domain_model(model, k):
    """Reconstruct the fitted, outcome-blind sklearn model without refitting."""
    from sklearn.mixture import GaussianMixture

    shapes = {
        'weights': (k,), 'means': (k, 2),
        'covariances': (k, 2, 2), 'precisions_cholesky': (k, 2, 2),
    }
    arrays = {}
    for name, shape in shapes.items():
        array = np.asarray(model[name], dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'Invalid pooled K={k} model array: {name}')
        arrays[name] = array
    weights, covariance = arrays['weights'], arrays['covariances']
    precision = arrays['precisions_cholesky']
    if (np.any(weights <= 0) or not np.isclose(weights.sum(), 1)
            or not np.allclose(covariance, covariance.transpose(0, 2, 1))
            or np.any(np.linalg.eigvalsh(covariance) <= 0)
            or not np.allclose(
                precision @ precision.transpose(0, 2, 1),
                np.linalg.inv(covariance), rtol=1e-6, atol=1e-8)):
        raise ValueError(f'Inconsistent pooled K={k} covariance/precision/weights')
    fitted = GaussianMixture(
        n_components=k, covariance_type='full', random_state=model['seed'],
    )
    fitted.weights_ = weights
    fitted.means_ = arrays['means']
    fitted.covariances_ = covariance
    fitted.precisions_cholesky_ = precision
    fitted.converged_ = bool(model['converged'])
    fitted.n_features_in_ = 2
    return fitted


def _candidate_domain_pixel_units(metadata: ET.Element) -> dict[str, Any]:
    """Use CZI Distance/Value meters, never the GUI DefaultUnitFormat.

    ZISRAW (CZI) File Format, Release 1.02, §6.9.1.1, p.49 defines
    Distance Value in meters; DefaultUnitFormat is only a GUI display
    preference, not the metadata Value unit. Explicit nonstandard unit
    declarations must agree with the format-defined meter convention.
    Both axes must have one finite positive value before any physical output.
    """
    meter_units = {'m', 'meter', 'metre', 'meters', 'metres'}
    evidence, scales = {}, {}
    for axis in ('X', 'Y'):
        nodes = metadata.findall(f'.//Scaling/Items/Distance[@Id="{axis}"]')
        evidence[axis] = [ET.tostring(node, encoding='unicode') for node in nodes]
        if len(nodes) != 1:
            return {
                'physical_scale_status': f'PIXELS_ONLY_{axis}_DISTANCE_NOT_UNIQUE',
                'distance_xml_json': json.dumps(evidence, sort_keys=True),
            }
        node = nodes[0]
        values = node.findall('Value')
        if len(values) != 1:
            return {
                'physical_scale_status': f'PIXELS_ONLY_{axis}_VALUE_NOT_UNIQUE',
                'distance_xml_json': json.dumps(evidence, sort_keys=True),
            }
        value = values[0]
        declarations = [
            node.get('Unit'), node.get('unit'),
            value.get('Unit'), value.get('unit'),
            *(child.text or '' for child in node.findall('Unit')),
            *(child.text or '' for child in node.findall('unit')),
        ]
        units = [item.strip().lower() for item in declarations if item is not None]
        try:
            number = float(value.text)
        except (TypeError, ValueError):
            number = float('nan')
        if (any(unit not in meter_units for unit in units)
                or not np.isfinite(number) or number <= 0):
            return {
                'physical_scale_status': f'PIXELS_ONLY_{axis}_UNIT_OR_VALUE_INVALID',
                'distance_xml_json': json.dumps(evidence, sort_keys=True),
            }
        scales[axis] = number * 1e6
        if not np.isfinite(scales[axis]) or scales[axis] <= 0:
            return {
                'physical_scale_status': f'PIXELS_ONLY_{axis}_SCALE_INVALID',
                'distance_xml_json': json.dumps(evidence, sort_keys=True),
            }
    area = scales['X'] * scales['Y']
    if not np.isfinite(area) or area <= 0:
        return {
            'physical_scale_status': 'PIXELS_ONLY_XY_AREA_INVALID',
            'distance_xml_json': json.dumps(evidence, sort_keys=True),
        }
    return {
        'physical_scale_status': 'CZI_SPEC_XY_DISTANCE_VALUES_METERS_PARSED',
        'physical_scale_rule': (
            'ZISRAW 6.9.1.1 Distance/Value meters; DefaultUnitFormat display-only; '
            'explicit unit declarations must agree with meters'
        ),
        'physical_scale_limitation': (
            'CZI metadata-derived; not independently calibrated '
            'against a stage micrometer'
        ),
        'pixel_width_um': scales['X'], 'pixel_height_um': scales['Y'],
        'pixel_area_um2': area,
        'distance_xml_json': json.dumps(evidence, sort_keys=True),
    }


def _candidate_domain_distribution(areas, pixel_area_um2=None):
    """Describe components, never treating pixels/components as biological n."""
    names = ('min', 'q25', 'median', 'q75', 'p95', 'max')
    probabilities = (0, .25, .5, .75, .95, 1)
    output = {'n_components': int(areas.size)}
    specifications = [
        ('area_pixels', 1., False),
        ('equivalent_diameter_pixels', 1., True),
    ]
    if pixel_area_um2 is not None:
        specifications.extend([
            ('area_um2', pixel_area_um2, False),
            ('equivalent_diameter_um', pixel_area_um2, True),
        ])
    for prefix, scale, diameter in specifications:
        if areas.size:
            values = areas.astype(np.float64)
            values *= scale
            if diameter:
                values *= 4 / np.pi
                np.sqrt(values, out=values)
            quantiles = np.quantile(values, probabilities, method='linear')
            output.update({f'{prefix}_{name}': float(value)
                           for name, value in zip(names, quantiles)})
        else:
            output.update({f'{prefix}_{name}': None for name in names})
    return output


def _candidate_domain_save_png(array, destination):
    """Publish native label/RGB PNGs atomically."""
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix='.png', delete=False) as handle:
            temporary = Path(handle.name)
        image = Image.fromarray(array)
        try:
            image.save(temporary, format='PNG')
        finally:
            image.close()
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _candidate_domain_figure(labels, row, k, destination):
    """Render native labels as internal-review-only candidate spectral domains."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _register_project_arial()
    palette = np.zeros((k + 1, 3), dtype=np.uint8)
    palette[1:] = np.rint(
        mpl.colormaps['tab20'](np.arange(k))[:, :3] * 255,
    ).astype(np.uint8)
    rgb = palette[labels]
    png_path = destination.with_suffix('.png')
    _candidate_domain_save_png(rgb, png_path)
    with mpl.rc_context(_pseudocolor_style()):
        figure = plt.figure(figsize=(90 / 25.4, 100 / 25.4))
        try:
            _draw_pseudocolor_frame(figure, [.05, .15, .90, .81], rgb)
            figure.text(
                .05, .97,
                f"{row['assigned_tma']} {row['confirmed_core_id']} | pooled K={k}",
                va='top',
            )
            figure.text(
                .05, .14,
                'Internal review only, not Cell manuscript figures\n'
                '4-connected components; no tissue mask or smoothing\n'
                'Black: undefined zero-signal phasor; colors: model labels\n'
                'Native mosaic overlap composition may differ from fit tiles',
                va='top', linespacing=1.25,
            )
            for cluster in range(k):
                figure.text(
                    .05 + .9 * cluster / k, .035, str(cluster + 1),
                    color=palette[cluster + 1] / 255, va='bottom',
                )
            figure.text(
                .05, .015, 'Generator: 02_scripts/clinical_roi_hyperspectral.py',
            )
            figure.savefig(destination, format='pdf', dpi=300)
        finally:
            plt.close(figure)
    _verify_pseudocolor_pdf(
        destination, (90, 100),
        native_size=(labels.shape[1], labels.shape[0]),
    )
    return png_path


def _candidate_domain_validate_cache(value):
    """Validate cached native disk images and source-bound review PDFs."""
    from PIL import Image

    for artifact in value['artifacts']:
        path = Path(artifact['path'])
        if _full_file_hash(path) != artifact['sha256']:
            raise ValueError(f'Candidate-domain artifact hash mismatch: {path}')
        if artifact['kind'] == 'pdf':
            _verify_pseudocolor_pdf(
                path, (90, 100), native_size=tuple(artifact['size']),
            )
        else:
            with Image.open(path) as image:
                if (image.size != tuple(artifact['size'])
                        or image.mode != artifact['mode']):
                    raise ValueError(f'Candidate-domain image mismatch: {path}')
                image.verify()


def _h1_channel_weights(channel_count):
    """Match PhasorPy's positive-sine first-harmonic convention."""
    if type(channel_count) is not int or channel_count < 2:
        raise ValueError('At least two contiguous spectral channels are required')
    phase = (
        2 * np.pi * np.arange(channel_count, dtype=np.float64)
        / channel_count
    )
    return np.cos(phase), np.sin(phase)


def _normalize_h1_sums(total, real_sum, imag_sum):
    """Normalize accumulated Fourier sums; preserve undefined zero-signal pixels."""
    if (total.shape != real_sum.shape or total.shape != imag_sum.shape
            or not np.isfinite(total).all()
            or not np.isfinite(real_sum).all()
            or not np.isfinite(imag_sum).all()
            or (total < 0).any()):
        raise ValueError('H1 sums must be finite with matching shapes and signal >= 0')
    valid = total > 0
    if ((~valid) & ((real_sum != 0) | (imag_sum != 0))).any():
        raise ValueError('Zero-signal H1 sums must both be zero')
    np.divide(real_sum, total, out=real_sum, where=valid)
    np.divide(imag_sum, total, out=imag_sum, where=valid)
    real_sum[~valid] = np.nan
    imag_sum[~valid] = np.nan
    return valid


def _candidate_domain_native(
        row: dict[str, Any], channels: pd.DataFrame,
        parameters: dict[str, Any], models: dict[int, dict[str, Any]],
        image_dir: Path, plot_dir: Path | None,
) -> dict[str, Any]:
    """Stream raw H1 sums on the exact full-scale pseudocolor mosaic geometry."""
    from aicspylibczi import CziFile
    from scipy import ndimage
    from threadpoolctl import threadpool_limits

    reader = CziFile(str(_require_raw_hs(row)))
    metadata, current_rows = inspect_raw_metadata(row, reader=reader)
    dimensions = _spectral_layout(metadata)
    current = _spectral_channel_grid(current_rows)
    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm',
        'detection_band_center_nm',
    ]
    if (metadata['metadata_xml_sha256'] != parameters['metadata_xml_sha256']
            or metadata['dimensions_json'] != parameters['dimensions_json']
            or not current[columns].equals(channels[columns].reset_index(drop=True))):
        raise ValueError('Current CZI metadata differs from audited channel map')
    base = {
        'assigned_tma': row['assigned_tma'],
        'confirmed_core_id': row['confirmed_core_id'], 'path': row['path'],
        'claim_label': 'Exploratory',
        'purpose': 'Internal review only, not Cell manuscript figures',
        'connectivity': 4, 'scale_factor': 1.,
        'overlap_policy': (
            'Zeiss/libCZI native read_mosaic stage composition; may differ from '
            'unstitched overlapping M-tile observations used for pooled fit'
        ),
        'interpretation': (
            'Descriptive candidate spectral domains, not cells/molecules; '
            'components/pixels are not independent biological n'
        ),
    }
    base.update(_candidate_domain_pixel_units(reader.meta))

    def blocked(reason: str) -> dict[str, Any]:
        return {
            'rows': [base | {'K': k, 'cluster_id': None, 'status': 'BLOCKED',
                             'blocked_reason': reason} for k in (8, 16)],
            'artifacts': [],
        }

    if not reader.is_mosaic():
        return blocked('Raw CZI is not a native mosaic; no fallback is allowed')
    box = reader.get_mosaic_bounding_box()
    raw_region = (box.x, box.y, box.w, box.h)
    if any(not np.isfinite(value) or int(value) != value for value in raw_region):
        return blocked(f'Nonintegral/nonfinite native mosaic geometry: {raw_region}')
    region = tuple(int(value) for value in raw_region)
    x, y, width, height = region
    if width <= 0 or height <= 0:
        return blocked(f'Nonpositive native mosaic geometry: {region}')
    pixels = width * height
    native_pixels = prod(
        stop - start for axis, (start, stop) in dimensions.items() if axis != 'C'
    )
    chunk = min(pixels, parameters['chunk_pixels'])
    # Conservative managed-array/decoder allowance, not process RSS. Includes
    # 3 float64 accumulators, labels, masks, worst-case component/bincount/
    # quantile copies, RGB/PDF copies and chunked sklearn log probabilities.
    working_bound = pixels * 192 + native_pixels * 16 + chunk * (128 + 16 * 64)
    base.update({
        'mosaic_x': x, 'mosaic_y': y,
        'mosaic_width_pixels': width, 'mosaic_height_pixels': height,
        'working_allocation_bound_bytes': working_bound,
    })
    if pixels > np.iinfo(np.intp).max or working_bound > parameters['max_working_bytes']:
        return blocked(
            f'Native mosaic {width}x{height} requires {working_bound} bounded bytes; '
            f"limit={parameters['max_working_bytes']}; no downsampling fallback"
        )
    selectors = {
        axis: start for axis, (start, stop) in dimensions.items()
        if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}
    }
    try:
        total = np.zeros((height, width), dtype=np.float64)
        real = np.zeros_like(total)
        imag = np.zeros_like(total)
        cosine_weights, sine_weights = _h1_channel_weights(len(current))

        for channel in current.to_dict('records'):
            try:
                plane = reader.read_mosaic(
                    C=int(channel['channel_index']), scale_factor=1.,
                    region=region, background_color=(0., 0., 0.), **selectors,
                )
            except MemoryError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Native CZI mosaic decode failed for "
                    f"C={channel['channel_index']}: {type(exc).__name__}: {exc}"
                ) from exc
            if (plane.dtype != np.dtype('uint16')
                    or plane.shape[-2:] != (height, width) or plane.size != pixels):
                return blocked(
                    f'Native scale=1 mosaic shape/dtype mismatch: actual={plane.shape},'
                    f'{plane.dtype}; expected=singleton axes plus {(height, width)},uint16'
                )
            channel_index = int(channel['channel_index'])
            cosine = cosine_weights[channel_index]
            sine = sine_weights[channel_index]
            for start in range(0, pixels, chunk):
                stop = min(start + chunk, pixels)
                signal = plane.reshape(-1)[start:stop].astype(np.float64)
                total.reshape(-1)[start:stop] += signal
                real.reshape(-1)[start:stop] += signal * cosine
                imag.reshape(-1)[start:stop] += signal * sine
            del plane
        finite_pixels = 0
        for start in range(0, pixels, chunk):
            stop = min(start + chunk, pixels)
            valid = _normalize_h1_sums(
                total.reshape(-1)[start:stop],
                real.reshape(-1)[start:stop],
                imag.reshape(-1)[start:stop],
            )
            finite_pixels += int(np.count_nonzero(valid))
        del total
        base.update({
            'finite_phasor_pixels': finite_pixels,
            'undefined_zero_signal_pixels': pixels - finite_pixels,
        })
        rows, artifacts = [], []
        structure = ndimage.generate_binary_structure(2, 1)
        for k in (8, 16):
            fitted = _candidate_domain_model(models[k], k)
            if not fitted.converged_:
                rows.append(base | {
                    'K': k, 'cluster_id': None, 'status': 'BLOCKED',
                    'blocked_reason': 'Source pooled model did not converge; no refitting',
                })
                continue
            labels = np.zeros((height, width), dtype=np.uint8)
            with threadpool_limits(limits=parameters['num_threads']):
                for start in range(0, pixels, chunk):
                    stop = min(start + chunk, pixels)
                    g = real.reshape(-1)[start:stop]
                    s = imag.reshape(-1)[start:stop]
                    valid = np.isfinite(g) & np.isfinite(s)
                    if valid.any():
                        coordinates = np.column_stack((g[valid], s[valid]))
                        labels.reshape(-1)[start:stop][valid] = (
                            fitted.predict(coordinates).astype(np.uint8) + 1
                        )
            if int(np.count_nonzero(labels)) != finite_pixels:
                raise ValueError('Native predicted label count differs from finite H1 count')
            for cluster in range(k):
                mask = labels == cluster + 1
                components = np.empty((height, width), dtype=np.int64)
                count = ndimage.label(mask, structure=structure, output=components)
                counts = np.bincount(components.reshape(-1), minlength=count + 1)
                areas = counts[1:]
                component_pixels = int(areas.sum(dtype=np.int64))
                del mask, components
                rows.append(base | {
                    'K': k, 'cluster_id': cluster + 1, 'status': 'COMPLETE',
                    'blocked_reason': '', 'cluster_pixels': component_pixels,
                    'cluster_fraction_finite': (
                        component_pixels / finite_pixels if finite_pixels else None
                    ),
                    **_candidate_domain_distribution(areas, base.get('pixel_area_um2')),
                })
                del counts, areas
            label_path = image_dir / f'K{k}_native_labels.png'
            _candidate_domain_save_png(labels, label_path)
            size = [width, height]
            artifacts.append({
                'kind': 'labels', 'path': str(label_path.resolve()),
                'sha256': _full_file_hash(label_path), 'mode': 'L', 'size': size,
                'K': k, 'label_encoding': '0 undefined; 1..K pooled cluster_id',
            })
            if plot_dir is not None:
                pdf_path = plot_dir / (
                    f"{row['assigned_tma']}_{row['confirmed_core_id']}"
                    f'_candidate_spectral_domains_K{k}_{image_dir.name[:12]}.pdf'
                )
                png_path = _candidate_domain_figure(labels, row, k, pdf_path)
                for kind, artifact_path in (('rgb', png_path), ('pdf', pdf_path)):
                    artifacts.append({
                        'kind': kind, 'path': str(artifact_path.resolve()),
                        'sha256': _full_file_hash(artifact_path),
                        'mode': 'RGB', 'size': size, 'K': k,
                    })
            del labels
        return {'rows': rows, 'artifacts': artifacts}
    except MemoryError as exc:
        raise MemoryError(
            f'Native mosaic allocation failed within the {working_bound}-byte '
            'managed bound; no blocked result was cached'
        ) from exc


def analyze_tma_candidate_spectral_domains(
        final_manifest: pd.DataFrame, channel_map: pd.DataFrame,
        cluster_composition: pd.DataFrame, *, assigned_tma: str, cache: HSCache,
        table_dir: str | Path, plot_dir: str | Path | None,
        max_working_bytes: int, num_threads: int = 1,
) -> tuple[pd.DataFrame, list[Path]]:
    """Describe native K8/K16 domains from source-bound pooled TMA models.

    A None plot_dir omits review PNG/PDF rendering; native label PNGs remain
    source-bound cache artifacts. Unsafe native cores return BLOCKED K rows.
    """
    if assigned_tma not in {'TMA1', 'TMA2'}:
        raise ValueError('assigned_tma must be TMA1 or TMA2; never pool TMAs')
    if (type(max_working_bytes) is not int or max_working_bytes <= 0
            or type(num_threads) is not int or num_threads <= 0):
        raise ValueError('Positive integer memory and thread limits required')
    inputs = _validated_spectral_manifest(final_manifest, cache)
    audited, _ = audit_spectral_channel_maps(inputs, cache=cache)
    if not _spectral_channel_maps_equal(audited, channel_map):
        raise ValueError('Supplied map differs from current exact spectral channel audit')
    selected = inputs.loc[inputs['assigned_tma'].eq(assigned_tma)]
    if (len(selected) != 9 or set(selected['confirmed_core_id']) != set(TMA1_CORES)):
        raise ValueError('Require one selected A1-C3 acquisition for this TMA')
    if (not {'path', 'assigned_tma'}.issubset(cluster_composition.columns)
            or not cluster_composition['assigned_tma'].eq(assigned_tma).all()
            or set(cluster_composition['path']) != set(selected['path'])):
        raise ValueError('Cluster composition does not cover the same single-TMA handoff')
    attrs = cluster_composition.attrs
    receipt_path = Path(attrs['analysis_receipt_path'])
    receipt_hash = attrs['analysis_receipt_sha256']
    if _full_file_hash(receipt_path) != receipt_hash:
        raise ValueError('Source pooled-cluster analysis receipt changed')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    cluster_parameters = _candidate_domain_json(attrs['cluster_parameters'])
    context = _candidate_domain_json(attrs['sample_context'])
    models = attrs['pooled_models']
    if (receipt['assigned_tma'] != assigned_tma
            or receipt['helper_code_sha256'] != cache.code_sha256
            or receipt['package_versions'] != cache.package_versions
            or receipt['final_manifest']['sha256'] != cache.manifest_sha256
            or receipt['parameters'] != cluster_parameters
            or cluster_parameters['models'] != _candidate_domain_json(models)
            or cluster_parameters['base_receipt_sha256'] != context['receipt_sha256']
            or cluster_parameters['base_result_sha256'] != context['result_sha256']
            or cluster_parameters['base_parameters_and_exact_band_maps']
            != context['parameters_by_path']):
        raise ValueError('Source pooled method/models/base-result context changed')
    for k in (8, 16):
        _candidate_domain_model(models[k], k)
    for binding in receipt['source_bindings'].values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Pooled-cluster source binding changed')
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    event_start = len(cache.events)
    rows, artifacts, parameters_by_path, raw_hashes, signatures = [], {}, {}, {}, {}
    pdf_paths = []
    for row in selected.to_dict('records'):
        path = _require_raw_hs(row)
        digest, signature = cache.raw_identity(path)
        if receipt['raw_czi_full_sha256'].get(str(path)) != digest:
            raise ValueError(f'Current raw CZI differs from pooled-fit receipt: {path}')
        cache.require_unchanged(path, tuple(context['raw_signatures'][str(path)]))
        raw_hashes[str(path)], signatures[str(path)] = digest, signature
        channels = audited.loc[audited['path'].eq(str(path))].reset_index(drop=True)
        spectral = _spectral_phasor_parameters(
            row, channels, max_working_bytes=max_working_bytes, num_threads=num_threads,
        )
        parameters = {
            key: spectral[key] for key in (
                'row', 'channel_map', 'metadata_xml_sha256', 'dimensions_json',
                'max_working_bytes', 'num_threads',
            )
        } | {
            'schema': 'clinical_hs_native_candidate_spectral_domains_v1',
            'source_cluster_receipt_sha256': receipt_hash,
            'cluster_parameters': cluster_parameters, 'sample_context': context,
            'models': _candidate_domain_json({k: models[k] for k in (8, 16)}),
            'reader': 'read_mosaic scale_factor=1.0; full native stage bounding box',
            'decoder_threads': 'read_mosaic has no per-call thread control',
            'transform': 'raw H1 sum(C*cos(2*pi*C/N))/sum(C); analogous sine',
            'normalization': 'positive sum only; zero signal undefined',
            'chunk_pixels': 65536, 'connectivity': 4,
            'component_policy': 'each predicted label separately; all finite H1 pixels',
            'threshold': None, 'tissue_mask': None, 'smoothing': None,
            'outcome_segmentation': None, 'refit': False,
            'physical_units': (
                'CZI ZISRAW 1.02 section 6.9.1.1 Distance/Value meters; '
                'DefaultUnitFormat display-only; unique finite positive X/Y; '
                'explicit units must agree with meters'
            ),
            'quantiles': 'numpy.quantile method=linear; component-weighted',
            'label_encoding': '0 undefined; 1..K pooled cluster_id',
            'figure_policy': (
                'Internal review only, not Cell manuscript figures; '
                'native labels; tab20; no reduction'
            ),
            'plot_dir': str(plot_dir.resolve()) if plot_dir is not None else None,
            'limitations': (
                'Native mosaic overlap composition may differ from unstitched fit; '
                'managed allocation bound excludes external-library process RSS; '
                'no biological attribution, P values, classifier or cross-TMA comparison; '
                'CZI metadata-derived sizes, not independently calibrated '
                'against a stage micrometer'
            ),
        }
        artifact_key = sha256(_json_bytes({
            'context': cache._context(), 'source_sha256': digest,
            'parameters': parameters,
        })).hexdigest()
        image_dir = cache.root / 'native_candidate_domains' / artifact_key

        def compute() -> dict[str, object]:
            value = _candidate_domain_native(
                row, channels, parameters, models, image_dir, plot_dir,
            )
            cache.require_unchanged(path, signature)
            return value

        result, status, key = cache.result(
            'phasor_gmm', digest, parameters, compute,
            validate=_candidate_domain_validate_cache,
        )
        cache.require_unchanged(path, signature)
        parameters_by_path[str(path)] = parameters
        for domain in result['rows']:
            rows.append(domain | {
                'full_file_sha256': digest,
                'metadata_xml_sha256': parameters['metadata_xml_sha256'],
                'source_cluster_receipt_sha256': receipt_hash,
                'cache_status': status, 'cache_key': key,
            })
        for index, artifact in enumerate(result['artifacts']):
            artifact_path = Path(artifact['path'])
            artifacts[f"{row['confirmed_core_id']}_artifact_{index}"] = artifact_path
            if artifact['kind'] == 'pdf':
                pdf_paths.append(artifact_path)
    output = pd.DataFrame(rows)
    stem = f'{assigned_tma.lower()}_candidate_spectral_domains'
    table_path = table_dir / f'{stem}.csv'
    output.to_csv(table_path, index=False)
    artifacts['domain_summary_csv'] = table_path
    artifacts['source_pooled_cluster_receipt'] = receipt_path
    if _full_file_hash(receipt_path) != receipt_hash:
        raise ValueError('Source pooled-cluster receipt changed before publication')
    for path, signature in signatures.items():
        cache.require_unchanged(path, signature)
    destination = table_dir / f'{stem}_receipt.json'
    _write_hs_exploration_receipt(
        selected, cache=cache,
        parameters={
            'schema': 'clinical_hs_native_candidate_spectral_domains_v1',
            'source_cluster_receipt_sha256': receipt_hash,
            'parameters_and_exact_band_maps': parameters_by_path,
            'purpose': (
                'Internal review only, not Cell manuscript figures; '
                'descriptive candidate spectral domains'
            ),
        },
        artifacts=artifacts, destination=destination, event_start=event_start,
        raw_hashes=raw_hashes, source_bindings=receipt['source_bindings'],
    )
    output.attrs['analysis_receipt_path'] = str(destination)
    output.attrs['analysis_receipt_sha256'] = _full_file_hash(destination)
    output.attrs['output_artifacts'] = {
        name: str(path) for name, path in artifacts.items()
    }
    return output, pdf_paths


def _native_centroid_summary(
        row: dict[str, Any], channels: pd.DataFrame, parameters: dict[str, Any],
) -> dict[str, Any]:
    """First native pass: retain only a bounded histogram and scalar summaries."""
    total, weighted, summary = _mosaic_channel_sums(row, channels, parameters)
    lower, upper = parameters['wavelength_range_nm']
    histogram = np.zeros(parameters['centroid_histogram_bins'], dtype=np.int64)
    positive_count, minimum, maximum = 0, None, None
    flat_total, flat_weighted = total.reshape(-1), weighted.reshape(-1)
    for start in range(0, total.size, 65536):
        stop = min(start + 65536, total.size)
        signal, weighted_chunk = flat_total[start:stop], flat_weighted[start:stop]
        positive = signal > 0
        centroid = weighted_chunk[positive] / signal[positive]
        if centroid.size:
            if (not np.isfinite(centroid).all()
                    or (centroid < lower - 1e-6).any()
                    or (centroid > upper + 1e-6).any()):
                raise ValueError('Native centroid outside audited wavelength range')
            np.clip(centroid, lower, upper, out=centroid)
            current_min, current_max = float(centroid.min()), float(centroid.max())
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
            histogram += np.histogram(
                centroid, bins=histogram.size, range=(lower, upper),
            )[0]
            positive_count += int(centroid.size)
    summary.update({
        'positive_signal_pixels': positive_count,
        'zero_signal_pixels': int(total.size - positive_count),
        'centroid_min_nm': minimum, 'centroid_max_nm': maximum,
        'native_accumulation_dtype': 'float64',
    })
    return {'summary': summary, 'centroid_histogram': histogram.tolist()}


def _shared_centroid_limits(
        results: list[dict[str, Any]], wavelength_range: list[float],
) -> dict[str, Any]:
    """Inverse equal-core mixture CDF, uniform interpolation within each bin."""
    if not results:
        raise ValueError('At least one native centroid summary is required')
    lower, upper = map(float, wavelength_range)
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise ValueError('Finite increasing positive wavelength range required')
    bins = len(results[0]['centroid_histogram'])
    if bins == 0:
        raise ValueError('Nonempty centroid histogram required')
    nonempty = []
    for result in results:
        counts = np.asarray(result['centroid_histogram'])
        population = result['summary']['positive_signal_pixels']
        if (counts.shape != (bins,) or not np.issubdtype(counts.dtype, np.integer)
                or (counts < 0).any() or int(counts.sum()) != population):
            raise ValueError('Native centroid histogram/count mismatch')
        if population:
            summary = result['summary']
            minimum, maximum = summary['centroid_min_nm'], summary['centroid_max_nm']
            if (not np.isfinite([minimum, maximum]).all()
                    or not lower <= minimum <= maximum <= upper):
                raise ValueError('Native centroid extrema outside audited range')
            nonempty.append(result)
    width = (upper - lower) / bins
    method = {
        'percentiles': [1, 99], 'histogram_bins': bins,
        'histogram_range_nm': wavelength_range, 'histogram_bin_width_nm': width,
        'quantile_approximation': (
            'Mean of per-core normalized native positive-centroid histograms; '
            'inverse CDF with uniform interpolation inside the selected bin. '
            'Not numpy linear sample percentiles. Within-bin location is '
            'unresolved at the reported bin width, not a confidence interval; '
            'the float64 mixture CDF also has floating-point roundoff.'
        ),
        'positive_core_count': len(nonempty),
        'empty_core_count': len(results) - len(nonempty),
        'population': 'All positive native pixels, including background; no tissue mask',
        'core_weighting': 'Equal mass per nonempty core, shared across both TMAs',
    }
    if not nonempty:
        return method | {
            'centroid_hue_min_nm': lower, 'centroid_hue_max_nm': upper,
            'range_status': 'NO_POSITIVE_CENTROIDS; hue undefined; all pixels black',
        }
    minimum = min(r['summary']['centroid_min_nm'] for r in nonempty)
    maximum = max(r['summary']['centroid_max_nm'] for r in nonempty)
    if minimum == maximum:
        limits = [minimum, maximum]
        status = 'CONSTANT_CENTROID; all positive pixels share midpoint hue'
    else:
        mixture = np.zeros(bins, dtype=np.float64)
        for result in nonempty:
            counts = np.asarray(result['centroid_histogram'], dtype=np.float64)
            mixture += counts / result['summary']['positive_signal_pixels']
        mixture /= len(nonempty)
        cdf = np.cumsum(mixture)
        cdf[-1] = 1.0
        limits = []
        for probability in (.01, .99):
            index = min(int(np.searchsorted(cdf, probability, side='left')), bins - 1)
            previous = cdf[index - 1] if index else 0.0
            fraction = (probability - previous) / mixture[index]
            estimate = lower + (index + fraction) * width
            limits.append(float(np.clip(estimate, minimum, maximum)))
        status = 'APPROXIMATE_EQUAL_CORE_P01_P99'
        if limits[0] == limits[1]:
            status = 'DEGENERATE_QUANTILES; all positive pixels share midpoint hue'
    return method | {
        'centroid_hue_min_nm': limits[0], 'centroid_hue_max_nm': limits[1],
        'range_status': status,
    }


def render_hyperspectral_contrast_gallery(
        final_manifest: pd.DataFrame, channel_map: pd.DataFrame, *, cache: HSCache,
        table_dir: str | Path, plot_dir: str | Path,
        max_working_bytes: int, num_threads: int = 1,
) -> tuple[pd.DataFrame, Path]:
    """Bounded native two-pass shared-centroid contrast, not histology."""
    if (type(max_working_bytes) is not int or max_working_bytes <= 0
            or type(num_threads) is not int or num_threads <= 0):
        raise ValueError('Positive integer memory and thread limits required')
    inputs = _validated_spectral_manifest(final_manifest, cache)
    audited, _ = audit_spectral_channel_maps(inputs, cache=cache)
    if not _spectral_channel_maps_equal(audited, channel_map):
        raise ValueError('Supplied map differs from current exact spectral audit')
    for tma in ('TMA1', 'TMA2'):
        selected = inputs.loc[inputs['assigned_tma'].eq(tma)]
        if len(selected) != 9 or set(selected['confirmed_core_id']) != set(TMA1_CORES):
            raise ValueError('Both TMAs require exactly one selected A1-C3 core')
    wavelength_range = [float(audited['detection_band_center_nm'].min()),
                        float(audited['detection_band_center_nm'].max())]
    histogram_results, parameters_by_path, raw_hashes, signatures = [], {}, {}, {}
    event_start = len(cache.events)
    for row in inputs.to_dict('records'):
        channels = audited.loc[audited['path'].eq(row['path'])].reset_index(drop=True)
        base = _spectral_phasor_parameters(
            row, channels, max_working_bytes=max_working_bytes,
            num_threads=num_threads,
        )
        parameters = {key: base[key] for key in (
            'row', 'channel_map', 'metadata_xml_sha256', 'dimensions_json',
            'max_working_bytes', 'num_threads',
        )} | {
            'schema': 'clinical_hs_native_centroid_summary_v1',
            'wavelength_range_nm': wavelength_range, 'centroid_histogram_bins': 65536,
            'reader': 'CziFile.read_mosaic; scale_factor=1; full stage bounding box',
            'decoder_threads': 'read_mosaic has no per-call thread control',
            'dtype': 'float64 channel sum and wavelength-weighted sum',
            'centroid_roundoff': 'clip only audited endpoints within 1e-6 nm',
            'population': 'all positive pixels including background; no tissue mask',
            'threshold': None, 'mask': None, 'registration': None,
            'correction': None, 'smoothing': None, 'rotation': None,
        }
        path = _require_raw_hs(row)
        digest, signature = cache.raw_identity(path)
        raw_hashes[str(path)], signatures[str(path)] = digest, signature
        parameters_by_path[str(path)] = parameters

        def compute_summary() -> dict[str, Any]:
            value = _native_centroid_summary(row, channels, parameters)
            cache.require_unchanged(path, signature)
            return value

        def validate_summary(value: dict[str, Any]) -> None:
            expected_bins = parameters['centroid_histogram_bins']
            if len(value['centroid_histogram']) != expected_bins:
                raise ValueError('Cached native centroid histogram size changed')
            _shared_centroid_limits([value], wavelength_range)

        value, status, key = cache.result(
            'centroid_summary', digest, parameters, compute_summary,
            validate=validate_summary,
        )
        cache.require_unchanged(path, signature)
        histogram_results.append(value | {'summary_cache_status': status,
                                         'summary_cache_key': key})
    method = _shared_centroid_limits(histogram_results, wavelength_range) | {
        'schema': 'clinical_hs_shared_centroid_contrast_v1',
        'hue': 'HSV 0.72*(1-clipped shared centroid fraction); saturation=1',
        'constant_hue': 'midpoint hue 0.36; no artificial hue variation',
        'clipped_counts': 'strictly below/above shared limits among positive pixels',
        'zero_signal': 'black; centroid undefined, excluded only from hue distribution',
        'brightness': 'independent per-core exact linear all-pixel log1p(sum) p1/p99',
        'constant_brightness': 'positive pixels full brightness; zero black',
        'native_strategy': 'Two native channel passes cold; histogram JSON + RGB PNG only',
        'native_scalar_cache_bytes': 0,
        'histogram_count_array_bytes': len(inputs) * 65536 * np.dtype('int64').itemsize,
        'histogram_python_working_allowance_bytes': len(inputs) * 65536 * 64,
        'limitations': [
            'Display contrast, not histology, molecular assignment or calibrated true color.',
            'Positive background included; no tissue mask or intensity cutoff.',
            'Brightness not comparable across cores/TMAs; hue limits shared.',
            'No H&E registration, rotation, correction, filtering or biological inference.',
            'Zeiss/libCZI mosaic overlap composition is not quantitative deduplication.',
            'Native PNG/per-core PDF; overview only reduced nearest-neighbor.',
            'Managed working bound excludes external-library process RSS.',
        ],
    }
    population = [{
        'path': value['summary']['path'],
        'full_file_sha256': raw_hashes[value['summary']['path']],
        'histogram_result_sha256': sha256(_json_bytes({
            'summary': value['summary'], 'centroid_histogram': value['centroid_histogram'],
        })).hexdigest(),
    } for value in histogram_results]
    results, summaries, artifacts = [], [], {}
    for row, histogram in zip(inputs.to_dict('records'), histogram_results):
        path = row['path']
        cache.require_unchanged(path, signatures[path])
        channels = audited.loc[audited['path'].eq(path)].reset_index(drop=True)
        parameters = parameters_by_path[path] | {
            'schema': 'clinical_hs_shared_centroid_contrast_v1',
            'method': method, 'population': population,
            'centroid_hue_range_nm': [
                method['centroid_hue_min_nm'], method['centroid_hue_max_nm'],
            ],
        }
        image_key = sha256(_json_bytes({
            'context': cache._context(), 'source_sha256': raw_hashes[path],
            'parameters': parameters,
        })).hexdigest()
        image_path = cache.root / 'pseudocolor_rgb' / f'{image_key}.png'

        def compute_rgb() -> dict[str, Any]:
            rgb, summary = _mosaic_pseudocolor(row, channels, parameters)
            cache.require_unchanged(path, signatures[path])
            _candidate_domain_save_png(rgb, image_path)
            return {
                'summary': summary, 'rgb_png_path': str(image_path.resolve()),
                'rgb_png_sha256': _full_file_hash(image_path),
            }

        result, status, key = cache.result(
            'pseudocolor', raw_hashes[path], parameters, compute_rgb,
            validate=_validate_pseudocolor_cache,
        )
        _validate_pseudocolor_cache(result)
        cache.require_unchanged(path, signatures[path])
        results.append(result)
        summary = histogram['summary'] | result['summary'] | {
            'full_file_sha256': raw_hashes[path],
            'metadata_xml_sha256': parameters['metadata_xml_sha256'],
            'centroid_hue_min_nm': method['centroid_hue_min_nm'],
            'centroid_hue_max_nm': method['centroid_hue_max_nm'],
            'rgb_png_path': result['rgb_png_path'],
            'rgb_png_sha256': result['rgb_png_sha256'],
            'cache_status': status, 'cache_key': key,
            'summary_cache_status': histogram['summary_cache_status'],
            'summary_cache_key': histogram['summary_cache_key'],
        }
        summaries.append(summary)
        core = f"{summary['assigned_tma']}_{summary['confirmed_core_id']}"
        artifacts[f'{core}_rgb_png'] = Path(result['rgb_png_path'])
    table_dir, plot_dir = Path(table_dir), Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    hue_range = [method['centroid_hue_min_nm'], method['centroid_hue_max_nm']]
    artifacts.update(_pseudocolor_figures(results, hue_range, plot_dir, contrast=True))
    summary = pd.DataFrame(summaries)
    summary['full_resolution_pdf'] = [
        str(artifacts[f"{row['assigned_tma']}_{row['confirmed_core_id']}_pdf"])
        for row in summaries
    ]
    table_path = table_dir / 'hyperspectral_centroid_contrast_acquisitions.csv'
    summary.to_csv(table_path, index=False)
    artifacts['acquisition_summary_csv'] = table_path
    for path, signature in signatures.items():
        cache.require_unchanged(path, signature)
    receipt_path = table_dir / 'hyperspectral_centroid_contrast_receipt.json'
    _write_hs_exploration_receipt(
        inputs, cache=cache, parameters={
            'method': method, 'parameters_and_exact_band_maps': parameters_by_path,
            'native_histogram_population': population,
            'cache_footprint': {
                'native_scalar_cache_bytes': 0,
                'histogram_json_bytes': sum(
                    Path(event['cache_path']).stat().st_size
                    for event in cache.events[event_start:]
                    if event['stage'] == 'centroid_summary'
                ),
                'native_rgb_png_bytes': sum(
                    Path(result['rgb_png_path']).stat().st_size
                    for result in results
                ),
            },
        }, artifacts=artifacts, destination=receipt_path,
        event_start=event_start, raw_hashes=raw_hashes,
    )
    summary.attrs.update(
        analysis_receipt_path=str(receipt_path),
        analysis_receipt_sha256=_full_file_hash(receipt_path),
        output_artifacts={name: str(path) for name, path in artifacts.items()},
        contrast_method=method,
    )
    return summary, artifacts['combined_grid_pdf']


def _candidate_gallery_inputs(
        domain_results: dict[str, pd.DataFrame], cluster_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Resolve exact native artifacts through hashed receipts/cache envelopes."""
    if set(domain_results) != {'TMA1', 'TMA2'} or cluster_count not in (8, 16):
        raise ValueError('Require both TMA results and cluster_count=8 or 16')
    panels, bindings = [], {}
    manifest_binding = None
    for tma in ('TMA1', 'TMA2'):
        table = domain_results[tma]
        receipt_path = Path(table.attrs['analysis_receipt_path'])
        receipt_hash = table.attrs['analysis_receipt_sha256']
        if _full_file_hash(receipt_path) != receipt_hash:
            raise ValueError('Candidate-domain source receipt changed')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt['assigned_tma'] != tma
                or receipt['parameters']['schema']
                != 'clinical_hs_native_candidate_spectral_domains_v1'):
            raise ValueError('Wrong TMA or candidate-domain receipt schema')
        if (manifest_binding is not None
                and receipt['final_manifest'] != manifest_binding):
            raise ValueError('TMA domain results use different final manifests')
        manifest_binding = receipt['final_manifest']
        bindings[tma] = {'path': str(receipt_path.resolve()), 'sha256': receipt_hash}
        source_bindings = [
            receipt['final_manifest'], *receipt['source_bindings'].values(),
        ]
        for binding in source_bindings:
            if _full_file_hash(binding['path']) != binding['sha256']:
                raise ValueError('Candidate-domain source binding changed')
        for artifact in receipt['output_artifacts'].values():
            if _full_file_hash(artifact['path']) != artifact['sha256']:
                raise ValueError('Candidate-domain receipt artifact changed')
        selected = receipt['selected_acquisitions']
        if (len(selected) != 9
                or {row['confirmed_core_id'] for row in selected} != set(TMA1_CORES)
                or any(row['assigned_tma'] != tma for row in selected)):
            raise ValueError('Candidate-domain receipt must contain nine TMA-qualified cores')
        current = table.loc[table['K'].eq(cluster_count)]
        if (set(current['path']) != {row['path'] for row in selected}
                or not current['assigned_tma'].eq(tma).all()):
            raise ValueError('Supplied domain results differ from receipt selection')
        published = {(a['path'], a['sha256'])
                     for a in receipt['output_artifacts'].values()}
        for row in selected:
            path, core = row['path'], row['confirmed_core_id']
            events = [event for event in receipt['cache_events']
                      if event['path'] == path
                      and event['parameters'].get('schema')
                      == 'clinical_hs_native_candidate_spectral_domains_v1']
            if len(events) != 1:
                raise ValueError('Require one exact native-domain cache event per core')
            event = events[0]
            envelope = json.loads(
                Path(event['cache_path']).read_text(encoding='utf-8'),
            )
            identity, result = envelope['identity'], envelope['result']
            parameters = receipt['parameters']['parameters_and_exact_band_maps'][path]
            result_hash = sha256(_json_bytes(result)).hexdigest()
            raw_hash = receipt['raw_czi_full_sha256'][path]
            final_hash = receipt['final_manifest']['sha256']
            if (sha256(_json_bytes(identity)).hexdigest() != event['key']
                    or result_hash != envelope['result_sha256']
                    or identity['parameters'] != parameters
                    or event['parameters'] != parameters
                    or identity['input_sha256'] != raw_hash
                    or identity['code_sha256'] != receipt['helper_code_sha256']
                    or identity['package_versions'] != receipt['package_versions']
                    or identity['manifest_sha256'] != final_hash):
                raise ValueError('Candidate-domain cache/source identity mismatch')
            native_rows = [
                item for item in result['rows'] if item['K'] == cluster_count
            ]
            statuses = {item['status'] for item in native_rows}
            supplied = current.loc[current['path'].eq(path)]
            supplied_hashes = supplied['full_file_sha256']
            if (not native_rows or len(statuses) != 1
                    or set(supplied['status']) != statuses
                    or not statuses.issubset({'COMPLETE', 'BLOCKED'})
                    or not supplied['confirmed_core_id'].eq(core).all()
                    or not supplied_hashes.eq(identity['input_sha256']).all()):
                raise ValueError('Supplied domain status/core differs from native cache')
            panel = {
                'assigned_tma': tma, 'confirmed_core_id': core,
                'path': path, 'K': cluster_count,
                'status': next(iter(statuses)), 'cache_key': event['key'],
            }
            if panel['status'] == 'BLOCKED':
                panel['blocked_reason'] = native_rows[0]['blocked_reason']
            else:
                artifacts = [item for item in result['artifacts']
                             if item['K'] == cluster_count and item['kind'] == 'rgb']
                if not artifacts:
                    artifacts = [item for item in result['artifacts']
                                 if item['K'] == cluster_count and item['kind'] == 'labels']
                if len(artifacts) != 1:
                    raise ValueError('Require one receipt-bound native RGB or label artifact')
                artifact = artifacts[0]
                if (artifact['path'], artifact['sha256']) not in published:
                    raise ValueError('Native display artifact is absent from domain receipt')
                panel['artifact'] = artifact
                expected_size = [native_rows[0]['mosaic_width_pixels'],
                                 native_rows[0]['mosaic_height_pixels']]
                if artifact['size'] != expected_size:
                    raise ValueError('Native domain image size differs from cached geometry')
            panels.append(panel)
    return panels, bindings


def render_candidate_spectral_domain_gallery(
        domain_results: dict[str, pd.DataFrame], *, cluster_count: int,
        plot_dir: str | Path,
) -> Path:
    """Equal-frame review grid from exact source-bound candidate-domain artifacts."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    panels, bindings = _candidate_gallery_inputs(domain_results, cluster_count)
    _register_project_arial()
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    destination = plot_dir / (
        f'hyperspectral_candidate_domains_k{cluster_count}_tma1_tma2.pdf'
    )
    palette = np.zeros((cluster_count + 1, 3), dtype=np.uint8)
    palette[1:] = np.rint(
        mpl.colormaps['tab20'](np.arange(cluster_count))[:, :3] * 255,
    ).astype(np.uint8)
    with mpl.rc_context(_pseudocolor_style()):
        figure = plt.figure(figsize=(180 / 25.4, 139 / 25.4))
        try:
            for panel in panels:
                tma, core = panel['assigned_tma'], panel['confirmed_core_id']
                block = 0 if tma == 'TMA1' else 1
                column, row = int(core[1]) - 1, 'ABC'.index(core[0])
                x_mm, y_mm = 5 + block * 87 + column * 27, 106 - row * 28
                bounds = [x_mm / 180, y_mm / 139, 25 / 180, 25 / 139]
                if panel['status'] == 'BLOCKED':
                    axis = figure.add_axes(bounds, facecolor='black')
                    axis.set(xticks=[], yticks=[])
                    axis.text(.5, .5, 'BLOCKED\nNative map unavailable',
                              color='white', ha='center', va='center', fontsize=5)
                    for spine in axis.spines.values():
                        spine.set_linewidth(.5)
                else:
                    artifact = panel['artifact']
                    if _full_file_hash(artifact['path']) != artifact['sha256']:
                        raise ValueError('Candidate-domain native image changed')
                    with Image.open(artifact['path']) as image:
                        if (image.size != tuple(artifact['size'])
                                or image.mode != artifact['mode']):
                            raise ValueError(
                                'Candidate-domain native geometry/mode changed'
                            )
                        image.thumbnail((768, 768), resample=Image.Resampling.NEAREST)
                        preview = np.asarray(image).copy()
                    if artifact['kind'] == 'labels':
                        if preview.max() > cluster_count:
                            raise ValueError('Native domain label exceeds requested K')
                        preview = palette[preview]
                    _draw_pseudocolor_frame(figure, bounds, preview)
                figure.text(x_mm / 180, (y_mm + 25.8) / 139, core, va='bottom')
            for block, tma in enumerate(('TMA1', 'TMA2')):
                left = 5 + block * 87
                figure.text(left / 180, 136 / 139,
                            f'{tma} model | K={cluster_count}', va='top')
                figure.text(left / 180, 45 / 139, f'{tma} model cluster ID', va='top')
                for index in range(cluster_count):
                    column, row = index % 8, index // 8
                    x_mm, y_mm = left + column * 10, 37 - row * 6
                    figure.add_artist(Rectangle(
                        (x_mm / 180, y_mm / 139), 2.5 / 180, 2.5 / 139,
                        transform=figure.transFigure, facecolor=palette[index + 1] / 255,
                        edgecolor='black', linewidth=.5,
                    ))
                    figure.text(
                        (x_mm + 3.3) / 180, y_mm / 139,
                        str(index + 1), va='bottom',
                    )
            figure.text(
                .028, .135,
                'Candidate spectral labels, not histology or anatomical STUs. '
                'Same ID/color across TMAs does not imply the same biology.\n'
                'All positive-signal pixels, including background; no tissue mask. '
                'Black: undefined zero-signal phasor.\n'
                'Nearest-neighbor overview only; native maps unchanged. '
                'No H&E registration; matching positions are not matched patients.\n'
                'Internal review only, not Cell manuscript figures. '
                'Generator: clinical_roi_hyperspectral.py',
                va='top', linespacing=1.3,
            )
            figure.savefig(destination)
        finally:
            plt.close(figure)
    _verify_pseudocolor_pdf(destination, (180, 139))
    for binding in bindings.values():
        if _full_file_hash(binding['path']) != binding['sha256']:
            raise ValueError('Candidate-domain receipt changed before gallery publication')
    receipt = {
        'schema': 'clinical_hs_candidate_domain_gallery_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cluster_count': cluster_count, 'source_bindings': bindings,
        'native_panels': panels, 'canvas_mm': [180, 139], 'frame_mm': [25, 25],
        'overview_max_edge_pixels': 768, 'resampling': 'nearest neighbor only',
        'palette': 'tab20; per-TMA model identifiers, not cross-TMA identity',
        'helper_code_sha256': {
            'clinical_roi_hyperspectral.py': _full_file_hash(Path(__file__)),
            'image_previews.py': _full_file_hash(
                Path(__file__).with_name('image_previews.py'),
            ),
        },
        'plot_style_sha256': _plot_style_sha256(),
        'package_versions': {
            'python': sys.version,
            **{package: importlib.metadata.version(package)
               for package in ('numpy', 'matplotlib', 'Pillow', 'PyMuPDF')},
        },
        'figure_policy': 'Internal review only; not Cell manuscript figures',
        'limitations': [
            'Candidate spectral labels are not histology or anatomical STUs.',
            'Native labels include positive background; no tissue mask or smoothing.',
            'Native mosaic composition and speckle are not anatomical boundaries.',
            'No H&E registration; same IDs across TMAs do not imply shared biology.',
        ],
        'output_artifacts': {
            'combined_grid_pdf': {
                'path': str(destination.resolve()),
                'sha256': _full_file_hash(destination),
            },
        },
    }
    receipt_path = destination.with_suffix('.receipt.json')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=plot_dir, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(receipt))
        temporary.replace(receipt_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _trial_memory_bound(
        cube_bytes: int, pixels: int, chunk_pixels: int,
        max_working_bytes: int, model_vectors: int = 0,
) -> int:
    """Bound managed trial arrays plus decoder/model/rendering allowances, not RSS."""
    if (not isinstance(max_working_bytes, int) or max_working_bytes <= 0
            or not isinstance(chunk_pixels, int) or chunk_pixels <= 0
            or not isinstance(model_vectors, int) or model_vectors < 0):
        raise ValueError('Positive integer memory/chunk limits and nonnegative model vectors required')
    bound = (
        cube_bytes + pixels * 160 + min(pixels, chunk_pixels) * 2048
        + min(pixels, model_vectors) * 4096 + 8 * 1024**2
    )
    if bound > max_working_bytes:
        raise MemoryError(
            f'Full-spectrum trial requires {bound} bounded bytes; '
            f'limit={max_working_bytes}; no downsampling fallback'
        )
    return bound


def load_spectral_trial_for_acquisition(
        row: dict[str, Any], channel_map: pd.DataFrame, *,
        cache: HSCache, max_working_bytes: int,
) -> dict[str, Any]:
    """Read any audited acquisition row's full-stage C20 display cube into RAM."""
    from aicspylibczi import CziFile

    channels = channel_map.loc[channel_map['path'].eq(row['path'])].reset_index(drop=True)
    if len(channels) != 20:
        raise ValueError('Acquisition requires the complete audited 20-band channel map')
    source_path = _require_raw_hs(row)
    full_file_sha256, signature = cache.raw_identity(source_path)
    reader = CziFile(str(source_path))
    metadata, channel_rows = inspect_raw_metadata(row, reader=reader)
    dimensions = _spectral_layout(metadata)
    current = _spectral_channel_grid(channel_rows)
    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm',
        'detection_band_center_nm',
    ]
    if (channels['metadata_xml_sha256'].nunique() != 1
            or channels['metadata_xml_sha256'].iloc[0] != metadata['metadata_xml_sha256']
            or channels['dimensions_json'].nunique() != 1
            or channels['dimensions_json'].iloc[0] != metadata['dimensions_json']
            or not current[columns].equals(channels[columns])):
        raise ValueError('Current acquisition metadata differs from the audited channel map')
    box = reader.get_mosaic_bounding_box()
    region = tuple(int(value) for value in (box.x, box.y, box.w, box.h))
    x, y, width, height = region
    if width <= 0 or height <= 0:
        raise ValueError('Acquisition stage bounding box must have positive extent')
    shape = (20, 1, height, width)
    pixels = width * height
    bound = _trial_memory_bound(pixels * 40, pixels, 65536, max_working_bytes)
    selectors = {
        axis: start for axis, (start, stop) in dimensions.items()
        if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}
    }
    cube = np.empty(shape, dtype=np.uint16)
    for channel in range(20):
        plane = reader.read_mosaic(
            C=channel, scale_factor=1.0, region=region,
            background_color=(0., 0., 0.), **selectors,
        )
        if (plane.dtype != np.uint16 or plane.size != pixels
                or plane.shape[-2:] != (height, width)):
            raise ValueError('Full-scale mosaic channel shape/dtype mismatch')
        cube[channel, 0] = plane.reshape(height, width)
        del plane
    cache.require_unchanged(source_path, signature)
    cube.flags.writeable = False
    return {
        'cube': cube, 'channels': channels, 'max_working_bytes': max_working_bytes,
        'summary': {
            'assigned_tma': row.get('assigned_tma', ''),
            'confirmed_core_id': row.get('confirmed_core_id', row.get('core_id', '')),
            'path': row['path'],
            'metadata_xml_sha256': metadata['metadata_xml_sha256'],
            'full_file_sha256': full_file_sha256,
            'cube_axes': 'C,display_frame,Y,X', 'cube_shape': shape,
            'channel_count': 20,
            'native_tile_count': int(dimensions['M'][1] - dimensions['M'][0]) if 'M' in dimensions else 1,
            'mosaic_x': x, 'mosaic_y': y, 'scale_factor': 1.0,
            'working_allocation_bound_bytes': bound,
            'overlap_policy': 'Zeiss/libCZI read_mosaic stage composition; display only',
            'display_layout': 'single complete native-scale stage bounding box',
            'raw_storage': 'CZI untouched; C20 display cube read-only RAM, never cached',
            'native_tiles': 'original M tiles/overlaps remain in the untouched raw CZI',
            'generating_script': str(Path(__file__).resolve()),
            'plot_style_sha256': cache.plot_style_sha256,
            'claim_label': 'Exploratory',
        },
    }


def load_c2_spectral_trial(
        final_manifest: pd.DataFrame, channel_map: pd.DataFrame, *,
        cache: HSCache, max_working_bytes: int,
) -> dict[str, Any]:
    """Read one full-stage C20 display cube into RAM, never into the raw cache."""
    final = _validated_spectral_manifest(final_manifest, cache)
    selected = final.loc[
        final['assigned_tma'].eq('TMA1') & final['confirmed_core_id'].eq('C2')
    ]
    if len(selected) != 1:
        raise ValueError('Require exactly one selected TMA1-C2 acquisition')
    return load_spectral_trial_for_acquisition(
        selected.iloc[0].to_dict(), channel_map, cache=cache, max_working_bytes=max_working_bytes,
    )

def _trial_cube(
        cube: np.ndarray, *, max_working_bytes: int,
        chunk_pixels: int, model_vectors: int = 0,
) -> tuple[np.ndarray, int]:
    """Validate every input band in bounded chunks without copying the cube."""
    if not isinstance(cube, np.ndarray):
        raise TypeError('A NumPy C20 display cube is required')
    if cube.ndim != 4 or cube.shape[0] != 20 or min(cube.shape[1:]) <= 0:
        raise ValueError('Expected cube shape (20, frames, Y, X) with positive extents')
    if cube.dtype.kind not in 'uif' or not cube.flags.c_contiguous:
        raise ValueError('Cube must be real numeric and C-contiguous; no implicit cube copy')
    pixels = prod(cube.shape[1:])
    bound = _trial_memory_bound(
        cube.nbytes, pixels, chunk_pixels, max_working_bytes, model_vectors,
    )
    flat = cube.reshape(20, pixels)
    for start in range(0, pixels, chunk_pixels):
        block = flat[:, start:start + chunk_pixels]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError('Full-spectrum input must be finite and nonnegative')
    return flat, bound


def _trial_features(block: np.ndarray, normalization: str) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero C20 vectors and the exact zero/nonzero selector."""
    values = block.T.astype(np.float64)
    with np.errstate(over='ignore'):
        totals = values.sum(axis=1)
    if not np.isfinite(totals).all():
        raise ValueError('Full-spectrum pixel totals exceed floating-point range')
    positive = totals > 0
    values = values[positive]
    if normalization == 'log1p':
        np.log1p(values, out=values)
    elif normalization == 'l1':
        values /= values.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Normalization must be 'log1p' or 'l1'")
    return values, positive


def full_spectrum_pca(
        cube: np.ndarray, *, normalization: str = 'log1p',
        max_working_bytes: int, chunk_pixels: int = 65536,
) -> dict[str, Any]:
    """Exact streamed C20 covariance PCA; one symmetric RGB score scale, no equalization."""
    if normalization not in {'log1p', 'l1'}:
        raise ValueError("Normalization must be 'log1p' or 'l1'")
    flat, bound = _trial_cube(
        cube, max_working_bytes=max_working_bytes, chunk_pixels=chunk_pixels,
    )
    pixels = flat.shape[1]
    total = np.zeros(20, dtype=np.float64)
    count = 0
    minima = np.full(20, np.inf)
    maxima = np.full(20, -np.inf)
    for start in range(0, pixels, chunk_pixels):
        values, _ = _trial_features(flat[:, start:start + chunk_pixels], normalization)
        total += values.sum(axis=0)
        count += len(values)
        if len(values):
            minima = np.minimum(minima, values.min(axis=0))
            maxima = np.maximum(maxima, values.max(axis=0))
    mean = total / max(count, 1)
    if count and np.array_equal(minima, maxima):
        mean = minima.copy()
    scatter = np.zeros((20, 20), dtype=np.float64)
    for start in range(0, pixels, chunk_pixels):
        values, _ = _trial_features(flat[:, start:start + chunk_pixels], normalization)
        values -= mean
        scatter += values.T @ values
    eigenvalues, vectors = np.linalg.eigh(scatter / max(count - 1, 1))
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    components = vectors[:, ::-1].T.copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    variance_sum = float(eigenvalues.sum())
    ratios = eigenvalues / variance_sum if variance_sum else np.zeros(20)
    scores = np.full((pixels, 3), np.nan, dtype=np.float32)
    half_range = 0.0
    component_ranges = np.zeros(3, dtype=np.float64)
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        values, positive = _trial_features(flat[:, start:stop], normalization)
        projected = (values - mean) @ components[:3].T
        scores[start:stop][positive] = projected
        if projected.size:
            component_ranges = np.maximum(component_ranges, np.abs(projected).max(axis=0))
            half_range = float(component_ranges.max())
    rgb = np.zeros((pixels, 3), dtype=np.uint8)
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        block = scores[start:stop]
        positive = np.isfinite(block[:, 0])
        colors = np.full((int(positive.sum()), 3), .5)
        if half_range:
            colors += block[positive].astype(np.float64) / (2 * half_range)
        # Reserve exact black for zero signal, including after uint8 quantization.
        rgb[start:stop][positive] = (
            1 + np.rint(np.clip(colors, 0, 1) * 254)
        ).astype(np.uint8)
    return {
        'rgb': rgb.reshape((*cube.shape[1:], 3)),
        'scores': scores.reshape((*cube.shape[1:], 3)),
        'components': pd.DataFrame(
            components.T, columns=[f'PC{index + 1}' for index in range(20)],
        ).rename_axis('channel_index').reset_index(),
        'explained_variance_ratio': ratios,
        'variance': pd.DataFrame({
            'component': np.arange(1, 21), 'eigenvalue': eigenvalues,
            'explained_variance_ratio': ratios, 'cumulative_ratio': ratios.cumsum(),
        }),
        'feature_mean': mean,
        'summary': {
            'normalization': normalization, 'channel_count': 20,
            'positive_signal_pixels': count, 'zero_signal_pixels': pixels - count,
            'rgb_score_half_range': half_range,
            'component_score_half_ranges': tuple(float(value) for value in component_ranges),
            'rgb_mapping': (
                'R=PC1 G=PC2 B=PC3; code=1+round(254*(0.5+score/(2*shared maximum))); '
                'zero signal alone has code 0'
            ),
            'fit_population': 'all nonzero input pixels; no sampling, band scaling or mask',
            'status': 'ZERO_SIGNAL' if not count else (
                'CONSTANT_SPECTRA' if not variance_sum else 'OK'
            ),
            'working_allocation_bound_bytes': bound,
        },
    }


def full_spectrum_clusters(
        cube: np.ndarray, *, k: int, max_working_bytes: int,
        normalization: str = 'l1', chunk_pixels: int = 65536,
) -> dict[str, Any]:
    """Fit every positive C20 vector with L1 shape or magnitude-aware log1p.

    Log1p retains brightness differences without per-band scaling. Its
    partitions may reflect illumination, not histology. Raw/L1 summaries
    always describe the input spectra, independently of the fit transform.
    """
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    if not isinstance(k, int) or not 2 <= k <= 18:
        raise ValueError('Categorical trial K must be an integer from 2 to 18')
    if normalization not in {'l1', 'log1p'}:
        raise ValueError("Normalization must be 'log1p' or 'l1'")
    feature_name = 'L1' if normalization == 'l1' else 'log1p'
    flat, bound = _trial_cube(
        cube, max_working_bytes=max_working_bytes, chunk_pixels=chunk_pixels,
        model_vectors=k,
    )
    pixels = flat.shape[1]
    count = distinct = 0
    initial = np.empty((k, 20), dtype=np.float64)
    # Retain at most K exact-distinct vectors from the complete population.
    # A bounded initialization sample could miss a rare vector and wrongly
    # reject a population that genuinely contains K distinct vectors.
    for start in range(0, pixels, chunk_pixels):
        values, _ = _trial_features(
            flat[:, start:start + chunk_pixels], normalization,
        )
        count += len(values)
        if distinct == k:
            continue
        unseen = np.ones(len(values), dtype=bool)
        for center in initial[:distinct]:
            unseen &= ~np.all(values == center, axis=1)
        while distinct < k and np.any(unseen):
            initial[distinct] = values[np.argmax(unseen)]
            unseen &= ~np.all(values == initial[distinct], axis=1)
            distinct += 1
    centers = np.zeros((k, 20), dtype=np.float64)
    fit_updates = fitted_count = 0
    if count:
        if distinct < k:
            raise ValueError('Insufficient distinct full-spectrum population vectors for requested K')
        model = MiniBatchKMeans(
            n_clusters=k, init=initial, n_init=1, random_state=0,
            batch_size=max(k, min(chunk_pixels, pixels)),
            compute_labels=False, reassignment_ratio=0.0,
        )
        # Bootstrap partial_fit with K real population rows, not synthetic seed
        # observations. Every positive row then participates exactly once.
        pending = np.empty((k, 20), dtype=np.float64)
        pending_count = 0
        with threadpool_limits(limits=1):
            for start in range(0, pixels, chunk_pixels):
                values, _ = _trial_features(
                    flat[:, start:start + chunk_pixels], normalization,
                )
                if pending_count < k:
                    take = min(k - pending_count, len(values))
                    pending[pending_count:pending_count + take] = values[:take]
                    pending_count += take
                    values = values[take:]
                    if pending_count == k:
                        model.partial_fit(pending)
                        fitted_count += k
                        fit_updates += 1
                if pending_count == k and len(values):
                    model.partial_fit(values)
                    fitted_count += len(values)
                    fit_updates += 1
        centers = model.cluster_centers_
        order = np.lexsort(tuple(centers[:, index] for index in range(19, -1, -1)))
        centers = centers[order]
    labels = np.zeros(pixels, dtype=np.uint8)
    counts = np.zeros(k + 1, dtype=np.int64)
    raw_sums = np.zeros((k, 20), dtype=np.float64)
    shape_sums = np.zeros_like(raw_sums)
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        values, positive = _trial_features(flat[:, start:stop], normalization)
        counts[0] += int((~positive).sum())
        if not len(values):
            continue
        best = np.full(len(values), np.inf)
        assigned = np.zeros(len(values), dtype=np.uint8)
        for index, center in enumerate(centers):
            distance = np.square(values - center).sum(axis=1)
            closer = distance < best
            assigned[closer] = index + 1
            best[closer] = distance[closer]
        labels[start:stop][positive] = assigned
        raw = flat[:, start:stop].T[positive].astype(np.float64)
        shapes = (
            values if normalization == 'l1'
            else raw / raw.sum(axis=1, keepdims=True)
        )
        for index in range(k):
            members = assigned == index + 1
            counts[index + 1] += int(members.sum())
            raw_sums[index] += raw[members].sum(axis=0)
            shape_sums[index] += shapes[members].sum(axis=0)
    denominator = counts[1:, None]
    raw_means = np.full((k, 20), np.nan)
    shape_means = np.full_like(raw_means, np.nan)
    np.divide(raw_sums, denominator, out=raw_means, where=denominator > 0)
    np.divide(shape_sums, denominator, out=shape_means, where=denominator > 0)
    return {
        'labels': labels.reshape(cube.shape[1:]), 'centers': centers,
        'counts': pd.DataFrame({
            'cluster_id': np.arange(k + 1), 'pixel_count': counts,
            'feature_transform': normalization,
        }),
        'spectra': pd.DataFrame({
            'cluster_id': np.repeat(np.arange(1, k + 1), 20),
            'channel_index': np.tile(np.arange(20), k),
            'pixel_count': np.repeat(counts[1:], 20),
            'mean_raw_intensity': raw_means.reshape(-1),
            'mean_l1_fraction': shape_means.reshape(-1),
            'feature_transform': normalization,
        }),
        'summary': {
            'normalization': normalization, 'channel_count': 20,
            'requested_cluster_count': k,
            'actual_cluster_count': int(np.count_nonzero(counts[1:])),
            'positive_signal_pixels': count, 'zero_signal_pixels': pixels - count,
            'fit_population_count': fitted_count,
            'fit_population': (
                f'every positive C20 {feature_name} vector, once in input flat order'
            ),
            'training_passes': 1 if count else 0,
            'algorithm': 'MiniBatchKMeans',
            'init_method': (
                f'first K exact-distinct {feature_name} vectors from full population scan'
            ),
            'init_count': distinct,
            'fit_method': (
                f'20-band {feature_name} MiniBatchKMeans partial_fit; seed=0 n_init=1; '
                'reassignment_ratio=0; one complete population pass; single thread'
            ),
            'fit_iterations': fit_updates,
            'label_policy': 'lexicographic centroid IDs; local to this C2/K fit, not biology',
            'status': 'OK' if count else 'ZERO_SIGNAL',
            'working_allocation_bound_bytes': bound,
        },
    }


def plot_c2_trial_map(
        result: dict[str, Any], *, source: dict[str, Any], destination: str | Path,
        component: int | None = None, scale_mode: str = 'shared',
) -> Path:
    """Save C2 maps; PCA RGB defaults to shared, optionally per-component full ranges."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Patch

    _register_project_arial()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    categorical = 'labels' in result
    if component is not None and (categorical or component not in {1, 2, 3}):
        raise ValueError('Component maps require PCA component 1, 2 or 3')
    if scale_mode not in {'shared', 'per_component'}:
        raise ValueError("Scale mode must be 'shared' or 'per_component'")
    if scale_mode != 'shared' and (categorical or component is not None):
        raise ValueError('Per-component scales require a PCA RGB composite')
    display_data = result['labels'] if categorical else (
        result['rgb'] if component is None else result['scores'][..., component - 1]
    )
    if scale_mode == 'per_component':
        # Remap scores, never the already-quantized shared RGB or source cube.
        scores = result['scores'].reshape(-1, 3)
        ranges = np.asarray(result['summary']['component_score_half_ranges'])
        display_data = np.zeros(result['scores'].shape, dtype=np.uint8)
        rgb = display_data.reshape(-1, 3)
        for start in range(0, len(scores), 65536):
            stop = min(start + 65536, len(scores))
            block = scores[start:stop]
            positive = np.isfinite(block[:, 0])
            colors = np.zeros((int(positive.sum()), 3), dtype=np.float64)
            np.divide(block[positive], 2 * ranges, out=colors, where=ranges > 0)
            colors += .5
            # As in the shared view, only exact all-band zeros stay black.
            rgb[start:stop][positive] = (
                1 + np.rint(np.clip(colors, 0, 1) * 254)
            ).astype(np.uint8)
    tiles = display_data.shape[0]
    columns = min(2, tiles)
    rows = (tiles + columns - 1) // columns
    style = _pseudocolor_style()
    with plt.rc_context(style):
        width_mm = 230 if categorical else 180
        height_mm = (75 if categorical else 40) + 65 * rows
        figure, axes = plt.subplots(
            rows, columns, squeeze=False,
            figsize=(width_mm / 25.4, height_mm / 25.4),
        )
        figure.subplots_adjust(
            left=.04, right=.68 if categorical else .96,
            bottom=.30 if categorical else .24, top=.88, hspace=.08,
        )
        image = None
        cmap = None
        if categorical:
            k = result['summary']['requested_cluster_count']
            palette = plt.get_cmap('tab20').colors[:k]
            cmap = ListedColormap([(0., 0., 0.), *palette])
            norm = BoundaryNorm(np.arange(k + 2) - .5, k + 1)
        elif component is not None:
            observed_range = result['summary']['component_score_half_ranges'][component - 1]
            half_range = observed_range or 1.
            cmap = plt.get_cmap('coolwarm').copy()
            cmap.set_bad('black')
            norm = Normalize(-half_range, half_range)
        for index, axis in enumerate(axes.flat):
            axis.set_axis_off()
            if index >= tiles:
                continue
            options = {} if cmap is None else {'cmap': cmap, 'norm': norm}
            image = axis.imshow(display_data[index], interpolation='nearest', **options)
            axis.text(.01, .99, 'Full-stage C2' if tiles == 1 else f'Display frame {index}',
                      color='white',
                      va='top', transform=axis.transAxes)
        if categorical:
            counts = result['counts'].set_index('cluster_id')['pixel_count']
            handles = [
                Patch(
                    facecolor=cmap(index),
                    label=f'{index}: {int(counts[index]):,} pixels',
                )
                for index in range(k + 1)
            ]
            legend_axis = figure.add_axes((.72, .30, .26, .58))
            legend_axis.set_axis_off()
            legend_axis.legend(
                handles=handles, loc='upper left', ncol=1,
                frameon=False, borderaxespad=0, labelspacing=.65,
                title='Cluster ID: exact display-pixel count',
            )
            description = (
                f'C2 full 20-band L1 spectral shape, K={k}; categorical IDs only'
            )
            notes = (
                'Fit: L1-normalized full 20-band spectral shapes.\n'
                'Qualitative false colors and IDs are local to this fit; '
                'not anatomy or histology.'
            )
            if result['summary']['normalization'] == 'log1p':
                description = (
                    f'C2 full 20-band raw-intensity log1p, K={k}; '
                    'categorical IDs only'
                )
                notes = (
                    'Fit: log1p of all 20 raw bands; no per-band scaling.\n'
                    'Illumination/brightness may drive clusters; '
                    'IDs do not imply anatomy or histology.\n'
                    'Qualitative false colors are local to this fit; '
                    'raw data and the L1 view remain available.'
                )
            figure.text(.04, .20, notes, va='top')
        else:
            normalization = result['summary']['normalization']
            description = f'C2 full 20-band PCA; {normalization}; '
            if component is None:
                description += 'R=PC1, G=PC2, B=PC3'
                if scale_mode == 'per_component':
                    description += '; independent component scales'
                    ranges = result['summary']['component_score_half_ranges']
                    limits = '; '.join(
                        f'{channel}/PC{index + 1}: -{limit:.5g} to +{limit:.5g}'
                        for index, (channel, limit) in enumerate(zip('RGB', ranges))
                    )
                    figure.text(
                        .04, .14,
                        'Independent full observed symmetric linear ranges:\n'
                        f'{limits}\n'
                        'Zero score = mid-gray; codes 1–255 reserve black for zero signal.\n'
                        'No equalization or tail clipping; relative PC magnitudes are not comparable.\n'
                        'RGB colors are PCA display coordinates, not wavelengths or histology.',
                    )
                else:
                    half_range = result['summary']['rgb_score_half_range']
                    figure.text(
                        .04, .15,
                        f'All RGB channels: score -{half_range:.5g} to +{half_range:.5g}; '
                        'zero score = mid-gray.\nCodes 1–255 reserve black for zero signal; '
                        'one shared linear scale, no equalization or tail clipping.',
                    )
            else:
                description += f'signed PC{component} score'
                if observed_range:
                    color_axis = figure.add_axes((.30, .14, .4, .025))
                    figure.colorbar(image, cax=color_axis, orientation='horizontal')
                else:
                    figure.text(.30, .14, 'All observed positive-pixel scores = 0')
        figure.text(
            .04, .96, description + f'\nStatus: {result["summary"]["status"]}', va='top',
        )
        figure.text(
            .04, .075,
            'Black = exact all-band zero signal, not a tissue mask. Positive background retained.\n'
            'Zeiss/libCZI stage overlap composition is display-only. Colors are not histology/STUs.',
        )
        figure.text(.04, .04, f'Generated by {Path(__file__).name}; internal review only.')
        figure.text(.04, .015, f'CZI SHA-256: {source["full_file_sha256"]}')
        figure.savefig(destination, format='pdf')
        plt.close(figure)
    return destination


def plot_c2_cluster_spectra(
        result: dict[str, Any], channels: pd.DataFrame, *,
        source: dict[str, Any], destination: str | Path, normalized: bool = False,
) -> Path:
    """Plot every 20-band cluster mean; raw and per-pixel L1 means stay separate."""
    import matplotlib.pyplot as plt

    _register_project_arial()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    spectra = result['spectra'].merge(
        channels[['channel_index', 'detection_band_center_nm']],
        on='channel_index', validate='many_to_one',
    )
    field = 'mean_l1_fraction' if normalized else 'mean_raw_intensity'
    with plt.rc_context(_pseudocolor_style()):
        figure, axis = plt.subplots(figsize=(120 / 25.4, 85 / 25.4))
        top = .84 if result['summary']['normalization'] == 'log1p' else .90
        figure.subplots_adjust(left=.14, right=.76, bottom=.24, top=top)
        for cluster_id, frame in spectra.groupby('cluster_id', sort=True):
            axis.plot(
                frame['detection_band_center_nm'], frame[field],
                color=plt.get_cmap('tab20').colors[int(cluster_id) - 1],
                label=f'{cluster_id}: {int(frame["pixel_count"].iloc[0]):,} pixels',
            )
        axis.set_xlabel('Recorded detection-band center (nm); all 20 bands')
        axis.set_ylabel(
            'Mean per-pixel L1 fraction' if normalized else 'Mean stage-composed input intensity'
        )
        axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
        figure.text(.14, .96, f'C2 K={result["summary"]["requested_cluster_count"]} display-ID spectra')
        if result['summary']['normalization'] == 'log1p':
            figure.text(
                .14, .91,
                'Fit: all 20 raw bands, log1p; no band scaling.\n'
                'Illumination/brightness may drive IDs; not histology.',
            )
        figure.text(
            .03, .05, 'Unsmoothed within-image means, not independent biological replicates.\n'
            f'Generated by {Path(__file__).name}; internal review only.',
        )
        figure.text(.03, .015, f'CZI SHA-256: {source["full_file_sha256"]}')
        figure.savefig(destination, format='pdf')
        plt.close(figure)
    return destination


def full_spectrum_h1_phase(
        cube: np.ndarray, *, max_working_bytes: int, chunk_pixels: int = 65536,
) -> dict[str, Any]:
    """Project every C20 spectrum to normalized positive-sine H1 and atan2 phase.

    This is a supplementary one-number spectral-shape projection, not an
    emission-wavelength estimate or a substitute for the full spectral vector.
    Only exact all-band zeros are undefined; no positive pixel is excluded.
    """
    flat, bound = _trial_cube(
        cube, max_working_bytes=max_working_bytes, chunk_pixels=chunk_pixels,
    )
    pixels = flat.shape[1]
    cosine, sine = _h1_channel_weights(20)
    g = np.full(pixels, np.nan, dtype=np.float64)
    s = np.full(pixels, np.nan, dtype=np.float64)
    phase = np.full(pixels, np.nan, dtype=np.float64)
    magnitude = np.full(pixels, np.nan, dtype=np.float64)
    positive_count = 0
    magnitude_sum = 0.
    magnitude_min, magnitude_max = np.inf, -np.inf
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        values = flat[:, start:stop].astype(np.float64)
        with np.errstate(over='ignore'):
            total = values.sum(axis=0)
        if not np.isfinite(total).all():
            raise ValueError('Full-spectrum pixel totals exceed floating-point range')
        positive = total > 0
        count = int(np.count_nonzero(positive))
        positive_count += count
        # Normalize the standard H1 weighted sums only by the pixel total:
        # no log transform, brightness encoding, band scaling or selection.
        real_sum = cosine @ values
        imag_sum = sine @ values
        np.divide(real_sum, total, out=g[start:stop], where=positive)
        np.divide(imag_sum, total, out=s[start:stop], where=positive)
        np.arctan2(s[start:stop], g[start:stop], out=phase[start:stop])
        np.hypot(g[start:stop], s[start:stop], out=magnitude[start:stop])
        if count:
            radii = magnitude[start:stop][positive]
            magnitude_sum += float(radii.sum())
            magnitude_min = min(magnitude_min, float(radii.min()))
            magnitude_max = max(magnitude_max, float(radii.max()))
    return {
        'g': g.reshape(cube.shape[1:]),
        's': s.reshape(cube.shape[1:]),
        'phase_radians': phase.reshape(cube.shape[1:]),
        'magnitude': magnitude.reshape(cube.shape[1:]),
        'summary': {
            'channel_count': 20, 'channel_indices': tuple(range(20)),
            'transform': 'atan2(S, G); G=sum(I*cos(2*pi*c/20))/sum(I); '
                         'S=sum(I*sin(2*pi*c/20))/sum(I)',
            'normalization': 'per-pixel total only; standard normalized H1 phasor',
            'phase_range_radians': (-np.pi, np.pi),
            'positive_signal_pixels': positive_count,
            'zero_signal_pixels': pixels - positive_count,
            'magnitude_min': magnitude_min if positive_count else None,
            'magnitude_mean': magnitude_sum / positive_count if positive_count else None,
            'magnitude_max': magnitude_max if positive_count else None,
            'phase_limit': 'One-number cyclic spectral-shape projection, not mean emission '
                           'wavelength; phase is unstable near zero H1 magnitude',
            'status': 'OK' if positive_count else 'ZERO_SIGNAL',
            'working_allocation_bound_bytes': bound,
        },
    }


def signal_weighted_h1_phase(
        cube: np.ndarray, phase_result: dict[str, Any], *,
        max_working_bytes: int, chunk_pixels: int = 65536,
) -> dict[str, Any]:
    """Keep H1 hue; encode log1p full-band total as HSV value for display only."""
    from matplotlib.colors import hsv_to_rgb

    flat, bound = _trial_cube(
        cube, max_working_bytes=max_working_bytes, chunk_pixels=chunk_pixels,
    )
    phase = phase_result['phase_radians']
    if phase.shape != cube.shape[1:]:
        raise ValueError('H1 phase and C20 cube must have the same display shape')
    pixels = flat.shape[1]
    total = np.empty(pixels, dtype=np.float64)
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        with np.errstate(over='ignore'):
            total[start:stop] = flat[:, start:stop].sum(
                axis=0, dtype=np.float64,
            )
        if not np.isfinite(total[start:stop]).all():
            raise ValueError('Full-spectrum pixel totals exceed floating-point range')
    observed_max = float(total.max())
    log_max = float(np.log1p(observed_max))
    value = np.log1p(total)
    if log_max > 0:
        value /= log_max
    rgb = np.zeros((pixels, 3), dtype=np.uint8)
    phases = phase.reshape(-1)
    positive_count = 0
    for start in range(0, pixels, chunk_pixels):
        stop = min(start + chunk_pixels, pixels)
        positive = total[start:stop] > 0
        count = int(positive.sum())
        positive_count += count
        if not count:
            continue
        angles = phases[start:stop][positive]
        if not np.isfinite(angles).all():
            raise ValueError('Every positive-signal pixel requires finite H1 phase')
        hsv = np.ones((count, 3), dtype=np.float64)
        hsv[:, 0] = (angles + np.pi) / (2 * np.pi)
        hsv[:, 2] = value[start:stop][positive]
        colors = hsv_to_rgb(hsv)
        encoded = np.rint(colors * 255).astype(np.uint8)
        # This is uint8 raster quantization, not a cutoff on numeric V/signal.
        # Reserve exact black for all-band zeros even at tiny positive V.
        rounded_black = ~encoded.any(axis=1)
        rows = np.flatnonzero(rounded_black)
        encoded[rows, np.argmax(colors[rows], axis=1)] = 1
        rgb[start:stop][positive] = encoded
    summary = dict(phase_result['summary'])
    summary.update({
        'display_projection': 'full cyclic H1 hue; full C20 total-signal value',
        'value_transform': 'log1p(sum_20_bands(I))/log1p(observed_max_total)',
        'observed_max_total_signal': observed_max,
        'observed_max_log1p_total_signal': log_max,
        'positive_signal_pixels': positive_count,
        'zero_signal_pixels': pixels - positive_count,
        'value_min': float(value.min()), 'value_max': float(value.max()),
        'saturation': 1.,
        'signal_cutoff': None, 'spatial_mask': False,
        'smoothing': False, 'clipping': False,
        'wavelength_selection': 'all 20 recorded bands',
        'zero_signal_value': 0.,
        'rgb_encoding': 'round(255*HSV_RGB); positive rounded-black pixels '
                        'use dominant-channel code 1; no signal cutoff',
        'working_allocation_bound_bytes': bound,
    })
    return {
        'phase_radians': phase, 'total_signal': total.reshape(phase.shape),
        'value': value.reshape(phase.shape),
        'rgb': rgb.reshape((*phase.shape, 3)), 'summary': summary,
    }


def plot_c2_h1_phase(
        result: dict[str, Any], *, source: dict[str, Any], destination: str | Path,
) -> Path:
    """Save a supplementary C2 phase map with a full cyclic HSV legend and SHA."""
    return _plot_c2_h1_phase(
        result, source=source, destination=destination, signal_weighted=False,
    )


def plot_c2_h1_signal_weighted(
        result: dict[str, Any], *, source: dict[str, Any], destination: str | Path,
) -> Path:
    """Save a separate signal-weighted phase PDF; never replace the phase map."""
    return _plot_c2_h1_phase(
        result, source=source, destination=destination, signal_weighted=True,
    )


def _plot_c2_h1_phase(
        result: dict[str, Any], *, source: dict[str, Any],
        destination: str | Path, signal_weighted: bool,
) -> Path:
    """Render a cyclic hue key and separate notes/source footer for H1 displays."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb

    _register_project_arial()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Include both identical red endpoints, unlike the truncated default HSV LUT.
    hue = np.linspace(0., 1., 257)
    cmap = ListedColormap(hsv_to_rgb(np.column_stack((hue, np.ones((257, 2))))))
    cmap.set_bad('black')
    norm = Normalize(-np.pi, np.pi)
    phase = result['phase_radians']
    frames = phase.shape[0]
    columns = min(2, frames)
    rows = (frames + columns - 1) // columns
    with plt.rc_context(_pseudocolor_style()):
        height_mm = (75 if signal_weighted else 55) + 65 * rows
        figure, axes = plt.subplots(
            rows, columns, squeeze=False,
            figsize=(180 / 25.4, height_mm / 25.4),
        )
        figure.subplots_adjust(
            left=.04, right=.96, bottom=.40 if signal_weighted else .32,
            top=.88, hspace=.08,
        )
        for index, axis in enumerate(axes.flat):
            axis.set_axis_off()
            if index >= frames:
                continue
            if signal_weighted:
                axis.imshow(result['rgb'][index], interpolation='nearest')
            else:
                axis.imshow(
                    phase[index], cmap=cmap, norm=norm,
                    interpolation='nearest',
                )
            axis.text(
                .01, .99, 'Full-stage C2' if frames == 1 else f'Display frame {index}',
                color='white', va='top', transform=axis.transAxes,
            )
        color_axis = figure.add_axes(
            (.25, .33 if signal_weighted else .245, .5, .025),
        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
            orientation='horizontal',
        )
        colorbar.set_ticks(
            [-np.pi, -np.pi / 2, 0., np.pi / 2, np.pi],
            labels=['−π', '−π/2', '0', 'π/2', 'π'],
        )
        colorbar.set_label('H1 phase angle (radians); cyclic HSV hue, −π = π in color')
        summary = result['summary']
        if summary['positive_signal_pixels']:
            magnitude_text = (
                f'H1 magnitude |G+iS|: min {summary["magnitude_min"]:.5g}; '
                f'mean {summary["magnitude_mean"]:.5g}; max {summary["magnitude_max"]:.5g}'
            )
        else:
            magnitude_text = 'H1 magnitude undefined: exact all-band zero signal throughout.'
        heading = 'C2 supplementary full 20-band H1 spectral phase: atan2(S, G)'
        if signal_weighted:
            heading = 'C2 signal-weighted full 20-band H1 phase display projection'
            figure.text(
                .04, .27,
                'Hue = full cyclic atan2(S, G); saturation = 1.\n'
                'Value = log1p(full 20-band total) / log1p(observed maximum total).\n'
                f'Observed maximum total: '
                f'{summary["observed_max_total_signal"]:.8g}; '
                'no cutoff, clipping or wavelength restriction.',
                va='top',
            )
        figure.text(
            .04, .96,
            heading + '\n'
            f'Full cyclic range [−π, π] radians; status: {summary["status"]}', va='top',
        )
        figure.text(
            .04, .19 if signal_weighted else .15,
            magnitude_text + '\n'
            'G/S: positive-sine H1 weights over channel indices 0–19; divide only by pixel total.\n'
            'One-number spectral-shape projection, not average emission wavelength or histology.\n'
            'Phase is unstable near zero H1 magnitude; no magnitude threshold is applied.',
            va='top' if signal_weighted else 'baseline',
        )
        figure.text(
            .04, .075,
            'Black = exact all-band zero; all positive background retained. '
            'No mask or smoothing.\n'
            + (
                'Uint8 display quantization reserves code 1 for positive values '
                'rounding to black; numeric V is unchanged.'
                if signal_weighted else
                'Zeiss/libCZI stage composition is display-only; '
                'full cube/PCA/cluster views are retained.'
            ),
        )
        figure.text(.04, .04, f'Generated by {Path(__file__).name}; internal review only.')
        figure.text(.04, .015, f'CZI SHA-256: {source["full_file_sha256"]}')
        figure.savefig(destination, format='pdf')
        plt.close(figure)
    return destination


# ==============================================================================
# Cohort hyperspectral processing, shared PCA, and candidate cluster comparison
# ==============================================================================


def _render_cohort_spectrum_pdf(
        l1_by_core: dict[str, np.ndarray],
        band_centers: np.ndarray,
        destination: str | Path,
) -> Path:
    """Render cohort mean L1 spectra for TMA1 and TMA2 adhering to project plot contract."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    destination = Path(destination)
    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()):
        fig, ax = plt.subplots(1, 1, figsize=(180 / 25.4, 110 / 25.4))

        tma1_spectra = [s.mean(axis=0) for k, s in l1_by_core.items() if k.startswith('TMA1') and len(s) > 0]
        tma2_spectra = [s.mean(axis=0) for k, s in l1_by_core.items() if k.startswith('TMA2') and len(s) > 0]

        # Subtle per-core dashed lines
        for k, s in l1_by_core.items():
            if len(s) > 0:
                mean_s = s.mean(axis=0)
                color = '#56B4E9' if k.startswith('TMA1') else '#E69F00'
                ax.plot(band_centers, mean_s, color=color, alpha=0.3, linewidth=0.5, linestyle='--')

        # TMA1 mean +/- std
        if tma1_spectra:
            m1 = np.mean(tma1_spectra, axis=0)
            std1 = np.std(tma1_spectra, axis=0)
            ax.plot(band_centers, m1, color='#0072B2', linewidth=1.0, label='TMA1 (Dylan, 9 cores) Mean')
            ax.fill_between(band_centers, m1 - std1, m1 + std1, color='#0072B2', alpha=0.15)

        # TMA2 mean +/- std
        if tma2_spectra:
            m2 = np.mean(tma2_spectra, axis=0)
            std2 = np.std(tma2_spectra, axis=0)
            ax.plot(band_centers, m2, color='#D55E00', linewidth=1.0, label='TMA2 (Sean, 9 cores) Mean')
            ax.fill_between(band_centers, m2 - std2, m2 + std2, color='#D55E00', alpha=0.15)

        ax.set_xlabel('Recorded detection-band center (nm); all 20 bands')
        ax.set_ylabel('L1 normalized spectral shape (unit sum)')
        ax.legend(loc='upper right', frameon=False, fontsize=5)
        ax.set_title('Cohort L1 Spectral Profiles Across 18 Selected Cores (TMA1 vs TMA2)', fontsize=5)

        fig.text(
            0.05, 0.02,
            f'Generated by {Path(__file__).name}; Arial 5 pt font, 0.5 pt strokes; claim label: Exploratory.',
            fontsize=5,
        )
        fig.savefig(destination, format='pdf')
        plt.close(fig)
    return destination


def prepare_hs_cohort(
        final_manifest: pd.DataFrame,
        channel_map: pd.DataFrame,
        *,
        cache: HSCache,
        object_dir: str | Path,
        table_dir: str | Path,
        plot_dir: str | Path,
        max_working_bytes: int,
        samples_per_core: int = 5000,
        display_stride: int = 8,
        random_state: int = 42,
) -> dict[str, Any]:
    """Source-bound persisted per-core L1 samples and spatial display inputs for all 18 cores.

    One core in memory at a time; full raw SHA verified before and unchanged after.
    Deterministic core-balanced capped sampling. No raw full cubes cached.
    """
    from aicspylibczi import CziFile

    object_dir = Path(object_dir)
    table_dir = Path(table_dir)
    plot_dir = Path(plot_dir)
    object_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if display_stride < 1:
        raise ValueError('display_stride must be an integer >= 1')
    if samples_per_core < 1:
        raise ValueError('samples_per_core must be an integer >= 1')

    final = _validated_spectral_manifest(final_manifest, cache)
    if len(final) != 18 or final['path'].duplicated().any():
        raise ValueError('Spectral cohort requires the complete 18-path final manifest')

    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm',
        'detection_band_center_nm',
    ]

    cos_w, sin_w = _h1_channel_weights(20)
    cos_w = cos_w.astype(np.float32)
    sin_w = sin_w.astype(np.float32)

    l1_by_core: dict[str, np.ndarray] = {}
    gs_by_core: dict[str, np.ndarray] = {}
    display_inputs: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for core_idx, (_, row) in enumerate(final.iterrows()):
        tma = str(row['assigned_tma'])
        core_id = str(row['confirmed_core_id'])
        core_key = f'{tma}_{core_id}'
        source_path = _require_raw_hs(row)
        full_file_sha256, signature = cache.raw_identity(source_path)

        cache_stem = f'core_{core_key}_stride{display_stride}_n{samples_per_core}_seed{random_state}'
        cache_file = object_dir / f'{cache_stem}.npz'

        # Check warm cache for this core
        loaded_from_cache = False
        if not cache.force and cache_file.is_file():
            try:
                npz = np.load(cache_file, allow_pickle=False)
                meta = json.loads(str(npz['metadata_json']))
                if (
                    meta.get('full_file_sha256') == full_file_sha256
                    and meta.get('display_stride') == display_stride
                    and meta.get('samples_per_core') == samples_per_core
                    and meta.get('random_state') == random_state
                ):
                    l1_by_core[core_key] = npz['sampled_l1']
                    gs_by_core[core_key] = npz['sampled_gs']
                    disp_shape = tuple(int(v) for v in npz['disp_shape'])
                    display_inputs[core_key] = {
                        'shape': disp_shape,
                        'positive_mask': npz['disp_pos_mask'],
                        'l1_spectra': npz['disp_l1'],
                        'h1_gs': npz['disp_gs'],
                        'assigned_tma': tma,
                        'confirmed_core_id': core_id,
                        'mosaic_x': meta['mosaic_x'],
                        'mosaic_y': meta['mosaic_y'],
                        'scale_factor': meta['scale_factor'],
                    }
                    summary_rows.append(meta['summary'])
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if loaded_from_cache:
            continue

        channels = channel_map.loc[channel_map['path'].eq(row['path'])].reset_index(drop=True)
        if len(channels) != 20:
            raise ValueError(f'Core {core_key} requires complete audited 20-band channel map')

        reader = CziFile(str(source_path))
        metadata, channel_rows = inspect_raw_metadata(row, reader=reader)
        dimensions = _spectral_layout(metadata)
        current = _spectral_channel_grid(channel_rows)

        if (
            channels['metadata_xml_sha256'].nunique() != 1
            or channels['metadata_xml_sha256'].iloc[0] != metadata['metadata_xml_sha256']
            or channels['dimensions_json'].nunique() != 1
            or channels['dimensions_json'].iloc[0] != metadata['dimensions_json']
            or not current[columns].equals(channels[columns])
        ):
            raise ValueError(f'Core {core_key} metadata differs from audited channel map')

        box = reader.get_mosaic_bounding_box()
        region = tuple(int(v) for v in (box.x, box.y, box.w, box.h))
        x, y, width, height = region
        if width <= 0 or height <= 0:
            raise ValueError(f'Core {core_key} bounding box must have positive extent')
        pixels = width * height
        bound = _trial_memory_bound(pixels * 40, pixels, 65536, max_working_bytes)

        selectors = {
            axis: start for axis, (start, stop) in dimensions.items()
            if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}
        }
        cube = np.empty((20, height, width), dtype=np.uint16)
        for channel in range(20):
            plane = reader.read_mosaic(
                C=channel, scale_factor=1.0, region=region,
                background_color=(0., 0., 0.), **selectors,
            )
            if (
                plane.dtype != np.uint16 or plane.size != pixels
                or plane.shape[-2:] != (height, width)
            ):
                raise ValueError(f'Mosaic channel {channel} shape/dtype mismatch for {core_key}')
            cube[channel] = plane.reshape(height, width)
            del plane
        del reader

        cache.require_unchanged(source_path, signature)

        total_signal = cube.sum(axis=0)
        pos_mask = total_signal > 0
        pos_count = int(np.count_nonzero(pos_mask))

        # Core-balanced sampling
        if pos_count == 0:
            sampled_l1 = np.empty((0, 20), dtype=np.float32)
            sampled_gs = np.empty((0, 2), dtype=np.float32)
        else:
            pos_y, pos_x = np.where(pos_mask)
            rng = np.random.RandomState(random_state + core_idx)
            n_take = min(pos_count, samples_per_core)
            if pos_count <= samples_per_core:
                take_idx = np.arange(pos_count)
            else:
                take_idx = rng.choice(pos_count, size=n_take, replace=False)
                take_idx.sort()
            raw_samples = cube[:, pos_y[take_idx], pos_x[take_idx]].T.astype(np.float64)
            sum_samples = raw_samples.sum(axis=1, keepdims=True)
            sampled_l1 = (raw_samples / np.maximum(sum_samples, 1e-12)).astype(np.float32)
            sampled_g = sampled_l1 @ cos_w
            sampled_s = sampled_l1 @ sin_w
            sampled_gs = np.column_stack([sampled_g, sampled_s])

        # Spatial display inputs at display_stride
        disp_cube = cube[:, ::display_stride, ::display_stride]
        disp_h, disp_w = disp_cube.shape[1], disp_cube.shape[2]
        disp_total = disp_cube.sum(axis=0)
        disp_pos_mask = disp_total > 0
        disp_pos_count = int(np.count_nonzero(disp_pos_mask))
        if disp_pos_count == 0:
            disp_l1 = np.empty((0, 20), dtype=np.float32)
            disp_gs = np.empty((0, 2), dtype=np.float32)
        else:
            disp_raw = disp_cube[:, disp_pos_mask].T.astype(np.float64)
            disp_sum = disp_raw.sum(axis=1, keepdims=True)
            disp_l1 = (disp_raw / np.maximum(disp_sum, 1e-12)).astype(np.float32)
            disp_g = disp_l1 @ cos_w
            disp_s = disp_l1 @ sin_w
            disp_gs = np.column_stack([disp_g, disp_s])

        del cube, disp_cube, total_signal

        tile_count = int(dimensions['M'][1] - dimensions['M'][0]) if 'M' in dimensions else 1
        core_summary = {
            'assigned_tma': tma,
            'confirmed_core_id': core_id,
            'analysis_core_id': core_key,
            'path': row['path'],
            'full_file_sha256': full_file_sha256,
            'positive_pixel_count': pos_count,
            'sampled_pixel_count': len(sampled_l1),
            'display_height': disp_h,
            'display_width': disp_w,
            'display_positive_pixels': disp_pos_count,
            'native_tile_count': tile_count,
            'display_stride': display_stride,
            'samples_per_core': samples_per_core,
            'working_allocation_bound_bytes': bound,
            'claim_label': 'Exploratory',
        }

        meta_to_save = {
            'full_file_sha256': full_file_sha256,
            'display_stride': display_stride,
            'samples_per_core': samples_per_core,
            'random_state': random_state,
            'mosaic_x': x,
            'mosaic_y': y,
            'scale_factor': 1.0,
            'summary': core_summary,
        }
        np.savez_compressed(
            cache_file,
            sampled_l1=sampled_l1,
            sampled_gs=sampled_gs,
            disp_shape=np.array([disp_h, disp_w], dtype=np.int32),
            disp_pos_mask=disp_pos_mask,
            disp_l1=disp_l1,
            disp_gs=disp_gs,
            metadata_json=json.dumps(meta_to_save),
        )

        l1_by_core[core_key] = sampled_l1
        gs_by_core[core_key] = sampled_gs
        display_inputs[core_key] = {
            'shape': (disp_h, disp_w),
            'positive_mask': disp_pos_mask,
            'l1_spectra': disp_l1,
            'h1_gs': disp_gs,
            'assigned_tma': tma,
            'confirmed_core_id': core_id,
            'mosaic_x': x,
            'mosaic_y': y,
            'scale_factor': 1.0,
        }
        summary_rows.append(core_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = table_dir / 'hs_cohort_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)

    band_centers = (
        channel_map.groupby('channel_index', as_index=False)['detection_band_center_nm']
        .first()
        .sort_values('channel_index')['detection_band_center_nm']
        .to_numpy(dtype=float)
    )

    spectrum_pdf_path = plot_dir / 'hs_cohort_l1_spectra.pdf'
    _render_cohort_spectrum_pdf(l1_by_core, band_centers, spectrum_pdf_path)

    valid_l1 = [arr for arr in l1_by_core.values() if len(arr) > 0]
    valid_gs = [arr for arr in gs_by_core.values() if len(arr) > 0]
    pooled_l1 = np.vstack(valid_l1) if valid_l1 else np.empty((0, 20), dtype=np.float32)
    pooled_gs = np.vstack(valid_gs) if valid_gs else np.empty((0, 2), dtype=np.float32)

    return {
        'summary': summary_df,
        'spectrum_pdf': spectrum_pdf_path,
        'cohort_samples': {
            'l1_by_core': l1_by_core,
            'gs_by_core': gs_by_core,
            'pooled_l1': pooled_l1,
            'pooled_gs': pooled_gs,
            'core_keys': list(l1_by_core.keys()),
        },
        'display_inputs': display_inputs,
        'parameters': {
            'samples_per_core': samples_per_core,
            'display_stride': display_stride,
            'random_state': random_state,
            'max_working_bytes': max_working_bytes,
        },
        'final_manifest': final.copy(),
        'channel_map': channel_map.copy(),
        'wavelengths': band_centers,
        'cache': cache,
    }


def fit_hs_cohort_pca(
        cohort: dict[str, Any],
        *,
        table_dir: str | Path,
        plot_dir: str | Path,
) -> dict[str, Any]:
    """Fit ONE pooled core-balanced shared PCA on L1 spectra across all 18 cores."""
    from sklearn.decomposition import PCA
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    table_dir = Path(table_dir)
    plot_dir = Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pooled_l1 = cohort['cohort_samples']['pooled_l1']
    if len(pooled_l1) == 0:
        raise ValueError('No positive pixel spectra available in cohort to fit PCA')

    random_state = cohort.get('parameters', {}).get('random_state', 42)
    pca = PCA(n_components=20, random_state=random_state)
    pca.fit(pooled_l1)

    components = pca.components_.copy()
    for i in range(len(components)):
        if components[i, np.argmax(np.abs(components[i]))] < 0:
            components[i] *= -1
    pca.components_ = components

    variance_df = pd.DataFrame({
        'component': np.arange(1, 21),
        'eigenvalue': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
    })
    variance_df.to_csv(table_dir / 'hs_cohort_pooled_pca_variance.csv', index=False)

    wavelengths = cohort.get('wavelengths', np.arange(20))
    loadings_dict: dict[str, Any] = {
        'channel_index': np.arange(20),
        'wavelength_nm': wavelengths,
    }
    for i in range(20):
        loadings_dict[f'PC{i+1}'] = components[i]
    loadings_df = pd.DataFrame(loadings_dict)
    loadings_df.to_csv(table_dir / 'hs_cohort_pooled_pca_loadings.csv', index=False)

    pdf_path = plot_dir / 'hs_cohort_pooled_pca_diagnostics.pdf'
    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()):
        fig, (ax_scree, ax_loadings) = plt.subplots(1, 2, figsize=(180 / 25.4, 90 / 25.4))

        comps = np.arange(1, 21)
        ax_scree.bar(comps, pca.explained_variance_ratio_, color='#0072B2', width=0.6, label='Individual')
        ax_scree.plot(
            comps, np.cumsum(pca.explained_variance_ratio_),
            color='#D55E00', marker='o', markersize=2, linewidth=0.5, label='Cumulative',
        )
        ax_scree.set_xlabel('Principal Component')
        ax_scree.set_ylabel('Explained Variance Ratio')
        ax_scree.set_xticks(np.arange(1, 21, 2))
        ax_scree.set_ylim(0, 1.05)
        ax_scree.legend(loc='center right', frameon=False, fontsize=5)
        ax_scree.set_title('Scree Plot: Pooled Core-Balanced L1 PCA', fontsize=5)

        colors = ['#0072B2', '#D55E00', '#009E73']
        for i in range(3):
            ax_loadings.plot(
                wavelengths, components[i], color=colors[i], linewidth=0.8,
                label=f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)',
            )
        ax_loadings.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        ax_loadings.set_xlabel('Recorded detection-band center (nm)')
        ax_loadings.set_ylabel('Component Loading')
        ax_loadings.legend(loc='upper right', frameon=False, fontsize=5)
        ax_loadings.set_title('Top 3 Spectral Loadings', fontsize=5)

        fig.text(
            0.05, 0.02,
            f'Generated by {Path(__file__).name}; Arial 5 pt font, 0.5 pt strokes; claim label: Exploratory.',
            fontsize=5,
        )
        fig.savefig(pdf_path, format='pdf')
        plt.close(fig)

    pooled_scores = pca.transform(pooled_l1)

    return {
        'summary': variance_df,
        'pdf': pdf_path,
        'pca': pca,
        'loadings': loadings_df,
        'variance': variance_df,
        'components': components,
        'mean': pca.mean_,
        'pooled_scores': pooled_scores,
    }


def _render_cohort_cluster_gallery(
        label_maps: dict[str, np.ndarray],
        joined_manifest: pd.DataFrame,
        *,
        method: str,
        k: int,
        display_stride: int,
        destination: Path,
) -> Path:
    """Render 18-core label maps (TMA1 3x3 + TMA2 3x3 side-by-side) with shared palette."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()):
        fig, axes = plt.subplots(
            3, 8,
            figsize=(340 / 25.4, 180 / 25.4),
            gridspec_kw={'width_ratios': [1, 1, 1, 0.25, 1, 1, 1, 0.85]},
        )

        palette = plt.get_cmap('tab20').colors[:k]
        cmap = ListedColormap([(0., 0., 0.), *palette])
        norm = BoundaryNorm(np.arange(k + 2) - 0.5, k + 1)

        row_letters = ['A', 'B', 'C']
        col_numbers = ['1', '2', '3']

        for r in range(3):
            axes[r, 3].set_axis_off()
            axes[r, 7].set_axis_off()

        tmas = [('TMA1', 0), ('TMA2', 4)]
        for tma, start_col in tmas:
            for r_idx, r_letter in enumerate(row_letters):
                for c_idx, c_num in enumerate(col_numbers):
                    core_id = f'{r_letter}{c_num}'
                    core_key = f'{tma}_{core_id}'
                    ax = axes[r_idx, start_col + c_idx]
                    ax.set_facecolor('black')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color('#555555')
                        spine.set_linewidth(0.5)

                    match = joined_manifest.loc[
                        joined_manifest['assigned_tma'].eq(tma)
                        & joined_manifest['confirmed_core_id'].eq(core_id)
                    ]
                    group = match['progression_group'].iloc[0] if len(match) > 0 else 'unknown'
                    group_abbrev = 'P' if group == 'progressor' else ('NP' if group == 'non_progressor' else group)

                    if core_key in label_maps:
                        lmap = label_maps[core_key]
                        ax.imshow(lmap, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
                        ax.set_title(f'{core_id} ({group_abbrev})', fontsize=5, color='black', pad=2)
                    else:
                        ax.text(0.5, 0.5, 'NO DATA', color='white', ha='center', va='center', fontsize=5)
                        ax.set_title(f'{core_id} ({group_abbrev})', fontsize=5, color='black', pad=2)

        fig.text(0.24, 0.96, 'TMA1 (Dylan) — 3x3 Grid', ha='center', va='top', fontsize=6, weight='bold')
        fig.text(0.66, 0.96, 'TMA2 (Sean) — 3x3 Grid', ha='center', va='top', fontsize=6, weight='bold')

        legend_ax = axes[0, 7]
        legend_ax.set_axis_off()
        legend_patches = [
            Patch(facecolor='black', edgecolor='#555555', linewidth=0.5, label='0: Zero / Background'),
        ]
        for c_idx in range(k):
            legend_patches.append(
                Patch(facecolor=palette[c_idx], edgecolor='none', label=f'Cluster {c_idx + 1}')
            )
        legend_ax.legend(
            handles=legend_patches,
            loc='upper left',
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=4.5,
            title=f'{method}\nK = {k}',
            title_fontsize=5,
        )

        footer_text = (
            f'Method: {method} | Clusters: K={k} | Stride: {display_stride}x nearest\n'
            'Shared model fit on core-balanced capped samples across ALL 18 cores; identical colors represent same cluster ID.\n'
            'Outcome labels: P = progressor, NP = non-progressor. Analysis unit: core (n=18); IDs are exploratory candidate domains, not histology.\n'
            f'Generated by {Path(__file__).name} | Arial 5 pt font, 0.5 pt strokes | claim label: Exploratory.'
        )
        fig.text(0.04, 0.02, footer_text, ha='left', va='bottom', fontsize=4.5)

        fig.savefig(destination, format='pdf')
        plt.close(fig)
    return destination


def _compute_within_tma_permutation_ranking(
        composition_df: pd.DataFrame,
        methods: list[str],
        cluster_counts: tuple[int, ...],
        random_state: int = 42,
        n_permutations: int = 1000,
) -> pd.DataFrame:
    """Descriptive outcome comparison and maxT within-TMA permutation ranking across candidate models."""
    core_meta = (
        composition_df[['assigned_tma', 'confirmed_core_id', 'analysis_core_id', 'progression_group']]
        .drop_duplicates()
        .set_index('analysis_core_id')
    )

    tma1_cores = [c for c in core_meta.index if c.startswith('TMA1')]
    tma2_cores = [c for c in core_meta.index if c.startswith('TMA2')]

    tma1_labels = core_meta.loc[tma1_cores, 'progression_group'].to_numpy()
    tma2_labels = core_meta.loc[tma2_cores, 'progression_group'].to_numpy()

    obs_stats: dict[tuple[str, int], dict[str, Any]] = {}
    props_by_model: dict[tuple[str, int], tuple[np.ndarray, list[Any]]] = {}

    for method in methods:
        for k in cluster_counts:
            sub = composition_df.loc[composition_df['method'].eq(method) & composition_df['k'].eq(k)]
            piv = sub.pivot(index='analysis_core_id', columns='cluster_id', values='cluster_proportion').fillna(0.0)
            piv = piv.reindex(core_meta.index, fill_value=0.0)
            props_by_model[(method, k)] = (piv.to_numpy(dtype=np.float64), list(piv.columns))

    p_mask = core_meta['progression_group'].eq('progressor').to_numpy()
    np_mask = core_meta['progression_group'].eq('non_progressor').to_numpy()

    for (method, k), (mat, clusters) in props_by_model.items():
        mean_p = mat[p_mask].mean(axis=0) if p_mask.any() else np.zeros(k)
        mean_np = mat[np_mask].mean(axis=0) if np_mask.any() else np.zeros(k)
        diffs = np.abs(mean_p - mean_np)
        signed_diffs = mean_p - mean_np
        max_idx = int(np.argmax(diffs))
        obs_stats[(method, k)] = {
            'observed_statistic': float(diffs[max_idx]),
            'max_cluster_abs_diff': float(diffs[max_idx]),
            'mean_cluster_abs_diff': float(diffs.mean()),
            'top_cluster_id': int(clusters[max_idx]),
            'top_cluster_p_mean': float(mean_p[max_idx]),
            'top_cluster_np_mean': float(mean_np[max_idx]),
            'top_cluster_diff': float(signed_diffs[max_idx]),
        }

    rng = np.random.RandomState(random_state)
    max_null_distribution = np.zeros(n_permutations, dtype=np.float64)

    tma1_indices = np.array([core_meta.index.get_loc(c) for c in tma1_cores])
    tma2_indices = np.array([core_meta.index.get_loc(c) for c in tma2_cores])

    for b in range(n_permutations):
        perm_t1 = rng.permutation(tma1_labels)
        perm_t2 = rng.permutation(tma2_labels)

        perm_p_mask = np.zeros(len(core_meta), dtype=bool)
        perm_p_mask[tma1_indices] = (perm_t1 == 'progressor')
        perm_p_mask[tma2_indices] = (perm_t2 == 'progressor')
        perm_np_mask = ~perm_p_mask

        max_stat_b = 0.0
        for (method, k), (mat, _) in props_by_model.items():
            mp = mat[perm_p_mask].mean(axis=0)
            mnp = mat[perm_np_mask].mean(axis=0)
            max_stat_b = max(max_stat_b, float(np.max(np.abs(mp - mnp))))
        max_null_distribution[b] = max_stat_b

    null_p95 = float(np.percentile(max_null_distribution, 95))

    rows = []
    for (method, k), stat_dict in obs_stats.items():
        obs = stat_dict['observed_statistic']
        adj_p = float((np.sum(max_null_distribution >= obs) + 1) / (n_permutations + 1))

        if null_p95 >= obs:
            note = (
                f'Descriptive ranking; permutation maximum null p95 ({null_p95:.4f}) exceeds '
                f'observed ({obs:.4f}): no significant separation after 12-candidate search.'
            )
        else:
            note = f'Observed ({obs:.4f}) exceeds permutation maximum null p95 ({null_p95:.4f}) over 12 candidates.'

        row = {
            'method': method,
            'k': k,
            'observed_statistic': obs,
            'max_cluster_abs_diff': stat_dict['max_cluster_abs_diff'],
            'mean_cluster_abs_diff': stat_dict['mean_cluster_abs_diff'],
            'top_cluster_id': stat_dict['top_cluster_id'],
            'top_cluster_p_mean': stat_dict['top_cluster_p_mean'],
            'top_cluster_np_mean': stat_dict['top_cluster_np_mean'],
            'top_cluster_diff': stat_dict['top_cluster_diff'],
            'null_p95': null_p95,
            'selection_adjusted_P': adj_p,
            'inferential_unit': 'core (n=18)',
            'permutation_type': 'exact within-TMA permutation maximum across 12 candidates',
            'claim_label': 'Exploratory',
            'selection_note': note,
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows).sort_values('observed_statistic', ascending=False).reset_index(drop=True)
    comp_df.insert(0, 'rank', np.arange(1, len(comp_df) + 1))
    return comp_df


def compare_hs_cohort_clusters(
        cohort: dict[str, Any],
        pooled_pca: dict[str, Any],
        *,
        table_dir: str | Path,
        plot_dir: str | Path,
        cluster_counts: tuple[int, ...] = (9, 12, 15, 18),
        random_state: int = 42,
        confirmation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit all 3 methods across K=(9, 12, 15, 18) on pooled samples, render galleries, rank with maxT permutations."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    table_dir = Path(table_dir)
    plot_dir = Path(plot_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = table_dir / 'hs_cohort_cluster_comparison_receipt.json'
    comp_csv_path = table_dir / 'hs_cohort_cluster_comparison.csv'
    pos_csv_path = table_dir / 'hs_cohort_cluster_composition.csv'

    methods = [
        'MiniBatchKMeans_L1_spectra',
        'GaussianMixture_L1_PCA_scores',
        'GaussianMixture_H1_GS',
    ]

    expected_gallery_pdfs = [
        plot_dir / f'hs_cohort_gallery_{method}_k{k}.pdf'
        for method in methods
        for k in cluster_counts
    ]

    # Warm cache guard with comprehensive parameter validation
    force = cohort.get('cache', None) is not None and getattr(cohort['cache'], 'force', False)
    if not force and receipt_path.is_file() and comp_csv_path.is_file() and pos_csv_path.is_file():
        if all(p.is_file() and p.stat().st_size > 0 for p in expected_gallery_pdfs):
            try:
                receipt_data = json.loads(receipt_path.read_text(encoding='utf-8'))
                cached_params = receipt_data.get('parameters', {})
                if (
                    cached_params.get('cluster_counts') == list(cluster_counts)
                    and cached_params.get('random_state') == random_state
                    and cached_params.get('methods') == methods
                    and cached_params.get('samples_per_core') == cohort.get('parameters', {}).get('samples_per_core', 5000)
                    and cached_params.get('display_stride') == cohort.get('parameters', {}).get('display_stride', 8)
                ):
                    comparison_df = pd.read_csv(comp_csv_path)
                    composition_df = pd.read_csv(pos_csv_path)
                    return {
                        'comparison': comparison_df,
                        'composition': composition_df,
                        'gallery_pdfs': expected_gallery_pdfs,
                        'receipt_path': receipt_path,
                    }
            except Exception:
                pass
    if confirmation_path is None:
        confirmation_path = cohort.get('confirmation_path') or (
            Path(__file__).resolve().parents[3]
            / '02_data_sources/01_self/annotations/tma_patient_independence_confirmation.json'
        )
    joined_manifest, bindings = _join_hyperspectral_progression(cohort['final_manifest'], confirmation_path)

    pooled_l1 = cohort['cohort_samples']['pooled_l1']
    pooled_gs = cohort['cohort_samples']['pooled_gs']
    display_inputs = cohort['display_inputs']
    display_stride = cohort.get('parameters', {}).get('display_stride', 8)

    pca_mean = pooled_pca['mean']
    pca_comp = pooled_pca['components'][:3]
    pooled_pca_3 = (pooled_l1 - pca_mean) @ pca_comp.T

    composition_rows: list[dict[str, Any]] = []
    gallery_pdfs: list[Path] = []
    models_fitted: dict[tuple[str, int], Any] = {}
    label_maps_all: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    with threadpool_limits(limits=1):
        for method in methods:
            for k in cluster_counts:
                # 1. Fit ONE shared model across ALL pooled samples
                if method == 'MiniBatchKMeans_L1_spectra':
                    model = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=random_state,
                        batch_size=2048,
                        n_init='auto' if hasattr(MiniBatchKMeans(), 'n_init') else 3,
                    )
                    model.fit(pooled_l1)
                elif method == 'GaussianMixture_L1_PCA_scores':
                    model = GaussianMixture(
                        n_components=k,
                        random_state=random_state,
                        covariance_type='full',
                        max_iter=200,
                        n_init=1,
                    )
                    model.fit(pooled_pca_3)
                elif method == 'GaussianMixture_H1_GS':
                    model = GaussianMixture(
                        n_components=k,
                        random_state=random_state,
                        covariance_type='full',
                        max_iter=200,
                        n_init=1,
                    )
                    model.fit(pooled_gs)
                else:
                    raise ValueError(f'Unknown method: {method}')

                models_fitted[(method, k)] = model

                # 2. Predict on each core's spatial display grid
                core_label_maps: dict[str, np.ndarray] = {}
                for core_key, disp in display_inputs.items():
                    shape = disp['shape']
                    pos_mask = disp['positive_mask']
                    lmap = np.zeros(shape, dtype=np.int32)

                    if pos_mask.any():
                        if method == 'MiniBatchKMeans_L1_spectra':
                            preds = model.predict(disp['l1_spectra'])
                        elif method == 'GaussianMixture_L1_PCA_scores':
                            core_pca = (disp['l1_spectra'] - pca_mean) @ pca_comp.T
                            preds = model.predict(core_pca)
                        elif method == 'GaussianMixture_H1_GS':
                            preds = model.predict(disp['h1_gs'])
                        lmap[pos_mask] = preds + 1

                    core_label_maps[core_key] = lmap

                    # Compute cluster proportions for this core
                    tot_pos = int(np.count_nonzero(pos_mask))
                    tma, core_id = core_key.split('_')
                    match = joined_manifest.loc[
                        joined_manifest['assigned_tma'].eq(tma)
                        & joined_manifest['confirmed_core_id'].eq(core_id)
                    ]
                    group = match['progression_group'].iloc[0] if len(match) > 0 else 'unknown'

                    for c in range(1, k + 1):
                        c_count = int(np.count_nonzero(lmap == c))
                        c_prop = c_count / max(tot_pos, 1)
                        composition_rows.append({
                            'method': method,
                            'k': k,
                            'cluster_id': c,
                            'assigned_tma': tma,
                            'confirmed_core_id': core_id,
                            'analysis_core_id': core_key,
                            'progression_group': group,
                            'pixel_count': c_count,
                            'total_core_positive_pixels': tot_pos,
                            'cluster_proportion': c_prop,
                        })

                label_maps_all[(method, k)] = core_label_maps

                # 3. Render bright 18-core gallery
                gal_pdf = plot_dir / f'hs_cohort_gallery_{method}_k{k}.pdf'
                _render_cohort_cluster_gallery(
                    core_label_maps,
                    joined_manifest,
                    method=method,
                    k=k,
                    display_stride=display_stride,
                    destination=gal_pdf,
                )
                gallery_pdfs.append(gal_pdf)

    composition_df = pd.DataFrame(composition_rows)
    composition_df.to_csv(pos_csv_path, index=False)

    # 4. Descriptive outcome comparison & within-TMA maxT permutation ranking
    comparison_df = _compute_within_tma_permutation_ranking(
        composition_df, methods=methods, cluster_counts=cluster_counts, random_state=random_state, n_permutations=1000,
    )
    comparison_df.to_csv(comp_csv_path, index=False)

    # 5. Persist receipt
    artifacts_list: list[dict[str, Any]] = [
        {
            'path': str(comp_csv_path),
            'sha256': _full_file_hash(comp_csv_path),
            'size_bytes': comp_csv_path.stat().st_size,
        },
        {
            'path': str(pos_csv_path),
            'sha256': _full_file_hash(pos_csv_path),
            'size_bytes': pos_csv_path.stat().st_size,
        },
    ]
    for gpdf in gallery_pdfs:
        artifacts_list.append({
            'path': str(gpdf),
            'sha256': _full_file_hash(gpdf),
            'size_bytes': gpdf.stat().st_size,
        })

    receipt = {
        'schema_version': 'hs_cohort_cluster_comparison_v1',
        'claim_label': 'Exploratory',
        'generating_script': str(Path(__file__).resolve()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'methods': methods,
            'cluster_counts': list(cluster_counts),
            'random_state': random_state,
            'samples_per_core': cohort.get('parameters', {}).get('samples_per_core', 5000),
            'display_stride': display_stride,
            'n_permutations': 1000,
        },
        'source_bindings': bindings,
        'code_sha256': {
            'clinical_roi_hyperspectral.py': _full_file_hash(Path(__file__).resolve()),
        },
        'artifacts': artifacts_list,
        'interpretation': (
            'Descriptive comparison of 3 spectral clustering methods across K=(9, 12, 15, 18). '
            'Shared models fit on core-balanced pooled samples across all 18 cores; identical colors represent same cluster ID. '
            'Outcome inference uses within-TMA permutation maximum over 12 candidate models; inferential unit is core (n=18). '
            'No biological compartment identities or meaningful P values treating pixels as n.'
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')

    return {
        'comparison': comparison_df,
        'composition': composition_df,
        'gallery_pdfs': gallery_pdfs,
        'receipt_path': receipt_path,
        'models': models_fitted,
        'label_maps': label_maps_all,
    }


def _hs_array_sha256(array: np.ndarray) -> str:
    """Bind numeric review inputs including shape and dtype, without pickling."""
    contiguous = np.ascontiguousarray(array)
    digest = sha256(_json_bytes({'shape': contiguous.shape, 'dtype': contiguous.dtype.str}))
    digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def _hs_channel_map_sha256(channel_map: pd.DataFrame) -> str:
    """Bind scientific metadata, excluding cache transport and column presentation."""
    stable = channel_map.drop(columns=['cache_status', 'cache_key'], errors='ignore')
    stable = stable.sort_index(axis=1).reset_index(drop=True)
    payload = {
        'dtypes': {column: str(dtype) for column, dtype in stable.dtypes.items()},
        'values': stable.to_json(orient='split', double_precision=15),
    }
    return sha256(_json_bytes(payload)).hexdigest()


def _hs_intensity_texture(
        intensity: np.ndarray, labels: np.ndarray, *, cluster_count: int,
        percentiles: tuple[float, float] = (1., 96.), exponent: float = .7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Dylan's grayscale-over-color display rule; never alter cluster labels."""
    import matplotlib as mpl

    if intensity.shape != labels.shape or intensity.ndim != 2:
        raise ValueError('Intensity and labels must share one YX grid')
    if (not np.isfinite(intensity).all() or (intensity < 0).any()
            or not np.isfinite(labels).all() or labels.dtype.kind not in 'iu'
            or labels.min() < 0 or labels.max() > cluster_count):
        raise ValueError('Nonnegative finite intensity and valid integer labels required')
    if not 0 <= percentiles[0] < percentiles[1] <= 100 or exponent <= 0:
        raise ValueError('Ordered display percentiles and positive exponent required')
    low, high = np.percentile(intensity, percentiles)
    clipped = np.clip(intensity.astype(np.float64), low, high)
    gray = np.zeros_like(clipped)
    alpha = np.ones_like(clipped)
    status = 'CONSTANT_DISPLAY_RANGE'
    if high == low and high > 0:
        # A constant positive image has no contrast to supply: retain its labels.
        gray.fill(1.)
        alpha.fill(0.)
    if high > low:
        gray = (clipped - low) / (high - low)
        powered = clipped ** exponent
        if powered.max() > 0:
            alpha = 1. - powered / powered.max()
            if alpha.max() > 0:
                alpha /= alpha.max()
        status = 'PERCENTILE_DISPLAY_RANGE'
    palette = np.vstack((np.zeros((1, 3)), np.asarray(mpl.colormaps['tab20'].colors[:cluster_count])))
    # Greys_r reproduces the source's grayscale LUT, including its nonlinear ramp.
    gray_rgb = mpl.colormaps['Greys_r'](gray)[..., :3]
    rgb = (1. - alpha[..., None]) * palette[labels] + alpha[..., None] * gray_rgb
    rgb[labels == 0] = 0.
    return gray.astype(np.float32), alpha.astype(np.float32), rgb.astype(np.float32), {
        'display_p_low': float(low), 'display_p_high': float(high),
        'display_range_status': status,
    }


def _hs_read_review_intensity(
        row: dict[str, Any], channels: pd.DataFrame, display: dict[str, Any], *,
        cache: HSCache, display_stride: int, max_working_bytes: int,
) -> np.ndarray:
    """Read matching native mosaic pixels one channel at a time and check L1 equality."""
    from aicspylibczi import CziFile

    source = _require_raw_hs(row)
    _, signature = cache.raw_identity(source)
    reader = CziFile(str(source))
    metadata, channel_rows = inspect_raw_metadata(row, reader=reader)
    dimensions = _spectral_layout(metadata)
    current = _spectral_channel_grid(channel_rows)
    columns = [
        'channel_index', 'channel_id', 'channel_xml_fields_json',
        'detection_band_start_nm', 'detection_band_end_nm', 'detection_band_center_nm',
    ]
    if (len(channels) != 20 or channels['metadata_xml_sha256'].nunique() != 1
            or channels['metadata_xml_sha256'].iloc[0] != metadata['metadata_xml_sha256']
            or channels['dimensions_json'].nunique() != 1
            or channels['dimensions_json'].iloc[0] != metadata['dimensions_json']
            or not current[columns].equals(channels[columns])):
        raise ValueError('Review source differs from the audited channel map')
    box = reader.get_mosaic_bounding_box()
    region = tuple(int(v) for v in (box.x, box.y, box.w, box.h))
    x, y, width, height = region
    shape = ((height + display_stride - 1) // display_stride,
             (width + display_stride - 1) // display_stride)
    if (width <= 0 or height <= 0 or tuple(display['shape']) != shape
            or display['mosaic_x'] != x or display['mosaic_y'] != y
            or display['scale_factor'] != 1.):
        raise ValueError('Review source and cohort preview mosaic geometry differ')
    # Native channel plus libCZI allowance; only the small strided cube is retained.
    memory_bound = width * height * 8 + prod(shape) * 20 * 16 + 64 * 1024**2
    if not isinstance(max_working_bytes, int) or memory_bound > max_working_bytes:
        raise MemoryError(f'Intensity review bound {memory_bound} exceeds {max_working_bytes}')
    selectors = {
        axis: start for axis, (start, stop) in dimensions.items()
        if axis not in {'C', 'M', 'Y', 'X', 'A', 'S'}
    }
    sampled = np.empty((20, *shape), dtype=np.uint16)
    for channel in range(20):
        plane = reader.read_mosaic(
            C=channel, scale_factor=1., region=region,
            background_color=(0., 0., 0.), **selectors,
        )
        if (plane.dtype != np.uint16 or plane.size != width * height
                or plane.shape[-2:] != (height, width)):
            raise ValueError('Review native channel shape or dtype differs')
        sampled[channel] = plane.reshape(height, width)[::display_stride, ::display_stride]
        del plane
    intensity = sampled.sum(axis=0, dtype=np.uint64)
    positive = intensity > 0
    if not np.array_equal(positive, display['positive_mask']):
        raise ValueError('Review intensity and cohort positive-pixel masks differ')
    raw = sampled[:, positive].T.astype(np.float64)
    l1 = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    if not np.array_equal(l1, display['l1_spectra']):
        raise ValueError('Freshly decoded raw spectra do not reproduce cohort display L1 spectra')
    cache.require_unchanged(source, signature)
    return intensity


def _hs_render_intensity_review_pdf(
        object_paths: list[Path], *, destination: Path, cluster_count: int,
        display_stride: int, fit_core_count: int,
) -> None:
    """Three cores per page, identical YX grid across raw, flat, and textured views."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    _register_project_arial()
    palette = np.vstack((np.zeros((1, 3)), np.asarray(mpl.colormaps['tab20'].colors[:cluster_count])))
    with mpl.rc_context(_pseudocolor_style()), PdfPages(destination) as pdf:
        for start in range(0, len(object_paths), 3):
            fig, axes = plt.subplots(3, 3, figsize=(180 / 25.4, 210 / 25.4))
            fig.subplots_adjust(left=.045, right=.985, top=.94, bottom=.14, hspace=.10, wspace=.04)
            for column, heading in enumerate(('Summed HS intensity', 'Flat spectral labels', 'Same labels + intensity texture')):
                bounds = axes[0, column].get_position()
                fig.text((bounds.x0 + bounds.x1) / 2, .965, heading, ha='center', fontsize=5)
            for row_index in range(3):
                index = start + row_index
                if index >= len(object_paths):
                    for ax in axes[row_index]:
                        ax.set_axis_off()
                    continue
                with np.load(object_paths[index], allow_pickle=False) as saved:
                    gray, labels, rgb = saved['gray'], saved['labels'], saved['rgb']
                    core_key = str(saved['core_key'])
                bounds = axes[row_index, 0].get_position()
                fig.text(bounds.x0, bounds.y1 + .006, core_key, fontsize=5)
                for column, view in enumerate((mpl.colormaps['Greys_r'](gray)[..., :3], palette[labels], rgb)):
                    ax = axes[row_index, column]
                    ax.imshow(view, interpolation='none', origin='upper', aspect='equal')
                    ax.set(xticks=[], yticks=[])
                    ax.grid(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(.5)
                        spine.set_edgecolor('#444444')
            handles = [Patch(facecolor=palette[i], label=str(i)) for i in range(1, cluster_count + 1)]
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .10),
                       ncol=min(cluster_count, 10), frameon=False, fontsize=5,
                       title='Model-local cluster ID (zero/undefined is black)', title_fontsize=5)
            footer = (
                f'HS only | shared L1 MiniBatchKMeans K={cluster_count}; fit scope: {fit_core_count} cores | display stride: {display_stride}\n'
                'Per-core p1-p96 contrast; grayscale alpha from intensity exponent 0.7. Labels unchanged; no tissue mask or smoothing.\n'
                'Review grid, not native-resolution segmentation; colors are exploratory spectral IDs, not histology or FLIM classes.\n'
                f'Generated by {Path(__file__).name}; native YX orientation; no outcome-based fitting or selection.'
            )
            fig.text(.045, .025, footer, fontsize=5, va='bottom')
            pdf.savefig(fig)
            plt.close(fig)


def _hs_render_textured_gallery(
        object_paths: list[Path], *, destination: Path,
        cluster_count: int, display_stride: int,
) -> None:
    """Compact TMA-separated texture gallery; consumes the identical saved review maps."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    lookup = {}
    for path in object_paths:
        with np.load(path, allow_pickle=False) as saved:
            lookup[str(saved['core_key'])] = path
    _register_project_arial()
    with mpl.rc_context(_pseudocolor_style()), PdfPages(destination) as pdf:
        for tma in sorted({key.split('_')[0] for key in lookup}):
            fig, axes = plt.subplots(3, 3, figsize=(180 / 25.4, 200 / 25.4))
            fig.subplots_adjust(left=.045, right=.985, top=.94, bottom=.15, hspace=.08, wspace=.04)
            fig.text(.045, .965, f'{tma}: HS spectral labels with intensity texture', fontsize=5)
            for row_index, letter in enumerate('ABC'):
                for column_index, number in enumerate('123'):
                    ax = axes[row_index, column_index]
                    key = f'{tma}_{letter}{number}'
                    if key not in lookup:
                        ax.set_axis_off()
                        continue
                    with np.load(lookup[key], allow_pickle=False) as saved:
                        ax.imshow(saved['rgb'], interpolation='none', origin='upper', aspect='equal')
                    ax.set(xticks=[], yticks=[])
                    ax.grid(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(.5)
                        spine.set_edgecolor('#444444')
                    bounds = ax.get_position()
                    fig.text(bounds.x0, bounds.y1 + .004, key, fontsize=5)
            palette = mpl.colormaps['tab20'].colors[:cluster_count]
            handles = [Patch(facecolor=color, label=str(i + 1)) for i, color in enumerate(palette)]
            fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .10),
                       ncol=min(cluster_count, 10), frameon=False, fontsize=5,
                       title='Model-local cluster ID', title_fontsize=5)
            footer = (
                f'Same pooled HS-only L1 KMeans labels; K={cluster_count}; stride={display_stride}; native YX.\n'
                'Per-core p1-p96 intensity contrast, exponent 0.7; display only. No mask or label smoothing.\n'
                f'Exploratory spectral IDs, not histology. Generated by {Path(__file__).name}.'
            )
            fig.text(.045, .025, footer, fontsize=5, va='bottom')
            pdf.savefig(fig)
            plt.close(fig)


def render_hs_cluster_intensity_review(
        cohort: dict[str, Any], *, object_dir: str | Path, table_dir: str | Path,
        plot_dir: str | Path, max_working_bytes: int,
        core_selection: str | tuple[str, str] | list[tuple[str, str]] = 'ALL',
        cluster_count: int = 9, random_state: int = 42,
) -> dict[str, Any]:
    """Compare flat HS labels with source-derived intensity texture, holding labels fixed.

    One unchanged pooled L1 MiniBatchKMeans recipe is refit deterministically.
    The review does not change the original comparison, exclude positive background,
    infer tissue identities, or recreate Dylan's multimodal PCA/FLIM feature model.
    """
    import inspect
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    if not isinstance(cluster_count, int) or not 2 <= cluster_count <= 20:
        raise ValueError('Review cluster count must be an integer between 2 and 20')
    if not isinstance(max_working_bytes, int) or max_working_bytes <= 0:
        raise ValueError('Positive integer working-memory limit required')
    cache = cohort['cache']
    final = _validated_spectral_manifest(cohort['final_manifest'], cache)
    full_keys = [f'{row.assigned_tma}_{row.confirmed_core_id}' for row in final.itertuples()]
    if isinstance(core_selection, str):
        if core_selection != 'ALL':
            raise ValueError('Use ALL or TMA-qualified (TMA, core) pairs')
        selected_keys = full_keys
    else:
        pairs = [core_selection] if isinstance(core_selection, tuple) else core_selection
        selected_keys = [f'{tma}_{core}' for tma, core in pairs]
        if (not selected_keys or len(set(selected_keys)) != len(selected_keys)
                or not set(selected_keys).issubset(full_keys)):
            raise ValueError('Review scope must contain unique selected TMA-qualified cores')
        selected_keys = [key for key in full_keys if key in selected_keys]
    stride = cohort['parameters']['display_stride']
    pooled = cohort['cohort_samples']['pooled_l1']
    if pooled.ndim != 2 or pooled.shape[1] != 20 or len(pooled) < cluster_count:
        raise ValueError('Complete pooled 20-band L1 training samples required')
    if not np.isfinite(pooled).all() or (pooled < 0).any():
        raise ValueError('Finite nonnegative pooled features required')
    for directory in (object_dir, table_dir, plot_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    root = Path(object_dir) / 'hs_cluster_intensity_review'
    root.mkdir(parents=True, exist_ok=True)
    summary_by_key = cohort['summary'].set_index('analysis_core_id')
    source_records, raw_signatures = [], {}
    for key, row in zip(full_keys, final.to_dict('records')):
        source = _require_raw_hs(row)
        digest, signature = cache.raw_identity(source)
        if digest != summary_by_key.loc[key, 'full_file_sha256']:
            raise ValueError(f'{key}: raw source changed since cohort preparation')
        raw_signatures[str(source)] = signature
        source_records.append({'core_key': key, 'path': str(source), 'sha256': digest})
    selected_displays = {key: cohort['display_inputs'][key] for key in selected_keys}
    display_identity = {
        key: {
            'shape': list(display['shape']), 'mosaic_x': display['mosaic_x'],
            'mosaic_y': display['mosaic_y'], 'scale_factor': display['scale_factor'],
            'l1_sha256': _hs_array_sha256(display['l1_spectra']),
            'positive_mask_sha256': _hs_array_sha256(display['positive_mask']),
        } for key, display in selected_displays.items()
    }
    identity = {
        'schema': 'hs_cluster_intensity_review_v1', 'sources': source_records,
        'final_manifest_sha256': _full_file_hash(cache.manifest_path),
        'channel_map_sha256': _hs_channel_map_sha256(cohort['channel_map']),
        'pooled_l1_sha256': _hs_array_sha256(pooled), 'display_inputs': display_identity,
        'selected_keys': selected_keys, 'fit_keys': full_keys,
        'parameters': {'cluster_count': cluster_count, 'random_state': random_state,
                       'display_stride': stride, 'percentiles': [1., 96.], 'exponent': .7,
                       'batch_size': 2048, 'n_init': 'auto', 'transform': 'PIXEL_L1'},
        'code_sha256': {
            function.__name__: sha256(inspect.getsource(function).encode()).hexdigest()
            for function in (
                _hs_array_sha256, _hs_channel_map_sha256, _hs_intensity_texture, _hs_read_review_intensity,
                _hs_render_intensity_review_pdf, _hs_render_textured_gallery,
                render_hs_cluster_intensity_review, inspect_raw_metadata,
                _spectral_layout, _spectral_channel_grid, _require_raw_hs,
            )
        },
        'plot_style_sha256': _plot_style_sha256(), 'versions': cache.package_versions,
    }
    key = sha256(_json_bytes(identity)).hexdigest()
    scope = 'all_cores' if selected_keys == full_keys else f'partial_{key[:12]}'
    stem = f'hs_cluster_intensity_review_{scope}_k{cluster_count}'
    receipt_path = Path(table_dir) / f'{stem}_receipt.json'
    summary_path = Path(table_dir) / f'{stem}_summary.csv'
    pdf_path = Path(plot_dir) / f'{stem}.pdf'
    gallery_path = Path(plot_dir) / f'{stem}_textured_gallery.pdf'
    if not cache.force and receipt_path.is_file():
        try:
            prior = json.loads(receipt_path.read_text())
            valid = prior['identity'] == json.loads(_json_bytes(identity))
            for artifact in prior['artifacts']:
                path = Path(artifact['path'])
                valid = valid and path.is_file() and _full_file_hash(path) == artifact['sha256']
            if valid:
                for source, signature in raw_signatures.items():
                    cache.require_unchanged(source, signature)
                print(f'CACHE_HIT intensity review: {key[:12]}; verified inputs and artifacts')
                return {'pdf': pdf_path, 'gallery_pdf': gallery_path,
                        'summary': pd.read_csv(summary_path),
                        'receipt_path': receipt_path, 'cache_status': 'CACHE_HIT',
                        'object_paths': [Path(p) for p in prior['object_paths']]}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    status = 'FORCE_RECOMPUTE' if cache.force else 'CACHE_MISS_RECOMPUTED'
    print(f'{status} intensity review: {key[:12]}; unchanged pooled L1 KMeans recipe')
    with threadpool_limits(limits=1):
        model = MiniBatchKMeans(n_clusters=cluster_count, random_state=random_state,
                                batch_size=2048, n_init='auto')
        model.fit(pooled)
    model_path = root / f'{key}_model.npz'
    np.savez_compressed(model_path, centers=model.cluster_centers_,
                        parameters_json=json.dumps(identity['parameters']))
    object_paths, rows = [], []
    reader_code = sha256(inspect.getsource(_hs_read_review_intensity).encode()).hexdigest()
    for row, core_key in zip(final.to_dict('records'), full_keys):
        if core_key not in selected_keys:
            continue
        disp = selected_displays[core_key]
        channels = cohort['channel_map'].loc[cohort['channel_map']['path'].eq(row['path'])].reset_index(drop=True)
        intensity_identity = {
            'source': next(r for r in source_records if r['core_key'] == core_key),
            'display': display_identity[core_key], 'stride': stride,
            'channels': _hs_channel_map_sha256(channels),
            'reader_code_sha256': reader_code, 'versions': cache.package_versions,
        }
        intensity_key = sha256(_json_bytes(intensity_identity)).hexdigest()
        intensity_path = root / f'intensity_{intensity_key}.npz'
        intensity = None
        if intensity_path.is_file() and not cache.force:
            try:
                with np.load(intensity_path, allow_pickle=False) as saved:
                    candidate = saved['intensity']
                    if (str(saved['sha256']) == _hs_array_sha256(candidate)
                            and tuple(candidate.shape) == tuple(disp['shape'])):
                        intensity = candidate
            except (OSError, ValueError, KeyError):
                pass
        if intensity is None:
            print(f'CACHE_MISS_RECOMPUTED intensity: {core_key}; reading native channels sequentially')
            intensity = _hs_read_review_intensity(
                row, channels, disp, cache=cache, display_stride=stride,
                max_working_bytes=max_working_bytes,
            )
            np.savez_compressed(intensity_path, intensity=intensity, sha256=_hs_array_sha256(intensity))
        else:
            print(f'CACHE_HIT intensity: {core_key}; verified cached source-bound sum')
        labels = np.zeros(disp['shape'], dtype=np.uint8)
        with threadpool_limits(limits=1):
            if disp['positive_mask'].any():
                labels[disp['positive_mask']] = model.predict(disp['l1_spectra']) + 1
        gray, alpha, rgb, contrast = _hs_intensity_texture(intensity, labels, cluster_count=cluster_count)
        object_path = root / f'{key}_{core_key}.npz'
        np.savez_compressed(object_path, core_key=core_key, intensity=intensity, labels=labels,
                            gray=gray, alpha=alpha, rgb=rgb,
                            identity_sha256=key, axes='YX', display_stride=stride)
        object_paths.append(object_path)
        counts = np.bincount(labels.ravel(), minlength=cluster_count + 1)
        positive_count = int(counts[1:].sum())
        rows.append({
            'analysis_core_id': core_key, 'path': row['path'],
            'full_file_sha256': summary_by_key.loc[core_key, 'full_file_sha256'],
            'display_height': labels.shape[0], 'display_width': labels.shape[1],
            'display_stride': stride, 'positive_display_pixels': positive_count,
            'dominant_cluster_id': int(counts[1:].argmax() + 1),
            'dominant_cluster_fraction_all_positive': float(counts[1:].max() / max(positive_count, 1)),
            'labels_sha256': _hs_array_sha256(labels),
            'intensity_sha256': _hs_array_sha256(intensity),
            'raw_to_cached_l1_check': 'EXACT_ARRAY_EQUALITY',
            'claim_label': 'Exploratory', **contrast,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    _hs_render_intensity_review_pdf(object_paths, destination=pdf_path,
                                   cluster_count=cluster_count, display_stride=stride,
                                   fit_core_count=len(full_keys))
    _hs_render_textured_gallery(object_paths, destination=gallery_path,
                               cluster_count=cluster_count, display_stride=stride)
    for source, signature in raw_signatures.items():
        cache.require_unchanged(source, signature)
    artifacts = [summary_path, pdf_path, gallery_path, model_path, *object_paths]
    receipt = {
        'identity': identity, 'identity_sha256': key, 'claim_label': 'Exploratory',
        'generating_script': str(Path(__file__).resolve()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'object_paths': [str(path) for path in object_paths],
        'artifacts': [{'path': str(path), 'sha256': _full_file_hash(path),
                       'size_bytes': path.stat().st_size} for path in artifacts],
        'interpretation': (
            'Flat and textured views use identical label arrays. Intensity contrast is per-core '
            'and display-only. Positive background remains in model training and assignment. '
            'This reproduces a grayscale-over-color display operation, not Dylan multimodal '
            'PCA/FLIM clusters, their historical IDs, or validated anatomical compartments. '
            'No outcome labels, tests, tissue mask, spatial label smoothing, or native-resolution '
            'prediction are used. TMAs retain differing wavelength grids and acquisition response.'
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')
    return {'pdf': pdf_path, 'gallery_pdf': gallery_path,
            'summary': summary, 'receipt_path': receipt_path,
            'object_paths': object_paths, 'cache_status': status}
