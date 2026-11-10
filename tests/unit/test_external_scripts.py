"""Integrity checks for the vendored FLIM and native hyperspectral source scripts."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import spatial_omics_toolkit
from spatial_omics_toolkit.application.workflows import flim, hyperspectral_native

VENDOR_DIR = Path(spatial_omics_toolkit.__file__).resolve().parent / "external_scripts"
PROVENANCE = VENDOR_DIR / "PROVENANCE.md"
FLIM_ENV = "SPATIAL_TOOLKIT_FLIM_SOURCE_DIR"
HS_ENV = "SPATIAL_TOOLKIT_HS_SOURCE_DIR"

REQUIRED_MODULES: tuple[str, ...] = (
    "clinical_roi_flim.py",
    "clinical_roi_flim_plots.py",
    "clinical_roi_hs_flim.py",
    "clinical_roi_hyperspectral.py",
    "dylan_k9_reconstruction.py",
    "clinical_roi_valis.py",
    "clinical_roi_cycif.py",
    "clinical_roi_tissue.py",
    "clinical_roi_pattern_recovery.py",
    "image_previews.py",
    "guided_serial_registration.py",
    "dylan_c2_image_orientation.py",
    "valis_supplied_transform.py",
    "cycif_zeb1_segment_quantify.py",
    "plotnine_style_helper.py",
)

_ROW = re.compile(
    r"^\| `(?P<name>[^`]+)` \| `(?P<original>[^`]+)` \| (?P<size>\d+) \| "
    r"`(?P<original_sha>[0-9a-f]{64})` \| `(?P<copy_sha>[0-9a-f]{64})` \| (?P<date>\d{4}-\d{2}-\d{2}) \|$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ProvenanceRow:
    name: str
    original: str
    size: int
    original_sha: str
    copy_sha: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_rows() -> list[ProvenanceRow]:
    text = PROVENANCE.read_text(encoding="utf-8")
    return [
        ProvenanceRow(
            name=match["name"],
            original=match["original"],
            size=int(match["size"]),
            original_sha=match["original_sha"],
            copy_sha=match["copy_sha"],
        )
        for match in _ROW.finditer(text)
    ]


def test_default_directories_resolve_to_vendored_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLIM_ENV, raising=False)
    monkeypatch.delenv(HS_ENV, raising=False)
    assert flim._source_dir() == VENDOR_DIR
    assert hyperspectral_native._source_dir() == VENDOR_DIR


def test_default_directory_contains_every_required_module() -> None:
    missing = [name for name in REQUIRED_MODULES if not (VENDOR_DIR / name).is_file()]
    assert missing == []


def test_provenance_lists_exactly_the_vendored_files() -> None:
    recorded = {row.name for row in _provenance_rows()}
    on_disk = {path.name for path in VENDOR_DIR.glob("*.py")}
    assert recorded == set(REQUIRED_MODULES)
    assert on_disk == set(REQUIRED_MODULES)


@pytest.mark.parametrize("row", _provenance_rows(), ids=lambda row: row.name)
def test_copy_hash_and_size_match_provenance(row: ProvenanceRow) -> None:
    copy = VENDOR_DIR / row.name
    assert _sha256(copy) == row.copy_sha
    assert copy.stat().st_size == row.size
    assert row.original_sha == row.copy_sha


@pytest.mark.parametrize(
    ("env_name", "resolver"),
    [(FLIM_ENV, flim._source_dir), (HS_ENV, hyperspectral_native._source_dir)],
)
def test_override_directory_wins(
    env_name: str,
    resolver: Callable[[], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "custom"
    override.mkdir()
    monkeypatch.setenv(env_name, str(override))
    assert resolver() == override.resolve()


def _copy_except(destination: Path, excluded: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_MODULES:
        if name != excluded:
            shutil.copy2(VENDOR_DIR / name, destination / name)


def test_flim_missing_module_error_names_file_and_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_except(tmp_path, "clinical_roi_flim_plots.py")
    monkeypatch.setenv(FLIM_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        flim._sources()
    message = str(excinfo.value)
    assert "clinical_roi_flim_plots.py" in message
    assert FLIM_ENV in message


@pytest.mark.parametrize("excluded", ["clinical_roi_hs_flim.py", "clinical_roi_hyperspectral.py"])
def test_hs_missing_module_error_names_file_and_variable(
    excluded: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_except(tmp_path, excluded)
    monkeypatch.setenv(HS_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        hyperspectral_native._reader()
    message = str(excinfo.value)
    assert excluded in message
    assert HS_ENV in message
