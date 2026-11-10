import csv
import io

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import xml.etree.ElementTree as ET

import pytest

from spatial_omics_toolkit.application.acquisition import (
    METADATA_COLUMNS,
    read_acquisition_csv_bytes,
    discover_native_metadata,
    read_acquisition_metadata,
    read_acquisition_relationships,
    resolve_metadata_value,
    resolve_source_path,
    write_acquisition_metadata,
    write_acquisition_relationships,
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project_samples.csv").write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\nunit-1,specimen-1,unknown,unknown\n",
        encoding="utf-8",
    )
    (project / "image.czi").write_bytes(b"native")
    (project / "other.czi").write_bytes(b"native")
    return project


def _row(acquisition: str = "a", source: str = "image.czi") -> dict[str, str]:
    return {"acquisition_id": acquisition, "modality": "HS", "source_path": source,
            "specimen_id": "specimen-1", "field": "pixel_size", "value": "1",
            "unit": "um", "provenance": "native", "source_detail": "native header", "reviewed": "false"}


def test_optional_files_and_round_trip(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert read_acquisition_metadata(project, tmp_path) == ([], [])
    assert read_acquisition_relationships(project, tmp_path) == ([], [])
    rows = [_row(), _row("b", "other.czi")]
    write_acquisition_metadata(project, tmp_path, rows)
    assert read_acquisition_metadata(project, tmp_path) == (rows, [])
    relationships = [{"acquisition_id": "a", "related_acquisition_id": "b", "relationship": "serial_section", "evidence": "slide log", "reviewed": "true"}]
    write_acquisition_relationships(project, tmp_path, relationships)
    assert read_acquisition_relationships(project, tmp_path) == (relationships, [])


def test_conflicts_require_reviewed_resolution(tmp_path: Path) -> None:
    project = _project(tmp_path)
    native = _row()
    other = {**native, "value": "2", "provenance": "user", "source_detail": "operator log"}
    assert resolve_metadata_value([native, other], "a", "pixel_size") is None
    reviewed = {**other, "reviewed": "true"}
    assert resolve_metadata_value([native, reviewed], "a", "pixel_size") == ("2", "um")
    write_acquisition_metadata(project, tmp_path, [native, reviewed])
    with pytest.raises(ValueError, match="conflicting reviewed"):
        write_acquisition_metadata(project, tmp_path, [{**native, "reviewed": "true"}, reviewed])
    assert read_acquisition_metadata(project, tmp_path)[0] == [native, reviewed]


def test_unknown_metadata_does_not_create_unreviewed_conflict() -> None:
    known = _row()
    unknown = {**known, "value": "unknown", "unit": "", "provenance": "unknown", "source_detail": ""}
    assert resolve_metadata_value([known, unknown], "a", "pixel_size") == ("1", "um")
    assert resolve_metadata_value([unknown], "a", "pixel_size") is None
    assert resolve_metadata_value([known, {**unknown, "reviewed": "true"}], "a", "pixel_size") is None


def test_canonical_specimens_and_relationship_evidence(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError, match="Unknown project specimen_id"):
        write_acquisition_metadata(project, tmp_path, [{**_row(), "specimen_id": "not-in-sheet"}])
    write_acquisition_metadata(project, tmp_path, [_row(), _row("b", "other.czi")])
    relation = {"acquisition_id": "a", "related_acquisition_id": "b", "relationship": "same_section", "evidence": "", "reviewed": "true"}
    with pytest.raises(ValueError, match="evidence"):
        write_acquisition_relationships(project, tmp_path, [relation])
    relation["evidence"] = "notebook entry"
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        write_acquisition_relationships(project, tmp_path, [relation, {**relation, "acquisition_id": "b", "related_acquisition_id": "a", "relationship": "serial_section"}])
    with pytest.raises(ValueError, match="unknown acquisition"):
        write_acquisition_relationships(project, tmp_path, [{**relation, "related_acquisition_id": "missing"}])


def test_rejects_bad_csv_and_unsafe_paths(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "acquisition_metadata.csv").write_text("acquisition_id,acquisition_id\na,b\n", encoding="utf-8")
    assert read_acquisition_metadata(project, tmp_path)[1]
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_text("secret", encoding="utf-8")
    (project / "escape.czi").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        resolve_source_path(project, tmp_path, "escape.czi")
    with pytest.raises(ValueError, match="project-relative"):
        resolve_source_path(project, tmp_path, "../outside")
    (project / "acquisition_metadata.csv").unlink()
    (project / "acquisition_metadata.csv").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe"):
        write_acquisition_metadata(project, tmp_path, [_row()])


def test_discovery_reads_headers_without_pixels_and_missing_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    module = ModuleType("aicspylibczi")
    class FakeCzi:
        def __init__(self, path: str) -> None:
            self.path = path
            self.meta = ET.fromstring('<Root><Information><Image><Dimensions><Channels><Channel Name="DAPI"/><Channel Name="FITC"/></Channels></Dimensions></Image></Information><Scaling><Items><Distance Id="X"><Value>0.000001</Value></Distance></Items></Scaling></Root>')
        def get_dims_shape(self) -> list[dict[str, tuple[int, int]]]:
            return [{"X": (0, 1024), "C": (0, 3)}]
        def is_mosaic(self) -> bool:
            return False
        def read_image(self) -> None:
            raise AssertionError("must not read pixels")
    module.CziFile = FakeCzi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aicspylibczi", module)
    rows = discover_native_metadata(project, tmp_path, "image.czi", "HS", "a")
    assert {row["field"]: row["value"] for row in rows} == {"dimension_X": "1024", "dimension_C": "3", "channel_names": '["DAPI", "FITC"]', "pixel_size_x": "0.000001"}
    assert next(row for row in rows if row["field"] == "pixel_size_x")["unit"] == "m"
    write_acquisition_metadata(project, tmp_path, rows)
    assert all(row["reviewed"] == "false" and row["specimen_id"] == "" for row in rows)
    (project / "unsupported.sdt").write_bytes(b"")
    unknown = discover_native_metadata(project, tmp_path, "unsupported.sdt", "FLIM", "b")
    assert unknown[0]["field"] == "metadata_status" and unknown[0]["value"] == "unknown"


@pytest.mark.parametrize(("bbox", "expected"), [
    (SimpleNamespace(x=-10, y=20, w=4096, h=3072), {"dimension_X": "4096", "dimension_Y": "3072"}),
    (None, {}),
    (SimpleNamespace(x=0, y=0, w=0, h=3072), {}),
    (SimpleNamespace(x=0, y=0, w=4096.5, h=3072), {}),
])
def test_czi_mosaic_dimensions_use_header_bounding_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bbox: object, expected: dict[str, str]
) -> None:
    project = _project(tmp_path)
    module = ModuleType("aicspylibczi")

    class FakeCzi:
        def __init__(self, path: str) -> None:
            self.meta = ET.fromstring("<Root/>")

        def get_dims_shape(self) -> list[dict[str, tuple[int, int]]]:
            return [{"X": (0, 512), "Y": (0, 256), "C": (0, 2)}]

        def is_mosaic(self) -> bool:
            return True

        def get_mosaic_bounding_box(self) -> object:
            if bbox is None:
                raise RuntimeError("bounding box unavailable")
            return bbox

        def read_image(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("must not read pixels")

        def read_mosaic(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("must not read pixels")

    module.CziFile = FakeCzi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aicspylibczi", module)
    rows = discover_native_metadata(project, tmp_path, "image.czi", "HS", "a")
    assert {row["field"]: row["value"] for row in rows} == {"dimension_C": "2", **expected}


@pytest.mark.parametrize("changes", [
    {"value": "NaN"}, {"value": "inf"}, {"value": "-1"}, {"unit": "px"},
    {"field": "unrecognized"}, {"modality": "unrecognized"},
    {"provenance": "unrecognized"}, {"provenance": "user", "source_detail": ""},
    {"field": "width", "value": "1.5", "unit": "px"},
    {"field": "channel_names", "value": "[1]", "unit": ""},
])
def test_rejects_invalid_typed_metadata(tmp_path: Path, changes: dict[str, str]) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        write_acquisition_metadata(project, tmp_path, [{**_row(), **changes}])


def test_selected_sample_sheet_and_absolute_workspace_source(tmp_path: Path) -> None:
    project = _project(tmp_path)
    outside_project = tmp_path / "another.czi"
    outside_project.write_bytes(b"header")
    selected = tmp_path / "samples.csv"
    selected.write_text("biological_unit_id,specimen_id,condition,tissue_source\nu,selected,unknown,unknown\n", encoding="utf-8")
    row = {**_row(), "source_path": str(outside_project), "specimen_id": "selected"}
    write_acquisition_metadata(project, tmp_path, [row], annotation_csv=selected)
    assert read_acquisition_metadata(project, tmp_path, annotation_csv=selected) == ([row], [])
    assert read_acquisition_metadata(project, tmp_path)[1]
    assert resolve_source_path(project, tmp_path, str(outside_project)) == outside_project


def test_openslide_native_scale_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    (project / "slide.svs").write_bytes(b"header")
    module = ModuleType("openslide")
    class FakeSlide:
        def __init__(self, path: str) -> None:
            self.dimensions = (400, 300)
            self.properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "nan"}
        def close(self) -> None:
            pass
    module.OpenSlide = FakeSlide  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openslide", module)
    rows = discover_native_metadata(project, tmp_path, "slide.svs", "H&E", "slide")
    by_field = {row["field"]: row for row in rows}
    assert by_field["openslide.mpp-x"]["value"] == "0.25"
    assert by_field["openslide.mpp-x"]["unit"] == "um"
    assert "openslide.mpp-y" not in by_field
    write_acquisition_metadata(project, tmp_path, rows)


@pytest.mark.parametrize(("field", "value", "unit"), [
    ("objective_magnification", "40", ""),
    ("exposure_time_ms", "25.5", "ms"),
    ("excitation_wavelength_nm", "488", "nm"),
    ("emission_wavelength_nm", "520", "nm"),
    ("laser_frequency_mhz", "80", "MHz"),
    ("instrument_name", "Microscope A", ""),
    ("detector_name", "Detector 1", ""),
    ("acquisition_date", "2024-02-29", ""),
])
def test_reviewed_acquisition_settings(tmp_path: Path, field: str, value: str, unit: str) -> None:
    project = _project(tmp_path)
    row = {**_row(), "field": field, "value": value, "unit": unit,
           "provenance": "user", "source_detail": "instrument log", "reviewed": "true"}
    write_acquisition_metadata(project, tmp_path, [row])
    assert read_acquisition_metadata(project, tmp_path) == ([row], [])
    unknown = {**row, "value": "unknown", "unit": "", "provenance": "unknown", "source_detail": "", "reviewed": "false"}
    write_acquisition_metadata(project, tmp_path, [unknown])


@pytest.mark.parametrize(("field", "value", "unit"), [
    ("objective_magnification", "0", ""),
    ("exposure_time_ms", "nan", "ms"),
    ("excitation_wavelength_nm", "488", "um"),
    ("emission_wavelength_nm", "-5", "nm"),
    ("laser_frequency_mhz", "inf", "MHz"),
    ("instrument_name", " ", ""),
    ("detector_name", "camera", "px"),
    ("acquisition_date", "2023-02-29", ""),
    ("acquisition_date", "20240229", ""),
])
def test_invalid_acquisition_settings(tmp_path: Path, field: str, value: str, unit: str) -> None:
    project = _project(tmp_path)
    row = {**_row(), "field": field, "value": value, "unit": unit,
           "provenance": "user", "source_detail": "instrument log", "reviewed": "true"}
    with pytest.raises(ValueError):
        write_acquisition_metadata(project, tmp_path, [row])


@pytest.mark.parametrize("field", [
    "custom:A", "custom:z", "custom:Exposure_2", "custom:a-b.c_9",
])
@pytest.mark.parametrize(("value", "unit"), [
    ("some text", ""), ("0", "arbitrary unit / µm"), ("NaN", "custom-unit"),
])
def test_custom_metadata_accepts_named_string_values_and_units(
    tmp_path: Path, field: str, value: str, unit: str,
) -> None:
    project = _project(tmp_path)
    row = {**_row(), "field": field, "value": value, "unit": unit,
           "provenance": "user", "source_detail": "operator notebook", "reviewed": "true"}
    write_acquisition_metadata(project, tmp_path, [row])
    assert read_acquisition_metadata(project, tmp_path) == ([row], [])
    assert resolve_metadata_value([row], "a", field) == (value, unit)


@pytest.mark.parametrize("field", [
    "custom:", "custom:1name", "custom:_name", "custom:-name", "custom:.name",
    "custom:name space", "custom:name:other", "custom:na/me", "custom:é",
    "custom:aé", "Custom:name", "custom:name\nother",
])
def test_custom_metadata_rejects_invalid_names(tmp_path: Path, field: str) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        write_acquisition_metadata(project, tmp_path, [{**_row(), "field": field}])


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_custom_metadata_rejects_empty_values(tmp_path: Path, value: str) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        write_acquisition_metadata(project, tmp_path, [{**_row(), "field": "custom:note", "value": value}])


@pytest.mark.parametrize("value", [None, 42, False])
def test_custom_metadata_rejects_non_string_values(tmp_path: Path, value: object) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        write_acquisition_metadata(project, tmp_path, [{**_row(), "field": "custom:note", "value": value}])  # type: ignore[dict-item]


def test_custom_metadata_csv_import_and_round_trip(tmp_path: Path) -> None:
    project = _project(tmp_path)
    rows = [
        {**_row(), "field": "custom:operator-note", "value": 'quoted "note", α',
         "unit": "counts, adjusted", "provenance": "user", "source_detail": "run log", "reviewed": "true"},
        {**_row(), "field": "custom:unavailable", "value": "unknown", "unit": "",
         "provenance": "unknown", "source_detail": "", "reviewed": "false"},
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=METADATA_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    imported, errors = read_acquisition_csv_bytes(buffer.getvalue().encode("utf-8"), "import.csv", METADATA_COLUMNS)
    assert errors == []
    assert imported == rows
    write_acquisition_metadata(project, tmp_path, imported)
    assert read_acquisition_metadata(project, tmp_path) == (rows, [])


def test_custom_metadata_review_conflicts_and_unknown_provenance(tmp_path: Path) -> None:
    project = _project(tmp_path)
    field = "custom:instrument_setting"
    native = {**_row(), "field": field, "value": "low", "unit": "level"}
    user = {**native, "value": "high", "provenance": "user", "source_detail": "operator log"}
    unknown = {**native, "value": "unknown", "unit": "", "provenance": "unknown", "source_detail": ""}
    assert resolve_metadata_value([native, unknown], "a", field) == ("low", "level")
    assert resolve_metadata_value([unknown], "a", field) is None
    assert resolve_metadata_value([native, user], "a", field) is None
    reviewed = {**user, "reviewed": "true"}
    rows = [native, reviewed, unknown]
    assert resolve_metadata_value(rows, "a", field) == ("high", "level")
    write_acquisition_metadata(project, tmp_path, rows)
    assert read_acquisition_metadata(project, tmp_path) == (rows, [])
    assert resolve_metadata_value([native, {**unknown, "reviewed": "true"}], "a", field) is None
    with pytest.raises(ValueError, match="conflicting reviewed"):
        write_acquisition_metadata(project, tmp_path, [{**native, "reviewed": "true"}, reviewed])
    with pytest.raises(ValueError, match="source_detail"):
        write_acquisition_metadata(project, tmp_path, [{**user, "source_detail": ""}])
    with pytest.raises(ValueError, match="unknown provenance"):
        write_acquisition_metadata(project, tmp_path, [{**unknown, "value": "measured"}])


def test_discovery_states_why_native_metadata_is_unknown(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "flim.sdt").write_bytes(b"not parsed")
    (project / "broken.tif").write_bytes(b"not a tiff")
    unsupported = discover_native_metadata(project, tmp_path, "flim.sdt", "FLIM", "f")
    assert [(row["field"], row["provenance"], row["value"]) for row in unsupported] == [("metadata_status", "unknown", "unknown")]
    assert unsupported[0]["source_detail"] == "unsupported suffix .sdt; enter values manually or mark unknown"
    unreadable = discover_native_metadata(project, tmp_path, "broken.tif", "H&E", "h")
    assert unreadable[0]["provenance"] == "unknown"
    assert unreadable[0]["source_detail"].startswith(("unreadable native file: ", "reader unavailable: "))
