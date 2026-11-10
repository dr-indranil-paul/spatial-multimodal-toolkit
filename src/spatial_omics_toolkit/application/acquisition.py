"""Structured acquisition declarations and header-only discovery.

Acquisition settings use finite positive numbers with the units in FIELD_UNITS,
nonempty instrument/detector names, or an ISO YYYY-MM-DD acquisition date.
The literal ``unknown`` is accepted for every field; it is never inferred.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from spatial_omics_toolkit.application.intake import read_project_samples, read_project_samples_bytes

METADATA_COLUMNS = ("acquisition_id", "modality", "source_path", "specimen_id", "field", "value", "unit", "provenance", "source_detail", "reviewed")
RELATIONSHIP_COLUMNS = ("acquisition_id", "related_acquisition_id", "relationship", "evidence", "reviewed")
METADATA_FILE = "acquisition_metadata.csv"
RELATIONSHIPS_FILE = "acquisition_relationships.csv"
MODALITIES = ("CycIF", "HS", "FLIM", "H&E", "Visium")
_DIMENSION_FIELDS = tuple(f"dimension_{axis}" for axis in "BCHIMRSTVXYZ")
_SIZE_FIELDS = ("pixel_size", "pixel_size_x", "pixel_size_y", "pixel_size_z", "openslide.mpp-x", "openslide.mpp-y")
_SETTING_UNITS: dict[str, tuple[str, ...]] = {
    "objective_magnification": ("",),
    "exposure_time_ms": ("ms",),
    "excitation_wavelength_nm": ("nm",),
    "emission_wavelength_nm": ("nm",),
    "laser_frequency_mhz": ("MHz",),
    "instrument_name": ("",),
    "detector_name": ("",),
    "acquisition_date": ("",),
}
_NUMERIC_SETTINGS = ("objective_magnification", "exposure_time_ms", "excitation_wavelength_nm", "emission_wavelength_nm", "laser_frequency_mhz")
METADATA_FIELDS = (*_DIMENSION_FIELDS, *_SIZE_FIELDS, *_SETTING_UNITS, "width", "height", "axes", "shape", "channel_names", "metadata_status")
_CUSTOM_FIELD = re.compile(r"custom:[A-Za-z][A-Za-z0-9_.-]*", re.ASCII)
FIELD_UNITS: dict[str, tuple[str, ...]] = {
    **{field: ("m", "mm", "um", "nm") for field in _SIZE_FIELDS},
    **{field: ("", "px") for field in (*_DIMENSION_FIELDS, "width", "height")},
    **_SETTING_UNITS,
    "axes": ("",), "shape": ("",), "channel_names": ("",), "metadata_status": ("",),
}


def _project(project_root: Path, workspace_root: Path) -> Path:
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    project = Path(project_root).expanduser().resolve(strict=True)
    if not workspace.is_dir() or not project.is_dir() or not project.is_relative_to(workspace):
        raise ValueError("Project must resolve to a directory within the allowed workspace")
    return project


def _project_file(project: Path, name: str) -> Path:
    path = project / name
    if path.is_symlink() or not path.resolve().is_relative_to(project):
        raise ValueError(f"Unsafe project file: {name}")
    return path


def resolve_source_path(project_root: Path, workspace_root: Path, source_path: str) -> Path:
    """Resolve a project-relative or absolute source within the workspace."""
    project = _project(project_root, workspace_root)
    candidate = Path(source_path).expanduser()
    if not source_path.strip() or (not candidate.is_absolute() and ".." in candidate.parts):
        raise ValueError("Source must be an absolute workspace path or a project-relative path without '..'")
    source = (project / candidate).resolve(strict=True)
    if not source.is_relative_to(Path(workspace_root).expanduser().resolve(strict=True)):
        raise ValueError("Source path resolves outside the allowed workspace")
    if not (source.is_file() or source.is_dir()):
        raise ValueError("Source path must identify a file or directory")
    return source


def _read_csv(path: Path, columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    try:
        return read_acquisition_csv_bytes(path.read_bytes(), path.name, columns)
    except OSError as exc:
        return [], [f"Cannot read {path.name}: {exc}"]


def read_acquisition_csv_bytes(content: bytes, name: str, columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with io.StringIO(content.decode("utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = reader.fieldnames
            if header is None or len(header) != len(set(header)) or set(header) != set(columns):
                return [], [f"{name}: expected exactly these unique columns: {', '.join(columns)}"]
            rows: list[dict[str, str]] = []
            for index, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    return [], [f"{name}:{index}: incorrect number of columns"]
                rows.append({key: value.strip() for key, value in row.items() if key is not None and value is not None})
            return rows, []
    except (UnicodeError, csv.Error) as exc:
        return [], [f"Cannot read {name}: {exc}"]


def _normalize(rows: Iterable[Mapping[str, str]], columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    result: list[dict[str, str]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, 2):
        if set(row) != set(columns) or any(not isinstance(v, str) for v in row.values()):
            errors.append(f"Row {index}: expected exactly the required string columns")
            continue
        result.append({key: row[key].strip() for key in columns})
    return result, errors


def _validate_field(row: Mapping[str, str], index: int) -> list[str]:
    field, value, unit = row["field"], row["value"], row["unit"]
    if _CUSTOM_FIELD.fullmatch(field):
        return [] if value.strip() else [f"Row {index}: invalid value for {field}"]
    if field not in METADATA_FIELDS:
        return [f"Row {index}: unsupported metadata field {field!r}"]
    if value == "unknown":
        return [] if unit == "" or unit in FIELD_UNITS[field] else [f"Row {index}: invalid unit for {field}"]
    if unit not in FIELD_UNITS[field]:
        return [f"Row {index}: invalid unit for {field}"]
    valid = True
    if field in (*_SIZE_FIELDS, *_NUMERIC_SETTINGS):
        try:
            number = float(value)
            valid = math.isfinite(number) and number > 0
        except ValueError:
            valid = False
    elif field in (*_DIMENSION_FIELDS, "width", "height"):
        valid = len(value) <= 20 and value.isascii() and value.isdecimal() and int(value) > 0
    elif field == "shape":
        parts = value.split(",")
        valid = bool(parts) and all(len(part) <= 20 and part.isascii() and part.isdecimal() and int(part) > 0 for part in parts)
    elif field == "axes":
        valid = bool(value) and value.isascii() and value.isalpha() and value.isupper()
    elif field == "channel_names":
        try:
            names = json.loads(value)
            valid = isinstance(names, list) and bool(names) and all(isinstance(name, str) and bool(name.strip()) for name in names)
        except (ValueError, TypeError, RecursionError):
            valid = False
    elif field == "acquisition_date":
        try:
            valid = date.fromisoformat(value).isoformat() == value
        except ValueError:
            valid = False
    elif field in {"instrument_name", "detector_name"}:
        valid = bool(value.strip())
    elif field == "metadata_status":
        valid = value in {"known", "unknown"}
    return [] if valid else [f"Row {index}: invalid value for {field}"]


def validate_acquisition_metadata(project_root: Path, workspace_root: Path, rows: Iterable[Mapping[str, str]], *, annotation_csv: Path | None = None, annotation_bytes: bytes | None = None) -> list[str]:
    records, errors = _normalize(rows, METADATA_COLUMNS)
    try:
        project = _project(project_root, workspace_root)
    except (OSError, ValueError, RuntimeError) as exc:
        return errors + [str(exc)]
    specimen_values = {row["specimen_id"] for row in records if row["specimen_id"] and row["specimen_id"].lower() != "unknown"}
    if specimen_values:
        try:
            if annotation_csv is None:
                samples_path = _project_file(project, "project_samples.csv")
            else:
                samples_path = (project / Path(annotation_csv).expanduser()).resolve(strict=True)
            if not samples_path.resolve().is_relative_to(Path(workspace_root).expanduser().resolve(strict=True)):
                raise ValueError("Sample sheet must resolve within the allowed workspace")
        except (OSError, ValueError, RuntimeError) as exc:
            return errors + [str(exc)]
        samples, sample_errors = (
            read_project_samples_bytes(annotation_bytes, str(samples_path))
            if annotation_bytes is not None else read_project_samples(samples_path)
        )
        errors.extend(sample_errors)
        known = {row.get("specimen_id", "") for row in samples}
        errors.extend(f"Unknown project specimen_id: {value}" for value in sorted(specimen_values - known))
    identities: dict[str, tuple[str, str, str]] = {}
    reviewed_values: dict[tuple[str, str], tuple[str, str]] = {}
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(records, 2):
        for key in ("acquisition_id", "modality", "source_path", "field", "value", "provenance", "reviewed"):
            if not row[key]:
                errors.append(f"Row {index}: {key} is required")
        if row["reviewed"] not in {"true", "false"}:
            errors.append(f"Row {index}: reviewed must be true or false")
        if row["modality"] not in MODALITIES:
            errors.append(f"Row {index}: unsupported modality {row['modality']!r}")
        if row["provenance"] not in {"native", "user", "unknown"}:
            errors.append(f"Row {index}: provenance must be native, user, or unknown")
        if row["provenance"] in {"native", "user"} and not row["source_detail"]:
            errors.append(f"Row {index}: source_detail is required for native or user provenance")
        if row["provenance"] == "unknown" and row["value"] != "unknown":
            errors.append(f"Row {index}: unknown provenance requires an unknown value")
        errors.extend(_validate_field(row, index))
        try:
            resolve_source_path(project, workspace_root, row["source_path"])
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"Row {index}: invalid source_path: {exc}")
        identity = (row["modality"], row["source_path"], row["specimen_id"])
        acquisition = row["acquisition_id"]
        if acquisition in identities and identities[acquisition] != identity:
            errors.append(f"Row {index}: conflicting identity for acquisition_id {acquisition}")
        identities[acquisition] = identity
        key = (acquisition, row["field"])
        value = (row["value"], row["unit"])
        if row["reviewed"] == "true":
            if key in reviewed_values and reviewed_values[key] != value:
                errors.append(f"Row {index}: conflicting reviewed values for {acquisition}/{row['field']}")
            reviewed_values[key] = value
        exact = tuple(row[column] for column in METADATA_COLUMNS)
        if exact in seen:
            errors.append(f"Row {index}: duplicate metadata record")
        seen.add(exact)
    return errors


def resolve_metadata_value(rows: Iterable[Mapping[str, str]], acquisition_id: str, field: str) -> tuple[str, str] | None:
    """Return a unique usable value; a reviewed unknown remains unresolved."""
    matching = [row for row in rows if row.get("acquisition_id") == acquisition_id and row.get("field") == field]
    reviewed = {(row["value"], row["unit"]) for row in matching if row.get("reviewed") == "true"}
    candidates = reviewed if reviewed else {(row["value"], row["unit"]) for row in matching if row["value"] != "unknown"}
    if len(candidates) != 1:
        return None
    value = next(iter(candidates))
    return None if value[0] == "unknown" else value


def validate_acquisition_relationships(project_root: Path, workspace_root: Path, rows: Iterable[Mapping[str, str]], *, annotation_csv: Path | None = None, metadata_rows: Iterable[Mapping[str, str]] | None = None) -> list[str]:
    records, errors = _normalize(rows, RELATIONSHIP_COLUMNS)
    if not records:
        return errors
    if metadata_rows is None:
        metadata, metadata_errors = read_acquisition_metadata(project_root, workspace_root, annotation_csv=annotation_csv)
        errors.extend(metadata_errors)
    else:
        metadata = list(metadata_rows)
    ids = {row["acquisition_id"] for row in metadata}
    pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(records, 2):
        left, right = row["acquisition_id"], row["related_acquisition_id"]
        if not left or not right or left == right:
            errors.append(f"Row {index}: relationship requires two distinct acquisition IDs")
        if left not in ids or right not in ids:
            errors.append(f"Row {index}: relationship references an unknown acquisition ID")
        if row["relationship"] not in {"same", "serial", "unknown"}:
            errors.append(f"Row {index}: relationship must be same, serial, or unknown")
        if not row["evidence"]:
            errors.append(f"Row {index}: relationship evidence is required")
        if row["reviewed"] not in {"true", "false"}:
            errors.append(f"Row {index}: reviewed must be true or false")
        pair = tuple(sorted((left, right)))
        if pair in pairs:
            errors.append(f"Row {index}: duplicate or conflicting relationship for {left} and {right}")
        pairs.add(pair)
    return errors


def read_acquisition_metadata(project_root: Path, workspace_root: Path, *, annotation_csv: Path | None = None) -> tuple[list[dict[str, str]], list[str]]:
    try:
        path = _project_file(_project(project_root, workspace_root), METADATA_FILE)
    except (OSError, ValueError, RuntimeError) as exc:
        return [], [str(exc)]
    rows, errors = _read_csv(path, METADATA_COLUMNS)
    return rows, errors or validate_acquisition_metadata(project_root, workspace_root, rows, annotation_csv=annotation_csv)


def read_acquisition_relationships(project_root: Path, workspace_root: Path, *, annotation_csv: Path | None = None) -> tuple[list[dict[str, str]], list[str]]:
    try:
        path = _project_file(_project(project_root, workspace_root), RELATIONSHIPS_FILE)
    except (OSError, ValueError, RuntimeError) as exc:
        return [], [str(exc)]
    rows, errors = _read_csv(path, RELATIONSHIP_COLUMNS)
    return rows, errors or validate_acquisition_relationships(project_root, workspace_root, rows, annotation_csv=annotation_csv)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        if path.is_symlink():
            raise ValueError(f"Unsafe project file: {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_acquisition_metadata(project_root: Path, workspace_root: Path, rows: Iterable[Mapping[str, str]], *, annotation_csv: Path | None = None) -> None:
    records, errors = _normalize(rows, METADATA_COLUMNS)
    errors.extend(validate_acquisition_metadata(project_root, workspace_root, records, annotation_csv=annotation_csv))
    if errors:
        raise ValueError("; ".join(errors))
    _write_csv(_project_file(_project(project_root, workspace_root), METADATA_FILE), METADATA_COLUMNS, records)


def write_acquisition_relationships(project_root: Path, workspace_root: Path, rows: Iterable[Mapping[str, str]], *, annotation_csv: Path | None = None) -> None:
    records, errors = _normalize(rows, RELATIONSHIP_COLUMNS)
    errors.extend(validate_acquisition_relationships(project_root, workspace_root, records, annotation_csv=annotation_csv))
    if errors:
        raise ValueError("; ".join(errors))
    _write_csv(_project_file(_project(project_root, workspace_root), RELATIONSHIPS_FILE), RELATIONSHIP_COLUMNS, records)

def discover_native_metadata(project_root: Path, workspace_root: Path, source_path: str, modality: str, acquisition_id: str) -> list[dict[str, str]]:
    """Read format headers only. Native properties never imply specimen pairing or registration."""
    source = resolve_source_path(project_root, workspace_root, source_path)
    base = {"acquisition_id": acquisition_id, "modality": modality, "source_path": source_path, "specimen_id": "", "unit": "", "provenance": "native", "source_detail": f"{source.suffix.lower()} native header", "reviewed": "false"}
    values: dict[str, str] = {}
    units: dict[str, str] = {}
    try:
        suffix = source.suffix.lower()
        if suffix == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(str(source))
            shapes = czi.get_dims_shape()
            try:
                is_mosaic = czi.is_mosaic()
            except (AttributeError, OSError, ValueError, RuntimeError):
                is_mosaic = None
            if len(shapes) == 1:
                for dimension, extent in shapes[0].items():
                    if dimension in {"X", "Y"} and is_mosaic is not False:
                        continue
                    if f"dimension_{dimension}" in _DIMENSION_FIELDS and extent[1] > extent[0]:
                        values[f"dimension_{dimension}"] = str(extent[1] - extent[0])
            if is_mosaic:
                try:
                    box = czi.get_mosaic_bounding_box()
                    raw_box = (box.x, box.y, box.w, box.h)
                    if all(math.isfinite(float(value)) and int(value) == value for value in raw_box):
                        width, height = int(box.w), int(box.h)
                        if width > 0 and height > 0:
                            values["dimension_X"] = str(width)
                            values["dimension_Y"] = str(height)
                except (AttributeError, OSError, ValueError, RuntimeError, TypeError, OverflowError):
                    pass
            metadata = getattr(czi, "meta", None)
            if metadata is not None:
                channels = metadata.findall(".//Information/Image/Dimensions/Channels/Channel")
                names = [node.findtext("Name", default=node.get("Name", "")) for node in channels]
                if names and all(names):
                    values["channel_names"] = json.dumps(names, ensure_ascii=False)
                for node in metadata.findall(".//Scaling/Items/Distance"):
                    axis = node.findtext("Id", default=node.get("Id", ""))
                    raw = node.findtext("Value", default="")
                    if axis not in {"X", "Y", "Z"} or not raw:
                        continue
                    try:
                        scale = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(scale) and scale > 0:
                        field = f"pixel_size_{axis.lower()}"
                        values[field] = raw
                        units[field] = "m"
        elif suffix in {".tif", ".tiff"}:
            import tifffile
            with tifffile.TiffFile(source) as tif:
                if tif.series:
                    values["axes"] = str(tif.series[0].axes)
                    values["shape"] = ",".join(str(size) for size in tif.series[0].shape)
        elif suffix == ".svs":
            import openslide
            slide = openslide.OpenSlide(str(source))
            try:
                values["width"] = str(slide.dimensions[0])
                values["height"] = str(slide.dimensions[1])
                for key in ("openslide.mpp-x", "openslide.mpp-y"):
                    raw = slide.properties.get(key)
                    if raw is not None:
                        try:
                            pitch = float(raw)
                        except ValueError:
                            continue
                        if math.isfinite(pitch) and pitch > 0:
                            values[key] = str(raw)
                            units[key] = "um"
            finally:
                slide.close()
    except (ImportError, OSError, ValueError, RuntimeError, AttributeError, IndexError):
        values = {}
    if not values:
        values["metadata_status"] = "unknown"
        base["provenance"] = "unknown"
        base["source_detail"] = ""
    return [{**base, "field": field, "value": value, "unit": units.get(field, "")} for field, value in values.items()]
