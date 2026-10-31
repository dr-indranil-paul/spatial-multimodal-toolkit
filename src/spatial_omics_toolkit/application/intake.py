from __future__ import annotations

import csv
import hashlib
import os
from io import StringIO
from pathlib import Path
from typing import Any


PROJECT_SAMPLE_COLUMNS = ("biological_unit_id", "specimen_id", "condition", "tissue_source")
ALLOWED_ROOT = Path("/home/groups/EmiliLab/pauli").resolve()


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def lexical_path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.absolute().relative_to(parent.absolute())
        return True
    except ValueError:
        return False


def validate_path_scope(path: Path, label: str) -> list[str]:
    if not path.exists():
        return [f"{label} does not exist: {path}"]
    if not lexical_path_is_under(path, ALLOWED_ROOT):
        return [f"{label} must be selected from under {ALLOWED_ROOT}: {path.absolute()}"]
    return []


def read_project_samples(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"project_samples.csv is not a file: {path}"]
    try:
        content = path.read_bytes()
    except OSError as exc:
        return [], [f"Unable to read project_samples.csv: {exc}"]
    return read_project_samples_bytes(content, str(path))


def read_project_samples_bytes(
    content: bytes, source: str = "project_samples.csv"
) -> tuple[list[dict[str, str]], list[str]]:
    blockers: list[str] = []
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(StringIO(text))
        headers = reader.fieldnames or []
        duplicate_headers = sorted({name for name in headers if headers.count(name) > 1})
        if duplicate_headers:
            return [], [
                f"{source} contains duplicate column names: " + ", ".join(duplicate_headers)
            ]
        missing = [name for name in PROJECT_SAMPLE_COLUMNS if name not in headers]
        if missing:
            return [], [
                f"{source} is missing required columns: " + ", ".join(missing)
            ]
        rows = [dict(row) for row in reader]
    except (UnicodeError, csv.Error) as exc:
        return [], [f"Unable to read {source}: {exc}"]
    if not rows:
        blockers.append(f"{source} contains no sample rows")
    seen_specimen_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        for column in PROJECT_SAMPLE_COLUMNS:
            if not (row.get(column) or "").strip():
                blockers.append(
                    f"{source} row {index} has an empty required value: {column}"
                )
        specimen_id = (row.get("specimen_id") or "").strip()
        if specimen_id:
            if specimen_id in seen_specimen_ids:
                blockers.append(
                    f"{source} row {index} repeats specimen_id {specimen_id!r}; "
                    "each physical specimen/section needs one unique row."
                )
            seen_specimen_ids.add(specimen_id)
    # biological_unit_id may repeat across specimens/sections from the same
    # independent unit; do not require one row per donor.
    return rows, blockers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_inventory(
    modality_dir: Path,
    project_root: Path,
    annotation_csv: Path,
    selected_input_paths: list[str] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    walk_errors: list[str] = []
    inventory_scope = "complete_modality_folder"

    if selected_input_paths is None:
        visited_directories: set[Path] = set()

        def record_walk_error(error: OSError) -> None:
            walk_errors.append(str(error))

        for root, directories, filenames in os.walk(
            modality_dir,
            followlinks=True,
            onerror=record_walk_error,
        ):
            root_path = Path(root)
            resolved_root = root_path.resolve()
            if resolved_root in visited_directories:
                directories[:] = []
                continue
            visited_directories.add(resolved_root)
            for name in sorted(directories):
                item = root_path / name
                if item.is_symlink():
                    entries.append(
                        {
                            "path": str(item),
                            "type": "symlink",
                            "target": str(item.resolve()),
                        }
                    )
                else:
                    entries.append({"path": str(item), "type": "directory"})
            for name in sorted(filenames):
                item = root_path / name
                if item.is_symlink():
                    resolved = item.resolve()
                    entry: dict[str, Any] = {
                        "path": str(item),
                        "type": "symlink",
                        "target": str(resolved),
                    }
                    file_path = resolved if resolved.is_file() else None
                else:
                    entry = {"path": str(item), "type": "file"}
                    file_path = item
                if file_path is not None:
                    try:
                        stat = file_path.stat()
                        entry.update(
                            {
                                "size_bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns,
                                "sha256": sha256_file(file_path),
                            }
                        )
                    except OSError as exc:
                        entry["checksum_error"] = str(exc)
                entries.append(entry)
    else:
        inventory_scope = "workflow_selected_input_files"
        entries.append(
            {
                "path": str(modality_dir.absolute()),
                "resolved_path": str(modality_dir.resolve()),
                "type": "directory",
            }
        )
        for raw_path in sorted(set(selected_input_paths)):
            item = Path(raw_path).expanduser().absolute()
            resolved = item.resolve()
            entry: dict[str, Any] = {
                "path": str(item),
                "resolved_path": str(resolved),
                "type": "symlink" if item.is_symlink() else "file",
                "exists": item.exists(),
            }
            if item.is_symlink():
                entry["target"] = str(resolved)
            if not item.exists():
                entry["checksum_error"] = "Selected workflow input does not exist."
            else:
                try:
                    stat = resolved.stat()
                    entry.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                    if resolved.is_file():
                        entry["sha256"] = sha256_file(resolved)
                    elif resolved.is_dir() and not item.is_symlink():
                        entry["type"] = "directory"
                except OSError as exc:
                    entry["checksum_error"] = str(exc)
            entries.append(entry)
    inventory: dict[str, Any] = {
        "modality_dir": str(modality_dir.absolute()),
        "modality_dir_resolved": str(modality_dir.resolve()),
        "project_root": str(project_root.absolute()),
        "project_root_resolved": str(project_root.resolve()),
        "annotation_csv": str(annotation_csv.absolute()),
        "annotation_csv_resolved": str(annotation_csv.resolve()),
        "files": entries,
        "inventory_errors": walk_errors,
        "inventory_scope": inventory_scope,
    }
    if selected_input_paths is not None:
        inventory["selected_input_paths"] = sorted(set(selected_input_paths))
    try:
        inventory["annotation_csv_sha256"] = sha256_file(annotation_csv)
    except OSError as exc:
        inventory["annotation_checksum_error"] = str(exc)
    return inventory
