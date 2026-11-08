from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from spatial_omics_toolkit.application.contracts import ParameterField, RunRequest, WorkflowDescriptor
from spatial_omics_toolkit.application.intake import ALLOWED_ROOT, path_is_under

PROFILE_KEY = "_settings_profile"
_MAX_BYTES = 1024 * 1024
_PROTECTED = re.compile(r"(^|[._])(confirm\w*|mapping|specimen|core_to|authoriz\w*|receipt|run_id|example|block_indices|pairing|pre_specif\w*|selection_after\w*|verified)([._]|$)", re.I)
_REVIEW_DECISION = re.compile(r"(^|[._])review\w*([._]|$)", re.I)


def resource_fields() -> list[ParameterField]:
    return [
        ParameterField(name="resources.cpus", label="SLURM job CPUs", kind="int", required=True, default=1, min_value=1, max_value=32, step=1, unit="cores", section="SLURM job resources", help_text="CPU allocation for the single Python workflow job.", advanced=True),
        ParameterField(name="resources.memory_gb", label="SLURM job memory", kind="int", required=True, default=8, min_value=1, max_value=512, unit="GiB", section="SLURM job resources", help_text="Memory allocation for the Python workflow job. Choose within your cluster allocation policy.", advanced=True),
        ParameterField(name="resources.time_limit", label="SLURM job wall time", kind="select", required=True, default="72h", choices=["8h", "24h", "48h", "72h", "168h"], unit="wall time", section="SLURM job resources", help_text="Maximum runtime for the analysis job.", advanced=True),
    ]

def profile_fields(descriptor: WorkflowDescriptor) -> dict[str, ParameterField]:
    """Reusable workflow parameters, excluding identity mappings and review decisions."""
    fields = {field.name: field for field in descriptor.parameter_fields
              if not _PROTECTED.search(field.name)
              and not (field.kind == "bool" and _REVIEW_DECISION.search(field.name))
              and field.section.lower() != "specimen mapping"}
    for field in resource_fields():
        fields.setdefault(field.name, field)
    return fields

def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError("Value cannot be represented as a settings TOML value")


def serialize_settings(descriptor: WorkflowDescriptor, values: dict[str, Any]) -> str:
    fields = profile_fields(descriptor)
    lines = ["# Spatial Toolkit settings; only reusable analysis parameters are accepted.",
             "# Mappings, authorizations and review decisions stay in the UI.",
             "workflow_id = " + _toml_value(descriptor.workflow_id), "", "[parameters]"]
    previous_group: tuple[str, str] | None = None
    for name, field in fields.items():
        group = (field.section, field.stage_id or "")
        if group != previous_group:
            lines.append("")
            stage = f" — stage: {field.stage_id}" if field.stage_id else ""
            lines.append("# " + field.section.replace("\n", " ") + stage.replace("\n", " "))
            previous_group = group
        details = [field.kind]
        if field.unit:
            details.append(f"unit: {field.unit}")
        if field.min_value is not None:
            details.append(f"minimum: {field.min_value}")
        if field.max_value is not None:
            details.append(f"maximum: {field.max_value}")
        if field.choices:
            details.append(f"choices: {field.choices!r}")
        lines.append("# " + field.label.replace("\n", " ") + " (" + "; ".join(details).replace("\n", " ") + ")")
        for help_line in field.help_text.splitlines():
            lines.append("# " + help_line)
        if name in values and values[name] is not None:
            lines.append(_toml_value(name) + " = " + _toml_value(values[name]))
        else:
            lines.append("# " + _toml_value(name) + " is unset")
    return "\n".join(lines) + "\n"


def default_settings(descriptor: WorkflowDescriptor) -> str:
    return serialize_settings(descriptor, {name: field.default for name, field in profile_fields(descriptor).items()})


def _validate_value(field: ParameterField, value: Any) -> None:
    name = field.name
    kind = field.kind
    if kind == "bool":
        valid = type(value) is bool
    elif kind == "int":
        valid = type(value) is int
    elif kind == "float":
        valid = type(value) in (int, float) and math.isfinite(value)
    elif kind in {"text", "path", "config_file"}:
        valid = isinstance(value, str)
    elif kind == "select":
        valid = type(value) in (str, int, float, bool) if field.choices else isinstance(value, str)
    elif kind == "multiselect":
        valid = isinstance(value, list) and all(type(item) in (str, int, float, bool) if field.choices else isinstance(item, str) for item in value)
    else:
        valid = False
    if not valid:
        raise ValueError(f"{name}: expected {kind} value")
    if field.choices:
        selected = value if kind == "multiselect" else [value]
        if kind in {"select", "multiselect"} and any(item not in field.choices or type(item) not in {type(choice) for choice in field.choices} for item in selected):
            raise ValueError(f"{name}: choose from {field.choices!r}")
    if kind in {"int", "float"}:
        if field.min_value is not None and value < field.min_value:
            raise ValueError(f"{name}: must be at least {field.min_value}")
        if field.max_value is not None and value > field.max_value:
            raise ValueError(f"{name}: must be at most {field.max_value}")


def parse_settings(descriptor: WorkflowDescriptor, text: str) -> dict[str, Any]:
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("Settings file exceeds 1 MiB")
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid TOML: {exc}") from exc
    if set(data) != {"workflow_id", "parameters"} or not isinstance(data.get("parameters"), dict):
        raise ValueError("Expected only workflow_id and a [parameters] table")
    if data["workflow_id"] != descriptor.workflow_id:
        raise ValueError(f"Settings workflow_id must be {descriptor.workflow_id!r}")
    fields = profile_fields(descriptor)
    values = data["parameters"]
    for name, value in values.items():
        if name not in fields:
            raise ValueError(f"{name}: unknown or protected setting; mappings and review decisions must be entered in the UI")
        _validate_value(fields[name], value)
    return dict(values)


def nested_get(parameters: dict[str, Any], name: str) -> Any:
    value: Any = parameters
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def apply_values(parameters: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(parameters)
    for name, value in values.items():
        parts = name.split(".")
        current = result
        for part in parts[:-1]:
            existing = current.get(part)
            if not isinstance(existing, dict):
                existing = {}
                current[part] = existing
            current = existing
        current[parts[-1]] = value
    return result


def reset_values(descriptor: WorkflowDescriptor, parameters: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(parameters)
    for name in profile_fields(descriptor):
        parts = name.split(".")
        current: Any = result
        for part in parts[:-1]:
            current = current.get(part) if isinstance(current, dict) else None
        if isinstance(current, dict):
            current.pop(parts[-1], None)
    return apply_values(result, parse_settings(descriptor, default_settings(descriptor)))


def profile_metadata(text: str, *, project_root: Path | None = None, name: str | None = None) -> dict[str, str]:
    result = {"source_toml": text, "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if project_root is not None and name is not None:
        result.update({"project_root": str(project_root.resolve()), "name": name})
    return result


def current_settings_fingerprint(parameters: dict[str, Any]) -> str:
    profile = parameters.get(PROFILE_KEY)
    if not isinstance(profile, dict):
        return ""
    name, root = profile.get("name"), profile.get("project_root")
    if isinstance(name, str) and isinstance(root, str):
        try:
            return hashlib.sha256(read_user_settings(Path(root), name).encode("utf-8")).hexdigest()
        except (OSError, ValueError, UnicodeError):
            return "unavailable"
    return hashlib.sha256(str(profile.get("source_toml", "")).encode("utf-8")).hexdigest()


def config_file_hashes(descriptor: WorkflowDescriptor, parameters: dict[str, Any], project_root: Path | None = None) -> dict[str, dict[str, str]]:
    files: dict[str, dict[str, str]] = {}
    for name, field in profile_fields(descriptor).items():
        value = nested_get(parameters, name)
        if field.kind != "config_file" or not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"{name}: configuration file does not exist: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[name] = {"path": str(path), "sha256": digest.hexdigest()}
    return files


def validate_frozen_config_files(request: RunRequest) -> list[str]:
    """Check original config paths and run snapshots against frozen provenance."""
    from spatial_omics_toolkit.application.workflows.registry import get_workflow

    profile = request.parameters.get(PROFILE_KEY)
    if profile is None:
        return []  # Existing runs may predate settings profiles.
    if not isinstance(profile, dict) or not isinstance(profile.get("config_files"), dict):
        return ["Missing frozen settings configuration provenance"]
    expected = profile["config_files"]
    try:
        _, descriptor = get_workflow(request.workflow_id)
        if config_file_hashes(descriptor, request.parameters, request.project_root) != expected:
            return ["Settings configuration paths or contents changed after preflight"]
        manifest_path = request.run_dir / "inputs" / "settings_configs.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = {
            name: {"path": f"inputs/settings_config_{index}.json", "sha256": info["sha256"]}
            for index, (name, info) in enumerate(sorted(expected.items()))
        }
        if manifest != expected_manifest:
            return ["Frozen settings configuration manifest changed after submission"]
        for name, info in expected_manifest.items():
            path = request.run_dir / info["path"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != info["sha256"]:
                return [f"Frozen configuration snapshot for {name} changed after submission"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [f"Cannot validate frozen settings configurations: {exc}"]
    return []


def settings_input_fingerprint(descriptor: WorkflowDescriptor, parameters: dict[str, Any], project_root: Path | None = None) -> str:
    try:
        files = config_file_hashes(descriptor, parameters, project_root)
    except (ValueError, OSError):
        return "unavailable"
    return hashlib.sha256(json.dumps([current_settings_fingerprint(parameters), files], sort_keys=True).encode("utf-8")).hexdigest()


def freeze_profile(descriptor: WorkflowDescriptor, parameters: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    profile = parameters.get(PROFILE_KEY)
    if profile is None:
        profile = profile_metadata(default_settings(descriptor))
    if not isinstance(profile, dict) or not isinstance(profile.get("source_toml"), str):
        raise ValueError("Invalid settings profile metadata")
    source = profile["source_toml"]
    parse_settings(descriptor, source)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if profile.get("source_sha256") != digest:
        raise ValueError("Settings source changed; load it and run preflight again")
    if PROFILE_KEY in parameters and current_settings_fingerprint(parameters) != digest:
        raise ValueError("Selected settings file changed or is unavailable; reload it and run preflight again")
    effective: dict[str, Any] = {}
    for name, field in profile_fields(descriptor).items():
        value = nested_get(parameters, name)
        if value is not None:
            if field.kind == "config_file" and isinstance(value, dict):
                value = json.dumps(value, sort_keys=True, separators=(",", ":"))
            _validate_value(field, value)
            effective[name] = value
    text = serialize_settings(descriptor, effective)
    return {**{key: str(value) for key, value in profile.items() if key in {"source_toml", "source_sha256", "project_root", "name"}},
            "effective_toml": text,
            "effective_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "config_files": config_file_hashes(descriptor, parameters, project_root)}


def settings_directory(project_root: Path) -> Path:
    root = project_root.expanduser().resolve(strict=True)
    if not root.is_dir() or not path_is_under(root, ALLOWED_ROOT):
        raise ValueError("Select a project directory under the permitted data root")
    directory = root / ".spatial_toolkit" / "settings"
    for path in (root / ".spatial_toolkit", directory):
        if path.is_symlink():
            raise ValueError("Settings storage cannot use symbolic links")
        path.mkdir(mode=0o700, exist_ok=True)
        if not path.is_dir() or path.stat().st_uid != os.getuid():
            raise ValueError("Settings storage must be a directory owned by the current user")
    return directory


def settings_path(project_root: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.toml", name) or name.lower() in {"default.toml", "defaults.toml"}:
        raise ValueError("Use a user profile name ending in .toml (letters, numbers, _ or -); default.toml is reserved")
    return settings_directory(project_root) / name


def read_user_settings(project_root: Path, name: str) -> str:
    path = settings_path(project_root, name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.getuid() or not __import__("stat").S_ISREG(stat.st_mode) or stat.st_nlink != 1 or stat.st_size > _MAX_BYTES:
            raise ValueError("Settings must be a user-owned, single-link regular file no larger than 1 MiB")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = -1
            return stream.read(_MAX_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)


def save_user_settings(project_root: Path, name: str, text: str, *, create: bool) -> None:
    data = text.encode("utf-8")
    if len(data) > _MAX_BYTES:
        raise ValueError("Settings file exceeds 1 MiB")
    path = settings_path(project_root, name)
    flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
        stat = os.fstat(fd)
        if stat.st_uid != os.getuid() or not __import__("stat").S_ISREG(stat.st_mode) or stat.st_nlink != 1:
            raise ValueError("Only a user-owned, unlinked regular profile may be edited")
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
    finally:
        if fd >= 0:
            os.close(fd)
