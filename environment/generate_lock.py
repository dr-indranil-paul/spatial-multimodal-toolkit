#!/usr/bin/env python3
"""Write environment/requirements.lock.txt from an already-installed environment.

Run under CPython 3.12 in the environment whose versions should be pinned::

    python environment/generate_lock.py [--extras all|a,b] [--pins FILE] [--strict]

Versions come only from ``importlib.metadata`` of the running interpreter. The
script never contacts an index. The pinned set is the transitive closure of:

* base dependencies and the selected extras declared in ``pyproject.toml``;
* every third-party top-level module imported anywhere under
  ``src/spatial_omics_toolkit`` (including ``external_scripts/``).

Requirement markers (``python_version``, ``sys_platform``, ``extra == ...``) are
evaluated for the running interpreter and the selected extras; entries whose
markers do not apply are dropped. The toolkit distribution itself is never pinned.

Exit status: 0 on success, including when entries are unresolved (they are listed
in the header and on stderr). 1 when the environment is inconsistent, an import
cannot be mapped to a distribution, or ``--pins`` disagrees with the environment.
With ``--strict``, also 1 when anything is unresolved or not index-installable.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import tomllib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

TARGET_PYTHON: Final[tuple[int, int]] = (3, 12)
TOOLKIT_DISTRIBUTION: Final[str] = "spatial-omics-toolkit"
SOURCE_LABEL: Final[str] = (
    "Emili lab environment (shared spatial-proteomics env with the spatial-toolkit-20261030 overlay)"
)
# Import names whose distribution name differs from the module name. Consulted only
# when the installed distribution map has no entry, e.g. for a package that is absent.
IMPORT_TO_DISTRIBUTION: Final[Mapping[str, str]] = {
    "PIL": "pillow",
    "SimpleITK": "simpleitk",
    "cv2": "opencv-python",
    "fitz": "pymupdf",
    "openslide": "openslide-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "spatialdata_io": "spatialdata-io",
    "tomli": "tomli",
    "valis": "valis-wsi",
}
DIRECT_URL_KINDS: Final[tuple[str, ...]] = ("dir_info", "vcs_info", "archive_info")


@dataclass(frozen=True)
class Edge:
    """A requirement to resolve against the installed environment."""

    name: str
    extras: frozenset[str]
    specifier: SpecifierSet
    requester: str


@dataclass(frozen=True)
class Pin:
    """An installed distribution selected for the lock."""

    display_name: str
    version: str
    origin: str  # "index" (default install), "overlay" (outside sys.prefix), "direct-url"
    detail: str


@dataclass
class Resolution:
    """Outcome of the closure walk."""

    pins: dict[str, Pin] = field(default_factory=dict)
    missing: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    conflicts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Options:
    repo: Path
    output: Path
    extras: frozenset[str] | None
    pins: Path | None
    strict: bool


def parse_args(argv: Sequence[str] | None = None) -> Options:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else "")
    parser.add_argument("--repo", type=Path, default=here.parent, help="repository root (default: parent of environment/)")
    parser.add_argument("--output", type=Path, default=here / "requirements.lock.txt")
    parser.add_argument("--extras", default="all", help="comma-separated extras, or 'all' (default)")
    parser.add_argument("--pins", type=Path, default=None, help="overlay pin file to cross-check against the environment")
    parser.add_argument("--strict", action="store_true", help="fail on unresolved, non-index, or unverified entries")
    ns = parser.parse_args(argv)
    extras: frozenset[str] | None = None
    if ns.extras != "all":
        extras = frozenset(canonicalize_name(item) for item in ns.extras.split(",") if item.strip())
    return Options(
        repo=ns.repo.resolve(),
        output=ns.output.resolve(),
        extras=extras,
        pins=ns.pins.resolve() if ns.pins else None,
        strict=bool(ns.strict),
    )


def load_pyproject(repo: Path) -> dict[str, Any]:
    with (repo / "pyproject.toml").open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    return data


def declared_edges(
    pyproject: Mapping[str, Any], selected: frozenset[str] | None
) -> tuple[list[Edge], frozenset[str], frozenset[str]]:
    """Return root edges, the extras resolved, and base dependencies inactive on this interpreter."""
    project: Mapping[str, Any] = pyproject["project"]
    optional: Mapping[str, list[str]] = project.get("optional-dependencies", {})
    declared = {canonicalize_name(name): name for name in optional}
    if selected is None:
        chosen = frozenset(declared)
    else:
        unknown = sorted(selected - frozenset(declared))
        if unknown:
            raise SystemExit(f"unknown extras requested: {', '.join(unknown)}")
        chosen = selected
    edges: list[Edge] = []
    inactive: set[str] = set()
    for text in project.get("dependencies", []):
        req = Requirement(text)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            edges.append(Edge(req.name, frozenset(req.extras), req.specifier, "pyproject dependencies"))
        else:
            inactive.add(canonicalize_name(req.name))
    for key in sorted(chosen):
        for text in optional[declared[key]]:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": key}):
                edges.append(Edge(req.name, frozenset(req.extras), req.specifier, f"pyproject extra '{key}'"))
    return edges, chosen, frozenset(inactive)


def scan_source(source_root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Map third-party top-level import names to the files that import them.

    Local names (the package itself and every module or directory beneath it) are
    excluded, including sibling modules that external_scripts import by bare name.
    """
    local: set[str] = {source_root.name}
    for path in source_root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        if path.is_dir():
            local.add(path.name)
        elif path.suffix == ".py":
            local.add(path.stem)
    imports: dict[str, set[str]] = defaultdict(set)
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
        for name in names:
            top = name.split(".", 1)[0]
            if top in sys.stdlib_module_names or top in local:
                continue
            imports[top].add(path.relative_to(source_root).as_posix())
    return imports, local


def import_edges(
    imports: Mapping[str, set[str]], inactive: frozenset[str]
) -> tuple[list[Edge], list[str], list[str]]:
    """Map third-party imports to distribution edges.

    Returns (edges, unmapped import names, imports skipped because their declared
    requirement is inactive on this interpreter, such as tomli on Python 3.12).
    """
    installed = metadata.packages_distributions()
    edges: list[Edge] = []
    unmapped: list[str] = []
    skipped: list[str] = []
    for top in sorted(imports):
        candidates = sorted(set(installed.get(top, [])))
        if not candidates and top in IMPORT_TO_DISTRIBUTION:
            candidates = [IMPORT_TO_DISTRIBUTION[top]]
        if not candidates:
            unmapped.append(top)
            continue
        if all(canonicalize_name(name) in inactive for name in candidates):
            skipped.append(top)
            continue
        first = sorted(imports[top])[0]
        for name in candidates:
            if canonicalize_name(name) in inactive:
                continue
            edges.append(
                Edge(name, frozenset(), SpecifierSet(), f"import {top} ({len(imports[top])} file(s), e.g. {first})")
            )
    return edges, unmapped, skipped


def installed_distribution(name: str) -> metadata.Distribution | None:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def classify(dist: metadata.Distribution, prefix: Path) -> tuple[str, str]:
    """Return (origin, detail) describing where the distribution was installed from."""
    direct = dist.read_text("direct_url.json")
    if direct is not None:
        info: dict[str, Any] = json.loads(direct)
        for kind in DIRECT_URL_KINDS:
            if kind in info:
                return "direct-url", f"{kind} in direct_url.json"
    location = Path(str(dist.locate_file(""))).resolve()
    if not location.is_relative_to(prefix):
        return "overlay", "installed from an overlay directory outside the interpreter prefix"
    return "index", ""


def resolve(edges: Iterable[Edge]) -> Resolution:
    """Walk the requirement graph, honouring markers and extras, against installed metadata."""
    result = Resolution()
    prefix = Path(sys.prefix).resolve()
    toolkit = canonicalize_name(TOOLKIT_DISTRIBUTION)
    processed: dict[str, set[str]] = defaultdict(set)
    queue: deque[Edge] = deque(edges)
    while queue:
        edge = queue.popleft()
        key = canonicalize_name(edge.name)
        if key == toolkit:
            continue
        dist = installed_distribution(edge.name)
        if dist is None:
            result.missing[key].add(edge.requester)
            continue
        display = str(dist.metadata["Name"])
        version = dist.version
        if not edge.specifier.contains(version, prereleases=True):
            result.conflicts.append(f"{display}=={version} violates '{edge.specifier}' required by {edge.requester}")
        if key not in result.pins:
            origin, detail = classify(dist, prefix)
            result.pins[key] = Pin(display, version, origin, detail)
        fresh = ({""} | set(edge.extras)) - processed[key]
        if not fresh:
            continue
        processed[key] |= fresh
        for raw in dist.requires or []:
            child = Requirement(raw)
            if child.marker is None:
                queue.append(Edge(child.name, frozenset(child.extras), child.specifier, display))
                continue
            active = {extra for extra in fresh if child.marker.evaluate({"extra": extra})}
            if active:
                extras = frozenset(child.extras) | frozenset(active - {""})
                queue.append(Edge(child.name, extras, child.specifier, display))
    return result


def check_pins(path: Path, resolution: Resolution) -> tuple[list[str], list[str]]:
    """Return (errors, notes) comparing an overlay pin file with the resolved environment."""
    errors: list[str] = []
    notes: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep:
            errors.append(f"unparseable pin line in {path.name}: {raw!r}")
            continue
        name = name.strip()
        version = version.strip()
        pin = resolution.pins.get(canonicalize_name(name))
        if pin is None:
            notes.append(f"{name}=={version} in {path.name} is not required by the declared closure")
        elif pin.version != version:
            errors.append(f"{pin.display_name}: environment {pin.version} != {path.name} {version}")
    return errors, notes


def render(
    resolution: Resolution,
    chosen: frozenset[str],
    unmapped: Sequence[str],
    skipped: Sequence[str],
    local_count: int,
    root_count: int,
    import_count: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    direct = sorted(
        (pin for pin in resolution.pins.values() if pin.origin == "direct-url"),
        key=lambda pin: canonicalize_name(pin.display_name),
    )
    overlay = sorted(
        (pin for pin in resolution.pins.values() if pin.origin == "overlay"),
        key=lambda pin: canonicalize_name(pin.display_name),
    )
    pinned = sorted(
        (pin for pin in resolution.pins.values() if pin.origin != "direct-url"),
        key=lambda pin: canonicalize_name(pin.display_name),
    )
    header = [
        "# spatial-omics-toolkit: exact dependency pins for CPython 3.12 (Linux x86_64 markers).",
        f"# Generated: {timestamp} by environment/generate_lock.py.",
        f"# Source interpreter: CPython {platform.python_version()} on {sys.platform} {platform.machine()}.",
        f"# Resolved from the {SOURCE_LABEL}.",
        "# Versions were read from installed distributions (importlib.metadata); no index was queried.",
        f"# Extras resolved: {', '.join(sorted(chosen)) or 'none'}.",
        (
            f"# Roots: {root_count} pyproject.toml requirements (base and selected extras) plus "
            f"{import_count} third-party import roots under src/spatial_omics_toolkit "
            f"({local_count} local module names excluded)."
        ),
        f"# Pinned: {len(pinned)} distributions.",
        "#",
        "# NOT PINNED: declared or imported, but absent from the source environment; excluded from the pins:",
    ]
    if resolution.missing:
        for key in sorted(resolution.missing):
            reasons = "; ".join(sorted(resolution.missing[key]))
            header.append(f"#   {key}  <- {reasons}")
    else:
        header.append("#   none")
    if unmapped:
        header.append("# UNMAPPED IMPORTS (no distribution known): " + ", ".join(unmapped))
    if skipped:
        header.append("# Imports skipped (declared requirement inactive on CPython 3.12): " + ", ".join(skipped))
    header.append("#")
    header.append("# NOT INSTALLABLE FROM AN INDEX (installed from a direct URL or local path; excluded from the pins):")
    if direct:
        for pin in direct:
            header.append(f"#   {pin.display_name}=={pin.version}  ({pin.detail})")
    else:
        header.append("#   none")
    header.append("#")
    header.append("# Pinned from an overlay directory outside the interpreter prefix (verify with a fresh install):")
    if overlay:
        for pin in overlay:
            header.append(f"#   {pin.display_name}=={pin.version}")
    else:
        header.append("#   none")
    header.append("")
    body = [f"{pin.display_name}=={pin.version}" for pin in pinned]
    return "\n".join(header + body) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    if sys.version_info[:2] != TARGET_PYTHON:
        print(f"generate_lock.py must run under CPython {TARGET_PYTHON[0]}.{TARGET_PYTHON[1]}", file=sys.stderr)
        return 2
    options = parse_args(argv)
    pyproject = load_pyproject(options.repo)
    roots, chosen, inactive = declared_edges(pyproject, options.extras)
    source_root = options.repo / "src" / "spatial_omics_toolkit"
    imports, local = scan_source(source_root)
    import_roots, unmapped, skipped = import_edges(imports, inactive)
    if unmapped:
        print(f"ERROR unmapped third-party imports: {', '.join(unmapped)}", file=sys.stderr)
        return 1
    resolution = resolve([*roots, *import_roots])
    failed = False
    for line in resolution.conflicts:
        print(f"ERROR inconsistent environment: {line}", file=sys.stderr)
    failed = failed or bool(resolution.conflicts)
    if options.pins is not None:
        errors, notes = check_pins(options.pins, resolution)
        for note in notes:
            print(f"NOTE {note}", file=sys.stderr)
        for error in errors:
            print(f"ERROR pin mismatch: {error}", file=sys.stderr)
        failed = failed or bool(errors)
    if failed:
        return 1
    text = render(resolution, chosen, unmapped, skipped, len(local), len(roots), len(import_roots))
    options.output.write_text(text, encoding="utf-8")
    direct_count = sum(1 for pin in resolution.pins.values() if pin.origin == "direct-url")
    overlay_count = sum(1 for pin in resolution.pins.values() if pin.origin == "overlay")
    pinned_count = len(resolution.pins) - direct_count
    print(
        f"wrote {options.output.name}: {pinned_count} pins ({overlay_count} overlay-located), "
        f"{len(resolution.missing)} declared/imported but not installed, {direct_count} not index-installable"
    )
    for key in sorted(resolution.missing):
        print(f"WARN not installed: {key} <- {'; '.join(sorted(resolution.missing[key]))}", file=sys.stderr)
    for pin in sorted(resolution.pins.values(), key=lambda item: canonicalize_name(item.display_name)):
        if pin.origin == "direct-url":
            print(f"WARN not index-installable: {pin.display_name}=={pin.version} ({pin.detail})", file=sys.stderr)
    if options.strict and (resolution.missing or direct_count):
        print("strict mode: unresolved entries present", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
