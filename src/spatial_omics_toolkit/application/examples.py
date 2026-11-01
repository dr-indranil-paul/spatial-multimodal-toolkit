from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatial_omics_toolkit.application.contracts import PreflightResult, RunRequest
from spatial_omics_toolkit.application.intake import sha256_file
from spatial_omics_toolkit.application.service import prepare_request, submit_request

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EXAMPLES_ROOT = _REPO_ROOT / "fixtures" / "example_data"
_CATALOG_NAME = "catalog.json"
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExampleSpec:
    example_id: str
    modality: str
    workflow_id: str
    title: str
    summary: str
    layout: str
    project_root: Path
    modality_dir: Path
    annotation_csv: Path
    parameters: dict[str, Any]
    input_help: str
    expected_outputs: tuple[str, ...]
    limitations: tuple[str, ...]
    source_paths: tuple[Path, ...]


def examples_root() -> Path:
    """Return the local example-data root, optionally overridden by the environment."""
    override = os.environ.get("SPATIAL_TOOLKIT_EXAMPLES")
    return Path(override).expanduser().resolve() if override else _DEFAULT_EXAMPLES_ROOT


def _catalog_path() -> Path:
    return examples_root() / "quickstart" / _CATALOG_NAME


def get_example(modality: str) -> ExampleSpec:
    """Load one prepared, fixed example without opening or hashing its raw inputs."""
    normalized = modality.strip().casefold()
    path = _catalog_path()
    if not path.is_file():
        raise FileNotFoundError(f"Example catalog is not prepared at {path}; run `python -m spatial_omics_toolkit.application.examples --prepare`.")
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read example catalog {path}: {exc}") from exc
    entries = catalog.get("examples") if isinstance(catalog, dict) else None
    if not isinstance(entries, dict):
        raise RuntimeError(f"Example catalog {path} has no valid 'examples' mapping; rerun --prepare.")
    entry = next((v for k, v in entries.items() if str(k).casefold() == normalized), None)
    if not isinstance(entry, dict):
        choices = ", ".join(sorted(str(k) for k in entries))
        raise ValueError(f"No prepared example for modality {modality!r}. Available: {choices}.")
    root = examples_root() / "quickstart" / entry["folder"]
    return ExampleSpec(
        example_id=entry["example_id"], modality=entry["modality"], workflow_id=entry["workflow_id"],
        title=entry["title"], summary=entry["summary"], layout=entry["layout"],
        project_root=root, modality_dir=root / "data", annotation_csv=root / "project_samples.csv",
        parameters=entry["parameters"], input_help=entry["input_help"],
        expected_outputs=tuple(entry["expected_outputs"]), limitations=tuple(entry["limitations"]),
        source_paths=tuple((examples_root() / Path(p)).resolve() for p in entry.get("source_paths", [])),
    )


def _configs(root: Path) -> dict[str, dict[str, Any]]:
    raw = root
    resources = {"cpus": 1, "memory_gb": 8, "time_limit": "8h"}
    return {
        "CycIF": {"example_id": "cycif.raw_intake", "workflow_id": "cycif-raw-czi", "title": "CyCIF raw CZI intake", "summary": "Technical intake QC of one extracted TMA core acquisition (round r1 only).", "layout": "One CZI in data/slide_a1/; project_samples.csv at project root.", "sources": [raw / "cycif_raw" / "slide_A1" / "cycif_tma_core_a1_r1.czi"], "input_help": "One native CZI; acquisition naming and channel count are explicitly checked from the source metadata.", "parameters": {"intake": {"filename_pattern": "^cycif_tma_core_(?P<slide>a1)_(?P<round>r1)(?P<markers>)(?P<date>)(?P<acq_id>)(?P<scene>)[.]czi$", "expected_rounds": ["r1"], "raw_extension": ".czi"}, "conversion": {"enabled": False}, "registration": {"enabled": False, "confirmed_grouping": False}, "segmentation": {"enabled": False}, "resources": resources}},
        "HS": {"example_id": "hs.native_spectral_qc", "workflow_id": "hs.native_spectral_qc", "title": "Hyperspectral native-file QC", "summary": "Native metadata and bounded technical QC of one extracted TMA core hyperspectral CZI.", "layout": "One CZI in data/; project_samples.csv at project root.", "sources": [raw / "03_optical_physics_flim_hs" / "optical_bodipy_hyperspectral_tma_core_a1.czi"], "input_help": "One native hyperspectral CZI; no spectral identity or biological interpretation is inferred.", "parameters": {"analysis_scope": "raw_qc", "max_working_bytes": 4294967296, "resources": resources}},
        "FLIM": {"example_id": "flim.raw_sdt", "workflow_id": "flim.raw_sdt", "title": "FLIM native SDT technical QC", "summary": "Technical QC of all recorded blocks in one native SDT; detector identity remains unknown.", "layout": "One SDT in data/; project_samples.csv at project root.", "sources": [raw / "03_optical_physics_flim_hs" / "optical_flim_sdt_tma_core_a1.sdt"], "input_help": "All recorded native blocks are included for technical QC only; detector identity, frequency, calibration, and lifetime interpretation remain unresolved.", "parameters": {"analysis_scope": "raw_qc", "raw_qc_all_blocks": False, "technical_qc_authorized": False, "resources": {"cpus": 1, "memory_gb": 16, "time_limit": "8h"}}},
        "H&E": {"example_id": "he.image_intake", "workflow_id": "he-image-intake", "title": "H&E whole-slide image intake", "summary": "Native thumbnail and technical QC of one whole-tissue biopsy slide.", "layout": "One SVS in data/; project_samples.csv at project root.", "sources": [raw / "04_histology_pathology" / "histology_wsi_he_biopsy.svs"], "input_help": "One native SVS whole-slide image; outputs describe image QC only.", "parameters": {"resources": resources}},
        "Visium": {"example_id": "visium.native_qc", "workflow_id": "visium-native-qc", "title": "Visium HD developer sample QC", "summary": "Technical bundle QC of one official 10x Visium HD developer sample at the coarsest complete bin.", "layout": "Selected native 16 µm bundle in data/; project_samples.csv at project root.", "sources": [], "input_help": "One complete native Visium HD 16 µm bundle and its required feature slice; no sample-level comparison.", "parameters": {"bin_size": 16, "resources": resources}},
        "Integration": {"example_id": "integration.serial_section_alignment", "workflow_id": "integration.serial_section_alignment", "title": "Adjacent-section H&E–CycIF technical alignment", "summary": "Exploratory geometry/QC of one candidate H&E–CycIF serial-section pair using the CyCIF DAPI channel; not confirmed identity.", "layout": "One H&E CZI and one CyCIF r1 CZI in data/; project_samples.csv at project root.", "sources": [raw / "06_adjacent_sections_registration" / "serial_modality_core_a1_hne.czi", raw / "06_adjacent_sections_registration" / "serial_modality_core_a1_cycif_r1.czi"], "input_help": "Confirm the displayed H&E and CyCIF r1 candidate source files for exploratory alignment only; this does not confirm biological pairing, same-cell correspondence, or an accepted transform.", "parameters": {"linkage": "serial_section_candidate", "candidate_sources_confirmed": False, "reference_channel_index": 0, "reference_channel_name": "", "search_radius": 16, "resources": resources}},
    }



def _symlink(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required real example source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) != relative:
            raise FileExistsError(f"Example view already links to a different source; refusing to replace it: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Example view already exists and will not be overwritten: {destination}")
    destination.symlink_to(relative)


def _channel_count(czi_path: Path) -> int:
    from aicspylibczi import CziFile
    czi = CziFile(str(czi_path))
    shapes = czi.get_dims_shape()
    if not shapes or "C" not in shapes[0]:
        raise RuntimeError(f"CZI metadata contains no channel dimension: {czi_path}")
    return int(shapes[0]["C"][1] - shapes[0]["C"][0])


def _channel_zero_name(czi_path: Path) -> str:
    from aicspylibczi import CziFile
    czi = CziFile(str(czi_path))
    nodes = czi.meta.findall(".//Information/Image/Dimensions/Channels/Channel")
    name = nodes[0].findtext("Name", default=nodes[0].get("Name", "")) if nodes else None
    if not name:
        raise RuntimeError(f"CZI metadata has no native name for channel index 0: {czi_path}")
    return name


def _flim_inventory(path: Path, cache_dir: Path) -> dict[str, Any]:
    from spatial_omics_toolkit.application.workflows.registry import get_workflow
    workflow, _ = get_workflow("flim.raw_sdt")
    flim, _ = workflow._sources()
    return flim.inspect_flim_inputs([str(path)], cache_dir=cache_dir, detector_indices=None)

def _stage_visium(root: Path, config: dict[str, Any]) -> list[Path]:
    src = root / "05_spatial_omics" / "visium_hd_tiny"
    bundle = src / "binned_outputs" / "square_016um"
    required = [
        bundle / "filtered_feature_bc_matrix.h5",
        bundle / "spatial" / "tissue_positions.parquet",
        bundle / "spatial" / "scalefactors_json.json",
        src / "feature_slice.h5",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official Visium HD 16 µm bundle is incomplete: " + ", ".join(missing))
    data_root = root / "quickstart" / "visium" / "data" / "visium_hd_tiny"
    staged: list[Path] = []
    for path in required:
        target = data_root / path.relative_to(src)
        _symlink(path, target)
        staged.append(path)
    lowres = src / "spatial" / "tissue_lowres_image.png"
    if lowres.is_file():
        _symlink(lowres, data_root / "spatial" / lowres.name)
        staged.append(lowres)
    config["parameters"]["bin_size"] = 16
    return staged


def _prepare() -> None:
    root = examples_root()
    quick = root / "quickstart"
    quick.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, Any] = {"schema_version": 1, "examples": {}}
    for modality, conf in _configs(root).items():
        folder = modality.casefold().replace("&", "e")
        project = quick / folder
        data = project / "data"
        project.mkdir(parents=True, exist_ok=True)
        with (project / "project_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("biological_unit_id", "specimen_id", "condition", "tissue_source"))
            writer.writerow((_UNKNOWN, _UNKNOWN, _UNKNOWN, _UNKNOWN))
        sources = list(conf["sources"])
        if modality == "Visium":
            sources = _stage_visium(root, conf)
        elif modality == "CycIF":
            _symlink(sources[0], data / "slide_a1" / sources[0].name)
            conf["parameters"]["intake"]["expected_channels"] = _channel_count(sources[0])
        elif modality == "Integration":
            _symlink(sources[0], data / sources[0].name)
            _symlink(sources[1], data / sources[1].name)
            conf["parameters"]["moving_he_path"] = str(Path("data") / sources[0].name)
            conf["parameters"]["reference_fluorescence_path"] = str(Path("data") / sources[1].name)
            conf["parameters"]["reference_channel_name"] = _channel_zero_name(sources[1])
        else:
            if sources:
                _symlink(sources[0], data / sources[0].name)
        native_inventory: list[dict[str, Any]] = []
        if modality == "FLIM":
            inspected = _flim_inventory(sources[0], project / "native_inventory")
            native_inventory = inspected.get("records", [])
        provenance_path = project / "source_provenance.json"
        mapping = {
            "schema_version": 1,
            "example_id": conf["example_id"],
            "technical_qc_only": True,
            "source_files": [
                {"path": p.resolve().relative_to(root.resolve()).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in sources
            ],
            "source_url": "https://www.10xgenomics.com/support/software/space-ranger/latest/resources/visium-hd-example-data" if modality == "Visium" else "Local toolkit exemplar; original source metadata is retained with the raw input.",
            "licence": "CC BY 4.0 (10x Visium HD developer sample)" if modality == "Visium" else "Local-only toolkit exemplar; redistribution rights not asserted.",
            "provenance_notes": conf["summary"],
            "native_inventory": native_inventory,
        }
        provenance_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode("utf-8")
        provenance_path.write_bytes(provenance_bytes)
        provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        params = dict(conf["parameters"])
        params["specimen_mapping_confirmed"] = False
        params["specimen_mapping"] = {}
        params["example_authorization"] = {
            "example_id": conf["example_id"],
            "title": conf["title"],
            "scope": "technical_qc_only",
            "provenance_path": "source_provenance.json",
            "provenance_sha256": provenance_sha256,
        }
        expected_outputs = {
            "CycIF": ["Intake coverage report and HTML QC plots", "Run-scoped source inventory"],
            "HS": ["Native spectral metadata/channel audit", "Bounded H1/H2 technical-QC PDFs"],
            "FLIM": ["Raw count and decay technical-QC tables/plots for all recorded blocks", "Source-bound all-block selection policy"],
            "H&E": ["Native whole-slide thumbnail", "Image technical-QC report"],
            "Visium": ["Per-bundle spatial count QC", "Source bundle inventory and spatial preview"],
            "Integration": ["Before/after alignment overlay PDF and PNG", "Alignment metrics/QC JSON"],
        }[modality]
        catalog["examples"][modality] = {
            "example_id": conf["example_id"], "modality": modality, "workflow_id": conf["workflow_id"],
            "title": conf["title"], "summary": conf["summary"], "layout": conf["layout"], "folder": folder,
            "parameters": params, "input_help": conf["input_help"], "expected_outputs": expected_outputs,
            "limitations": ["Technical QC and exploratory geometry only; not a biological analysis.", "No biological identity, condition, or cross-modality specimen relationship is asserted."],
            "source_paths": [p.resolve().relative_to(root.resolve()).as_posix() for p in sources],
            "source_provenance_path": "source_provenance.json", "source_provenance_sha256": provenance_sha256,
        }
    (_catalog_path()).write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def prepare_example(
    modality: str, output_root: Path, *, authorize_technical_qc: bool = False
) -> tuple[RunRequest, PreflightResult]:
    if authorize_technical_qc is not True:
        raise PermissionError("Example submission requires explicit authorization from the visible Run example action.")
    spec = get_example(modality)
    params = json.loads(json.dumps(spec.parameters))
    authorization = params.get("example_authorization", {})
    if not isinstance(authorization, dict):
        raise RuntimeError(f"Example {spec.example_id} has no valid provenance authorization record.")
    provenance_path = spec.project_root / "source_provenance.json"
    provenance_bytes = provenance_path.read_bytes()
    provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    if provenance_sha256 != authorization.get("provenance_sha256"):
        raise RuntimeError(f"Example provenance changed since catalog preparation: {provenance_path}")
    provenance = json.loads(provenance_bytes)
    source_records = provenance.get("source_files") if isinstance(provenance, dict) else None
    if not isinstance(source_records, list) or not source_records:
        raise RuntimeError(f"Example provenance has no source file records: {provenance_path}")
    expected_sources: list[dict[str, str]] = []
    examples_root_path = examples_root().resolve()
    for record in source_records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            raise RuntimeError(f"Example provenance contains an invalid source file record: {provenance_path}")
        source_path = (examples_root_path / record["path"]).resolve()
        if examples_root_path not in source_path.parents:
            raise RuntimeError(f"Example provenance source escapes examples root: {record['path']}")
        expected_sources.append({"path": str(source_path), "sha256": record["sha256"]})
    authorization.update({
        "example_id": spec.example_id,
        "title": spec.title,
        "scope": "technical_qc_only",
        "action": "Run example",
        "authorized_by": getpass.getuser(),
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "acquisition_group_mapping": "UNMAPPED",
        "provenance_path": str(provenance_path),
        "provenance_sha256": provenance_sha256,
        "source_files": expected_sources,
    })
    params["example_authorization"] = authorization
    params["specimen_mapping_confirmed"] = False
    params["specimen_mapping"] = {}
    if spec.modality == "FLIM":
        params["raw_qc_all_blocks"] = True
        params["technical_qc_authorized"] = True
    if spec.modality == "Integration":
        params["candidate_sources_confirmed"] = True
    request, first = prepare_request(spec.workflow_id, spec.project_root, spec.annotation_csv, spec.modality_dir, output_root, params)
    if not first.ready:
        return request, first
    groups = first.resolved_parameters.get("acquisition_groups", [])
    if not isinstance(groups, list):
        raise RuntimeError(f"Example {spec.example_id} preflight returned no acquisition_groups list.")
    group_keys: list[str] = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_key"), str) or not group["group_key"].strip():
            raise RuntimeError(f"Example {spec.example_id} preflight returned a group without canonical group_key.")
        group_keys.append(group["group_key"])
    if len(group_keys) != len(set(group_keys)):
        raise RuntimeError(f"Example {spec.example_id} preflight returned duplicate acquisition-group keys.")
    params["specimen_mapping_confirmed"] = True
    params["specimen_mapping"] = {key: "UNMAPPED" for key in group_keys}
    authorization["acquisition_groups"] = group_keys
    params["example_authorization"] = authorization
    request, result = prepare_request(spec.workflow_id, spec.project_root, spec.annotation_csv, spec.modality_dir, output_root, params)
    return request, result


def submit_example(
    modality: str, output_root: Path, *, authorize_technical_qc: bool = False
) -> tuple[RunRequest, str]:
    request, preflight = prepare_example(modality, output_root, authorize_technical_qc=authorize_technical_qc)
    if not preflight.ready:
        raise RuntimeError("Example preflight blocked: " + "; ".join(preflight.blockers))
    return request, submit_request(request, preflight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed, real-input spatial toolkit examples.")
    parser.add_argument("--prepare", action="store_true", help="Materialize local relative-symlink views and provenance catalog.")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("use --prepare")
    _prepare()


if __name__ == "__main__":
    main()
