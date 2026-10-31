import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.application.contracts import PreflightResult, RunRequest
from spatial_omics_toolkit.application.workflows import visium as visium_workflow

from spatial_omics_toolkit.application.workflows.visium import (
    _available_hd_bin_sizes,
    _acquisition_input_paths,
    _build_visium_hd_reader_view,
    _discover_bundles,
)


def test_visium_hd_bin_directories_are_not_misclassified_as_classic_bundles(tmp_path: Path) -> None:
    bundle_root = tmp_path / "spaceranger" / "outs"
    bin_root = bundle_root / "binned_outputs" / "square_008um"
    (bin_root / "spatial").mkdir(parents=True)
    (bin_root / "filtered_feature_bc_matrix.h5").touch()
    (bin_root / "spatial" / "tissue_positions.parquet").touch()

    (bin_root / "spatial" / "scalefactors_json.json").write_text("{}", encoding="utf-8")
    bundles, blockers = _discover_bundles(bundle_root, bin_size=8)

    assert blockers == []
    assert len(bundles) == 1
    assert bundles[0]["platform"] == "visium_hd"
    assert bundles[0]["bundle_root"] == str(bundle_root)
    assert bundles[0]["bin_size"] == 8


def test_visium_hd_discovery_requires_reader_supported_positions_and_bin_scales(tmp_path: Path) -> None:
    bundle_root = tmp_path / "spaceranger" / "outs"
    bin_root = bundle_root / "binned_outputs" / "square_008um"
    spatial_root = bin_root / "spatial"
    spatial_root.mkdir(parents=True)
    (bin_root / "filtered_feature_bc_matrix.h5").touch()
    (spatial_root / "tissue_positions.csv").touch()
    (spatial_root / "scalefactors_json.json").touch()

    bundles, blockers = _discover_bundles(bundle_root, bin_size=8)

    assert bundles == []
    assert any("tissue_positions.parquet" in blocker for blocker in blockers)

    (spatial_root / "tissue_positions.parquet").touch()
    (spatial_root / "scalefactors_json.json").unlink()
    bundles, blockers = _discover_bundles(bundle_root, bin_size=8)

    assert bundles == []
    assert any("scalefactors_json.json" in blocker for blocker in blockers)

    (spatial_root / "scalefactors_json.json").touch()
    bundles, blockers = _discover_bundles(bundle_root, bin_size=8)

    assert blockers == []
    assert len(bundles) == 1


def test_hd_resolution_choices_require_complete_common_bins_across_bundles(tmp_path: Path) -> None:
    modality_root = tmp_path / "Visium"
    for bundle_name, available_sizes in (("bundle-a", (8, 16)), ("bundle-b", (8, 16))):
        bundle_root = modality_root / bundle_name / "outs"
        bundle_root.mkdir(parents=True)
        (bundle_root / "feature_slice.h5").touch()
        for size in available_sizes:
            bin_root = bundle_root / "binned_outputs" / f"square_{size:03d}um"
            spatial_root = bin_root / "spatial"
            spatial_root.mkdir(parents=True)
            (bin_root / "filtered_feature_bc_matrix.h5").touch()
            (spatial_root / "tissue_positions.parquet").touch()
            if not (bundle_name == "bundle-b" and size == 16):
                (spatial_root / "scalefactors_json.json").touch()

    assert _available_hd_bin_sizes(modality_root) == [8]


def test_visium_hd_reader_view_only_exposes_selected_bounded_inputs(tmp_path: Path) -> None:
    bundle_root = tmp_path / "outs"
    bin_root = bundle_root / "binned_outputs" / "square_008um"
    spatial_root = bin_root / "spatial"
    spatial_root.mkdir(parents=True)
    (bundle_root / "sample_feature_slice.h5").touch()
    (bin_root / "filtered_feature_bc_matrix.h5").touch()
    (spatial_root / "tissue_positions.parquet").touch()
    (spatial_root / "scalefactors_json.json").write_text("{}", encoding="utf-8")
    original_spatial = bundle_root / "spatial"
    original_spatial.mkdir()
    (original_spatial / "scalefactors_json.json").write_text('{"tissue_lowres_scalef": 0.5}', encoding="utf-8")
    from PIL import Image

    Image.new("RGB", (16, 16)).save(original_spatial / "tissue_lowres_image.png")
    Image.new("RGB", (32, 32)).save(original_spatial / "tissue_hires_image.png")
    (bundle_root / "microscope_image").mkdir()
    (bundle_root / "microscope_image" / "slide.tif").touch()
    segmentation_root = bundle_root / "segmented_outputs"
    segmentation_root.mkdir()
    (segmentation_root / "cell_segmentations.geojson").touch()

    reader_root = tmp_path / "reader"
    reader_root.mkdir()
    _build_visium_hd_reader_view(bundle_root, 8, reader_root)

    staged_files = {
        path.relative_to(reader_root).as_posix()
        for path in reader_root.rglob("*")
        if path.is_file()
    }
    assert staged_files == {
        "sample_feature_slice.h5",
        "binned_outputs/square_008um/filtered_feature_bc_matrix.h5",
        "binned_outputs/square_008um/spatial/tissue_positions.parquet",
        "binned_outputs/square_008um/spatial/scalefactors_json.json",
        "segmented_outputs/spatial/scalefactors_json.json",
        "spatial/tissue_lowres_image.png",
    }
    Image.new("RGB", (2049, 1)).save(original_spatial / "tissue_lowres_image.png")
    Image.new("RGB", (16, 16)).save(spatial_root / "tissue_lowres_image.png")
    oversized_reader_root = tmp_path / "reader-oversized"
    oversized_reader_root.mkdir()
    _build_visium_hd_reader_view(bundle_root, 8, oversized_reader_root)
    assert (oversized_reader_root / "spatial" / "tissue_lowres_image.png").is_file()
    input_paths = _acquisition_input_paths(
        {"bundle_root": str(bundle_root), "platform": "visium_hd", "bin_size": 8}
    )
    assert str(bundle_root / "sample_feature_slice.h5") in input_paths
    assert str(bin_root / "filtered_feature_bc_matrix.h5") in input_paths
    assert str(spatial_root / "tissue_positions.parquet") in input_paths
    assert str(original_spatial / "scalefactors_json.json") in input_paths
    assert str(spatial_root / "scalefactors_json.json") in input_paths
    assert str(spatial_root / "tissue_lowres_image.png") in input_paths
    assert not any("tissue_hires" in path or "segmented_outputs" in path for path in input_paths)


def test_visium_hd_execution_without_lowres_image_keeps_optional_preview_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "outs"
    bin_root = bundle_root / "binned_outputs" / "square_008um"
    spatial_root = bin_root / "spatial"
    spatial_root.mkdir(parents=True)
    (bundle_root / "sample_feature_slice.h5").touch()
    (bin_root / "filtered_feature_bc_matrix.h5").touch()
    (spatial_root / "tissue_positions.parquet").touch()
    (spatial_root / "scalefactors_json.json").write_text("{}", encoding="utf-8")
    annotation_csv = tmp_path / "project_samples.csv"
    annotation_csv.write_text("biological_unit_id,specimen_id,condition,tissue_source\n", encoding="utf-8")
    request = RunRequest(
        workflow_id="visium.native_qc",
        modality="Visium",
        project_root=tmp_path,
        modality_dir=tmp_path / "input",
        annotation_csv=annotation_csv,
        output_root=tmp_path / "output",
        run_id=uuid4(),
        run_dir=tmp_path / "run",
    )
    bundle_key = "sample"
    preflight_result = PreflightResult(
        ready=True,
        resolved_parameters={
            "bundles": [
                {
                    "bundle_key": bundle_key,
                    "bundle_root": str(bundle_root),
                    "platform": "visium_hd",
                    "bin_size": 8,
                }
            ],
            "spatialdata_io_version": "0.6.0",
            "scanpy_version": "not installed",
            "specimen_mapping": {bundle_key: "UNMAPPED"},
            "specimen_mapping_confirmed": True,
            "preview_max_points": 1000,
        },
    )
    counts = np.asarray([[2, 0], [0, 3]])
    adata = SimpleNamespace(
        n_obs=2,
        n_vars=2,
        X=counts,
        obs=pd.DataFrame(index=["barcode-1", "barcode-2"]),
        obsm={},
    )
    reader_module = ModuleType("spatialdata_io")
    reader_module.visium_hd = lambda **kwargs: SimpleNamespace(tables={"square_008um": adata})
    monkeypatch.setattr(visium_workflow, "preflight", lambda _: preflight_result)
    monkeypatch.setattr(visium_workflow, "_version", lambda _: "test")
    monkeypatch.setitem(sys.modules, "spatialdata_io", reader_module)
    events = []

    result = visium_workflow.execute(request, events.append)

    assert result.status != "FAILED"
    assert any(event.stage_id == "bundle-intake" and event.status == "SUCCEEDED" for event in events)
    assert not any(Path(path).name == "tissue_image_lowres.png" for path in result.artifacts)
    receipt_path = next(Path(path) for path in result.artifacts if Path(path).name == "bundle_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["tissue_lowres_preview"] is None
