"""Unit tests for step04_export_registered and step05_segment_cells.

No real gigapixel images or MCMICRO/PIPEX containers are available in this
environment, so:
  - step04_export_registered is tested against small synthetic in-memory
    arrays (I/O + provenance JSON only — no data literals are asserted as
    scientific results, only structural facts about files written).
  - step05_segment_cells is tested via its command-builder functions
    directly (verifying the constructed subprocess command string given
    fake paths) and the bridge_mcmicro_to_pipex column-translation function
    against a small synthetic DataFrame. No container is ever invoked.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.cycif.config import ExportNamingConfig, MCMICROConfig, PIPEXConfig
from spatial_omics_toolkit.cycif.export_images import (
    build_composite_ometiff,
    extract_pipex_channels,
    qc_summary_table,
    save_scene_metadata,
    step04_export_registered,
)
from spatial_omics_toolkit.cycif.segment import (
    bridge_mcmicro_to_pipex,
    build_markers_csv,
    build_mcmicro_cmd,
    build_params_yml,
    build_pipex_segmentation_cmd,
    get_measure_markers,
    prepare_mcmicro_project,
    step05_segment_cells,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_result(h=8, w=8):
    """A minimal synthetic registration result matching step03's contract."""
    rng = np.random.default_rng(0)
    r1_channels = rng.integers(0, 100, size=(2, h, w)).astype(np.uint16)  # DAPI, MarkerA
    r2_channels = rng.integers(0, 100, size=(2, h, w)).astype(np.uint16)  # DAPI, MarkerB
    transform = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -0.5]])
    return {
        "pixel_size_um": 0.325,
        "reference_shape": (h, w),
        "params": {"n_features": 10000, "lowe_ratio": 0.75},
        "round_order": ["R1", "R2"],
        "rounds": {
            "R1": {
                "registered": True,
                "status": "reference",
                "inliers": 0,
                "matches": 0,
                "czi_path": "/fake/R1.czi",
                "czi_size_bytes": 12345,
                "mov_shape": (h, w),
                "transform": None,
                "channels": r1_channels,
                "dapi": r1_channels[0],
            },
            "R2": {
                "registered": True,
                "status": "ok",
                "inliers": 42,
                "matches": 50,
                "czi_path": "/fake/R2.czi",
                "czi_size_bytes": 23456,
                "mov_shape": (h, w),
                "transform": transform,
                "channels": r2_channels,
                "dapi": r2_channels[0],
            },
        },
    }


MARKER_MAP = {"R1": ["DAPI", "MarkerA"], "R2": ["DAPI", "MarkerB"]}


# ---------------------------------------------------------------------------
# step04_export_registered — building blocks
# ---------------------------------------------------------------------------

def test_save_scene_metadata_writes_provenance_json(tmp_path):
    result = _make_fake_result()
    path = save_scene_metadata(result, "sceneA1", tmp_path, extra={"operator": "test-suite"})

    assert Path(path).exists()
    with open(path) as fh:
        record = json.load(fh)

    assert record["scene_label"] == "sceneA1"
    assert record["pixel_size_um"] == pytest.approx(0.325)
    assert record["reference_shape"] == [8, 8]
    assert set(record["rounds"].keys()) == {"R1", "R2"}
    assert record["rounds"]["R2"]["inliers"] == 42
    assert record["rounds"]["R2"]["matches"] == 50
    assert record["rounds"]["R2"]["inlier_rate"] == pytest.approx(42 / 50)
    assert record["rounds"]["R1"]["transform_2x3"] is None
    assert record["rounds"]["R2"]["transform_2x3"] is not None
    assert record["operator"] == "test-suite"
    assert "created_utc" in record
    assert "software" in record and "numpy" in record["software"]


def test_save_scene_metadata_custom_filename_template(tmp_path):
    result = _make_fake_result()
    path = save_scene_metadata(
        result, "sceneA1", tmp_path, metadata_filename_template="{scene_label}_prov.json"
    )
    assert Path(path).name == "sceneA1_prov.json"


def test_build_composite_ometiff_writes_calibrated_stack(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    result = _make_fake_result()
    out_path = tmp_path / "sceneA1_composite.ome.tif"

    channel_names = build_composite_ometiff(result, out_path, MARKER_MAP)

    assert out_path.exists()
    assert channel_names == ["R1_DAPI", "R1_MarkerA", "R2_DAPI", "R2_MarkerB"]

    # Read structural metadata only (page/series shape+dtype), not decoded
    # pixel data: this environment's tifffile/numpy combination has a
    # reshape() incompatibility inside imread()/asarray() on zlib-compressed
    # data (an environment defect unrelated to the code under test).
    with tifffile.TiffFile(str(out_path)) as tif:
        assert len(tif.pages) == 4
        assert tif.series[0].shape == (4, 8, 8)
        assert tif.series[0].dtype == np.uint16


def test_build_composite_ometiff_excludes_failed_rounds(tmp_path):
    pytest.importorskip("tifffile")
    result = _make_fake_result()
    result["rounds"]["R2"]["channels"] = None  # simulate a failed round
    out_path = tmp_path / "sceneA1_composite.ome.tif"

    channel_names = build_composite_ometiff(result, out_path, MARKER_MAP)
    assert channel_names == ["R1_DAPI", "R1_MarkerA"]


def test_build_composite_ometiff_raises_when_all_rounds_failed(tmp_path):
    pytest.importorskip("tifffile")
    result = _make_fake_result()
    result["rounds"]["R1"]["channels"] = None
    result["rounds"]["R2"]["channels"] = None

    with pytest.raises(ValueError):
        build_composite_ometiff(result, tmp_path / "out.ome.tif", MARKER_MAP)


def test_extract_pipex_channels_uses_naming_template(tmp_path):
    pytest.importorskip("tifffile")
    result = _make_fake_result()

    written = extract_pipex_channels(
        result, tmp_path, MARKER_MAP, channel_filename_template="{round}__{marker}.tif"
    )
    names = sorted(Path(p).name for p in written)
    assert names == ["R1__DAPI.tif", "R1__MarkerA.tif", "R2__DAPI.tif", "R2__MarkerB.tif"]


def test_qc_summary_table_reports_registration_quality():
    result = _make_fake_result()
    table = qc_summary_table(result)

    assert list(table["Round"]) == ["R1", "R2"]
    assert table.loc[table["Round"] == "R1", "Status"].iloc[0] == "REFERENCE"
    assert table.loc[table["Round"] == "R2", "Status"].iloc[0] == "OK"
    assert table.loc[table["Round"] == "R2", "Inliers"].iloc[0] == 42


# ---------------------------------------------------------------------------
# step04_export_registered — orchestrator
# ---------------------------------------------------------------------------

def test_step04_export_registered_writes_expected_files(tmp_path):
    pytest.importorskip("tifffile")
    pytest.importorskip("cv2")
    result = _make_fake_result()
    output_dir = tmp_path / "registered"
    pipex_dir = tmp_path / "pipex"

    out = step04_export_registered(
        result,
        "sceneA1",
        output_dir,
        MARKER_MAP,
        pipex_dir=pipex_dir,
        verbose=False,
    )

    assert Path(out["composite_path"]).exists()
    assert Path(out["composite_path"]).name == "sceneA1_composite.ome.tif"
    assert out["channel_names"] == ["R1_DAPI", "R1_MarkerA", "R2_DAPI", "R2_MarkerB"]

    assert Path(out["metadata_path"]).exists()
    assert Path(out["metadata_path"]).parent == output_dir / "sceneA1"

    assert out["mask_paths"] is not None
    for p in out["mask_paths"].values():
        assert Path(p).exists()
    assert "combined" in out["mask_paths"]

    assert out["pipex_channel_paths"] is not None
    assert len(out["pipex_channel_paths"]) == 4
    for p in out["pipex_channel_paths"]:
        assert Path(p).exists()
        assert str(pipex_dir / "sceneA1") in p

    assert isinstance(out["qc_table"], pd.DataFrame)
    # write_qc_plot now defaults to True: every step should leave a human-inspectable
    # artifact behind unless explicitly disabled (see spatial-toolkit-package-design-principles memory).
    assert out["qc_plot_path"] is not None
    assert Path(out["qc_plot_path"]).exists()


def test_step04_export_registered_requires_pipex_dir_when_requested(tmp_path):
    result = _make_fake_result()
    with pytest.raises(ValueError):
        step04_export_registered(
            result, "sceneA1", tmp_path / "out", MARKER_MAP,
            pipex_dir=None, write_pipex_channels=True, verbose=False,
        )


def test_step04_export_registered_can_skip_pipex_and_masks(tmp_path):
    pytest.importorskip("tifffile")
    result = _make_fake_result()

    out = step04_export_registered(
        result, "sceneA1", tmp_path / "out", MARKER_MAP,
        write_pipex_channels=False, write_border_masks=False, verbose=False,
    )
    assert out["pipex_channel_paths"] is None
    assert out["mask_paths"] is None
    assert Path(out["composite_path"]).exists()


def test_step04_export_registered_custom_naming_template(tmp_path):
    pytest.importorskip("tifffile")
    result = _make_fake_result()
    naming = ExportNamingConfig(composite_filename_template="{scene_label}.composite.tif")

    out = step04_export_registered(
        result, "sceneA1", tmp_path / "out", MARKER_MAP,
        naming=naming, write_pipex_channels=False, write_border_masks=False, verbose=False,
    )
    assert Path(out["composite_path"]).name == "sceneA1.composite.tif"


# ---------------------------------------------------------------------------
# step05_segment_cells — MCMICRO command builders
# ---------------------------------------------------------------------------

def test_build_markers_csv_orders_rounds_numerically():
    marker_map = {"R2": ["DAPI", "CK5"], "R10": ["DAPI"], "R1": ["DAPI", "MUC1"]}
    csv_text = build_markers_csv(marker_map)
    lines = csv_text.splitlines()
    assert lines[0] == "cycle,marker_name"
    # Rounds sort numerically (R1 < R2 < R10, not alphabetically), and every
    # marker within a round shares that round's cycle number.
    assert lines[1:3] == ["1,R1_DAPI", "1,R1_MUC1"]
    assert lines[3:5] == ["2,R2_DAPI", "2,R2_CK5"]
    assert lines[5] == "3,R10_DAPI"


def test_build_params_yml_contains_cellpose_options():
    config = MCMICROConfig(
        mcmicro_bin="/fake/bin/mcmicro",
        mcmicro_nf="/fake/mcmicro/main.nf",
        dapi_channel=1,
        cellpose_model="cyto3",
        cellpose_expansion=10,
    )
    yml = build_params_yml(config)
    assert "start-at: segmentation" in yml
    assert "stop-at: quantification" in yml
    assert "--channel 1" in yml
    assert "--pretrained_model cyto3" in yml
    assert "--expand-size 10" in yml


def test_build_mcmicro_cmd_uses_config_paths_and_profile():
    config = MCMICROConfig(
        mcmicro_bin="/fake/bin/mcmicro",
        mcmicro_nf="/fake/mcmicro/main.nf",
        profile="exacloud",
        use_gpu=False,
    )
    cmd = build_mcmicro_cmd("/fake/project_dir", config)
    assert cmd == (
        "/fake/bin/mcmicro run /fake/mcmicro/main.nf "
        "--in /fake/project_dir -profile exacloud"
    )


def test_build_mcmicro_cmd_gpu_profile_and_extra_args():
    config = MCMICROConfig(
        mcmicro_bin="/fake/bin/mcmicro",
        mcmicro_nf="/fake/mcmicro/main.nf",
        profile="exacloud",
        use_gpu=True,
        extra_args="-resume",
    )
    cmd = build_mcmicro_cmd("/fake/project_dir", config)
    assert "-profile exacloud,gpu" in cmd
    assert cmd.endswith("-resume")


def test_prepare_mcmicro_project_creates_expected_layout(tmp_path):
    composite = tmp_path / "sceneA1_composite.ome.tif"
    composite.write_bytes(b"fake-tiff-bytes")
    config = MCMICROConfig(mcmicro_bin="/fake/bin/mcmicro", mcmicro_nf="/fake/main.nf")
    project_dir = tmp_path / "mcmicro_project"

    result_dir = prepare_mcmicro_project(
        "sceneA1", composite, MARKER_MAP, project_dir, config
    )

    assert result_dir == project_dir
    assert (project_dir / "markers.csv").exists()
    assert (project_dir / "params.yml").exists()
    link = project_dir / "registration" / "sceneA1.ome.tif"
    assert link.is_symlink()
    assert link.resolve() == composite.resolve()


# ---------------------------------------------------------------------------
# step05_segment_cells — PIPEX command builders
# ---------------------------------------------------------------------------

def test_get_measure_markers_excludes_dapi_and_excluded_rounds():
    marker_map = {"R1": ["DAPI", "MUC1", "MUC5AC"], "R2": ["DAPI", "CK5"], "R5Q": ["DAPI"]}
    markers = get_measure_markers(marker_map, exclude_rounds=["R5Q"])
    assert markers == ["R1_MUC1", "R1_MUC5AC", "R2_CK5"]


def test_build_pipex_segmentation_cmd_contains_all_flags():
    config = PIPEXConfig(
        pipex_home="/fake/pipex",
        nuclei_marker="R1_DAPI",
        nuclei_diameter=30,
        nuclei_expansion=10,
        membrane_markers=("R2_CK5",),
    )
    cmd = build_pipex_segmentation_cmd("/fake/data_dir", config, measure_markers=["R1_MUC1", "R2_CK5"])

    assert "segmentation.py" in cmd
    assert "-data=/fake/data_dir" in cmd
    assert "-nuclei_marker=R1_DAPI" in cmd
    assert "-nuclei_diameter=30" in cmd
    assert "-nuclei_expansion=10" in cmd
    assert "-membrane_markers=R2_CK5" in cmd
    assert "-measure_markers=R1_MUC1,R2_CK5" in cmd


def test_build_pipex_segmentation_cmd_falls_back_to_config_measure_markers():
    config = PIPEXConfig(
        pipex_home="/fake/pipex",
        nuclei_marker="R1_DAPI",
        measure_markers=("R1_MUC1",),
    )
    cmd = build_pipex_segmentation_cmd("/fake/data_dir", config)
    assert "-measure_markers=R1_MUC1" in cmd


# ---------------------------------------------------------------------------
# bridge_mcmicro_to_pipex
# ---------------------------------------------------------------------------

def test_bridge_mcmicro_to_pipex_renames_columns_and_keeps_markers(tmp_path):
    mcmicro_df = pd.DataFrame(
        {
            "CellID": [1, 2, 3],
            "X_centroid": [10.5, 20.1, 30.9],
            "Y_centroid": [5.0, 6.0, 7.0],
            "Area": [100, 150, 200],
            "Eccentricity": [0.1, 0.2, 0.3],
            "Solidity": [0.9, 0.85, 0.95],
            "R1_MUC1": [1.0, 2.0, 3.0],
            "R2_CK5": [4.0, 5.0, 6.0],
            "R1_DAPI": [999.0, 999.0, 999.0],  # not in analysis_markers -> dropped
        }
    )
    pipex_dir = tmp_path / "pipex_scene"

    out_path = bridge_mcmicro_to_pipex(mcmicro_df, pipex_dir, ["R1_MUC1", "R2_CK5"])

    assert Path(out_path) == pipex_dir / "analysis" / "cell_data.csv"
    assert Path(out_path).exists()

    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == [
        "cell_id", "size", "x", "y", "solidity", "eccentricity", "memref",
        "R1_MUC1", "R2_CK5",
    ]
    assert (out_df["memref"] == 0).all()
    assert out_df["x"].tolist() == [10.5, 20.1, 30.9]
    assert "R1_DAPI" not in out_df.columns


def test_bridge_mcmicro_to_pipex_raises_without_centroid_columns(tmp_path):
    mcmicro_df = pd.DataFrame({"CellID": [1, 2], "R1_MUC1": [1.0, 2.0]})
    with pytest.raises(ValueError):
        bridge_mcmicro_to_pipex(mcmicro_df, tmp_path / "pipex", ["R1_MUC1"])


def test_bridge_mcmicro_to_pipex_handles_alternate_centroid_names(tmp_path):
    mcmicro_df = pd.DataFrame(
        {
            "CellID": [1],
            "column_centroid": [11.0],
            "row_centroid": [22.0],
            "R1_MUC1": [7.0],
        }
    )
    out_path = bridge_mcmicro_to_pipex(mcmicro_df, tmp_path / "pipex", ["R1_MUC1"])
    out_df = pd.read_csv(out_path)
    assert out_df["x"].iloc[0] == 11.0
    assert out_df["y"].iloc[0] == 22.0


# ---------------------------------------------------------------------------
# step05_segment_cells orchestrator — dry run only (no container invoked)
# ---------------------------------------------------------------------------

def test_step05_segment_cells_mcmicro_dry_run_builds_cmd_without_running(tmp_path):
    composite = tmp_path / "sceneA1_composite.ome.tif"
    composite.write_bytes(b"fake-tiff-bytes")
    config = MCMICROConfig(mcmicro_bin="/fake/bin/mcmicro", mcmicro_nf="/fake/main.nf")
    project_dir = tmp_path / "mcmicro_project"

    out = step05_segment_cells(
        "mcmicro",
        scene_label="sceneA1",
        composite_ometiff_path=composite,
        marker_map=MARKER_MAP,
        project_dir=project_dir,
        mcmicro_config=config,
        run=False,
        verbose=False,
    )

    assert out["backend"] == "mcmicro"
    assert out["returncode"] is None
    assert out["cells"] is None
    assert "mcmicro run" in out["cmd"]
    assert str(project_dir) in out["cmd"]
    # Project should still have been prepared (I/O side, not a container run)
    assert (project_dir / "markers.csv").exists()
    assert (project_dir / "registration" / "sceneA1.ome.tif").is_symlink()


def test_step05_segment_cells_pipex_dry_run_builds_cmd_without_running(tmp_path):
    config = PIPEXConfig(pipex_home="/fake/pipex", nuclei_marker="R1_DAPI")
    data_dir = tmp_path / "pipex_data"

    out = step05_segment_cells(
        "pipex",
        data_dir=data_dir,
        pipex_config=config,
        marker_map=MARKER_MAP,
        run=False,
        verbose=False,
    )

    assert out["backend"] == "pipex"
    assert out["returncode"] is None
    assert out["cells"] is None
    assert "segmentation.py" in out["cmd"]
    assert "-nuclei_marker=R1_DAPI" in out["cmd"]
    # marker_map was supplied but pipex_config.measure_markers was None,
    # so measure markers should have been auto-derived (non-DAPI channels)
    assert "-measure_markers=R1_MarkerA,R2_MarkerB" in out["cmd"]


def test_step05_segment_cells_unknown_backend_raises():
    with pytest.raises(ValueError):
        step05_segment_cells("unknown_backend", run=False, verbose=False)


def test_step05_segment_cells_missing_required_args_raises_for_mcmicro():
    with pytest.raises(ValueError):
        step05_segment_cells("mcmicro", run=False, verbose=False)


def test_step05_segment_cells_missing_required_args_raises_for_pipex():
    with pytest.raises(ValueError):
        step05_segment_cells("pipex", run=False, verbose=False)


def test_step05_segment_cells_bridge_flag_rejected_for_pipex_backend(tmp_path):
    config = PIPEXConfig(pipex_home="/fake/pipex", nuclei_marker="R1_DAPI")
    with pytest.raises(ValueError):
        step05_segment_cells(
            "pipex",
            data_dir=tmp_path / "data",
            pipex_config=config,
            bridge_to_pipex_format=True,
            run=False,
            verbose=False,
        )


def test_step05_segment_cells_raises_on_nonzero_exit(tmp_path):
    """A failing subprocess must raise RuntimeError, never be swallowed."""
    composite = tmp_path / "sceneA1_composite.ome.tif"
    composite.write_bytes(b"fake-tiff-bytes")
    # mcmicro_bin is a shell command guaranteed to fail (exit 1), standing in
    # for a real container invocation without needing MCMICRO installed.
    config = MCMICROConfig(mcmicro_bin="false;", mcmicro_nf="/fake/main.nf")
    project_dir = tmp_path / "mcmicro_project"

    with pytest.raises(RuntimeError):
        step05_segment_cells(
            "mcmicro",
            scene_label="sceneA1",
            composite_ometiff_path=composite,
            marker_map=MARKER_MAP,
            project_dir=project_dir,
            mcmicro_config=config,
            run=True,
            verbose=False,
        )
