from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "apps" / "streamlit" / "app.py"


def _app(tmp_path: Path, page: str = "Home") -> AppTest:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    metadata = project / "project_samples.csv"
    metadata.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\nunit_a,sample_a,control,tissue\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["project_root_path"] = str(project)
    app.session_state["annotation_csv"] = str(metadata)
    app.session_state["output_root"] = str(tmp_path / "outputs")
    app.session_state["spatial_page"] = page
    for modality in ("CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"):
        app.session_state[f"data_mode_{modality}"] = "own"
    return app


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [item.message for item in app.exception]


def _bundle(output_root: Path, modality: str, workflow_id: str, name: str) -> Path:
    run_dir = output_root / modality / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": name, "modality": modality, "workflow_id": workflow_id,
                    "created_at": "2026-01-01T10:00:00+00:00"}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}), encoding="utf-8")
    return run_dir


def test_home_actions_open_project_and_modality_workspaces(tmp_path: Path) -> None:
    app = _app(tmp_path).run()
    _assert_clean(app)
    app.button(key="home_open_project").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Project"
    app.radio(key="spatial_page").set_value("Home").run()
    app.button(key="home_modality_CycIF").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "CycIF"
    assert app.session_state["workspace_section_CycIF"] == "Setup"


def test_modality_settings_survive_navigation_without_cross_contamination(tmp_path: Path) -> None:
    app = _app(tmp_path, "CycIF")
    cycif_raw = tmp_path / "cycif_raw"
    flim_raw = tmp_path / "flim_raw"
    cycif_raw.mkdir()
    flim_raw.mkdir()
    app.session_state["modality_folder_CycIF"] = str(cycif_raw)
    app.session_state["modality_folder_FLIM"] = str(flim_raw)
    app.session_state["modality_output_CycIF"] = str(tmp_path / "cycif_results")
    app.session_state["modality_output_FLIM"] = str(tmp_path / "flim_results")
    app.session_state["workflow_parameters_cycif-raw-czi"] = {
        "intake": {"filename_pattern": "(?P<round>R1)", "expected_channels": 4}
    }
    app.run()
    _assert_clean(app)
    app.radio(key="workspace_section_CycIF").set_value("Results").run()
    app.radio(key="spatial_page").set_value("FLIM").run()
    _assert_clean(app)
    app.radio(key="workspace_section_FLIM").set_value("Run").run()
    app.radio(key="spatial_page").set_value("CycIF").run()
    _assert_clean(app)
    assert app.session_state["workspace_section_CycIF"] == "Results"
    app.radio(key="spatial_page").set_value("FLIM").run()
    _assert_clean(app)
    assert app.radio(key="workspace_section_FLIM").value == "Run"
    assert app.session_state["modality_folder_CycIF"] == str(cycif_raw)
    assert app.session_state["modality_folder_FLIM"] == str(flim_raw)
    assert app.session_state["modality_output_CycIF"] == str(tmp_path / "cycif_results")
    assert app.session_state["modality_output_FLIM"] == str(tmp_path / "flim_results")
    assert app.session_state["workflow_parameters_cycif-raw-czi"]["intake"]["expected_channels"] == 4


def test_results_are_local_to_selected_modality_and_deduplicate_images(tmp_path: Path) -> None:
    from PIL import Image

    output = tmp_path / "outputs"
    cycif_run = _bundle(output, "CycIF", "cycif-raw-czi", "cycif-one")
    flim_run = _bundle(output, "FLIM", "flim.raw_sdt", "flim-one")
    image = cycif_run / "qc-image.png"
    Image.new("RGB", (16, 16), "navy").save(image)
    (cycif_run / "events.jsonl").write_text(
        json.dumps({"stage_id": "intake", "status": "SUCCEEDED", "message": "Image intake complete", "artifacts": [str(image)]}) + "\n",
        encoding="utf-8",
    )
    (cycif_run / "artifacts.json").write_text(
        json.dumps({"artifacts": [{"path": str(image), "kind": "image", "stage_id": "intake"}]}), encoding="utf-8"
    )
    app = _app(tmp_path, "CycIF")
    app.session_state["workspace_section_CycIF"] = "Results"
    app.session_state["workspace_section_FLIM"] = "Results"
    app.session_state["active_run_dirs"] = {"CycIF": str(cycif_run), "FLIM": str(flim_run)}
    app.session_state["selected_run_dirs"] = {"CycIF": str(cycif_run), "FLIM": str(flim_run)}
    app.run()
    _assert_clean(app)
    assert len(app.get("imgs")) == 1
    app.radio(key="spatial_page").set_value("FLIM").run()
    _assert_clean(app)
    assert len(app.get("imgs")) == 0
    assert app.session_state["selected_run_dirs"]["CycIF"] == str(cycif_run)
    assert app.session_state["selected_run_dirs"]["FLIM"] == str(flim_run)
    app.radio(key="spatial_page").set_value("CycIF").run()
    _assert_clean(app)
    assert len(app.get("imgs")) == 1


def test_cleared_optional_calibration_stays_cleared(tmp_path: Path) -> None:
    import hashlib

    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]

    workflow_id = "flim.raw_sdt"
    app = _app(tmp_path, "FLIM")
    app.session_state[f"workflow_parameters_{workflow_id}"] = {
        "reference_calibration": {"path": str(tmp_path / "old-reference.csv")}
    }
    app.run()
    _assert_clean(app)
    widget_key = f"{key(workflow_id)}_parameter_{key(workflow_id, 'reference_calibration.path')}"
    app.text_input(key=widget_key).set_value("").run()
    _assert_clean(app)
    assert not app.session_state[f"workflow_parameters_{workflow_id}"].get("reference_calibration", {}).get("path")
    app.radio(key="workspace_section_FLIM").set_value("Run").run()
    app.radio(key="workspace_section_FLIM").set_value("Setup").run()
    _assert_clean(app)
    assert app.text_input(key=widget_key).value == ""


def test_rejected_saved_run_does_not_render_other_modality_results(tmp_path: Path) -> None:
    from PIL import Image

    output = tmp_path / "outputs"
    wrong_run = _bundle(output, "FLIM", "flim.raw_sdt", "flim-run")
    image = wrong_run / "foreign-image.png"
    Image.new("RGB", (16, 16), "red").save(image)
    (wrong_run / "events.jsonl").write_text(json.dumps({
        "stage_id": "flim_phasors", "status": "SUCCEEDED", "artifacts": [str(image)],
    }) + "\n", encoding="utf-8")
    app = _app(tmp_path, "CycIF")
    app.session_state["workspace_section_CycIF"] = "Results"
    app.session_state["selected_run_dirs"] = {"CycIF": str(wrong_run)}
    app.session_state["active_run_dirs"] = {"CycIF": str(wrong_run)}
    app.run()
    _assert_clean(app)
    assert not app.get("imgs")


def test_project_dialog_cancel_keeps_original_path(tmp_path: Path) -> None:
    app = _app(tmp_path, "Project").run()
    _assert_clean(app)
    original = app.session_state["project_root_path"]
    alternative = tmp_path / "alternative"
    alternative.mkdir()
    app.button(key="browse_project_root").click().run()
    _assert_clean(app)
    app.text_input(key="_server_picker_path_project_root_path").set_value(str(alternative)).run()
    app.button(key="_server_picker_go_project_root_path").click().run()
    _assert_clean(app)
    assert app.session_state["project_root_path"] == original
    app.button(key="_server_picker_cancel_project_root_path").click().run()
    _assert_clean(app)
    assert app.session_state["project_root_path"] == original
    assert not app.session_state["_server_picker_open"]


def test_raw_intake_rounds_can_be_entered_before_first_check(tmp_path: Path) -> None:
    import hashlib

    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]

    raw = tmp_path / "raw" / "slide_A1"
    raw.mkdir(parents=True)
    # Quick discovery reads names only; this test never decodes or analyzes the file.
    (raw / "r1_A1_scene1.czi").write_bytes(b"name-discovery-only")
    workflow_id = "cycif-raw-czi"
    app = _app(tmp_path, "CycIF")
    app.session_state["modality_folder_CycIF"] = str(raw.parent)
    app.session_state[f"workflow_parameters_{workflow_id}"] = {
        "intake": {
            "filename_pattern": r"^(?P<round>r\d+)_(?P<slide>A1)_(?P<scene>scene1)(?P<markers>)(?P<date>)(?P<acq_id>)\.czi$",
            "expected_channels": 4,
        },
        "registration": {"enabled": False},
    }
    app.run()
    _assert_clean(app)
    rounds_key = f"{key(workflow_id)}_parameter_{key(workflow_id, 'intake.expected_rounds')}"
    app.multiselect(key=rounds_key).set_value(["r1"]).run()
    app.button(key="continue_to_run_CycIF").click().run()
    app.button(key="check_inputs_CycIF").click().run()
    _assert_clean(app)
    receipt = app.session_state["preflight_CycIF"]
    assert receipt[1]["resolved_parameters"]["available_rounds"] == ["r1"]
    assert receipt[1]["ready"]
    assert app.session_state["active_run_dirs"] == {}


def test_reviewed_tma_map_populates_exact_groups_and_changes_invalidate_checks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "r1_A1_scene1.czi").write_bytes(b"filename inventory only; no decoding in this test")
    core_map = tmp_path / "core_map.csv"
    core_map.write_text(
        "group_key,specimen_id,slide_id,core_id,section_id\nslide=A1;scene=scene1,sample_a,A1,core_1,section_1\n",
        encoding="utf-8",
    )
    app = _app(tmp_path, "CycIF")
    app.session_state["modality_folder_CycIF"] = str(raw)
    app.session_state["data_layout_CycIF"] = "Tissue microarray (TMA)"
    app.session_state["workflow_parameters_cycif-raw-czi"] = {
        "intake": {
            "filename_pattern": r"^(?P<round>r\d+)_(?P<slide>A1)_(?P<scene>scene1)(?P<markers>)(?P<date>)(?P<acq_id>)\.czi$",
            "expected_rounds": ["r1"], "expected_channels": 4,
        },
        "registration": {"enabled": False},
    }
    app.run()
    _assert_clean(app)
    app.text_input(key="tma_core_map_path_widget_CycIF").set_value(str(core_map)).run()
    _assert_clean(app)
    assert app.session_state["workflow_parameters_valid_cycif-raw-czi"]
    app.button(key="continue_to_run_CycIF").click().run()
    app.button(key="check_inputs_CycIF").click().run()
    _assert_clean(app)
    assert app.session_state["preflight_CycIF"][1]["ready"]
    group_selector = next(widget for widget in app.selectbox if widget.label == "Project specimen")
    assert group_selector.value == "sample_a"
    assert not any(button.key == "submit_CycIF" for button in app.button)
    next(box for box in app.checkbox if box.key and box.key.startswith("specimen_mapping_confirmed_CycIF")).check().run()
    assert app.button(key="submit_CycIF")
    core_map.write_text("group_key,specimen_id\nslide=A1;scene=scene1,UNMAPPED\n", encoding="utf-8")
    app.run()
    _assert_clean(app)
    assert not any(button.key == "submit_CycIF" for button in app.button)
    assert app.session_state["active_run_dirs"] == {}


@pytest.mark.parametrize("contents", [
    "group_key\ncore_a\n",
    "group_key,specimen_id,specimen_id\ncore_a,sample_a,sample_b\n",
    "group_key,specimen_id\ncore_a,absent_from_project\n",
])
def test_invalid_tma_map_blocks_settings_without_crashing(tmp_path: Path, contents: str) -> None:
    app = _app(tmp_path, "H&E")
    app.session_state["data_layout_H&E"] = "Tissue microarray (TMA)"
    app.run()
    _assert_clean(app)
    assert app.session_state["workflow_parameters_valid_he-image-intake"]
    core_map = tmp_path / "invalid-core-map.csv"
    core_map.write_text(contents, encoding="utf-8")
    app.text_input(key="tma_core_map_path_widget_H&E").set_value(str(core_map)).run()
    _assert_clean(app)
    assert not app.session_state["workflow_parameters_valid_he-image-intake"]
    assert app.session_state["active_run_dirs"] == {}
