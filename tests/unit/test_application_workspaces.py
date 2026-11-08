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


def _click_label(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run()
    _assert_clean(app)


def _open_modality(app: AppTest, modality: str) -> None:
    _click_label(app, "Home")
    key = "home_integration" if modality == "Integration" else f"home_modality_{modality}"
    app.button(key=key).click().run()
    _assert_clean(app)


def _bundle(output_root: Path, modality: str, workflow_id: str, name: str) -> Path:
    run_dir = output_root / modality / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": name, "modality": modality, "workflow_id": workflow_id,
                    "created_at": "2026-01-01T10:00:00+00:00"}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("modality", ["CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"])
def test_home_actions_open_project_and_modality_workspaces(tmp_path: Path, modality: str) -> None:
    app = _app(tmp_path).run()
    _assert_clean(app)
    app.button(key="home_open_project").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Project"
    _open_modality(app, modality)
    assert app.session_state["spatial_page"] == modality
    assert app.session_state[f"workspace_section_{modality}"] == "Introduction"
    _click_label(app, "Project")
    assert app.session_state["spatial_page"] == "Project"
    _click_label(app, "Home")
    assert app.session_state["spatial_page"] == "Home"




def test_home_exposes_all_destinations_and_current_navigation(tmp_path: Path) -> None:
    app = _app(tmp_path).run()
    _assert_clean(app)
    for modality in ("CycIF", "HS", "FLIM", "H&E", "Visium"):
        assert app.button(key=f"home_modality_{modality}")
    assert app.button(key="home_integration")
    app.button(key="home_integration").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Integration"
    app.button(key="navigate_project").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Project"
    app.button(key="navigate_home").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Home"


@pytest.mark.parametrize("settings", [None, {}, {"mode": "hpc"}, {"datasets_url": "https://drive.google.com/drive/folders/test-folder"}])
def test_full_workspace_without_slurm_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, str] | None
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("shutil.which", lambda executable: None)
    app = _app(tmp_path)
    if settings is not None:
        app.secrets["toolkit"] = settings
    app.run()
    _assert_clean(app)
    assert any("sbatch was not found" in item.value for item in app.info)
    if settings and "datasets_url" in settings:
        assert any(settings["datasets_url"] in item.value for item in app.markdown)
    app.button(key="home_open_project").click().run()
    _assert_clean(app)
    assert app.text_input(key="project_root_widget")
    _open_modality(app, "CycIF")
    section = app.radio(key="workspace_section_CycIF")
    assert section.options == ["Introduction", "Data & settings", "Run", "Quality control", "Results", "Files"]
    for value in ("Setup", "Run", "Quality control", "Results", "Files", "Introduction"):
        app.radio(key="workspace_section_CycIF").set_value(value).run()
        _assert_clean(app)
        assert app.radio(key="workspace_section_CycIF").value == value
    assert not any("view-only" in item.value for item in app.info)
    assert not app.error


def test_read_only_home_shows_only_configured_google_drive_url(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.secrets["toolkit"] = {
        "mode": "read_only",
        "datasets_url": "https://drive.google.com/drive/folders/test-folder",
    }
    app.run()
    _assert_clean(app)
    drive_link = "https://drive.google.com/drive/folders/test-folder"
    rendered_links = [
        item.value for item in app.markdown if '<a class="dataset-link"' in item.value
    ]
    assert len(rendered_links) == 1
    assert drive_link in rendered_links[0]

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid = _app(invalid_root)
    invalid.secrets["toolkit"] = {
        "mode": "read_only",
        "datasets_url": "https://example.org/datasets",
    }
    invalid.run()
    _assert_clean(invalid)
    assert not any('<a class="dataset-link"' in item.value for item in invalid.markdown)


@pytest.mark.parametrize("modality", ["CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"])
def test_read_only_navigation_exposes_introductions_but_no_execution_controls(
    tmp_path: Path, modality: str
) -> None:
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    app = _app(tmp_path)
    app.secrets["toolkit"] = {"mode": "read_only"}
    app.session_state["_server_picker_open"] = True
    app.session_state[f"workspace_section_{modality}"] = "Run"
    app.run()
    _assert_clean(app)
    assert any("view-only" in item.value for item in app.info)
    assert not app.text_input
    app.button(key="home_open_project").click().run()
    _assert_clean(app)
    assert app.session_state["spatial_page"] == "Project"
    assert not app.text_input
    assert not app.selectbox
    _open_modality(app, modality)
    assert any(item.title in [header.value for header in app.subheader]
               for item in list_workflows() if item.modality == modality)
    assert not app.radio
    assert not app.text_input
    assert not app.checkbox
    if modality == "CycIF":
        alternatives = [item for item in list_workflows() if item.modality == modality]
        app.selectbox[0].set_value(alternatives[1].title).run()
        _assert_clean(app)
        assert alternatives[1].title in [header.value for header in app.subheader]


@pytest.mark.parametrize("settings", [{"mode": "invalid"}, {"mode": None}, "invalid"])
def test_invalid_cloud_mode_reports_error_without_hiding_workspace(tmp_path: Path, settings: Any) -> None:
    app = _app(tmp_path, "Project")
    app.secrets["toolkit"] = settings
    app.run()
    _assert_clean(app)
    assert any("Invalid [toolkit]" in item.value for item in app.error)
    assert app.text_input(key="project_root_widget")
    assert not any("view-only" in item.value for item in app.info)




@pytest.mark.parametrize("modality", ["CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"])
def test_introduction_does_not_submit_example_and_setup_state_survives_return(
    tmp_path: Path, modality: str
) -> None:
    app = _app(tmp_path, modality)
    app.session_state[f"data_mode_{modality}"] = "example"
    app.run()
    _assert_clean(app)
    assert app.radio(key=f"workspace_section_{modality}").value == "Introduction"
    assert app.session_state["active_run_dirs"] == {}
    assert not (tmp_path / "outputs").exists()
    assert all(button.key != f"run_example_setup_{modality}" for button in app.button)

    app.radio(key=f"data_mode_{modality}").set_value("own").run()
    _assert_clean(app)
    app.radio(key=f"workspace_section_{modality}").set_value("Setup").run()
    _assert_clean(app)
    app.selectbox(key=f"data_layout_{modality}").set_value("Multiple samples").run()
    _assert_clean(app)
    app.radio(key=f"workspace_section_{modality}").set_value("Introduction").run()
    _assert_clean(app)
    app.radio(key=f"workspace_section_{modality}").set_value("Setup").run()
    _assert_clean(app)
    assert app.selectbox(key=f"data_layout_{modality}").value == "Multiple samples"


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
    _open_modality(app, "FLIM")
    _assert_clean(app)
    app.radio(key="workspace_section_FLIM").set_value("Run").run()
    _open_modality(app, "CycIF")
    _assert_clean(app)
    assert app.session_state["workspace_section_CycIF"] == "Results"
    _open_modality(app, "FLIM")
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
    _open_modality(app, "FLIM")
    _assert_clean(app)
    assert len(app.get("imgs")) == 0
    assert app.session_state["selected_run_dirs"]["CycIF"] == str(cycif_run)
    assert app.session_state["selected_run_dirs"]["FLIM"] == str(flim_run)
    _open_modality(app, "CycIF")
    _assert_clean(app)
    assert len(app.get("imgs")) == 1


def test_cleared_optional_calibration_stays_cleared(tmp_path: Path) -> None:
    import hashlib

    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]

    workflow_id = "flim.raw_sdt"
    app = _app(tmp_path, "FLIM")
    app.session_state["workspace_section_FLIM"] = "Setup"
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
    app.session_state["workspace_section_CycIF"] = "Setup"
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
    app.session_state["workspace_section_CycIF"] = "Setup"
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
    app.session_state["workspace_section_H&E"] = "Setup"
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


def test_applied_profile_updates_visible_and_submitted_parameters_and_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spatial_omics_toolkit.application import settings

    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    app = _app(tmp_path, "H&E")
    app.session_state["workspace_section_H&E"] = "Setup"
    settings.save_user_settings(
        tmp_path / "project", "reviewed-preview.toml",
        'workflow_id = "he-image-intake"\n[parameters]\n"preview_max_dimension" = 1024\n',
        create=True,
    )
    app.run()
    _assert_clean(app)
    next(widget for widget in app.selectbox if widget.label == "User settings file").set_value("reviewed-preview.toml").run()
    next(button for button in app.button if button.label == "Apply saved settings").click().run()
    _assert_clean(app)
    preview = next(widget for widget in app.number_input if widget.label == "Maximum preview dimension (pixels)")
    assert preview.value == 1024
    assert app.session_state["workflow_parameters_he-image-intake"]["preview_max_dimension"] == 1024
    app.radio(key="workspace_section_H&E").set_value("Introduction").run()
    app.radio(key="workspace_section_H&E").set_value("Setup").run()
    _assert_clean(app)
    assert next(widget for widget in app.number_input if widget.label == "Maximum preview dimension (pixels)").value == 1024
    next(button for button in app.button if button.label == "Reset to defaults").click().run()
    _assert_clean(app)
    assert next(widget for widget in app.number_input if widget.label == "Maximum preview dimension (pixels)").value == 2048
    assert app.session_state["workflow_parameters_he-image-intake"]["preview_max_dimension"] == 2048
