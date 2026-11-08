from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from spatial_omics_toolkit.application import examples
from spatial_omics_toolkit.application.examples import ExampleSpec

APP = Path(__file__).resolve().parents[2] / "apps" / "streamlit" / "app.py"


@pytest.mark.parametrize(("modality", "workflow_id"), [
    ("CycIF", "cycif-raw-czi"), ("HS", "hs.native_spectral_qc"),
    ("FLIM", "flim.raw_sdt"), ("H&E", "he-image-intake"),
    ("Visium", "visium-native-qc"), ("Integration", "integration.serial_section_alignment"),
])
def test_examples_default_without_overwriting_own_project_or_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modality: str, workflow_id: str,
) -> None:
    example_project = tmp_path / "example"
    example_project.mkdir()
    example = ExampleSpec(
        example_id=f"{modality}.test", modality=modality, workflow_id=workflow_id,
        title="File-backed example", summary="Technical quality-control example",
        layout="Single source", project_root=example_project, modality_dir=example_project / "data",
        annotation_csv=example_project / "project_samples.csv", parameters={},
        input_help="Choose the provided source file.", expected_outputs=("QC report",),
        limitations=("No biological conclusions.",), source_paths=(),
    )
    monkeypatch.setattr(examples, "get_example", lambda selected_modality: example)
    own_project = tmp_path / "own"
    own_project.mkdir()
    own_input = own_project / "raw"
    own_input.mkdir()
    metadata = own_project / "project_samples.csv"
    metadata.write_text("biological_unit_id,specimen_id,condition,tissue_source\nunit,specimen,control,tissue\n", encoding="utf-8")
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["spatial_page"] = modality
    app.session_state["project_root_path"] = str(own_project)
    app.session_state["annotation_csv"] = str(metadata)
    app.session_state[f"modality_folder_{modality}"] = str(own_input)
    app.session_state[f"modality_output_{modality}"] = str(own_project / "results")
    app.session_state[f"workflow_parameters_{workflow_id}"] = {"user_setting_to_preserve": "unchanged"}
    app.session_state["active_run_dirs"] = {modality: str(own_project / "prior-run")}
    app.session_state[f"workspace_section_{modality}"] = "Setup"
    app.session_state["selected_run_dirs"] = {modality: str(own_project / "prior-run")}
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key=f"data_mode_{modality}").value == "example"
    assert app.button(key=f"run_example_setup_{modality}").disabled is False
    assert app.session_state["project_root_path"] == str(own_project)
    assert app.session_state["annotation_csv"] == str(metadata)
    assert app.session_state[f"modality_folder_{modality}"] == str(own_input)
    assert app.session_state[f"workflow_parameters_{workflow_id}"] == {"user_setting_to_preserve": "unchanged"}
    app.radio(key=f"data_mode_{modality}").set_value("own").run()
    assert not app.exception, [item.message for item in app.exception]
    before = copy.deepcopy(app.session_state[f"workflow_parameters_{workflow_id}"])
    app.radio(key=f"data_mode_{modality}").set_value("example").run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.session_state[f"workflow_parameters_{workflow_id}"] == before
    assert app.session_state["active_run_dirs"][modality] == str(own_project / "prior-run")
    assert app.session_state["selected_run_dirs"][modality] == str(own_project / "prior-run")


@pytest.mark.parametrize("modality", ["CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"])
def test_missing_catalog_defaults_to_own_data_and_keeps_workspace_accessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modality: str,
) -> None:
    monkeypatch.setenv("SPATIAL_TOOLKIT_EXAMPLES", str(tmp_path / "missing_examples"))
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["spatial_page"] = modality
    app.session_state["output_root"] = str(tmp_path / "outputs")
    app.run()

    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key=f"data_mode_{modality}").value == "own"
    assert any("Example unavailable" in item.value and "own-data controls remain available" in item.value
               for item in app.info)
    for section in ("Introduction", "Setup", "Run", "Quality control", "Results", "Files"):
        app.radio(key=f"workspace_section_{modality}").set_value(section).run()
        assert not app.exception, [item.message for item in app.exception]
        assert app.radio(key=f"workspace_section_{modality}").value == section
        assert app.radio(key=f"data_mode_{modality}").value == "own"
    assert app.session_state["active_run_dirs"] == {}
    assert not (tmp_path / "outputs").exists()

    app.radio(key=f"data_mode_{modality}").set_value("example").run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key=f"data_mode_{modality}").value == "example"
    assert any("Example unavailable" in item.value for item in app.error)
    app.radio(key=f"data_mode_{modality}").set_value("own").run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key=f"workspace_section_{modality}").value == "Files"


def test_real_prepared_catalog_defaults_to_example_and_preserves_own_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "examples"
    quickstart = root / "quickstart"
    quickstart.mkdir(parents=True)
    entry = {
        "folder": "he", "example_id": "he.image_intake", "modality": "H&E",
        "workflow_id": "he-image-intake", "title": "H&E example",
        "summary": "Technical QC", "layout": "Single source", "parameters": {},
        "input_help": "Native image", "expected_outputs": ["QC"], "limitations": ["Technical QC only"],
    }
    (quickstart / "catalog.json").write_text(
        json.dumps({"examples": {"H&E": entry}}), encoding="utf-8",
    )
    monkeypatch.setenv("SPATIAL_TOOLKIT_EXAMPLES", str(root))
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["spatial_page"] = "H&E"
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key="data_mode_H&E").value == "example"
    app.radio(key="data_mode_H&E").set_value("own").run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key="data_mode_H&E").value == "own"
    (quickstart / "catalog.json").unlink()
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key="data_mode_H&E").value == "own"
    assert any("Example unavailable" in item.value for item in app.info)


def test_catalog_without_modality_defaults_to_own_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "examples"
    quickstart = root / "quickstart"
    quickstart.mkdir(parents=True)
    (quickstart / "catalog.json").write_text(json.dumps({"examples": {}}), encoding="utf-8")
    monkeypatch.setenv("SPATIAL_TOOLKIT_EXAMPLES", str(root))
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["spatial_page"] = "HS"
    app.run()
    assert not app.exception, [item.message for item in app.exception]
    assert app.radio(key="data_mode_HS").value == "own"
    assert any("No prepared example for modality" in item.value for item in app.info)
