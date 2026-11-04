from __future__ import annotations

import copy
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
