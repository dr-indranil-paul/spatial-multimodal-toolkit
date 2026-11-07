from __future__ import annotations

import json
from html import escape, unescape
from html.parser import HTMLParser
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
    assert app.session_state[f"workspace_section_{modality}"] == "Setup"
    _click_label(app, "Project")
    assert app.session_state["spatial_page"] == "Project"
    _click_label(app, "Home")
    assert app.session_state["spatial_page"] == "Home"


def test_project_setup_groups_controls_and_review_status(tmp_path: Path) -> None:
    app = _app(tmp_path, "Project").run()
    _assert_clean(app)
    assert [item.value for item in app.subheader] == [
        "1. Choose the study folder",
        "2. Add the sample sheet",
        "3. Choose where results are saved",
    ]
    assert {item.label for item in app.text_input} >= {
        "Project root directory", "Project sample metadata (CSV)", "Default output root",
    }
    assert app.button(key="browse_project_root")
    assert app.button(key="browse_project_metadata")
    assert any("Sample-sheet structure checked: 1 row" in item.value for item in app.success)
    assert app.dataframe


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
    assert section.options == ["Data & settings", "Run", "Quality control", "Results", "Files", "Introduction"]
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
def test_introduction_describes_selected_workflow(tmp_path: Path, modality: str) -> None:
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    descriptor = next(item for item in list_workflows() if item.modality == modality)
    app = _app(tmp_path, modality)
    app.session_state[f"workspace_section_{modality}"] = "Introduction"
    app.run()
    _assert_clean(app)
    assert app.radio(key=f"workspace_section_{modality}").value == "Introduction"
    assert descriptor.title in [item.value for item in app.subheader]
    rendered = "\n".join(
        str(item.value)
        for collection in (app.markdown, app.caption, app.info, app.warning)
        for item in collection
    )
    rendered = unescape(rendered)
    assert descriptor.description in rendered
    assert all(stage.title in rendered and stage.purpose in rendered for stage in descriptor.stages)
    assert all(tool.name in rendered for stage in descriptor.stages for tool in stage.tools)
    diagram = next(item.value for item in app.markdown if '<section class="stage-flow"' in item.value)
    assert diagram.count('class="stage-detail"') == len(descriptor.stages)
    for stage in descriptor.stages:
        assert escape(stage.title) in diagram
        assert escape(stage.purpose) in diagram
        assert all(escape(tool.name) in diagram for tool in stage.tools)
    limitations = next(item.value for item in app.markdown if item.value.startswith("- " + descriptor.limitations[0]))
    assert limitations.splitlines() == [f"- {limitation}" for limitation in descriptor.limitations]
    assert not app.warning


@pytest.mark.parametrize(("modality", "workflow_id", "input_facts", "output_facts"), [
    ("CycIF", "cycif-raw-czi", ("immediate subfolder", "sidecars", "expected-round"), ("intake_inventory.csv", "registration", "enabled and successful")),
    ("CycIF", "cycif.tabular_analysis", ("already-segmented", "coordinate", "specimen-id"), ("selected stages", "spatial-null", "exploratory")),
    ("HS", "hs.native_spectral_qc", ("one file is processed per run", "five uniformly spaced", "unmapped"), ("spectral-band audit", "technical qc", "not molecular identity")),
    ("HS", "hs.clinical-roi-hyperspectral", ("manifest", "18 acquisitions", "tma1 and tma2"), ("complete confirmed specimen mappings", "separate tma1 and tma2", "exploratory")),
    ("FLIM", "flim.raw_sdt", ("`.sdt`", "block", "raw qc"), ("qc_pending", "raw qc", "calibration")),
    ("H&E", "he-image-intake", ("format-specific reader", "no image-to-specimen mapping", ".tiff"), ("missing metadata", "tiff has metadata only", "preview")),
    ("Visium", "visium-native-qc", ("filtered_feature_bc_matrix.h5", "2, 8, or 16", "mapping"), ("spot_bin_qc.csv", "only when usable coordinates", "unnormalized")),
    ("Integration", "integration.serial_section_alignment", ("reference channel index", "integration:moving_he", "integration:reference"), ("translation", "visual review", "not accepted")),
])
def test_introduction_input_output_bullets_cover_workflow_facts(
    tmp_path: Path, modality: str, workflow_id: str,
    input_facts: tuple[str, ...], output_facts: tuple[str, ...],
) -> None:
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    descriptors = [item for item in list_workflows() if item.modality == modality]
    selected = next(item for item in descriptors if item.workflow_id == workflow_id)
    app = _app(tmp_path, modality)
    app.session_state[f"workspace_section_{modality}"] = "Introduction"
    app.run()
    _assert_clean(app)
    if len(descriptors) > 1:
        app.selectbox[0].set_value(selected.title).run()
        _assert_clean(app)
    blocks = [str(item.value) for item in app.markdown]
    for facts in (input_facts, output_facts):
        block = next(value for value in blocks if all(fact in value.lower() for fact in facts))
        lines = block.strip().splitlines()
        assert len(lines) >= 2
        assert all(line.startswith("- ") for line in lines)
    shared = next(value for value in blocks if "project_samples.csv" in value and "1–512 GB" in value)
    for fact in ("tissue_source", "biological_unit_id", "specimen_id"):
        assert fact in shared
    if workflow_id == "cycif.tabular_analysis":
        assert "existing specimen-ID column" in shared
    elif workflow_id == "he-image-intake":
        assert "mapping is not required" in shared
    else:
        assert "UNMAPPED" in shared
    assert any("Included examples use their own fixed inputs" in item.value for item in app.markdown)
    assert any("Inputs and settings determine which paths run" in item.value for item in app.markdown)
    assert not any("Inputs and settings determine which paths run" in item.value for item in app.caption)


class _GraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.edges: set[tuple[str, str]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li" and "data-source" in attributes and "data-target" in attributes:
            self.edges.add((attributes["data-source"], attributes["data-target"]))


_EXPECTED_PATHS = {
    "cycif-raw-czi": {
        ("Intake", "Conversion"), ("Intake", "Registration"),
        ("Registration", "Marker export"), ("Marker export", "Segmentation"),
    },
    "cycif.tabular_analysis": {
        ("Selected cell table", "Intensity QC"), ("Intensity QC", "Cell classification"),
        ("Cell classification", "Program scoring"), ("Program scoring", "Current cell frame"),
        ("Current cell frame", "Aggregation"), ("Current cell frame", "Spatial features"),
        ("Current cell frame", "Spatial null"), ("Aggregation", "Group comparison"),
    },
    "hs.clinical-roi-hyperspectral": {
        ("Metadata audit", "Specimen mapping"), ("Specimen mapping", "TMA1 phasors"),
        ("Specimen mapping", "TMA2 phasors"), ("TMA1 phasors", "TMA1 filter"),
        ("TMA1 phasors", "TMA1 GMM"), ("TMA2 phasors", "TMA2 filter"),
        ("TMA2 phasors", "TMA2 GMM"),
    },
    "hs.native_spectral_qc": {
        ("Native reader call", "Inventory and band audit"), ("Native reader call", "Moment maps"),
        ("Moment maps", "Intensity plots"), ("Moment maps", "H1/H2 plots"),
    },
    "flim.raw_sdt": {
        ("Target inventory", "Block review"), ("Block review", "Raw decay QC"),
        ("Raw decay QC", "Raw QC stop"), ("Block review", "Target phasors"),
        ("Target phasors", "Count mask"), ("Count mask", "Phasor filter"),
        ("Measured reference input", "Reference inventory"), ("Reference inventory", "Reference QC"),
        ("Reference inventory", "Reference phasors"), ("Reference phasors", "Reference calibration"),
        ("Count mask", "Reference calibration"), ("Phasor filter", "Reference calibration"),
        ("Reference calibration", "Lifetimes"), ("Count mask", "Lifetimes"),
        ("Phasor filter", "Lifetimes"),
    },
    "he-image-intake": {
        ("File selection", "SVS reader"), ("File selection", "CZI reader"),
        ("File selection", "TIFF reader"), ("SVS reader", "Inventory and bounded preview"),
        ("CZI reader", "Inventory and bounded preview"), ("TIFF reader", "Inventory and bounded preview"),
    },
    "visium-native-qc": {
        ("Bundle intake", "Classic Visium reader"), ("Bundle intake", "HD native bin"),
        ("HD native bin", "HD reader"), ("Classic Visium reader", "Native count QC"),
        ("HD reader", "Native count QC"), ("Native count QC", "Bounded preview"),
    },
    "integration.serial_section_alignment": {
        ("Moving H&E CZI", "Source and channel authorization"),
        ("Reference fluorescence/HS CZI", "Source and channel authorization"),
        ("Source and channel authorization", "Metadata and physical scale"),
        ("Metadata and physical scale", "Bounded previews"),
        ("Bounded previews", "Translation-only registration"),
        ("Translation-only registration", "Metrics and human review"),
    },
}


@pytest.mark.parametrize(("modality", "workflow_id", "note"), [
    ("CycIF", "cycif-raw-czi", "original CZI"),
    ("CycIF", "cycif.tabular_analysis", "current cell table"),
    ("FLIM", "flim.raw_sdt", "QC_PENDING"),
    ("HS", "hs.clinical-roi-hyperspectral", "Incomplete or unconfirmed specimen mapping"),
    ("HS", "hs.native_spectral_qc", "One read produces"),
    ("H&E", "he-image-intake", "format-specific reader"),
    ("Visium", "visium-native-qc", "HD requires an explicit bin size"),
    ("Integration", "integration.serial_section_alignment", "images need source and channel confirmation"),
])
def test_introduction_explains_conditional_workflow_paths(
    tmp_path: Path, modality: str, workflow_id: str, note: str
) -> None:
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    descriptors = [item for item in list_workflows() if item.modality == modality]
    selected = next(item for item in descriptors if item.workflow_id == workflow_id)
    app = _app(tmp_path, modality)
    app.session_state[f"workspace_section_{modality}"] = "Introduction"
    app.run()
    _assert_clean(app)
    if len(descriptors) > 1:
        app.selectbox[0].set_value(selected.title).run()
        _assert_clean(app)
    notes_block = next(item.value for item in app.markdown if item.value.strip().startswith("- "))
    notes_lines = notes_block.strip().splitlines()
    assert note in notes_block
    assert len(notes_lines) > 1
    assert all(line.lstrip().startswith("- ") for line in notes_lines)
    diagram = next(item.value for item in app.markdown if '<section class="stage-flow"' in item.value)
    graph = _GraphParser()
    graph.feed(diagram)
    assert _EXPECTED_PATHS[workflow_id] <= graph.edges
    assert 'aria-hidden="true">→' in diagram
    for stage in selected.stages:
        assert escape(stage.title) in diagram and escape(stage.purpose) in diagram
        for tool in stage.tools:
            assert escape(tool.name) in diagram
            assert escape(tool.role) in diagram
            version = "runtime version recorded in the run record" if tool.name == "MCMICRO" else tool.version
            assert escape(version) in diagram
    forbidden = {
        "cycif-raw-czi": {("Conversion", "Registration"), ("Registration", "Conversion")},
        "cycif.tabular_analysis": {("Spatial features", "Spatial null")},
        "hs.clinical-roi-hyperspectral": {("TMA1 filter", "TMA1 GMM"), ("TMA2 filter", "TMA2 GMM")},
        "hs.native_spectral_qc": {("Inventory and band audit", "Moment maps")},
        "flim.raw_sdt": {("Target phasors", "Reference inventory"), ("Reference QC", "Target phasors")},
        "he-image-intake": {("SVS reader", "CZI reader"), ("CZI reader", "TIFF reader")},
    }
    assert not (forbidden.get(workflow_id, set()) & graph.edges)
    if workflow_id in {"cycif-raw-czi", "cycif.tabular_analysis", "flim.raw_sdt"}:
        assert "Optional" in diagram
    if workflow_id in {"hs.clinical-roi-hyperspectral", "he-image-intake", "visium-native-qc"}:
        assert "alternative" in diagram.lower()
    if workflow_id in {"cycif-raw-czi", "hs.clinical-roi-hyperspectral", "flim.raw_sdt", "integration.serial_section_alignment"}:
        assert "Review required" in diagram
    if workflow_id == "visium-native-qc":
        assert "Selection required" in diagram
    assert '>Gate<' not in diagram
    if workflow_id == "cycif.tabular_analysis":
        assert 'data-node="Current cell frame"' in diagram
        assert "<h3>Current cell table</h3>" in diagram
        assert "<h3>Spatial null model</h3>" in diagram
    if workflow_id == "hs.clinical-roi-hyperspectral":
        assert "<h3>Metadata review</h3>" in diagram
        assert "(filtering or GMM)" in diagram
        assert "Gaussian mixture model (GMM)" in notes_block
    if workflow_id == "flim.raw_sdt":
        assert "<h3>Stop after raw QC</h3>" in diagram
        assert "(raw-QC mode)" in diagram
    if workflow_id == "integration.serial_section_alignment":
        assert "<h3>H&amp;E CZI to align</h3>" in diagram
        assert "(initial checks passed)" in diagram


def test_stage_flow_escapes_descriptor_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.ui import presentation
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    original = next(item for item in list_workflows() if item.workflow_id == "cycif-raw-czi")
    stage = original.stages[0]
    tool = stage.tools[0].model_copy(update={"name": "<tool>", "version": '<v "1">', "role": "<role>"})
    changed = stage.model_copy(update={"title": "<title>", "purpose": "<purpose>", "tools": [tool]})
    descriptor = original.model_copy(update={"stages": [changed, *original.stages[1:]]})
    rendered: list[str] = []
    monkeypatch.setattr(presentation.st, "markdown", lambda text, **kwargs: rendered.append(text))
    monkeypatch.setattr(presentation.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "info", lambda *args, **kwargs: None)
    presentation.render_stage_flow(descriptor)
    diagram = next(item for item in reversed(rendered) if '<section class="stage-flow"' in item)
    for value in ("<title>", "<purpose>", "<tool>", '<v "1">', "<role>"):
        assert escape(value) in diagram
        assert value not in diagram


def test_introduction_tracks_selected_workflow(tmp_path: Path) -> None:
    from spatial_omics_toolkit.application.workflows.registry import list_workflows

    descriptors = [item for item in list_workflows() if item.modality == "CycIF"]
    assert len(descriptors) > 1
    app = _app(tmp_path, "CycIF")
    app.session_state["workspace_section_CycIF"] = "Introduction"
    app.run()
    _assert_clean(app)
    alternative = descriptors[1]
    app.selectbox[0].set_value(alternative.title).run()
    _assert_clean(app)
    assert alternative.title in [item.value for item in app.subheader]
    rendered = unescape("\n".join(str(item.value) for item in app.markdown))
    assert all(stage.title in rendered and stage.purpose in rendered for stage in alternative.stages)
    assert all(tool.name in rendered for stage in alternative.stages for tool in stage.tools)
    assert alternative.input_description in rendered
    assert alternative.output_description in rendered


@pytest.mark.parametrize("modality", ["CycIF", "HS", "FLIM", "H&E", "Visium", "Integration"])
def test_introduction_does_not_submit_example_and_setup_state_survives_return(
    tmp_path: Path, modality: str
) -> None:
    app = _app(tmp_path, modality)
    app.session_state[f"data_mode_{modality}"] = "example"
    app.run()
    _assert_clean(app)
    assert app.radio(key=f"workspace_section_{modality}").value == "Setup"
    assert app.session_state["active_run_dirs"] == {}
    assert not (tmp_path / "outputs").exists()
    app.radio(key=f"workspace_section_{modality}").set_value("Introduction").run()
    _assert_clean(app)
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
