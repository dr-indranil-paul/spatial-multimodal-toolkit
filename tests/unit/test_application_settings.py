from __future__ import annotations

from pathlib import Path
import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from spatial_omics_toolkit.application.contracts import ParameterField, PreflightResult, RunRequest, WorkflowDescriptor
from spatial_omics_toolkit.application import service, settings


def descriptor() -> WorkflowDescriptor:
    return WorkflowDescriptor(
        workflow_id="test.workflow", modality="Test", title="Test", description="Test",
        stages=[], input_description="", output_description="", parameter_fields=[
            ParameterField(name="analysis.threshold", label="Threshold", kind="float", default=0.5, min_value=0, max_value=1),
            ParameterField(name="analysis.enabled", label="Enabled", kind="bool", default=False),
            ParameterField(name="analysis.method", label="Method", kind="select", choices=["a", "b"], default="a"),
            ParameterField(name="input_path", label="Input", kind="path"),
            ParameterField(name="config", label="Config", kind="config_file"),
            ParameterField(name="mapping_confirmed", label="Mapping review", kind="bool", default=False),
            ParameterField(name="reviewed_image_path", label="Reviewed image", kind="path", review_required=True),
            ParameterField(name="analysis_reviewed", label="Review decision", kind="bool", review_required=True),
        ],
    )


def test_defaults_are_generated_from_descriptor() -> None:
    fields = descriptor()
    values = settings.parse_settings(fields, settings.default_settings(fields))
    assert values["analysis.threshold"] == 0.5
    assert values["analysis.enabled"] is False
    assert values["resources.cpus"] == 1
    assert "mapping_confirmed" not in values
    fields.parameter_fields[0].default = 0.7
    assert settings.parse_settings(fields, settings.default_settings(fields))["analysis.threshold"] == 0.7


def test_review_required_selection_does_not_grant_review() -> None:
    text = 'workflow_id = "test.workflow"\n[parameters]\n"reviewed_image_path" = "/data/image"\n'
    assert settings.parse_settings(descriptor(), text)["reviewed_image_path"] == "/data/image"
    with pytest.raises(ValueError, match="protected"):
        settings.parse_settings(descriptor(), text + '"analysis_reviewed" = true\n')


@pytest.mark.parametrize("line", [
    '"analysis.enabled" = 1',
    '"analysis.threshold" = true',
    '"analysis.threshold" = 2.0',
    '"analysis.method" = "missing"',
    '"mapping_confirmed" = true',
    '"unknown" = 1',
])
def test_invalid_or_protected_fields_are_rejected(line: str) -> None:
    with pytest.raises(ValueError):
        settings.parse_settings(descriptor(), f'workflow_id = "test.workflow"\n[parameters]\n{line}\n')


def test_syntax_and_workflow_mismatch_are_actionable() -> None:
    with pytest.raises(ValueError, match="TOML"):
        settings.parse_settings(descriptor(), "[")
    with pytest.raises(ValueError, match="workflow_id"):
        settings.parse_settings(descriptor(), 'workflow_id = "other"\n[parameters]\n')


def test_profile_apply_reset_and_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    text = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.8\n"input_path" = "/data/source"\n'
    settings.save_user_settings(project, "mine.toml", text, create=True)
    params = settings.apply_values({"analysis": {"enabled": True}, "mapping_confirmed": True}, settings.parse_settings(descriptor(), text))
    params[settings.PROFILE_KEY] = settings.profile_metadata(text, project_root=project, name="mine.toml")
    frozen = settings.freeze_profile(descriptor(), params)
    assert frozen is not None
    assert settings.parse_settings(descriptor(), frozen["effective_toml"])["analysis.threshold"] == 0.8
    assert params["mapping_confirmed"] is True
    reset = settings.reset_values(descriptor(), params)
    assert reset["analysis"]["threshold"] == 0.5
    assert "input_path" not in reset
    settings.save_user_settings(project, "mine.toml", text + "\n# changed\n", create=False)
    with pytest.raises(ValueError, match="changed"):
        settings.freeze_profile(descriptor(), params)


def test_files_cannot_overwrite_defaults_or_source_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.toml"
    source.write_text("original", encoding="utf-8")
    directory = settings.settings_directory(project)
    (directory / "linked.toml").symlink_to(source)
    with pytest.raises((ValueError, OSError)):
        settings.save_user_settings(project, "linked.toml", "changed", create=False)
    with pytest.raises(ValueError):
        settings.save_user_settings(project, "defaults.toml", "changed", create=True)
    with pytest.raises(ValueError):
        settings.save_user_settings(project, "../source.toml", "changed", create=False)
    assert source.read_text(encoding="utf-8") == "original"


def test_config_file_mutation_changes_frozen_provenance(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"threshold": 1}', encoding="utf-8")
    defaults = settings.default_settings(descriptor())
    params = {"config": str(config), settings.PROFILE_KEY: settings.profile_metadata(defaults)}
    frozen = settings.freeze_profile(descriptor(), params)
    assert frozen is not None
    assert frozen["config_files"]["config"]["path"] == str(config)
    original = settings.settings_input_fingerprint(descriptor(), params)
    config.write_text('{"threshold": 2}', encoding="utf-8")
    assert settings.settings_input_fingerprint(descriptor(), params) != original
    assert settings.freeze_profile(descriptor(), params) != frozen


def test_submit_rejects_effective_settings_swapped_after_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fields = descriptor()
    monkeypatch.setattr(service, "get_workflow", lambda _: (None, fields))
    params = {"analysis": {"threshold": 0.2}, settings.PROFILE_KEY: settings.profile_metadata(settings.default_settings(fields))}
    params[settings.PROFILE_KEY] = settings.freeze_profile(fields, params)
    request = RunRequest(
        workflow_id=fields.workflow_id, modality=fields.modality, project_root=tmp_path,
        modality_dir=tmp_path, annotation_csv=tmp_path / "samples.csv", output_root=tmp_path,
        run_id=uuid4(), run_dir=tmp_path / "run", parameters=params,
    )
    result = PreflightResult(ready=True, resolved_parameters=request.model_copy(deep=True).parameters)
    request.parameters["analysis"]["threshold"] = 0.9
    with pytest.raises(ValueError, match="changed after preflight"):
        service.submit_request(request, result)


def test_worker_checks_original_and_config_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.workflows import registry

    fields = descriptor()
    monkeypatch.setattr(registry, "get_workflow", lambda _: (None, fields))
    config = tmp_path / "config.json"
    config.write_text('{"value": 1}', encoding="utf-8")
    params = {"config": str(config), settings.PROFILE_KEY: settings.profile_metadata(settings.default_settings(fields))}
    params[settings.PROFILE_KEY] = settings.freeze_profile(fields, params)
    request = RunRequest(
        workflow_id=fields.workflow_id, modality=fields.modality, project_root=tmp_path,
        modality_dir=tmp_path, annotation_csv=tmp_path / "samples.csv", output_root=tmp_path,
        run_id=uuid4(), run_dir=tmp_path / "run", parameters=params,
    )
    inputs = request.run_dir / "inputs"
    inputs.mkdir(parents=True)
    snapshot = inputs / "settings_config_0.json"
    snapshot.write_bytes(config.read_bytes())
    digest = request.parameters[settings.PROFILE_KEY]["config_files"]["config"]["sha256"]
    (inputs / "settings_configs.json").write_text(
        json.dumps({"config": {"path": "inputs/settings_config_0.json", "sha256": digest}}), encoding="utf-8",
    )
    assert settings.validate_frozen_config_files(request) == []
    snapshot.write_text("changed", encoding="utf-8")
    assert settings.validate_frozen_config_files(request)
    snapshot.write_bytes(config.read_bytes())
    config.write_text("changed", encoding="utf-8")
    assert settings.validate_frozen_config_files(request)


def test_relative_config_and_default_snapshot(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    config.write_text('{"value": 1}', encoding="utf-8")
    params = {"config": "settings.json", "analysis": {"threshold": 0.8}}
    frozen = settings.freeze_profile(descriptor(), params, tmp_path)
    assert frozen["config_files"]["config"]["path"] == str(config)
    assert settings.parse_settings(descriptor(), frozen["source_toml"])["analysis.threshold"] == 0.5
    assert settings.parse_settings(descriptor(), frozen["effective_toml"])["analysis.threshold"] == 0.8


def test_sparse_profile_starts_from_defaults() -> None:
    old = {"analysis": {"threshold": 0.9, "enabled": True}, "input_path": "/old"}
    fresh = settings.reset_values(descriptor(), old)
    applied = settings.apply_values(fresh, {"analysis.method": "b"})
    assert applied["analysis"] == {"threshold": 0.5, "enabled": False, "method": "b"}
    assert "input_path" not in applied


def test_profile_editor_switches_project_without_cross_project_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.ui import settings as settings_ui

    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project_a, project_b = tmp_path / "a", tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    text_a = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.2\n'
    text_b = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.8\n'
    settings.save_user_settings(project_a, "mine.toml", text_a, create=True)
    settings.save_user_settings(project_b, "mine.toml", text_b, create=True)
    ui = MagicMock()
    ui.session_state = {}
    ui.selectbox.return_value = "mine.toml"
    ui.file_uploader.return_value = None
    shown: list[str] = []
    def editor(label: str, *, key: str, height: int, **kwargs: object) -> str:
        text = ui.session_state[key]
        shown.append(text)
        return text
    ui.text_area.side_effect = editor
    ui.button.side_effect = lambda label, **kwargs: label == "Save user file" and len(shown) == 2
    monkeypatch.setattr(settings_ui, "st", ui)
    settings_ui.render_settings(descriptor(), project_a, {}, "test")
    settings_ui.render_settings(descriptor(), project_b, {}, "test")
    assert shown == [text_a, text_b]
    assert settings.read_user_settings(project_b, "mine.toml") == text_b


def test_profile_editor_draft_survives_widget_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.ui import settings as settings_ui

    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    original = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.2\n'
    draft = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.8\n'
    settings.save_user_settings(project, "mine.toml", original, create=True)
    ui = MagicMock()
    ui.session_state = {}
    ui.selectbox.return_value = "mine.toml"
    ui.file_uploader.return_value = None
    ui.button.return_value = False
    shown: list[str] = []
    widget_keys: list[str] = []
    def editor(label: str, *, key: str, height: int, on_change: object, args: tuple[str, str]) -> str:
        widget_keys.append(key)
        shown.append(ui.session_state[key])
        return ui.session_state[key]
    ui.text_area.side_effect = editor
    monkeypatch.setattr(settings_ui, "st", ui)
    settings_ui.render_settings(descriptor(), project, {}, "test")
    ui.session_state[widget_keys[-1]] = draft
    callback = ui.text_area.call_args.kwargs["on_change"]
    callback(*ui.text_area.call_args.kwargs["args"])
    del ui.session_state[widget_keys[-1]]
    settings_ui.render_settings(descriptor(), project, {}, "test")
    assert shown == [original, draft]
    assert settings.read_user_settings(project, "mine.toml") == original


def test_new_profile_filename_and_draft_survive_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.ui import settings as settings_ui

    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    ui = MagicMock()
    ui.session_state = {}
    ui.selectbox.side_effect = lambda label, options, *, key, **kwargs: ui.session_state[key]
    ui.text_input.side_effect = lambda label, *, key, **kwargs: ui.session_state[key]
    ui.text_area.side_effect = lambda label, *, key, **kwargs: ui.session_state[key]
    ui.file_uploader.return_value = None
    ui.button.return_value = False
    monkeypatch.setattr(settings_ui, "st", ui)
    settings_ui.render_settings(descriptor(), project, {}, "test")
    name_call = ui.text_input.call_args.kwargs
    ui.session_state[name_call["key"]] = "validation-settings.toml"
    name_call["on_change"](*name_call["args"])
    settings_ui.render_settings(descriptor(), project, {}, "test")
    draft = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.8\n'
    editor_call = ui.text_area.call_args.kwargs
    ui.session_state[editor_call["key"]] = draft
    editor_call["on_change"](*editor_call["args"])
    select_call = ui.selectbox.call_args.kwargs
    for key in (name_call["key"], editor_call["key"], select_call["key"]):
        del ui.session_state[key]
    settings_ui.render_settings(descriptor(), project, {}, "test")
    assert ui.session_state[ui.text_input.call_args.kwargs["key"]] == "validation-settings.toml"
    assert ui.session_state[ui.text_area.call_args.kwargs["key"]] == draft


def test_selected_profile_survives_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatial_omics_toolkit.application.ui import settings as settings_ui

    monkeypatch.setattr(settings, "ALLOWED_ROOT", tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    source = 'workflow_id = "test.workflow"\n[parameters]\n"analysis.threshold" = 0.8\n'
    settings.save_user_settings(project, "mine.toml", source, create=True)
    ui = MagicMock()
    ui.session_state = {}
    ui.selectbox.side_effect = lambda label, options, *, key, **kwargs: ui.session_state[key]
    ui.text_input.side_effect = lambda label, *, key, **kwargs: ui.session_state[key]
    ui.text_area.side_effect = lambda label, *, key, **kwargs: ui.session_state[key]
    ui.file_uploader.return_value = None
    ui.button.return_value = False
    monkeypatch.setattr(settings_ui, "st", ui)
    settings_ui.render_settings(descriptor(), project, {}, "test")
    select = ui.selectbox.call_args.kwargs
    ui.session_state[select["key"]] = "mine.toml"
    select["on_change"](*select["args"])
    settings_ui.render_settings(descriptor(), project, {}, "test")
    editor = ui.text_area.call_args.kwargs
    del ui.session_state[select["key"]]
    del ui.session_state[editor["key"]]
    settings_ui.render_settings(descriptor(), project, {}, "test")
    assert ui.session_state[select["key"]] == "mine.toml"
    assert ui.session_state[editor["key"]] == source
