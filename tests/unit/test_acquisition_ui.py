from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import pandas as pd

from spatial_omics_toolkit.application.ui import acquisition as acquisition_ui
from spatial_omics_toolkit.application.ui.acquisition import (
    _choose_conflict,
    _conflicts,
    _guidance_issues,
    _modality_acquisitions,
    _supported_pair,
    _merge_discovered,
)


def _row(acquisition: str, source: str, modality: str = "HS") -> dict[str, str]:
    return {
        "acquisition_id": acquisition, "modality": modality, "source_path": source,
        "specimen_id": "specimen", "field": "pixel_size_x", "value": "1", "unit": "um",
        "provenance": "native", "source_detail": "source header", "reviewed": "false",
    }


def test_modality_selection_filters_acquisitions_and_reports_supported_pairs() -> None:
    rows = [_row("h", "he.czi", "H&E"), _row("s", "hs.czi", "HS"), _row("f", "flim.sdt", "FLIM")]
    assert _modality_acquisitions(rows, "HS") == ["s"]
    assert _modality_acquisitions(rows, "CycIF") == []
    assert _supported_pair("H&E", "CycIF")
    assert _supported_pair("H&E", "HS")
    assert not _supported_pair("HS", "FLIM")
    assert not _supported_pair("HS", "H&E")


def test_conflict_choice_preserves_provenance_and_all_evidence() -> None:
    native = _row("a", "a.czi")
    user = {**native, "value": "2", "provenance": "user", "source_detail": "log"}
    assert len(_conflicts([native, user])) == 1
    result = _choose_conflict([native, user], "a", "pixel_size_x", 1)
    assert result == [native, {**user, "reviewed": "true"}]
    assert native["reviewed"] == "false"
    with pytest.raises(ValueError):
        _choose_conflict([native, user], "a", "pixel_size_y", 1)


def test_unknown_is_not_competing_measured_evidence() -> None:
    known = _row("a", "a.czi")
    unknown = {**known, "value": "unknown", "unit": "", "provenance": "unknown", "source_detail": ""}
    assert _conflicts([known, unknown]) == []
    other = {**known, "value": "2", "provenance": "user", "source_detail": "log"}
    assert len(_conflicts([known, unknown, other])) == 1
    with pytest.raises(ValueError):
        _choose_conflict([known, unknown, other], "a", "pixel_size_x", 1)


def test_custom_conflict_review_preserves_evidence() -> None:
    native = {**_row("a", "a.czi"), "field": "custom:operator-note", "value": "first", "unit": ""}
    user = {**native, "value": "second", "provenance": "user", "source_detail": "notebook"}
    unknown = {**native, "value": "unknown", "provenance": "unknown", "source_detail": ""}
    assert _conflicts([native, unknown]) == []
    assert len(_conflicts([native, user, unknown])) == 1
    assert _choose_conflict([native, user, unknown], "a", "custom:operator-note", 1) == [
        native, {**user, "reviewed": "true"}, unknown,
    ]


def test_discovery_merges_with_existing_identity_without_replacing_records(tmp_path: Path) -> None:
    source = tmp_path / "a.czi"
    source.write_bytes(b"header")
    existing = _row("a", "a.czi")
    discovered = {**existing, "source_path": str(source), "specimen_id": "", "field": "pixel_size_y"}
    merged = _merge_discovered([existing], [discovered], tmp_path, tmp_path)
    assert merged == [existing, {**discovered, "source_path": "a.czi", "specimen_id": "specimen"}]
    assert _merge_discovered(merged, [discovered], tmp_path, tmp_path) == merged
    with pytest.raises(ValueError, match="different modality"):
        _merge_discovered([existing], [{**discovered, "modality": "FLIM"}], tmp_path, tmp_path)


def test_guidance_reports_missing_evidence_and_does_not_verify_csv_claims(tmp_path: Path) -> None:
    (tmp_path / "he.czi").write_bytes(b"header")
    (tmp_path / "hs.czi").write_bytes(b"header")
    moving = _row("he", "he.czi", "H&E")
    reference = _row("hs", "hs.czi", "HS")
    rows = [moving, reference, {**reference, "value": "2", "provenance": "user", "source_detail": "log"}]
    issues = _guidance_issues(rows, [], "he", "hs", tmp_path, tmp_path)
    assert any("unreviewed" in issue for issue in issues)
    assert any("Y scale" in issue and "missing" in issue for issue in issues)
    assert any("channel names" in issue and "missing" in issue for issue in issues)
    assert any("Relationship" in issue and "missing" in issue for issue in issues)
    assert any("declaration, not independent verification" in issue for issue in issues)
    assert any("Source authorization" in issue for issue in issues)


def test_guidance_marks_unsupported_modality_and_missing_source(tmp_path: Path) -> None:
    rows = [_row("a", "missing.czi", "HS"), _row("b", "missing2.czi", "FLIM")]
    issues = _guidance_issues(rows, [], "a", "b", tmp_path, tmp_path)
    assert sum("unsupported" in issue for issue in issues) == 2
    assert sum("source is missing" in issue for issue in issues) == 2


class _Rerun(Exception):
    pass


class _EditorStreamlit:
    """Model Streamlit's cumulative editor patch and widget cleanup."""

    def __init__(self) -> None:
        self.session_state = {}
        self.upload = None
        self.pressed = None

    def warning(self, message):
        pass

    def error(self, message):
        raise AssertionError(message)

    def file_uploader(self, *args, **kwargs):
        return self.upload

    def button(self, label, **kwargs):
        return kwargs["key"] == self.pressed

    def data_editor(self, frame, **kwargs):
        key = kwargs["key"]
        patch = self.session_state.setdefault(key, {"edited_rows": {}, "added_rows": []})
        result = frame.copy()
        for index, changes in patch["edited_rows"].items():
            for column, value in changes.items():
                result.loc[index, column] = value
        if patch["added_rows"]:
            result = pd.concat([result, pd.DataFrame(patch["added_rows"])], ignore_index=True)
        return result

    def download_button(self, *args, **kwargs):
        pass

    def rerun(self):
        raise _Rerun


@pytest.mark.parametrize("name", ["metadata", "relationships"])
def test_editor_preserves_draft_across_navigation_and_project_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    st = _EditorStreamlit()
    monkeypatch.setattr(acquisition_ui, "st", st)
    columns = ["acquisition_id", "value"]
    loaded = [{"acquisition_id": "original", "value": "saved"}]

    def render(scope: str, initial=loaded):
        return acquisition_ui._editor(tmp_path, tmp_path, scope, name, columns, initial, [], lambda *_: None)

    key = f"acquisition_one_{name}_editor_0"
    assert render("one") == loaded
    st.session_state[key] = {
        "edited_rows": {0: {"value": "unsaved"}},
        "added_rows": [{"acquisition_id": "added", "value": "new"}],
    }
    draft = [{"acquisition_id": "original", "value": "unsaved"},
             {"acquisition_id": "added", "value": "new"}]
    assert render("one") == draft
    # Reapplying the cumulative widget patch must not append the new row twice.
    assert render("one") == draft
    assert render("two", [{"acquisition_id": "other", "value": "project"}]) == [
        {"acquisition_id": "other", "value": "project"}
    ]
    # Streamlit removes widget-owned state after its page is no longer rendered.
    del st.session_state[key]
    assert render("one") == draft
    assert render("one") == draft


@pytest.mark.parametrize("field", ["pixel_size_x", "custom:operator-note"])
def test_import_replaces_draft_and_discovery_merges_latest_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    st = _EditorStreamlit()
    monkeypatch.setattr(acquisition_ui, "st", st)
    columns = acquisition_ui.METADATA_COLUMNS
    original = {**_row("a", "a.czi"), "field": field}
    source = tmp_path / "a.czi"
    source.write_bytes(b"header")

    def render():
        return acquisition_ui._editor(tmp_path, tmp_path, "one", "metadata", columns, [original], [], lambda *_: None)

    render()
    key = "acquisition_one_metadata_editor_0"
    st.session_state[key]["edited_rows"] = {0: {"value": "edited"}}
    rows = render()
    discovered = {**original, "source_path": str(source), "specimen_id": "", "field": "pixel_size_y"}
    merged = _merge_discovered(rows, [discovered], tmp_path, tmp_path)
    st.session_state["acquisition_one_metadata_draft"] = merged
    st.session_state["acquisition_one_metadata_generation"] += 1
    assert render() == merged
    assert render()[0]["value"] == "edited"

    class Upload:
        def getvalue(self):
            return acquisition_ui._csv_text([original], columns).encode()

    st.upload = Upload()
    st.pressed = "acquisition_one_metadata_import"
    with pytest.raises(_Rerun):
        render()
    st.pressed = None
    assert render() == [original]
