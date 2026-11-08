from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from spatial_omics_toolkit.application.ui import implementation
from spatial_omics_toolkit.application.ui import results
from spatial_omics_toolkit.application.worker import _render_pdf_previews



@pytest.mark.parametrize("stage_id", ["hs_native_moments", "preview_hs_native_qc_plots"])
def test_saved_source_render_merges_workflow_receipts_and_event_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_id: str,
) -> None:
    monkeypatch.setattr(results, "ALLOWED_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "native_source_metadata.json").write_text(json.dumps({
        "workflow_id": "hs.native_spectral_qc",
        "canonical_source_sha256": {
            "clinical_roi_hs_flim.py": "f" * 64,
            "clinical_roi_hyperspectral.py": "1" * 64,
        },
    }), encoding="utf-8")
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from spatial_omics_toolkit.application.ui.results import _render_saved_sources\n"
        f"_render_saved_sources(Path({str(run_dir)!r}), "
        "{'workflow_id': 'hs.native_spectral_qc'}, "
        f"[{{'stage_id': {stage_id!r}, 'tool_versions': {{'qc-tool': '2.0'}}}}, "
        "{'stage_id': 'different_step', 'tool_versions': {'other-tool': '9.9'}}], "
        f"stage_id={stage_id!r}, section='results')\n",
        default_timeout=30,
    ).run()
    assert not app.exception, [error.message for error in app.exception]
    cells = {
        str(value) for frame in app.dataframe
        for row in frame.value.itertuples(index=False, name=None) for value in row
    }
    assert {"2.0", "f" * 64, "1" * 64} <= cells
    assert "9.9" not in cells
    if stage_id.startswith("preview_"):
        definition = _render_pdf_previews.__code__
        expected_location = f"{Path(definition.co_filename).resolve()}:{definition.co_firstlineno}"
        assert expected_location in {block.value for block in app.code}


def test_receipt_source_versions_collects_only_workflow_script_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(results, "ALLOWED_ROOT", tmp_path)
    flim_dir = tmp_path / "flim"
    flim_dir.mkdir()
    flim_hashes = {
        "clinical_roi_flim.py": "a" * 64,
        "clinical_roi_flim_plots.py": "b" * 64,
    }
    (flim_dir / "adapter_receipt.json").write_text(json.dumps({
        "workflow_id": "flim.raw_sdt", "run_id": "flim-run", "source_sha256": flim_hashes,
    }), encoding="utf-8")
    flim = results._receipt_source_versions(
        flim_dir, {"workflow_id": "flim.raw_sdt", "run_id": "flim-run"},
    )
    assert set(flim.values()) == set(flim_hashes.values())
    assert all("adapter_receipt.json" in key for key in flim)

    hs_dir = tmp_path / "hs"
    hs_dir.mkdir()
    hs_hashes = {
        "clinical_roi_hs_flim.py": "c" * 64,
        "clinical_roi_hyperspectral.py": "d" * 64,
    }
    (hs_dir / "native_source_metadata.json").write_text(json.dumps({
        "workflow_id": "hs.native_spectral_qc",
        "canonical_source_sha256": hs_hashes,
        "source_sha256": "e" * 64,
    }), encoding="utf-8")
    hs = results._receipt_source_versions(hs_dir, {"workflow_id": "hs.native_spectral_qc"})
    assert set(hs.values()) == set(hs_hashes.values())
    assert "e" * 64 not in hs.values()
    assert all("native_source_metadata.json" in key for key in hs)


@pytest.mark.parametrize(
    ("receipt_text", "metadata"),
    [
        (json.dumps({"workflow_id": "wrong", "source_sha256": {"clinical_roi_flim.py": "a" * 64}}),
         {"workflow_id": "flim.raw_sdt"}),
        ("{malformed", {"workflow_id": "flim.raw_sdt"}),
        (json.dumps({"workflow_id": "flim.raw_sdt", "run_id": "other", "source_sha256": {"clinical_roi_flim.py": "a" * 64}}),
         {"workflow_id": "flim.raw_sdt", "run_id": "selected"}),
    ],
)
def test_receipt_source_versions_rejects_invalid_or_mismatched_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt_text: str, metadata: dict[str, str],
) -> None:
    monkeypatch.setattr(results, "ALLOWED_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "adapter_receipt.json").write_text(receipt_text, encoding="utf-8")

    assert results._receipt_source_versions(run_dir, metadata) == {}


def test_receipt_source_versions_rejects_receipt_symlink_outside_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(results, "ALLOWED_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({
        "workflow_id": "flim.raw_sdt", "source_sha256": {"clinical_roi_flim.py": "a" * 64},
    }), encoding="utf-8")
    (run_dir / "adapter_receipt.json").symlink_to(outside)

    assert results._receipt_source_versions(run_dir, {"workflow_id": "flim.raw_sdt"}) == {}


def test_configured_source_directory_never_falls_back_to_another_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    configured = tmp_path / "configured"
    canonical.mkdir()
    configured.mkdir()
    (canonical / "analysis.py").write_text("def execute() -> None:\n    pass\n", encoding="utf-8")
    override = configured / "analysis.py"
    override.write_text("def execute() -> None:\n    return None\n", encoding="utf-8")
    monkeypatch.setattr(implementation, "_EXTERNAL_SOURCE_ROOT", canonical)
    module = SimpleNamespace(_source_dir=lambda: configured)
    assert implementation._source_path("external:analysis.py", module) == override
    override.unlink()
    assert implementation._source_path("external:analysis.py", module) is None


def test_function_location_uses_current_top_level_definition(tmp_path: Path) -> None:
    script = tmp_path / "source.py"
    script.write_text(
        "def wrapper() -> None:\n    def execute() -> None:\n        pass\n\nasync def execute() -> None:\n    pass\n",
        encoding="utf-8",
    )
    assert implementation._function_line(script, "execute") == 5
    script.write_text("\n\ndef execute() -> None:\n    pass\n", encoding="utf-8")
    assert implementation._function_line(script, "execute") == 3
    assert implementation._function_line(script, "absent") is None
    script.write_text("def execute(:\n", encoding="utf-8")
    assert implementation._function_line(script, "execute") is None
