from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from spatial_omics_toolkit.application import service
from spatial_omics_toolkit.application.contracts import PreflightResult, RunRequest

APP = Path(__file__).resolve().parents[2] / "apps" / "streamlit" / "app.py"


@pytest.mark.parametrize("final_outcome", ["ready", "blocked", "changed_metadata"])
def test_start_requires_confirmation_and_fresh_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, final_outcome: str,
) -> None:
    project = tmp_path / "project"
    raw = project / "images"
    raw.mkdir(parents=True)
    metadata = project / "project_samples.csv"
    metadata.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\nunit,specimen,condition,tissue\n",
        encoding="utf-8",
    )
    checked: list[RunRequest] = []
    submitted: list[RunRequest] = []

    def prepare(
        workflow_id: str, project_root: Path, annotation_csv: Path,
        modality_dir: Path, output_root: Path, parameters: dict[str, Any] | None = None,
    ) -> tuple[RunRequest, PreflightResult]:
        run_id = uuid4()
        request = RunRequest(
            workflow_id=workflow_id, modality="H&E", project_root=project_root,
            annotation_csv=annotation_csv, modality_dir=modality_dir,
            output_root=output_root, parameters=dict(parameters or {}),
            run_id=run_id, run_dir=output_root / "H&E" / "runs" / str(run_id),
        )
        checked.append(request)
        ready = len(checked) < 3 or final_outcome != "blocked"
        receipt = "changed" if len(checked) >= 3 and final_outcome == "changed_metadata" else "reviewed"
        return request, PreflightResult(
            ready=ready, blockers=[] if ready else ["Input changed during final validation"],
            resolved_parameters={**request.parameters, "acquisition_groups": [], "annotation_csv_sha256": receipt},
        )

    def submit(request: RunRequest, preflight: PreflightResult) -> str:
        assert preflight.ready
        submitted.append(request)
        request.run_dir.mkdir(parents=True)
        (request.run_dir / "run.json").write_text(json.dumps({
            "run_id": str(request.run_id), "workflow_id": request.workflow_id,
            "modality": request.modality,
        }), encoding="utf-8")
        (request.run_dir / "status.json").write_text('{"status":"CREATED"}', encoding="utf-8")
        return "test-job"

    monkeypatch.setattr(service, "prepare_request", prepare)
    monkeypatch.setattr(service, "submit_request", submit)
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["spatial_page"] = "H&E"
    app.session_state["data_mode_H&E"] = "own"
    app.session_state["project_root_path"] = str(project)
    app.session_state["annotation_csv"] = str(metadata)
    app.session_state["output_root"] = str(tmp_path / "outputs")
    app.session_state["modality_folder_H&E"] = str(raw)
    app.run()
    assert not app.exception
    app.button(key="continue_to_run_H&E").click().run()
    assert not app.exception
    assert submitted == []
    app.button(key="check_inputs_H&E").click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert len(checked) == 1
    assert submitted == []
    assert not any(button.key == "submit_H&E" and not button.disabled for button in app.button)

    next(box for box in app.checkbox if box.key and box.key.startswith("qc_only_confirmed_H&E")).check().run()
    assert len(checked) == 2  # Mapping confirmation is revalidated before Start is enabled.
    app.button(key="submit_H&E").click().run()

    assert not app.exception, [item.message for item in app.exception]
    assert len(checked) == 3
    assert len(submitted) == (1 if final_outcome == "ready" else 0)
    if final_outcome == "ready":
        assert app.session_state["active_run_dirs"]["H&E"] == str(submitted[0].run_dir)
    else:
        assert "H&E" not in app.session_state["active_run_dirs"]
