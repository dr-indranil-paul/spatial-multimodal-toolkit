from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, WorkflowResult
from spatial_omics_toolkit.application.workflows import integration


@pytest.fixture
def integration_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunRequest, list[str]]:
    moving = tmp_path / "he.czi"
    reference = tmp_path / "reference.czi"
    moving.write_bytes(b"test H&E source")
    reference.write_bytes(b"test reference source")
    calls: list[str] = []

    class Reader:
        def __init__(self, path: str) -> None:
            self.path = Path(path)
            self.pixel_type = "bgr24" if self.path == moving else "gray16"
            self.meta = ET.fromstring(
                "<Metadata><Information><Image><Dimensions><Channels><Channel Name='DAPI'/>"
                "</Channels></Dimensions></Image></Information><Scaling><Items>"
                "<Distance Id='X'><Value>1e-6</Value></Distance>"
                "<Distance Id='Y'><Value>1e-6</Value></Distance>"
                "</Items></Scaling></Metadata>"
            )

        def get_dims_shape(self) -> list[dict[str, tuple[int, int]]]:
            return [{"X": (0, 32), "Y": (0, 32), "C": (0, 1)}]

        def is_mosaic(self) -> bool:
            return False

        def read_image(self, **selectors: int) -> tuple[np.ndarray, str]:
            calls.append(self.path.name)
            y, x = np.indices((32, 32))
            if self.path == moving:
                image = np.stack([x * 4, y * 4, (x + y) * 2], axis=-1).astype(np.uint8)
            else:
                image = ((x + y) * 100).astype(np.uint16)
            return image, "YX"

    monkeypatch.setattr(integration, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(integration, "_version", lambda _name: "test-installed")
    monkeypatch.setattr(integration, "_czi_reader", lambda: Reader)
    from spatial_omics_toolkit.application import plotting
    monkeypatch.setattr(plotting, "ensure_arial", lambda: None)
    request = RunRequest(
        workflow_id="integration.serial_section_alignment", modality="Integration",
        project_root=tmp_path, modality_dir=tmp_path, annotation_csv=tmp_path / "samples.csv",
        output_root=tmp_path, run_id=uuid.uuid4(), run_dir=tmp_path / "run",
        parameters={"moving_he_path": str(moving), "reference_fluorescence_path": str(reference),
                    "linkage": "serial_section_candidate", "candidate_sources_confirmed": True,
                    "reference_channel_index": 0, "reference_channel_name": "DAPI", "search_radius": 1},
    )
    return request, calls


def test_integration_stages_execute_only_their_work(integration_sources: tuple[RunRequest, list[str]]) -> None:
    request, reads = integration_sources
    events: list[Any] = []
    upstream: dict[str, WorkflowResult] = {}
    stages = integration.describe().stages
    assert [stage.stage_id for stage in stages] == ["authorization", "metadata-scale", "bounded-previews", "translation-registration", "review"]
    assert [stage.depends_on for stage in stages] == [[], ["authorization"], ["metadata-scale"], ["bounded-previews"], ["translation-registration"]]
    for stage in stages:
        upstream[stage.stage_id] = integration.execute_stage(request, stage.stage_id, upstream, events.append)
        assert upstream[stage.stage_id].status == ("WARNING" if stage.stage_id == "review" else "SUCCEEDED")
        assert len(reads) == (0 if stage.stage_id in {"authorization", "metadata-scale"} else 2)
        json.dumps(upstream[stage.stage_id].state, allow_nan=False)
    output = request.run_dir / "integration_serial_section_alignment"
    qc = json.loads((output / "alignment_qc_report.json").read_text())
    assert qc["review_required"] and not qc["alignment_accepted"]
    assert qc["source_previews"]["moving_he"] == str(output / "moving_he_source_preview.png")
    assert (output / "source_inventory_and_native_metadata.json").is_file()
    assert (output / "before_after_alignment_overlay.pdf").is_file()
    assert (output / "before_after_alignment_overlay.png").is_file()
    assert upstream["review"].review_request["required"]


def test_authorization_and_channel_mismatch_block_later_work(integration_sources: tuple[RunRequest, list[str]]) -> None:
    request, reads = integration_sources
    request.parameters["candidate_sources_confirmed"] = False
    denied = integration.execute_stage(request, "authorization", {}, lambda event: None)
    assert denied.status == "BLOCKED"
    blocked = integration.execute_stage(request, "metadata-scale", {"authorization": denied}, lambda event: None)
    assert blocked.status == "BLOCKED" and not reads
    request.parameters["candidate_sources_confirmed"] = True
    request.parameters["reference_channel_name"] = "wrong"
    auth = integration.execute_stage(request, "authorization", {}, lambda event: None)
    failed = integration.execute_stage(request, "metadata-scale", {"authorization": auth}, lambda event: None)
    assert failed.status == "FAILED"
    assert "does not exactly match" in failed.warnings[0]
    blocked = integration.execute_stage(request, "bounded-previews", {"metadata-scale": failed}, lambda event: None)
    assert blocked.status == "BLOCKED" and not reads


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED", "QC_PENDING", "SKIPPED", "RUNNING"])
def test_incomplete_upstream_never_computes(integration_sources: tuple[RunRequest, list[str]], status: str) -> None:
    request, reads = integration_sources
    result = integration.execute_stage(request, "bounded-previews", {"metadata-scale": WorkflowResult(status=status)}, lambda event: None)
    assert result.status == "BLOCKED"
    assert not reads
