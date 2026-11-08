import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from spatial_omics_toolkit.application.contracts import RunRequest, StageEvent
from spatial_omics_toolkit.application.workflows import he


def _request(tmp_path: Path) -> RunRequest:
    source = tmp_path / "source"
    source.mkdir()
    return RunRequest(
        workflow_id="he-image-intake", modality="H&E", project_root=tmp_path,
        modality_dir=source, annotation_csv=tmp_path / "project_samples.csv",
        output_root=tmp_path / "outputs", run_id=uuid4(),
        run_dir=tmp_path / "outputs" / "run", parameters={},
    )


def test_he_stage_inventories_tiff_without_modifying_source(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    request = _request(tmp_path)
    source = request.modality_dir / "image.tif"
    tifffile.imwrite(source, np.zeros((8, 8), dtype=np.uint8))
    original = source.read_bytes()
    events: list[StageEvent] = []

    result = he.execute_stage(request, "image-intake", {}, events.append)

    assert result.status == "WARNING"
    assert source.read_bytes() == original
    inventory = json.loads((request.run_dir / "he_image_intake" / "image_inventory.json").read_text())
    assert inventory[0]["path"] == str(source)
    assert inventory[0]["metadata"]["pages"] == 1
    assert inventory[0]["preview"] is None
    assert events[-1].stage_id == "image-intake"


def test_he_stage_rejects_unknown_stage(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="Unknown H&E stage"):
        he.execute_stage(request, "not-a-stage", {}, lambda event: None)


def test_he_stage_blocks_empty_inventory(tmp_path: Path) -> None:
    pytest.importorskip("tifffile")
    request = _request(tmp_path)
    events: list[StageEvent] = []

    result = he.execute_stage(request, "image-intake", {}, events.append)

    assert result.status == "BLOCKED"
    assert "No supported H&E image files" in result.warnings[0]
    assert events[-1].status == "BLOCKED"
