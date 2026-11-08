from __future__ import annotations

from contextlib import nullcontext
from graphlib import TopologicalSorter
from html import escape
import re

import pytest

from spatial_omics_toolkit.application.ui import presentation
from spatial_omics_toolkit.application.ui.presentation import _stage_graph
from spatial_omics_toolkit.application.workflows.registry import list_workflows


def test_registered_diagrams_match_execution_dependencies() -> None:
    descriptors = list_workflows()
    assert descriptors
    for descriptor in descriptors:
        stages = descriptor.stages
        ids = [stage.stage_id for stage in stages]
        assert len(ids) == len(set(ids)), descriptor.workflow_id
        dependencies = {stage.stage_id: set(stage.depends_on) for stage in stages}
        assert all(deps <= set(ids) for deps in dependencies.values()), descriptor.workflow_id
        # This raises for a cycle, including self-dependencies.
        tuple(TopologicalSorter(dependencies).static_order())
        dot = _stage_graph(descriptor)
        nodes = {int(index) for index in re.findall(r'data-stage-index="(\d+)"', dot)}
        edges = {
            (int(source), int(target))
            for source, target in re.findall(r'data-source="(\d+)" data-target="(\d+)"', dot)
        }
        expected = {
            (ids.index(dependency), index)
            for index, stage in enumerate(stages)
            for dependency in stage.depends_on
        }
        assert nodes == set(range(len(stages))), descriptor.workflow_id
        assert edges == expected, descriptor.workflow_id


def test_graph_escapes_dynamic_labels_and_does_not_interpolate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = next(item for item in list_workflows() if item.stages and item.stages[0].tools)
    stage = descriptor.stages[0]
    payload = '</TD><SCRIPT>alert("x")</SCRIPT>'
    unsafe_id = 'stage"; evil -> node; //'
    tool = stage.tools[0].model_copy(update={"name": payload, "version": payload, "role": payload})
    changed = stage.model_copy(update={
        "stage_id": unsafe_id,
        "title": payload,
        "purpose": payload,
        "tools": [tool],
        "inputs": [payload],
        "outputs": [payload],
        "condition": payload,
        "depends_on": [],
    })
    changed_descriptor = descriptor.model_copy(update={"stages": [changed]})
    dot = _stage_graph(changed_descriptor)
    assert dot.count(escape(payload, quote=True)) == 2
    assert payload not in dot
    assert unsafe_id not in dot
    details: list[str] = []
    monkeypatch.setattr(presentation.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "html", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(presentation.st, "markdown", lambda text, **kwargs: details.append(text))
    presentation.render_stage_flow(changed_descriptor)
    assert len(details) == 1
    assert details[0].count(escape(payload, quote=True)) == 8


def test_unknown_dependencies_and_duplicate_ids_are_not_silently_drawn() -> None:
    descriptor = next(item for item in list_workflows() if item.stages)
    stage = descriptor.stages[0]
    missing = stage.model_copy(update={"depends_on": ["not-a-stage"]})
    with pytest.raises(ValueError, match="Unknown workflow stage dependency"):
        _stage_graph(descriptor.model_copy(update={"stages": [missing]}))
    with pytest.raises(ValueError, match="unique IDs"):
        _stage_graph(descriptor.model_copy(update={"stages": [stage, stage]}))
