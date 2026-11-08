from __future__ import annotations

from contextlib import nullcontext
from graphlib import TopologicalSorter
from html import escape
import re
import xml.etree.ElementTree as ET

import pytest

from spatial_omics_toolkit.application.contracts import WorkflowDescriptor
from spatial_omics_toolkit.application.ui import presentation
from spatial_omics_toolkit.application.ui.presentation import _stage_graph
from spatial_omics_toolkit.application.workflows.registry import list_workflows


SVG = "{http://www.w3.org/2000/svg}"


def _diagram(descriptor: WorkflowDescriptor) -> tuple[str, ET.Element]:
    markup = _stage_graph(descriptor)
    svg = markup[markup.index("<svg "):markup.index("</svg>") + len("</svg>")]
    return markup, ET.fromstring(svg)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _text(element: ET.Element) -> str:
    return _compact("".join(text.text or "" for text in element.iter(f"{SVG}text")))


def _node_bounds(node: ET.Element) -> tuple[float, float, float, float]:
    shape = node.find(f"{SVG}path")
    assert shape is not None
    coordinates = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", shape.attrib["d"])]
    assert len(coordinates) >= 6 and len(coordinates) % 2 == 0
    xs, ys = coordinates[::2], coordinates[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_registered_diagrams_match_execution_dependencies_and_annotations() -> None:
    descriptors = list_workflows()
    assert descriptors
    for descriptor in descriptors:
        stages = descriptor.stages
        ids = [stage.stage_id for stage in stages]
        assert len(ids) == len(set(ids)), descriptor.workflow_id
        dependencies = {stage.stage_id: set(stage.depends_on) for stage in stages}
        assert all(deps <= set(ids) for deps in dependencies.values()), descriptor.workflow_id
        tuple(TopologicalSorter(dependencies).static_order())
        _, svg = _diagram(descriptor)
        assert svg.attrib["width"].endswith("px")
        assert svg.attrib["height"].endswith("px")
        _, _, width, height = svg.attrib["viewBox"].split()
        assert float(svg.attrib["width"][:-2]) == float(width)
        assert float(svg.attrib["height"][:-2]) == float(height)
        nodes = {int(node.attrib["id"].split("-")[1]): node
                 for node in svg.iter(f"{SVG}g") if node.attrib.get("class") == "node"}
        edges = {tuple(map(int, edge.attrib["id"].split("-")[1:])): edge
                 for edge in svg.iter(f"{SVG}g") if edge.attrib.get("class") == "edge"}
        expected = {(ids.index(dependency), index)
                    for index, stage in enumerate(stages) for dependency in stage.depends_on}
        assert set(nodes) == set(range(len(stages))), descriptor.workflow_id
        assert set(edges) == expected, descriptor.workflow_id
        bounds = {index: _node_bounds(node) for index, node in nodes.items()}
        for index, (left, top, right, bottom) in bounds.items():
            for other, (other_left, other_top, other_right, other_bottom) in bounds.items():
                if index < other:
                    assert (right <= other_left or other_right <= left
                            or bottom <= other_top or other_bottom <= top), descriptor.workflow_id
            for text in nodes[index].iter(f"{SVG}text"):
                x, y = float(text.attrib["x"]), float(text.attrib["y"])
                assert left <= x <= right and top <= y <= bottom, descriptor.workflow_id
        for index, stage in enumerate(stages):
            content = _text(nodes[index])
            for value in (stage.title, stage.purpose, stage.condition, *stage.inputs, *stage.outputs):
                assert _compact(value) in content
            for tool in stage.tools:
                for value in (tool.name, tool.version, tool.role):
                    assert _compact(value) in content
        for (source, target), edge in edges.items():
            path = edge.find(f"{SVG}path")
            arrow = edge.find(f"{SVG}polygon")
            assert path is not None and path.attrib.get("d")
            assert arrow is not None and len(arrow.attrib["points"].split()) >= 3
            label = _text(edge)
            for value in (*stages[source].outputs, stages[target].condition):
                assert _compact(value) in label


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
    markup, svg = _diagram(changed_descriptor)
    node = next(element for element in svg.iter(f"{SVG}g") if element.attrib.get("class") == "node")
    assert _text(node).count(_compact(payload)) == 8
    assert payload not in markup
    assert unsafe_id not in markup
    assert not list(svg.iter(f"{SVG}script"))
    rendered: list[str] = []
    monkeypatch.setattr(presentation.st, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(presentation.st, "html", lambda text, **kwargs: rendered.append(text))
    monkeypatch.setattr(presentation.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(presentation.st, "markdown", lambda text, **kwargs: rendered.append(text))
    presentation.render_stage_flow(changed_descriptor)
    assert all(payload not in fragment for fragment in rendered)
    details = [fragment for fragment in rendered if "<h3>" in fragment]
    assert len(details) == 1
    assert details[0].count(escape(payload, quote=True)) == 8


def test_svg_bolds_only_field_prefixes_and_preserves_wrapped_values() -> None:
    descriptor = next(item for item in list_workflows() if item.stages and item.stages[0].tools)
    stage = descriptor.stages[0]
    payload = '<unsafe & "quoted">'
    long_value = " ".join(["wrappedfield"] * 20) + " " + payload
    identifier = "unbroken_identifier_" * 12
    tool = stage.tools[0].model_copy(update={"name": identifier, "version": "1", "role": long_value})
    upstream = stage.model_copy(update={
        "stage_id": "upstream",
        "title": "Upstream stage",
        "purpose": long_value,
        "tools": [tool],
        "inputs": [payload],
        "outputs": [long_value],
        "condition": payload,
        "depends_on": [],
    })
    downstream = stage.model_copy(update={
        "stage_id": "downstream",
        "title": "Downstream stage",
        "purpose": payload,
        "tools": [],
        "inputs": [payload],
        "outputs": [payload],
        "condition": long_value,
        "depends_on": ["upstream"],
    })
    markup, svg = _diagram(descriptor.model_copy(update={"stages": [upstream, downstream]}))
    groups = {element.attrib.get("id"): element for element in svg.iter(f"{SVG}g")}
    node = groups["stage-0"]
    edge = groups["edge-0-1"]

    def weighted_text(element: ET.Element, bold: bool) -> str:
        return _compact("".join(
            text.text or "" for text in element.iter(f"{SVG}text")
            if (text.attrib.get("font-weight") in {"bold", "700"}) == bold
        ))

    node_bold = weighted_text(node, True)
    node_normal = weighted_text(node, False)
    edge_bold = weighted_text(edge, True)
    edge_normal = weighted_text(edge, False)
    for prefix in ("Purpose:", "Tool:", "Inputs:", "Outputs:", "Condition / review:"):
        assert _compact(prefix) in node_bold
        assert _compact(prefix) not in node_normal
    for prefix in ("Upstream outputs (when produced):", "Downstream condition / review:"):
        assert _compact(prefix) in edge_bold
        assert _compact(prefix) not in edge_normal
    assert _compact(long_value) in node_normal
    assert _compact(long_value) in edge_normal
    assert _compact(payload) in node_normal
    assert _compact(payload) in edge_normal
    assert _compact(payload) not in node_bold + edge_bold
    assert _compact(long_value) not in node_bold + edge_bold
    normal_lines = [text.text or "" for text in node.iter(f"{SVG}text")
                    if text.attrib.get("font-weight") not in {"bold", "700"}]
    assert any(identifier in line for line in normal_lines)
    assert sum(line.count("wrappedfield") for line in normal_lines) == 60
    assert all("wrapped" not in line.replace("wrappedfield", "") for line in normal_lines)
    left, top, right, bottom = _node_bounds(node)
    for text in node.iter(f"{SVG}text"):
        assert left <= float(text.attrib["x"]) <= right
        assert top <= float(text.attrib["y"]) <= bottom
    assert payload not in markup
    assert not list(svg.iter(f"{SVG}script"))
    assert edge.find(f"{SVG}path") is not None
    assert edge.find(f"{SVG}polygon") is not None


def test_unknown_dependencies_and_duplicate_ids_are_not_silently_drawn() -> None:
    descriptor = next(item for item in list_workflows() if item.stages)
    stage = descriptor.stages[0]
    missing = stage.model_copy(update={"depends_on": ["not-a-stage"]})
    with pytest.raises(ValueError, match="Unknown workflow stage dependency"):
        _stage_graph(descriptor.model_copy(update={"stages": [missing]}))
    with pytest.raises(ValueError, match="unique IDs"):
        _stage_graph(descriptor.model_copy(update={"stages": [stage, stage]}))
