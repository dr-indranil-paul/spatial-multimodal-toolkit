from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


StageStatus = Literal["CREATED", "RUNNING", "SUCCEEDED", "REUSED", "WARNING", "FAILED", "BLOCKED", "QC_PENDING"]


class RunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str
    modality: str
    project_root: Path
    modality_dir: Path
    annotation_csv: Path
    output_root: Path
    run_id: UUID
    run_dir: Path
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolReference(BaseModel):
    name: str
    version: str
    role: str
    documentation_url: str | None = None


ParameterFieldKind = Literal["text", "int", "float", "bool", "select", "multiselect", "path", "config_file"]


class ParameterField(BaseModel):
    name: str
    label: str
    kind: ParameterFieldKind
    required: bool = False
    default: Any = None
    choices: list[Any] = Field(default_factory=list)
    dynamic_choices_key: str | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None
    unit: str | None = None
    section: str = "Parameters"
    stage_id: str | None = None
    help_text: str = ""
    advanced: bool = False
    review_required: bool = False


class FunctionReference(BaseModel):
    """A source function responsible for a workflow stage."""

    source_file: str
    function_name: str
    role: str


class StageDefinition(BaseModel):
    stage_id: str
    title: str
    purpose: str
    tools: list[ToolReference] = Field(default_factory=list)
    implementations: list[FunctionReference] = Field(default_factory=list)
    event_stage_ids: list[str] = Field(default_factory=list)

class WorkflowDescriptor(BaseModel):
    workflow_id: str
    modality: str
    title: str
    description: str
    stages: list[StageDefinition]
    input_description: str
    output_description: str
    parameter_fields: list[ParameterField] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PreflightResult(BaseModel):
    ready: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)


class StageEvent(BaseModel):
    stage_id: str
    status: StageStatus
    message: str
    tool_versions: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    status: str
    stages: list[StageEvent] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    review_request: dict[str, Any] | None = None


WorkflowEmitter = Callable[[StageEvent], None]
