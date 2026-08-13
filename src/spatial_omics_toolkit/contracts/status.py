"""Structured status, stage states, and machine-readable error response contracts."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ErrorCode(str, Enum):
    """Machine-readable error codes for API and AI agent consumption."""
    INVALID_MANIFEST = "INVALID_MANIFEST"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    INSUFFICIENT_BIOLOGICAL_REPLICATES = "INSUFFICIENT_BIOLOGICAL_REPLICATES"
    RESOURCE_ENVELOPE_EXCEEDED = "RESOURCE_ENVELOPE_EXCEEDED"
    PSEUDOBULK_AGGREGATION_FAILED = "PSEUDOBULK_AGGREGATION_FAILED"
    METHOD_FAILED_TERMINAL = "METHOD_FAILED_TERMINAL"
    SPATIAL_NULL_TEST_FAILED = "SPATIAL_NULL_TEST_FAILED"
    UNAUTHORIZED_PATH = "UNAUTHORIZED_PATH"


class StageState(BaseModel):
    """Pipeline stage execution state snapshot."""
    model_config = ConfigDict(extra="forbid")

    stage_name: str = Field(..., description="Stage name (e.g. intake_qc, pseudobulk_aggregation, differential_edger)")
    status: str = Field(..., description="Stage status (SUCCEEDED, FAILED, RUNNING, SKIPPED)")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Stage execution duration in seconds")
    error_code: Optional[ErrorCode] = Field(default=None, description="Error code if stage failed")
    detail: Optional[str] = Field(default=None, description="Detailed stage log message or exception summary")


class StructuredErrorResponse(BaseModel):
    """Machine-readable structured error contract for CLI and FastAPI responses."""
    model_config = ConfigDict(extra="forbid")

    error_code: ErrorCode = Field(..., description="Stable machine-readable error code")
    reason: str = Field(..., description="High-level description of what went wrong")
    remediation_hint: str = Field(..., description="Actionable advice on how to resolve the error")
    context: Dict[str, str] = Field(default_factory=dict, description="Additional context parameters")
