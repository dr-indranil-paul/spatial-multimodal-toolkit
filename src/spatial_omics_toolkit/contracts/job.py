"""Job specification and execution envelope contracts."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class JobStatus(str, Enum):
    """Job execution lifecycle states."""
    CREATED = "created"
    VALIDATED = "validated"
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    QC_PENDING = "qc_pending"
    CANONICAL = "canonical"
    NONCANONICAL_FALLBACK = "noncanonical_fallback"
    REJECTED_QC = "rejected_qc"
    FAILED_TERMINAL = "failed_terminal"
    CANCELLED = "cancelled"


class ResourceEnvelope(BaseModel):
    """Runtime resource limits compliant with HPC QOS policies (cpus <= 32)."""
    model_config = ConfigDict(extra="forbid")

    cpus: int = Field(default=8, ge=1, le=32, description="CPU cores requested (HPC max 32)")
    memory_gb: float = Field(default=32.0, ge=1.0, le=256.0, description="Memory requested in GB")
    time_limit_hours: float = Field(default=12.0, ge=0.5, le=72.0, description="Walltime limit in hours")
    use_gpu: bool = Field(default=False, description="GPU requirement flag")


class JobSpec(BaseModel):
    """Job specification contract for pipeline submission."""
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Unique job UUID")
    owner: str = Field(..., description="User or agent owner ID")
    study_id: str = Field(..., description="Associated study ID")
    manifest_path: str = Field(..., description="Path to validated intake_manifest.json")
    plugin_name: str = Field(..., description="Target plugin (e.g. scrna, visium, xenium)")
    resources: ResourceEnvelope = Field(default_factory=ResourceEnvelope, description="Runtime resource limits")
    allow_fallback: bool = Field(default=True, description="Allow non-canonical qualified fallback methods on primary failure")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Resolved scientific parameters")
    output_dir: str = Field(..., description="Output directory path beneath /home/groups/EmiliLab/pauli/")
