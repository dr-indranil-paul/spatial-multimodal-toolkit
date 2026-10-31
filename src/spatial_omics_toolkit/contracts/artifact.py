"""Artifact contract specifications."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ArtifactRole(str, Enum):
    """Semantic role of an output artifact."""

    RAW_COUNTS = "raw_counts"
    QC_METRICS = "qc_metrics"
    PSEUDOBULK_MATRIX = "pseudobulk_matrix"
    DIFFERENTIAL_RESULTS = "differential_results"
    SPATIAL_OME_ZARR = "spatial_ome_zarr"
    SPATIAL_NEIGHBORHOOD_GRAPH = "spatial_neighborhood_graph"
    HTML_REPORT = "html_report"
    PDF_REPORT = "pdf_report"
    RO_CRATE_METADATA = "ro_crate_metadata"


class ArtifactContract(BaseModel):
    """Artifact inventory contract for release bundle tracking."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(..., description="Unique artifact identifier")
    role: ArtifactRole = Field(..., description="Semantic artifact role")
    relative_path: str = Field(..., description="Path relative to run_<uuid> root")
    media_type: str = Field(
        ...,
        description="MIME or media format string (e.g. application/parquet, image/zarr)",
    )
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    sha256: str = Field(
        ..., pattern="^[a-fA-F0-9]{64}$", description="SHA-256 file checksum"
    )
    producing_stage: str = Field(
        ..., description="Pipeline stage that generated this file"
    )
    is_canonical: bool = Field(
        default=True,
        description="Flag indicating if produced by a canonical pipeline path",
    )
