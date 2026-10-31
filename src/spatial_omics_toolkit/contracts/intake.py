"""Intake manifest and specimen hierarchy contracts."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FeatureType(str, Enum):
    """Channel-agnostic feature modalities."""

    RNA = "rna"
    PROTEIN = "protein"
    ANTIBODY_CONTROL = "antibody_control"
    MORPHOLOGY = "morphology"
    GENOMIC_VARIANT = "genomic_variant"


class Modality(str, Enum):
    """Assay platform modalities."""

    SINGLE_CELL_RNA = "scrna"
    SINGLE_NUCLEUS_RNA = "snrna"
    VISIUM = "visium"
    XENIUM = "xenium"
    SPATIAL_PROTEOMICS = "spatial_proteomics"


class IntakeClass(str, Enum):
    """Dataset Intake Classifier (Classes A-G) per Institute Analytics Manual."""

    CLASS_A_RAW = "A_native_raw"
    CLASS_B_PROCESSED_NATIVE = "B_processed_native"
    CLASS_C_ANALYSIS_OBJECT = "C_analysis_object"
    CLASS_D_COUNT_MATRIX_ONLY = "D_count_matrix_only"
    CLASS_E_IMAGE_COORDINATES_ONLY = "E_image_coordinates_only"
    CLASS_F_SUMMARY_ONLY = "F_summary_only"
    CLASS_G_UNRESOLVED_QUARANTINE = "G_unresolved_quarantine"


class ModalityLinkage(str, Enum):
    """Physical modality linkage classification per Institute Analytics Manual."""

    SAME_MOLECULE = "same_molecule"
    SAME_CELL_OR_NUCLEUS = "same_cell_or_nucleus"
    SAME_SPOT_OR_PIXEL = "same_spot_or_pixel"
    SAME_SECTION = "same_section"
    SERIAL_SECTION = "serial_section"
    SAME_SPECIMEN = "same_specimen"
    SAME_DONOR = "same_donor"
    CONDITION_MATCHED = "condition_matched"
    EXTERNAL_REFERENCE = "external_reference"


class BiologicalSpecimen(BaseModel):
    """Biological specimen hierarchy unit representing an independent biological replicate (n=1)."""

    model_config = ConfigDict(extra="forbid")

    biological_unit_id: str = Field(
        ...,
        description="Unique independent biological unit ID (e.g. Patient ID, Core ID)",
    )
    specimen_id: str = Field(
        ..., description="Unique specimen identifier (e.g. Tissue Section ID)"
    )
    condition: str = Field(
        ..., description="Experimental condition or disease phenotype"
    )
    batch_or_block: str = Field(
        default="batch_1", description="Batch, block, or run cohort identifier"
    )
    organism: str = Field(default="Homo sapiens", description="Taxonomic species")
    tissue_source: str = Field(..., description="Anatomical tissue origin")


class InputFileSpec(BaseModel):
    """Input file specification with SHA-256 validation."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(..., description="Logical file identifier")
    relative_path: str = Field(..., description="Path relative to manifest directory")
    sha256: str = Field(
        ..., pattern="^[a-fA-F0-9]{64}$", description="HEX SHA-256 digest"
    )
    file_type: str = Field(
        ..., description="File format (e.g. h5ad, zarr, raw_matrix_dir)"
    )


class IntakeManifest(BaseModel):
    """Comprehensive intake manifest contract for spatial and single-cell deliveries."""

    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(..., description="Unique study or grant identifier")
    schema_version: str = Field(default="0.1.0", description="Contract schema version")
    modality: Modality = Field(..., description="Platform assay modality")
    intake_class: IntakeClass = Field(
        default=IntakeClass.CLASS_B_PROCESSED_NATIVE,
        description="Dataset intake class (A-G)",
    )
    modality_linkage: ModalityLinkage = Field(
        default=ModalityLinkage.SAME_CELL_OR_NUCLEUS,
        description="Physical modality linkage classification",
    )
    specimens: List[BiologicalSpecimen] = Field(
        ..., min_length=1, description="List of biological specimens"
    )
    input_files: List[InputFileSpec] = Field(
        ..., min_length=1, description="List of verified input files"
    )
    feature_types: List[FeatureType] = Field(
        default=[FeatureType.RNA], description="Active feature channels"
    )
    reference_genome: Optional[str] = Field(
        default="GRCh38", description="Reference genome build"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional key-value metadata"
    )
