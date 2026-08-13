"""Unit tests for Pydantic contracts and hierarchy validation."""

import pytest
from pydantic import ValidationError

from spatial_omics_toolkit.contracts.intake import (
    BiologicalSpecimen,
    InputFileSpec,
    IntakeManifest,
    Modality,
    FeatureType,
)
from spatial_omics_toolkit.contracts.job import ResourceEnvelope, JobSpec
from spatial_omics_toolkit.contracts.status import ErrorCode, StructuredErrorResponse
from spatial_omics_toolkit.domain.hierarchy import validate_biological_units


def test_resource_envelope_cpus_limit():
    """Verify resource envelope caps CPUs at 32 per HPC rules."""
    env = ResourceEnvelope(cpus=16, memory_gb=64.0)
    assert env.cpus == 16

    with pytest.raises(ValidationError):
        ResourceEnvelope(cpus=48)  # Exceeds max 32 CPUs


def test_biological_unit_hierarchy_validation():
    """Test enforcement of independent biological units (n)."""
    # Invalid hierarchy: Condition 'Tumor' has only 1 biological unit (Patient_1)
    specimens = [
        BiologicalSpecimen(
            biological_unit_id="Patient_1",
            specimen_id="Specimen_1A",
            condition="Tumor",
            tissue_source="Colon",
        ),
        BiologicalSpecimen(
            biological_unit_id="Patient_1",
            specimen_id="Specimen_1B",
            condition="Tumor",
            tissue_source="Colon",
        ),
        BiologicalSpecimen(
            biological_unit_id="Patient_2",
            specimen_id="Specimen_2A",
            condition="Normal",
            tissue_source="Colon",
        ),
        BiologicalSpecimen(
            biological_unit_id="Patient_3",
            specimen_id="Specimen_3A",
            condition="Normal",
            tissue_source="Colon",
        ),
    ]

    is_valid, errors = validate_biological_units(specimens)
    assert not is_valid
    assert len(errors) == 1
    assert errors[0].error_code == ErrorCode.INSUFFICIENT_BIOLOGICAL_REPLICATES


def test_intake_manifest_valid():
    """Test valid intake manifest model construction."""
    manifest = IntakeManifest(
        study_id="STUDY_001",
        modality=Modality.VISIUM,
        specimens=[
            BiologicalSpecimen(
                biological_unit_id="Patient_1",
                specimen_id="Spec_1",
                condition="Tumor",
                tissue_source="Lung",
            ),
            BiologicalSpecimen(
                biological_unit_id="Patient_2",
                specimen_id="Spec_2",
                condition="Tumor",
                tissue_source="Lung",
            ),
        ],
        input_files=[
            InputFileSpec(
                file_id="counts_1",
                relative_path="raw/counts.h5ad",
                sha256="a" * 64,
                file_type="h5ad",
            )
        ],
        feature_types=[FeatureType.RNA, FeatureType.PROTEIN],
    )
    assert manifest.study_id == "STUDY_001"
    assert FeatureType.PROTEIN in manifest.feature_types
