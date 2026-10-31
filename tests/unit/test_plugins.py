"""Unit tests for scientific plugins."""

from spatial_omics_toolkit.plugins.spatial_proteomics import SpatialProteomicsPlugin
from spatial_omics_toolkit.contracts.intake import (
    IntakeManifest,
    Modality,
    FeatureType,
    BiologicalSpecimen,
    InputFileSpec,
)
from spatial_omics_toolkit.contracts.job import JobSpec


def test_spatial_proteomics_plugin_validation():
    """Test spatial proteomics plugin validation logic."""
    plugin = SpatialProteomicsPlugin()
    assert plugin.name == "spatial_proteomics"

    valid_manifest = IntakeManifest(
        study_id="PREVAIL_001",
        modality=Modality.SPATIAL_PROTEOMICS,
        feature_types=[FeatureType.PROTEIN, FeatureType.MORPHOLOGY],
        specimens=[
            BiologicalSpecimen(
                biological_unit_id="Patient_1",
                specimen_id="TMA1_1",
                condition="Progressor",
                tissue_source="Esophagus",
            ),
            BiologicalSpecimen(
                biological_unit_id="Patient_2",
                specimen_id="TMA2_1",
                condition="Non_progressor",
                tissue_source="Esophagus",
            ),
        ],
        input_files=[
            InputFileSpec(
                file_id="cycif_1",
                relative_path="raw/cycif.ome.tiff",
                sha256="a" * 64,
                file_type="ome_tiff",
            )
        ],
    )

    is_valid, errors = plugin.validate_intake(valid_manifest)
    assert is_valid
    assert len(errors) == 0


def test_spatial_proteomics_execution_plan():
    """Test spatial proteomics execution planning."""
    plugin = SpatialProteomicsPlugin()
    job = JobSpec(
        job_id="JOB_123",
        owner="dr_paul",
        study_id="PREVAIL_001",
        manifest_path="intake_manifest.json",
        plugin_name="spatial_proteomics",
        output_dir="/home/groups/EmiliLab/pauli/scratch/work/run_123",
    )
    plan = plugin.plan_execution(job)
    assert plan["plugin"] == "spatial_proteomics"
    assert "cell_segmentation_cellpose_stardist" in plan["stages"]
    assert "pseudobulk_sample_aggregation" in plan["stages"]


def test_prevail_barretts_cycif_manifest_fixture():
    """Test loading and validating the real-world PREVAIL Barretts CyCIF manifest fixture."""
    from pathlib import Path
    from spatial_omics_toolkit.io.manifest import load_and_validate_manifest

    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "prevail_barretts_cycif_manifest.json"
    manifest, errors = load_and_validate_manifest(fixture_path)
    assert manifest is not None
    assert manifest.study_id == "PREVAIL_BARRETTS_ESOPHAGUS"
    assert manifest.modality == Modality.SPATIAL_PROTEOMICS
    assert len(errors) == 0
