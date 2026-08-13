"""Spatial Proteomics & Multiplex Imaging (CyCIF) plugin implementation."""

from typing import Dict, List, Tuple
from spatial_omics_toolkit.plugins.base import PluginInterface
from spatial_omics_toolkit.contracts.intake import IntakeManifest, Modality, FeatureType
from spatial_omics_toolkit.contracts.job import JobSpec
from spatial_omics_toolkit.contracts.status import StructuredErrorResponse, ErrorCode


class SpatialProteomicsPlugin(PluginInterface):
    """Plugin implementation for CyCIF, MCMICRO, and multiplexed spatial proteomics."""

    @property
    def name(self) -> str:
        return "spatial_proteomics"

    @property
    def version(self) -> str:
        return "0.1.0"

    def validate_intake(self, manifest: IntakeManifest) -> Tuple[bool, List[StructuredErrorResponse]]:
        """Validate intake manifest for spatial proteomics deliveries."""
        errors: List[StructuredErrorResponse] = []

        if manifest.modality not in (Modality.SPATIAL_PROTEOMICS, Modality.VISIUM):
            errors.append(
                StructuredErrorResponse(
                    error_code=ErrorCode.INVALID_MANIFEST,
                    reason=f"Modality '{manifest.modality.value}' is not supported by spatial_proteomics plugin.",
                    remediation_hint="Set manifest modality to 'spatial_proteomics'.",
                    context={"modality": manifest.modality.value}
                )
            )

        # Ensure PROTEIN feature type is present
        if FeatureType.PROTEIN not in manifest.feature_types:
            errors.append(
                StructuredErrorResponse(
                    error_code=ErrorCode.INVALID_MANIFEST,
                    reason="Feature type 'protein' is required for spatial proteomics analysis.",
                    remediation_hint="Add 'protein' to manifest feature_types list.",
                    context={"feature_types": [f.value for f in manifest.feature_types]}
                )
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def plan_execution(self, job_spec: JobSpec) -> Dict[str, str]:
        """Plan execution stages for multiplexed spatial proteomics."""
        return {
            "plugin": self.name,
            "version": self.version,
            "stages": [
                "validate_cycif_manifest",
                "cell_segmentation_cellpose_stardist",
                "protein_intensity_extraction",
                "tissue_mask_spatial_gsi_null",
                "pseudobulk_sample_aggregation",
                "risk_stratification_differential",
                "ome_zarr_pyramidal_tiling"
            ],
            "nextflow_process": "SPATIAL_PROTEOMICS_WORKFLOW"
        }
