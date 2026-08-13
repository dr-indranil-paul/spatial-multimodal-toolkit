"""Abstract base plugin interface for SpatialOmics Toolkit."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from spatial_omics_toolkit.contracts.intake import IntakeManifest
from spatial_omics_toolkit.contracts.job import JobSpec
from spatial_omics_toolkit.contracts.status import StructuredErrorResponse


class PluginInterface(ABC):
    """Abstract base class for all scientific modality plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin unique name (e.g. scrna, visium, xenium, spatial_proteomics)."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin semver string."""
        pass

    @abstractmethod
    def validate_intake(self, manifest: IntakeManifest) -> Tuple[bool, List[StructuredErrorResponse]]:
        """Validate assay-specific intake manifest parameters and files."""
        pass

    @abstractmethod
    def plan_execution(self, job_spec: JobSpec) -> Dict[str, str]:
        """Resolve Nextflow execution DAG and resource parameters for the plugin."""
        pass
