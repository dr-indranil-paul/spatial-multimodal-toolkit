"""Intake manifest loader and semantic validator."""

import json
from pathlib import Path
from typing import Tuple, List, Union, Optional
from pydantic import ValidationError

from spatial_omics_toolkit.contracts.intake import IntakeManifest
from spatial_omics_toolkit.contracts.status import ErrorCode, StructuredErrorResponse
from spatial_omics_toolkit.domain.hierarchy import validate_biological_units


def load_and_validate_manifest(
    manifest_path: Union[str, Path]
) -> Tuple[Optional[IntakeManifest], List[StructuredErrorResponse]]:
    """Load and validate an intake manifest JSON file.
    
    Performs Pydantic schema validation and biological unit hierarchy checks.
    """
    path = Path(manifest_path)
    errors: List[StructuredErrorResponse] = []

    if not path.is_file():
        errors.append(
            StructuredErrorResponse(
                error_code=ErrorCode.FILE_NOT_FOUND,
                reason=f"Manifest file not found at '{path}'",
                remediation_hint="Provide a valid absolute or relative path to intake_manifest.json",
                context={"path": str(path)}
            )
        )
        return None, errors

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = IntakeManifest.model_validate(data)
    except ValidationError as ve:
        errors.append(
            StructuredErrorResponse(
                error_code=ErrorCode.INVALID_MANIFEST,
                reason="Manifest failed Pydantic schema validation.",
                remediation_hint="Ensure all required fields (study_id, modality, specimens, input_files) match schema.",
                context={"validation_errors": str(ve)}
            )
        )
        return None, errors
    except json.JSONDecodeError as jde:
        errors.append(
            StructuredErrorResponse(
                error_code=ErrorCode.INVALID_MANIFEST,
                reason=f"Manifest is not valid JSON: {jde}",
                remediation_hint="Fix syntax errors in intake_manifest.json",
                context={"json_error": str(jde)}
            )
        )
        return None, errors

    # Biological hierarchy check
    valid_hierarchy, hierarchy_errors = validate_biological_units(manifest.specimens)
    errors.extend(hierarchy_errors)

    if not valid_hierarchy:
        return manifest, errors

    return manifest, []
