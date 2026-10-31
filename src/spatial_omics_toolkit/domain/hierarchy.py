"""Domain logic for independent biological unit hierarchy validation."""

from typing import Dict, List, Tuple
from spatial_omics_toolkit.contracts.intake import BiologicalSpecimen
from spatial_omics_toolkit.contracts.status import ErrorCode, StructuredErrorResponse


def validate_biological_units(
    specimens: List[BiologicalSpecimen], min_biological_units_per_condition: int = 2
) -> Tuple[bool, List[StructuredErrorResponse]]:
    """Validate that specimen hierarchy contains sufficient independent biological units (n).

    Prevents pseudoreplication by verifying that distinct biological_unit_ids exist per condition.
    """
    errors: List[StructuredErrorResponse] = []
    condition_to_units: Dict[str, set] = {}

    for spec in specimens:
        cond = spec.condition
        unit = spec.biological_unit_id
        if cond not in condition_to_units:
            condition_to_units[cond] = set()
        condition_to_units[cond].add(unit)

    for cond, units in condition_to_units.items():
        n_units = len(units)
        if n_units < min_biological_units_per_condition:
            errors.append(
                StructuredErrorResponse(
                    error_code=ErrorCode.INSUFFICIENT_BIOLOGICAL_REPLICATES,
                    reason=f"Condition '{cond}' has only {n_units} independent biological unit(s). Minimum required is {min_biological_units_per_condition}.",
                    remediation_hint=f"Add additional biological specimens (patients/cores) for condition '{cond}'. Cells/spots cannot be used as independent replicates.",
                    context={"condition": cond, "n_units": str(n_units)},
                )
            )

    is_valid = len(errors) == 0
    return is_valid, errors
