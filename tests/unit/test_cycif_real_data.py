"""Real-data validation: step07-step09 reproduce the PREVAIL hBE-TMA cohort's
own (already-published-internally, non-fabricated) classification output.

Skipped automatically off the EmiliLab HPC filesystem — these paths are real
patient-derived TMA data, not fixtures, and are never bundled with this
package. On HPC, this is the honest "does the port actually work on the real
data" check requested when this pipeline was built (2026-08-08); it is not a
manuscript result and asserts nothing about progressor vs. non-progressor
biology.
"""

from pathlib import Path

import pandas as pd
import pytest

from spatial_omics_toolkit.cycif.classify import (
    step07_classify_cells,
    step08_score_programs,
)
from spatial_omics_toolkit.cycif.aggregate import step09_aggregate_per_core
from spatial_omics_toolkit.cycif.config import (
    PREVAIL_BARRETTS_EMT_PROGRAMS,
    PREVAIL_BARRETTS_MARKER_CONFIG,
)

KOEI_BASE = Path(
    "/home/groups/EmiliLab/ChinData/Cyclic_Workflow/cmIF_2026-05-15_BE-TMA/Segmentation"
)
FEATURE_FILES = {
    "hBE-TMA1-1": KOEI_BASE
    / "hBE-TMA1-1_CellposeSegmentation"
    / "features_hBE-TMA1-1_Ecad_patched_MeanIntensity_Shape_DAPI1_DAPI5_subtractedregisteredimages.csv",
    "hBE-TMA2-1": KOEI_BASE
    / "hBE-TMA2-1_CellposeSegmentation"
    / "features_hBE-TMA2-1_Ecad_patched_MeanIntensity_Shape_DAPI1_DAPI5_subtractedregisteredimages.csv",
}
CLASSIFICATION_CSV = Path(
    "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/03_tables/tma_cell_classification.csv"
)

_HPC_DATA_AVAILABLE = (
    all(p.exists() for p in FEATURE_FILES.values()) and CLASSIFICATION_CSV.exists()
)

pytestmark = pytest.mark.skipif(
    not _HPC_DATA_AVAILABLE,
    reason="Real PREVAIL hBE-TMA data only available on the EmiliLab HPC filesystem",
)


@pytest.fixture(scope="module")
def real_raw_cells() -> pd.DataFrame:
    chunks = []
    for tma_id, path in FEATURE_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        df["tma_id"] = tma_id
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


@pytest.fixture(scope="module")
def real_reference_classification() -> pd.DataFrame:
    return pd.read_csv(CLASSIFICATION_CSV, low_memory=False)


def test_step07_matches_source_project_cell_type_exactly(
    real_raw_cells, real_reference_classification
):
    classified = step07_classify_cells(
        real_raw_cells,
        PREVAIL_BARRETTS_MARKER_CONFIG,
        core_col="slide_scene",
        verbose=False,
    )
    merged = real_reference_classification[
        ["slide_scene", "cellid", "cell_type"]
    ].merge(
        classified[["slide_scene", "cellid", "cell_type"]],
        on=["slide_scene", "cellid"],
        suffixes=("_reference", "_ported"),
    )
    assert len(merged) == len(real_reference_classification)
    agreement = (merged["cell_type_reference"] == merged["cell_type_ported"]).mean()
    assert (
        agreement == 1.0
    ), f"cell_type agreement with source project's own output was only {agreement:.4f}"


def test_step08_program_scores_match_source_project(
    real_raw_cells, real_reference_classification
):
    classified = step07_classify_cells(
        real_raw_cells,
        PREVAIL_BARRETTS_MARKER_CONFIG,
        core_col="slide_scene",
        verbose=False,
    )
    scored = step08_score_programs(
        classified, PREVAIL_BARRETTS_EMT_PROGRAMS, verbose=False
    )
    merged = real_reference_classification[
        ["slide_scene", "cellid", "EL_score", "IH_score", "ME_score"]
    ].merge(
        scored[["slide_scene", "cellid", "EL_score", "IH_score", "ME_score"]],
        on=["slide_scene", "cellid"],
        suffixes=("_reference", "_ported"),
    )
    for col in ("EL_score", "IH_score", "ME_score"):
        diff = (merged[f"{col}_reference"] - merged[f"{col}_ported"]).abs().max()
        assert (
            diff < 1e-6
        ), f"{col} differs from source project's own output by up to {diff}"


def test_step09_aggregate_matches_published_per_core_table(real_raw_cells):
    reference_cores = pd.read_csv(
        "/home/groups/EmiliLab/pauli/02_projects_current/02_spatial_proteomics/03_analysis/03_tables/tma_per_core_metrics.csv"
    )
    classified = step07_classify_cells(
        real_raw_cells,
        PREVAIL_BARRETTS_MARKER_CONFIG,
        core_col="slide_scene",
        verbose=False,
    )
    scored = step08_score_programs(
        classified, PREVAIL_BARRETTS_EMT_PROGRAMS, verbose=False
    )
    cores = step09_aggregate_per_core(scored, core_col="slide_scene")
    merged = reference_cores.merge(
        cores, on="slide_scene", suffixes=("_reference", "_ported")
    )
    assert len(merged) == len(reference_cores)
    for col in ("Epithelial", "Hybrid_epithelial", "immune_fraction", "total_cells"):
        diff = (merged[f"{col}_reference"] - merged[f"{col}_ported"]).abs().max()
        assert (
            diff < 1e-6
        ), f"{col} differs from published tma_per_core_metrics.csv by up to {diff}"
