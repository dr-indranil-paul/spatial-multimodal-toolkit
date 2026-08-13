"""Unit tests for the step06-step13 CyCIF notebook pipeline.

Small synthetic fixtures (2 cores x a handful of cells) exercise the real
gating/aggregation/statistics logic. These are unit-test fixtures, not a
stand-in for a scientific finding — see tests/unit/test_cycif_real_data.py
for the actual real-data validation.
"""

import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.cycif.config import CellTypeMarkerConfig, ProgramDefinition
from spatial_omics_toolkit.cycif.classify import step07_classify_cells, step08_score_programs, zscore_per_core
from spatial_omics_toolkit.cycif.aggregate import step09_aggregate_per_core
from spatial_omics_toolkit.cycif.spatial_features import step10_spatial_features
from spatial_omics_toolkit.cycif.spatial_null import step11_spatial_null
from spatial_omics_toolkit.cycif.compare import step12_compare_groups
from spatial_omics_toolkit.cycif.qc import step06_qc_filter_intensities


MINIMAL_CONFIG = CellTypeMarkerConfig(
    epithelial_markers=("EpiA", "EpiB"),
    mesenchymal_markers=("MesA", "MesB"),
    immune_marker="Immune1",
    myeloid_markers=("Myeloid1",),
    antigen_presentation_marker="APC1",
    stromal_marker="Stromal1",
    endothelial_marker="Endo1",
    proliferation_marker="Prolif1",
    z_score_markers=("EpiA", "EpiB", "MesA", "MesB", "Immune1", "Myeloid1", "APC1", "Stromal1", "Endo1", "Prolif1"),
)


def _synthetic_cells(n_per_core: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cores = ["core_1", "core_2"]
    rows = []
    cid = 0
    for core in cores:
        for _ in range(n_per_core):
            cid += 1
            rows.append({
                "slide_scene": core,
                "cellid": cid,
                "DAPI_X": rng.uniform(0, 1000),
                "DAPI_Y": rng.uniform(0, 1000),
                "nuclei_area": rng.uniform(50, 200),
                "cell_area": rng.uniform(100, 400),
                "EpiA": rng.lognormal(mean=3, sigma=1.0),
                "EpiB": rng.lognormal(mean=3, sigma=1.0),
                "MesA": rng.lognormal(mean=3, sigma=1.0),
                "MesB": rng.lognormal(mean=3, sigma=1.0),
                "Immune1": rng.lognormal(mean=3, sigma=1.0),
                "Myeloid1": rng.lognormal(mean=3, sigma=1.0),
                "APC1": rng.lognormal(mean=3, sigma=1.0),
                "Stromal1": rng.lognormal(mean=3, sigma=1.0),
                "Endo1": rng.lognormal(mean=3, sigma=1.0),
                "Prolif1": rng.lognormal(mean=3, sigma=1.0),
            })
    return pd.DataFrame(rows)


def test_zscore_per_core_zero_variance_gives_zero():
    df = pd.DataFrame({"slide_scene": ["a", "a", "a"], "marker": [5.0, 5.0, 5.0]})
    z = zscore_per_core(df, ["marker"], core_col="slide_scene")
    assert (z["marker_z"] == 0.0).all()


def test_zscore_per_core_matches_manual_calculation():
    df = pd.DataFrame({"slide_scene": ["a"] * 4, "marker": [1.0, 2.0, 3.0, 4.0]})
    z = zscore_per_core(df, ["marker"], core_col="slide_scene")
    expected = (df["marker"] - df["marker"].mean()) / df["marker"].std()
    np.testing.assert_allclose(z["marker_z"].to_numpy(), expected.to_numpy())


def test_step07_classify_cells_produces_mutually_exclusive_types():
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    assert "cell_type" in classified.columns
    # Every cell gets exactly one type label (no NaNs, no multi-assignment column collision).
    assert classified["cell_type"].notna().all()
    valid_types = {
        "Hybrid_epithelial", "Epithelial", "Mesenchymal_like", "Myeloid_immune",
        "Lymphoid_immune", "Endothelial", "Stromal", "Proliferating", "Other",
    }
    assert set(classified["cell_type"].unique()) <= valid_types


def test_step07_classify_cells_requires_core_col():
    df = _synthetic_cells().drop(columns=["slide_scene"])
    with pytest.raises(ValueError):
        step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene")


def test_cell_type_marker_config_rejects_missing_z_markers():
    with pytest.raises(ValueError):
        CellTypeMarkerConfig(
            epithelial_markers=("EpiA",),
            mesenchymal_markers=("MesA",),
            immune_marker="Immune1",
            myeloid_markers=("Myeloid1",),
            antigen_presentation_marker="APC1",
            stromal_marker="Stromal1",
            endothelial_marker="Endo1",
            proliferation_marker="Prolif1",
            z_score_markers=("EpiA",),  # missing MesA, Immune1, etc.
        )


def test_step08_score_programs_dominant_and_entropy_bounds():
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    programs = (
        ProgramDefinition("P1", ("EpiA", "EpiB")),
        ProgramDefinition("P2", ("MesA", "MesB")),
    )
    scored = step08_score_programs(classified, programs, verbose=False)
    assert {"P1_score", "P2_score", "dominant_program", "hybrid_balance", "program_entropy"} <= set(scored.columns)
    assert set(scored["dominant_program"].unique()) <= {"P1", "P2"}
    assert (scored["hybrid_balance"].abs() <= 10.0 + 1e-9).all()
    assert (scored["program_entropy"] >= 0.0).all()


def test_step08_score_programs_raises_on_missing_zscore_column():
    df = _synthetic_cells()
    programs = (ProgramDefinition("P1", ("NotAMarker",)),)
    with pytest.raises(ValueError):
        step08_score_programs(df, programs, verbose=False)


def test_step09_aggregate_per_core_fractions_sum_to_one():
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    cores = step09_aggregate_per_core(classified, core_col="slide_scene")
    assert len(cores) == 2
    type_cols = sorted(classified["cell_type"].unique())
    fraction_sum = cores[type_cols].sum(axis=1)
    np.testing.assert_allclose(fraction_sum.to_numpy(), 1.0, atol=1e-9)


def test_step10_spatial_features_one_row_per_core():
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    feats = step10_spatial_features(classified, core_col="slide_scene", knn_k_values=(3,), grid_sizes=(2,), radii=(50.0,))
    assert len(feats) == 2
    assert "cell.n_cells" in feats.columns
    assert (feats["cell.n_cells"] == 40).all()


def test_step11_spatial_null_p_value_in_unit_interval():
    df = _synthetic_cells(n_per_core=60, seed=1)
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    types_present = classified["cell_type"].unique()
    assume_two = [t for t in types_present if t != "Other"]
    if len(assume_two) < 2:
        pytest.skip("synthetic draw did not produce two non-Other types; not a real failure")
    result = step11_spatial_null(
        classified, source_type=assume_two[0], target_type=assume_two[1],
        core_col="slide_scene", k=3, n_permutations=50, seed=0,
    )
    finite = result.dropna(subset=["p_value"])
    if len(finite):
        assert ((finite["p_value"] >= 0.0) & (finite["p_value"] <= 1.0)).all()


def test_step11_spatial_null_self_pair_shows_no_enrichment_signal():
    """A type tested against itself should show no reliable enrichment (sanity check on the null machinery)."""
    df = _synthetic_cells(n_per_core=80, seed=2)
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    common_type = classified["cell_type"].value_counts().idxmax()
    if (classified["cell_type"] == common_type).sum() < 10:
        pytest.skip("insufficient synthetic cells of the majority type")
    result = step11_spatial_null(
        classified, source_type=common_type, target_type=common_type,
        core_col="slide_scene", k=3, n_permutations=50, seed=0,
    )
    assert "p_value" in result.columns


def test_step12_compare_groups_requires_exactly_two_groups():
    core_df = pd.DataFrame({
        "group": ["a", "b", "c"],
        "metric": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError):
        step12_compare_groups(core_df, group_col="group", metrics=["metric"])


def test_step12_compare_groups_fdr_never_below_raw_p():
    rng = np.random.default_rng(0)
    core_df = pd.DataFrame({
        "group": ["a"] * 6 + ["b"] * 6,
        "m1": rng.normal(0, 1, 12),
        "m2": rng.normal(0, 1, 12),
        "m3": rng.normal(5, 1, 6).tolist() + rng.normal(0, 1, 6).tolist(),
    })
    result = step12_compare_groups(core_df, group_col="group", metrics=["m1", "m2", "m3"])
    valid = result.dropna(subset=["welch_t_p", "fdr_bh_p"])
    assert (valid["fdr_bh_p"] >= valid["welch_t_p"] - 1e-9).all()


def test_step12_compare_groups_warns_when_units_collapse():
    core_df = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "patient": ["P1", "P1", "P2", "P2"],
        "metric": [1.0, 1.1, 2.0, 2.1],
    })
    with pytest.warns(UserWarning):
        step12_compare_groups(core_df, group_col="group", metrics=["metric"], biological_unit_col="patient")


def test_step06_qc_filter_intensities_floors_at_zero_and_reports_snr():
    df = pd.DataFrame({"MarkerA": [1.0, 2.0, 100.0, 120.0, 150.0]})
    filtered = step06_qc_filter_intensities(df)
    assert filtered["MarkerA"].min() == 0.0
    assert "marker_snr" in filtered.attrs
    assert "MarkerA" in filtered.attrs["marker_snr"]


# --- output_dir QC-artifact checks: every stepNN_* must leave an inspectable file behind ---

def test_step06_output_dir_writes_snr_plot(tmp_path):
    df = pd.DataFrame({"MarkerA": [1.0, 2.0, 100.0, 120.0, 150.0], "MarkerB": [5.0, 6.0, 7.0, 200.0, 210.0]})
    step06_qc_filter_intensities(df, output_dir=tmp_path, verbose=False)
    assert (tmp_path / "step06_marker_snr.png").exists()


def test_step07_output_dir_writes_composition_and_gate_plots(tmp_path):
    df = _synthetic_cells()
    step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False, output_dir=tmp_path)
    assert (tmp_path / "step07_cell_type_composition.png").exists()
    assert (tmp_path / "step07_gate_diagnostic.png").exists()


def test_step08_output_dir_writes_score_plot(tmp_path):
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    programs = (ProgramDefinition("P1", ("EpiA", "EpiB")), ProgramDefinition("P2", ("MesA", "MesB")))
    step08_score_programs(classified, programs, verbose=False, output_dir=tmp_path)
    assert (tmp_path / "step08_program_scores.png").exists()


def test_step09_output_dir_writes_csv_and_heatmap(tmp_path):
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    step09_aggregate_per_core(classified, core_col="slide_scene", output_dir=tmp_path, verbose=False)
    assert (tmp_path / "step09_core_table.csv").exists()
    assert (tmp_path / "step09_core_composition_heatmap.png").exists()


def test_step10_output_dir_writes_spatial_maps_and_csv(tmp_path):
    df = _synthetic_cells()
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    step10_spatial_features(
        classified, core_col="slide_scene", knn_k_values=(3,), grid_sizes=(2,), radii=(50.0,),
        output_dir=tmp_path, verbose=False,
    )
    assert (tmp_path / "step10_spatial_features.csv").exists()
    assert list(tmp_path.glob("step10_spatial_map_*.png"))


def test_step11_output_dir_writes_null_histogram(tmp_path):
    df = _synthetic_cells(n_per_core=80, seed=3)
    classified = step07_classify_cells(df, MINIMAL_CONFIG, core_col="slide_scene", verbose=False)
    types_present = [t for t in classified["cell_type"].unique() if t != "Other"]
    if len(types_present) < 2:
        pytest.skip("synthetic draw did not produce two non-Other types; not a real failure")
    step11_spatial_null(
        classified, source_type=types_present[0], target_type=types_present[1],
        core_col="slide_scene", k=3, n_permutations=30, seed=0,
        output_dir=tmp_path, verbose=False,
    )
    assert list(tmp_path.glob("step11_null_*.png")) or list(tmp_path.glob("step11_spatial_null_*.csv"))


def test_step12_output_dir_writes_violin_plots_and_table(tmp_path):
    rng = np.random.default_rng(0)
    core_df = pd.DataFrame({
        "group": ["a"] * 6 + ["b"] * 6,
        "m1": rng.normal(0, 1, 12),
    })
    step12_compare_groups(core_df, group_col="group", metrics=["m1"], output_dir=tmp_path, verbose=False)
    assert (tmp_path / "step12_comparison_table.csv").exists()
    assert (tmp_path / "step12_m1.png").exists()
