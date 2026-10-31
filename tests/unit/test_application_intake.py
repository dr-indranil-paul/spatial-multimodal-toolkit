from pathlib import Path

from spatial_omics_toolkit.application.intake import build_input_inventory, read_project_samples


def test_project_samples_allow_repeated_biological_units_and_unknown(tmp_path: Path) -> None:
    path = tmp_path / "project_samples.csv"
    path.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "donor-1,specimen-1,unknown,unknown\n"
        "donor-1,specimen-2,unknown,unknown\n",
        encoding="utf-8",
    )

    rows, blockers = read_project_samples(path)

    assert blockers == []
    assert len(rows) == 2
    assert {row["biological_unit_id"] for row in rows} == {"donor-1"}


def test_project_samples_reject_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "project_samples.csv"
    path.write_text(
        "biological_unit_id,specimen_id,specimen_id,condition,tissue_source\n"
        "donor-1,specimen-1,specimen-2,control,tissue\n",
        encoding="utf-8",
    )

    rows, blockers = read_project_samples(path)

    assert rows == []
    assert len(blockers) == 1
    assert "duplicate column names" in blockers[0]


def test_project_samples_reject_duplicate_specimen_ids(tmp_path: Path) -> None:
    path = tmp_path / "project_samples.csv"
    path.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "donor-1,specimen-1,control,tissue-a\n"
        "donor-1,specimen-1,control,tissue-a\n",
        encoding="utf-8",
    )

    rows, blockers = read_project_samples(path)

    assert len(rows) == 2
    assert len(blockers) == 1
    assert "repeats specimen_id" in blockers[0]


def test_project_samples_reject_empty_required_values(tmp_path: Path) -> None:
    path = tmp_path / "project_samples.csv"
    path.write_text(
        "biological_unit_id,specimen_id,condition,tissue_source\n"
        "donor-1,specimen-1,,tissue-a\n",
        encoding="utf-8",
    )

    rows, blockers = read_project_samples(path)

    assert len(rows) == 1
    assert any("empty required value: condition" in blocker for blocker in blockers)
def test_selected_inventory_hashes_declared_files_only(tmp_path: Path) -> None:
    modality_dir = tmp_path / "visium"
    modality_dir.mkdir()
    selected_matrix = modality_dir / "filtered_feature_bc_matrix.h5"
    selected_matrix.write_bytes(b"selected count matrix")
    unrelated_bam = modality_dir / "possorted_genome_bam.bam"
    unrelated_bam.write_bytes(b"large unrelated source")
    annotation_csv = tmp_path / "project_samples.csv"
    annotation_csv.write_text("biological_unit_id,specimen_id,condition,tissue_source\n", encoding="utf-8")

    inventory = build_input_inventory(
        modality_dir,
        tmp_path,
        annotation_csv,
        selected_input_paths=[str(selected_matrix)],
    )

    file_entries = {
        entry["path"]: entry
        for entry in inventory["files"]
        if entry.get("type") == "file"
    }
    assert inventory["inventory_scope"] == "workflow_selected_input_files"
    assert str(selected_matrix.absolute()) in file_entries
    assert file_entries[str(selected_matrix.absolute())]["sha256"]
    assert str(unrelated_bam.absolute()) not in file_entries
