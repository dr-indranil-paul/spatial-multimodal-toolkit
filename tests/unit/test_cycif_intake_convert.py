"""Unit tests for step01_scan_intake (cycif/intake.py) and step02_convert_raw_images
(cycif/convert.py).

These test pure logic against small synthetic inputs (fake filenames,
tiny in-memory DataFrames/arrays built on disk in a tmp_path fixture) —
no real gigapixel CZI/ND2 files are available in this environment, and
none of these tests pretend otherwise. Anything that genuinely requires
reading raw pixel data (aicspylibczi/nd2, not installed here) is verified
to fail loudly with a structured, informative result rather than raising
an unhandled exception or silently doing nothing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatial_omics_toolkit.cycif.config import (
    ImageConversionConfig,
    RawImageIntakeConfig,
    SlurmDispatchConfig,
)
from spatial_omics_toolkit.cycif.convert import (
    build_conversion_sbatch_script,
    convert_czi_file,
    convert_nd2_file,
    convert_raw_images_inprocess,
    create_rgb_composite,
    enhance_image,
    get_color_map,
    step02_convert_raw_images,
)
from spatial_omics_toolkit.cycif.intake import (
    build_intake_report,
    check_channel_counts,
    check_dimensions,
    check_pixel_sizes,
    check_round_scene_coverage,
    find_all_round_paths,
    find_sample_raw_file,
    parse_filenames,
    parse_intake_filename,
    scan_raw_images,
    step01_scan_intake,
)

# ── Fixtures: synthetic hBE-TMA-style filenames ─────────────────────────────

GOOD_FILENAMES = [
    "R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_2026_05_16__17930-sceneA1.czi",
    "R2_Ecad.Vim.CK5.CK19_hBE-TMA1-1_2026_05_17__17931-sceneA1.czi",
    "R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_2026_05_16__17932-sceneA2.czi",
]
BAD_FILENAME = "not_a_valid_filename.czi"


def _make_raw_dir(tmp_path: Path, filenames, slide="hBE-TMA1-1") -> Path:
    raw_dir = tmp_path / "raw"
    slide_dir = raw_dir / slide
    slide_dir.mkdir(parents=True)
    for fn in filenames:
        (slide_dir / fn).write_bytes(b"0" * 5_000_000)
    return raw_dir


# ── step01: filename parsing ─────────────────────────────────────────────


def test_parse_intake_filename_matches_expected_fields():
    parsed = parse_intake_filename(GOOD_FILENAMES[0])
    assert parsed is not None
    assert parsed["round"] == "R1"
    assert parsed["markers"] == "MUC1.MUC5AC.MUC2.MUC6"
    assert parsed["slide"] == "hBE-TMA1-1"
    assert parsed["date"] == "2026-05-16"
    assert parsed["acq_id"] == "17930"
    assert parsed["scene"] == "sceneA1"


def test_parse_intake_filename_rejects_non_matching():
    assert parse_intake_filename(BAD_FILENAME) is None


def test_parse_intake_filename_custom_pattern_is_a_parameter_not_hardcoded():
    # A totally different naming convention works as long as it defines
    # the required named groups -- proves the regex is a parameter, not
    # a hardcoded hBE-TMA constant.
    custom_pattern = (
        r"^(?P<round>Cycle\d+)_(?P<slide>[A-Za-z0-9]+)_(?P<scene>Pos\d+)_"
        r"(?P<markers>\w+)_(?P<date>\d{8})_(?P<acq_id>\d+)\.czi$"
    )
    fn = "Cycle3_SlideX_Pos7_CD45_20260101_42.czi"
    parsed = parse_intake_filename(fn, pattern=custom_pattern)
    # date normalization only rewrites underscores (matches the default
    # pattern's YYYY_MM_DD capture); a pattern that captures a differently
    # formatted date is passed through as captured.
    assert parsed == {
        "round": "Cycle3",
        "slide": "SlideX",
        "scene": "Pos7",
        "markers": "CD45",
        "date": "20260101",
        "acq_id": "42",
    }


def test_parse_intake_filename_custom_pattern_missing_group_raises():
    bad_pattern = r"^(?P<round>R\d+)_(?P<slide>\w+)\.czi$"
    with pytest.raises(ValueError, match="missing required named group"):
        parse_intake_filename("R1_slideA.czi", pattern=bad_pattern)


# ── step01: scanning + parsing over a synthetic directory ──────────────────


def test_scan_raw_images_missing_dir_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_raw_images(str(tmp_path / "does_not_exist"))


def test_scan_raw_images_finds_files_and_computes_size(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    df = scan_raw_images(str(raw_dir))
    assert len(df) == 3
    assert set(df["fname"]) == set(GOOD_FILENAMES)
    assert (df["size_gb"] > 0).all()
    assert all(Path(p).exists() for p in df["path"])


def test_parse_filenames_separates_parsed_and_unparsed(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES + [BAD_FILENAME])
    df_files = scan_raw_images(str(raw_dir))
    df_parsed, unparsed = parse_filenames(df_files)
    assert len(df_parsed) == 3
    assert unparsed == [BAD_FILENAME]
    # path is carried through -- not reconstructed from parsed pieces
    assert all(Path(p).exists() for p in df_parsed["path"])


def test_check_round_scene_coverage_flags_missing_rounds(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    df_files = scan_raw_images(str(raw_dir))
    df_parsed, _ = parse_filenames(df_files)
    pivot, missing = check_round_scene_coverage(df_parsed, expected_rounds=("R1", "R2"))
    # sceneA1 has R1+R2 -> fully covered; sceneA2 only has R1 -> missing R2
    missing_scenes = {(m["scene"], m["round"]) for m in missing}
    assert ("sceneA2", "R2") in missing_scenes
    assert ("sceneA1", "R1") not in missing_scenes
    assert ("sceneA1", "R2") not in missing_scenes


def test_check_round_scene_coverage_empty_input_returns_empty():
    pivot, missing = check_round_scene_coverage(
        pd.DataFrame(columns=["slide", "scene", "round", "size_gb"])
    )
    assert len(pivot) == 0
    assert missing == []


# ── step01: OME-XML checks (pure dict/DataFrame logic, no real XML files needed) ──


def test_check_channel_counts_counts_mismatches():
    xml_records = [
        {"n_channels": 5, "xml_file": "a.ome.xml"},
        {"n_channels": 5, "xml_file": "b.ome.xml"},
        {"n_channels": 4, "xml_file": "c.ome.xml"},
    ]
    counts, bad = check_channel_counts(xml_records, expected=5)
    assert counts == {5: ["a.ome.xml", "b.ome.xml"], 4: ["c.ome.xml"]}
    assert bad == 1


def test_check_pixel_sizes_flags_out_of_bounds():
    xml_records = [
        {"slide": "S1", "xml_file": "a.ome.xml", "pixel_size_um": 0.325},
        {"slide": "S1", "xml_file": "b.ome.xml", "pixel_size_um": 99.0},  # implausible
        {"slide": "S1", "xml_file": "c.ome.xml", "pixel_size_um": None},  # dropped
    ]
    df = check_pixel_sizes(xml_records, bounds_um=(0.05, 2.0))
    assert len(df) == 2  # None dropped
    assert df["within_bounds"].tolist() == [True, False]


def test_check_dimensions_extracts_size_fields():
    xml_records = [
        {
            "slide": "S1",
            "xml_file": "a.ome.xml",
            "dims": {"SizeX": "1024", "SizeY": "2048", "SizeC": "5", "SizeZ": "1"},
        }
    ]
    df = check_dimensions(xml_records)
    assert df.loc[0, "Width"] == 1024
    assert df.loc[0, "Height"] == 2048
    assert df.loc[0, "Channels"] == 5


# ── step01: report table never invents a default, and returns structured problems ──


def test_build_intake_report_skips_per_slide_count_when_not_specified(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    df_files = scan_raw_images(str(raw_dir))
    df_parsed, unparsed = parse_filenames(df_files)
    config = RawImageIntakeConfig()  # expected_files_per_slide=None by default
    report = build_intake_report(
        df_files, df_parsed, unparsed, [], pd.DataFrame(), [], config=config
    )
    row = report[report["Check"] == "Per-slide file count"].iloc[0]
    assert row["Status"] == "—"  # not silently defaulted to a fabricated pass/fail


def test_build_intake_report_flags_unparsed_and_missing_as_structured_problems(
    tmp_path,
):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES + [BAD_FILENAME])
    df_files = scan_raw_images(str(raw_dir))
    df_parsed, unparsed = parse_filenames(df_files)
    _, missing = check_round_scene_coverage(
        df_parsed, expected_rounds=("R1", "R2", "R3")
    )
    report = build_intake_report(
        df_files, df_parsed, unparsed, [], pd.DataFrame(), missing
    )
    parsing_row = report[report["Check"] == "Filename parsing"].iloc[0]
    assert parsing_row["Status"] == "⚠"
    coverage_row = report[report["Check"].str.contains("expected rounds present")].iloc[
        0
    ]
    assert coverage_row["Status"] == "⚠"
    assert str(len(missing)) in coverage_row["Detail"]


# ── step01: find_sample_raw_file / find_all_round_paths use carried path ───


def test_find_sample_raw_file_and_all_round_paths(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    df_files = scan_raw_images(str(raw_dir))
    df_parsed, _ = parse_filenames(df_files)

    path = find_sample_raw_file(
        df_parsed, slide="hBE-TMA1-1", scene="sceneA1", round_label="R1"
    )
    assert path is not None
    assert Path(path).exists()
    assert Path(path).name == GOOD_FILENAMES[0]

    missing = find_sample_raw_file(
        df_parsed, slide="hBE-TMA1-1", scene="sceneA1", round_label="R99"
    )
    assert missing is None

    rounds = find_all_round_paths(df_parsed, slide="hBE-TMA1-1", scene="sceneA1")
    assert set(rounds.keys()) == {"R1", "R2"}


# ── step01: full orchestrator end to end on synthetic data ─────────────────


def test_step01_scan_intake_end_to_end_without_html(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    reports_dir = tmp_path / "reports"
    result = step01_scan_intake(
        str(raw_dir),
        str(reports_dir),
        generate_html_report=False,
        verbose=False,
    )
    assert len(result.inventory) == 3
    assert len(result.parsed) == 3
    assert result.unparsed == []
    assert result.html_report_path is None
    assert "Filename parsing" in result.checks["Check"].tolist()


def test_step01_scan_intake_generates_html_report(tmp_path):
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES)
    reports_dir = tmp_path / "reports"
    result = step01_scan_intake(str(raw_dir), str(reports_dir), verbose=False)
    assert result.html_report_path is not None
    html_path = Path(result.html_report_path)
    assert html_path.exists()
    html = html_path.read_text()
    assert "<html" in html.lower()
    assert (reports_dir / "intake_inventory.csv").exists()
    assert (reports_dir / "intake_checks.csv").exists()


def test_step01_scan_intake_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        step01_scan_intake(str(tmp_path / "nope"), str(tmp_path / "reports"))


def test_step01_scan_intake_strict_raises_on_failed_check(tmp_path):
    # BAD_FILENAME guarantees the "Filename parsing" check fails.
    raw_dir = _make_raw_dir(tmp_path, GOOD_FILENAMES + [BAD_FILENAME])
    reports_dir = tmp_path / "reports"
    with pytest.raises(RuntimeError):
        step01_scan_intake(
            str(raw_dir),
            str(reports_dir),
            generate_html_report=False,
            strict=True,
            verbose=False,
        )


# ── step02: pure pixel-logic helpers (no raw-format reader needed) ─────────


def test_get_color_map_keyword_matching():
    assert get_color_map("DAPI") == [0.1, 0.4, 1.0]
    assert get_color_map("AF488") == [0.0, 1.0, 0.0]
    assert get_color_map("Cy5") == [1.0, 0.0, 0.0]
    assert get_color_map("SomeUnknownDye") == [1.0, 1.0, 1.0]


def test_get_color_map_custom_override_checked_first():
    custom = {"CUSTOMDYE": (0.5, 0.5, 0.5)}
    assert get_color_map("CustomDye_Ch0", custom_map=custom) == [0.5, 0.5, 0.5]
    # Falls through to built-in table when custom key doesn't match.
    assert get_color_map("DAPI", custom_map=custom) == [0.1, 0.4, 1.0]


def test_enhance_image_contrast_stretch_uint16_output():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, size=(32, 32)).astype(np.uint16)
    out = enhance_image(img, percentiles=(2.0, 98.0))
    assert out.dtype == np.uint16
    assert out.shape == img.shape
    assert out.max() <= 65535
    assert out.min() >= 0


def test_create_rgb_composite_shape_and_dtype():
    stack = np.zeros((2, 16, 16), dtype=np.uint16)
    stack[0] = 40000  # DAPI-like channel, high signal
    stack[1] = 0
    rgb = create_rgb_composite(stack, ["DAPI", "AF488"])
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8
    # DAPI channel is blue-weighted -> blue channel should dominate where DAPI is bright
    assert rgb[..., 2].mean() > rgb[..., 1].mean()


# ── step02: raw-format readers fail loudly and structured, not silently ────


def test_convert_czi_file_missing_dependency_returns_structured_failure(tmp_path):
    # Verifies that parsing invalid CZI data or missing dependency returns a structured
    # informative record instead of raising an unhandled crash or silently no-op'ing.
    fake_czi = tmp_path / "fake.czi"
    fake_czi.write_bytes(b"not a real czi")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = convert_czi_file(fake_czi, out_dir)
    assert result["status"] == "failed"
    assert result["outputs"] == []
    assert result["error"] is not None and len(result["error"]) > 0
    assert not (out_dir / "fake.tif").exists()


def test_convert_nd2_file_missing_dependency_returns_structured_failure(tmp_path):
    fake_nd2 = tmp_path / "fake.nd2"
    fake_nd2.write_bytes(b"not a real nd2")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = convert_nd2_file(fake_nd2, out_dir)
    assert result["status"] == "failed"
    assert result["outputs"] == []
    assert result["error"] is not None and len(result["error"]) > 0


def test_convert_raw_images_inprocess_strict_raises_on_failures(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "a.czi").write_bytes(b"not real")
    output_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="failed conversion"):
        convert_raw_images_inprocess(input_dir, output_dir, strict=True)


def test_convert_raw_images_inprocess_non_strict_returns_report(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "a.czi").write_bytes(b"not real")
    output_dir = tmp_path / "out"
    df = convert_raw_images_inprocess(input_dir, output_dir, strict=False)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "failed"


def test_convert_raw_images_inprocess_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_raw_images_inprocess(tmp_path / "nope", tmp_path / "out")


def test_convert_raw_images_inprocess_empty_folder_returns_empty_report(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    df = convert_raw_images_inprocess(input_dir, output_dir, strict=True)
    assert len(df) == 0


# ── step02: SLURM job building (no cluster / sbatch needed) ────────────────


def test_slurm_dispatch_config_rejects_too_many_cpus():
    with pytest.raises(ValueError, match="exceeds this cluster"):
        SlurmDispatchConfig(cpus_per_task=48)


def test_build_conversion_sbatch_script_contains_resource_directives(tmp_path):
    slurm_cfg = SlurmDispatchConfig(
        job_name="test_job", cpus_per_task=4, mem_gb=16, time_limit="01:00:00"
    )
    script = build_conversion_sbatch_script(
        tmp_path / "in",
        tmp_path / "out",
        slurm_config=slurm_cfg,
        conversion_config=ImageConversionConfig(generate_mip=False),
        log_dir=str(tmp_path / "logs"),
    )
    assert "#!/bin/bash" in script
    assert "--job-name=test_job" in script
    assert "--cpus-per-task=4" in script
    assert "--mem=16G" in script
    assert "--time=01:00:00" in script
    assert "python -m spatial_omics_toolkit.cycif.convert" in script
    assert "--no-mip" in script
    assert "--no-composite" not in script  # composite still enabled


def test_step02_convert_raw_images_slurm_dispatch_builds_script_without_blocking(
    tmp_path,
):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    # Deliberately no real CZI/ND2 files -- proves dispatch="slurm" never
    # touches pixel data or requires aicspylibczi/nd2 to be installed.
    output_dir = tmp_path / "out"
    result = step02_convert_raw_images(
        str(input_dir),
        str(output_dir),
        dispatch="slurm",
        submit=False,
        verbose=False,
    )
    assert result["dispatch"] == "slurm"
    assert result["submitted"] is False
    assert Path(result["script_path"]).exists()
    assert result["sbatch_cmd"].startswith("sbatch ")


def test_step02_convert_raw_images_invalid_dispatch_raises():
    with pytest.raises(ValueError, match="dispatch must be"):
        step02_convert_raw_images("in", "out", dispatch="not_a_real_mode")


def test_step02_convert_raw_images_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        step02_convert_raw_images(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_step02_convert_raw_images_inprocess_dispatch_runs_directly(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    df = step02_convert_raw_images(
        str(input_dir),
        str(output_dir),
        dispatch="inprocess",
        verbose=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
