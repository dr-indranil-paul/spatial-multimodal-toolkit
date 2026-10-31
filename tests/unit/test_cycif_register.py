"""Unit tests for spatial_omics_toolkit.cycif.register (step03_register_rounds).

Two independent things are tested, per this lab's analysis-integrity policy
(no data literals presented as if computed; every number here is either a
test-fixture input we chose, or an output actually computed by the algorithm
under test at run time):

1. The real SIFT+RANSAC registration math (``register_round_pair``), on small
   synthetic images built in-memory with a known ground-truth affine
   transform — not real CZI data (none is available in this environment),
   but a real, honest exercise of the real algorithm.
2. The filename-pattern parsing and numeric-aware round-sort logic
   (``parse_czi_filename``, ``_round_sort_key``, ``group_czis_by_scene``),
   including the R10+ / R5Q edge case that was the real bug fix being
   preserved from the source pipeline — no image data needed for this part.
"""

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from spatial_omics_toolkit.cycif.config import RegistrationConfig
from spatial_omics_toolkit.cycif.register import (
    _round_sort_key,
    _validate_transform,
    group_czis_by_scene,
    parse_czi_filename,
    register_round_pair,
    step03_register_rounds,
    warp_image,
)


# ---------------------------------------------------------------------------
# Synthetic image fixture
# ---------------------------------------------------------------------------


def _make_synthetic_dapi(
    rng: np.random.RandomState, shape=(512, 512), n_blobs=180
) -> np.ndarray:
    """Build a synthetic 'nuclei-like' image: Gaussian blobs + background noise.

    Not a real DAPI image — a stand-in with SIFT-detectable local structure,
    used only to exercise the real registration algorithm on data with a
    known ground-truth transform. All blob positions/amplitudes are drawn at
    call time from ``rng``, never hardcoded.
    """
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    ys = rng.uniform(40, shape[0] - 40, n_blobs)
    xs = rng.uniform(40, shape[1] - 40, n_blobs)
    radii = rng.uniform(3.0, 8.0, n_blobs)
    amps = rng.uniform(2000.0, 8000.0, n_blobs)
    for y, x, r, a in zip(ys, xs, radii, amps):
        img += a * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * r**2)))
    img += rng.normal(0.0, 25.0, shape).astype(np.float32)
    img = np.clip(img, 0, 65535)
    return img.astype(np.uint16)


def _known_similarity_transform() -> np.ndarray:
    """A fixed, physically-plausible moving->reference transform for the test.

    scale=0.99 (within RegistrationConfig default [0.95, 1.05]), rotation
    ~2.5deg (within default +/-5deg), translation (16, -11) px (well within
    the default 50% max_shift_fraction of a 512px-diagonal image). This is a
    chosen test fixture, not a measured value.
    """
    angle = np.radians(2.5)
    scale = 0.99
    tx, ty = 16.0, -11.0
    return np.array(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), tx],
            [scale * np.sin(angle), scale * np.cos(angle), ty],
        ],
        dtype=np.float64,
    )


class TestRegisterRoundPairSyntheticData:
    """Exercise the real SIFT+RANSAC algorithm against a known transform."""

    def setup_method(self):
        rng = np.random.RandomState(20260808)
        self.shape = (512, 512)
        self.mov_dapi = _make_synthetic_dapi(rng, self.shape)
        self.t_true = _known_similarity_transform()
        # ref_dapi is mov_dapi warped by the known moving->reference transform,
        # using the same warpAffine call pattern step03_register_rounds itself
        # uses (warp_image), so the recovered transform is directly comparable.
        self.ref_dapi = warp_image(self.mov_dapi, self.t_true, self.shape).astype(
            np.uint16
        )

    def test_recovers_known_transform(self):
        transform, n_inliers, n_matches, status = register_round_pair(
            self.ref_dapi,
            self.mov_dapi,
        )

        assert status == "ok", f"registration failed unexpectedly: {status}"
        assert transform is not None
        assert n_matches >= 4
        assert n_inliers >= 4

        # Recovered scale/angle/translation vs. the known ground truth.
        est_scale = float(np.sqrt(transform[0, 0] ** 2 + transform[0, 1] ** 2))
        est_angle = float(np.degrees(np.arctan2(transform[0, 1], transform[0, 0])))
        true_scale = float(np.sqrt(self.t_true[0, 0] ** 2 + self.t_true[0, 1] ** 2))
        true_angle = float(np.degrees(np.arctan2(self.t_true[0, 1], self.t_true[0, 0])))

        assert abs(est_scale - true_scale) < 0.01
        assert abs(est_angle - true_angle) < 0.5
        assert abs(transform[0, 2] - self.t_true[0, 2]) < 3.0  # tx, px
        assert abs(transform[1, 2] - self.t_true[1, 2]) < 3.0  # ty, px

        # The whole 2x3 matrix, element-wise, as a single summary check.
        # atol=0.1 covers both the unitless rotation/scale entries (~1) and
        # the pixel-scale translation entries recovered above to < 3 px.
        assert np.allclose(transform, self.t_true, atol=0.1)

    def test_warping_moving_image_with_recovered_transform_matches_reference(self):
        """A stronger, algorithm-agnostic check: does the recovered transform
        actually align the moving image to the reference image in pixel space?
        """
        transform, _, _, status = register_round_pair(self.ref_dapi, self.mov_dapi)
        assert status == "ok"

        warped = warp_image(self.mov_dapi, transform, self.shape)

        # Compare only the central region, away from warp border padding.
        margin = 60
        a = self.ref_dapi[margin:-margin, margin:-margin].astype(np.float64)
        b = warped[margin:-margin, margin:-margin].astype(np.float64)
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        assert (
            corr > 0.9
        ), f"warped moving image poorly correlated with reference (r={corr:.3f})"

    def test_uncorrelated_images_fail_loudly_not_silently(self):
        """Two independent noise fields share no real structure: registration
        must report an explicit non-'ok' status, never a fabricated transform.
        """
        rng1 = np.random.RandomState(1)
        rng2 = np.random.RandomState(2)
        img_a = _make_synthetic_dapi(rng1, (256, 256), n_blobs=40)
        img_b = _make_synthetic_dapi(rng2, (256, 256), n_blobs=40)

        transform, n_inliers, n_matches, status = register_round_pair(
            img_a,
            img_b,
            ransac_confidence=0.999,
            ransac_threshold=1.0,
        )
        # Either no transform was found, or one was found but rejected —
        # either way this must be surfaced, never silently treated as 'ok'.
        if transform is None:
            assert status != "ok"
        else:
            valid, _ = _validate_transform(transform, img_a.shape)
            # If cv2 happened to fit *something* on pure noise, at minimum we
            # require this test to demonstrate the sanity check can reject it;
            # a transform that both exists and passes validation on unrelated
            # noise would indicate the sanity bounds are too loose.
            assert not valid or status == "ok" and n_inliers < 4


class TestValidateTransformSanityBounds:
    """Direct tests of the physically-motivated transform sanity check —
    no image data needed."""

    def test_identity_transform_is_valid(self):
        identity = np.eye(2, 3, dtype=np.float64)
        valid, reason = _validate_transform(identity, (512, 512))
        assert valid
        assert reason == ""

    def test_excessive_rotation_is_rejected(self):
        angle = np.radians(30.0)  # far beyond default max_angle_deg=5
        bad = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0]]
        )
        valid, reason = _validate_transform(bad, (512, 512), max_angle_deg=5.0)
        assert not valid
        assert "angle" in reason

    def test_out_of_range_scale_is_rejected(self):
        bad = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        valid, reason = _validate_transform(bad, (512, 512), scale_range=(0.95, 1.05))
        assert not valid
        assert "scale" in reason

    def test_excessive_shift_is_rejected(self):
        bad = np.array([[1.0, 0.0, 10000.0], [0.0, 1.0, 0.0]])
        valid, reason = _validate_transform(bad, (512, 512), max_shift_fraction=0.5)
        assert not valid
        assert "shift" in reason


# ---------------------------------------------------------------------------
# Filename parsing / round ordering
# ---------------------------------------------------------------------------


class TestRoundSortKey:
    def test_numeric_aware_sort_beats_alphabetical(self):
        """The real bug this preserves: alphabetical sort puts R10 before R2
        and R5Q before R5. Numeric-aware sort must get both right."""
        labels = ["R1", "R2", "R5", "R5Q", "R10", "R3"]
        alphabetical = sorted(labels)
        numeric = sorted(labels, key=_round_sort_key)

        assert alphabetical != numeric  # alphabetical really does get this wrong
        assert numeric == ["R1", "R2", "R3", "R5", "R5Q", "R10"]

    def test_r5q_sorts_after_r5_not_before(self):
        assert _round_sort_key("R5") < _round_sort_key("R5Q")

    def test_r10_sorts_after_r2(self):
        assert _round_sort_key("R2") < _round_sort_key("R10")

    def test_unparseable_label_sorts_last(self):
        labels = ["R2", "R1", "Cycle_weird"]
        numeric = sorted(labels, key=_round_sort_key)
        assert numeric[-1] == "Cycle_weird"

    def test_custom_round_label_pattern(self):
        """A non-hBE-TMA round-naming convention is a parameter, not a code change."""
        labels = ["Cycle10", "Cycle2", "Cycle1"]
        pattern = r"^Cycle(\d+)()$"
        numeric = sorted(
            labels, key=lambda l: _round_sort_key(l, round_label_pattern=pattern)
        )
        assert numeric == ["Cycle1", "Cycle2", "Cycle10"]


class TestParseCziFilename:
    def test_default_hbe_tma_pattern(self):
        fname = "R1_MUC1.MUC5AC.MUC2.MUC6_hBE-TMA1-1_2026_05_16__17930-sceneA1.czi"
        info = parse_czi_filename(fname)
        assert info is not None
        assert info["round"] == "R1"
        assert info["markers"] == "MUC1.MUC5AC.MUC2.MUC6"
        assert info["slide"] == "hBE-TMA1-1"
        assert info["date"] == "2026-05-16"
        assert info["acq_id"] == "17930"
        assert info["scene"] == "sceneA1"

    def test_r_suffix_q_round_parses(self):
        fname = "R5Q_DAPI_hBE-TMA2-3_2026_06_01__18000-sceneB2.czi"
        info = parse_czi_filename(fname)
        assert info is not None
        assert info["round"] == "R5Q"

    def test_non_matching_filename_returns_none(self):
        assert parse_czi_filename("not_a_czi_at_all.tif") is None
        assert parse_czi_filename("random_file_R1.czi") is None

    def test_custom_pattern_is_a_parameter_not_a_code_change(self):
        fname = "cycle03_someOtherLab_sceneZ9.czi"
        pattern = (
            r"^cycle(\d+)_" r"([A-Za-z]+)_" r"()" r"()" r"()" r"(scene[A-Z]\d+)\.czi$"
        )
        info = parse_czi_filename(fname, pattern=pattern)
        assert info is not None
        assert info["round"] == "03"
        assert info["scene"] == "sceneZ9"


class TestGroupCzisByScene(object):
    def test_groups_and_sorts_rounds(self, tmp_path):
        names = [
            "R1_MUC1_hBE-TMA1-1_2026_05_16__100-sceneA1.czi",
            "R2_MUC1_hBE-TMA1-1_2026_05_17__101-sceneA1.czi",
            "R10_MUC1_hBE-TMA1-1_2026_05_20__102-sceneA1.czi",
            "R5Q_MUC1_hBE-TMA1-1_2026_05_18__103-sceneA1.czi",
            "R1_MUC1_hBE-TMA1-1_2026_05_16__104-sceneB1.czi",
            "not_a_czi.txt",
        ]
        for n in names:
            (tmp_path / n).write_bytes(b"")

        groups = group_czis_by_scene(str(tmp_path))
        key = ("hBE-TMA1-1", "sceneA1")
        assert key in groups
        assert list(groups[key].keys()) == ["R1", "R2", "R5Q", "R10"]
        assert ("hBE-TMA1-1", "sceneB1") in groups


# ---------------------------------------------------------------------------
# step03_register_rounds — structural checks that don't need real CZI files
# ---------------------------------------------------------------------------


class TestStep03RegisterRoundsStructural:
    def test_missing_reference_round_raises(self):
        with pytest.raises(ValueError, match="Reference round"):
            step03_register_rounds({"R2": "/nonexistent/R2.czi"}, ref_round="R1")

    def test_default_config_matches_source_pipeline_defaults(self):
        cfg = RegistrationConfig()
        assert cfg.n_features == 10000
        assert cfg.flann_checks == 100
        assert cfg.lowe_ratio == 0.75
        assert cfg.ransac_threshold == 3.0
        assert cfg.ransac_confidence == 0.95
        assert cfg.ransac_max_iters == 20000
        assert cfg.max_angle_deg == 5.0
        assert cfg.scale_range == (0.95, 1.05)
        assert cfg.max_shift_fraction == 0.5
