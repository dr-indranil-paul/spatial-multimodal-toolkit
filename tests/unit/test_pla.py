import numpy as np

from spatial_omics_toolkit.pla import (
    PLAProcessingConfig,
    detect_pla_spots,
    enhance_dapi,
    estimate_af_scale,
    segment_dapi_cells,
    subtract_autofluorescence,
)


def test_estimate_and_subtract_autofluorescence_recovers_background():
    af = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    pla = 7.0 + 2.5 * af
    pla[4:6, 4:6] += 500.0
    scale, intercept = estimate_af_scale(af, pla, regression_percentile=90.0)
    corrected = subtract_autofluorescence(af, pla, scale, intercept)
    assert np.isclose(scale, 2.5)
    assert np.isclose(intercept, 7.0)
    background = np.ones_like(af, dtype=bool)
    background[4:6, 4:6] = False
    assert np.max(corrected[background]) < 1e-4
    assert np.min(corrected[4:6, 4:6]) > 499.9


def test_detect_pla_spots_removes_singletons_and_large_regions():
    corrected = np.zeros((40, 40), dtype=np.float32)
    corrected[5:7, 5:7] = 30
    corrected[10, 10] = 35
    corrected[20:30, 20:30] = 25
    labels, threshold = detect_pla_spots(
        corrected,
        np.ones_like(corrected, dtype=bool),
        threshold_percentile=5.0,
        min_size=2,
        max_size=20,
    )
    assert threshold == 25.0
    assert labels.max() == 1
    assert np.count_nonzero(labels) == 4


def test_enhance_and_segment_dapi_preserves_nuclei():
    yy, xx = np.ogrid[:64, :64]
    dapi = np.full((64, 64), 100, dtype=np.uint16)
    dapi[(yy - 20) ** 2 + (xx - 20) ** 2 < 8**2] = 3000
    dapi[(yy - 43) ** 2 + (xx - 43) ** 2 < 9**2] = 4000
    enhanced = enhance_dapi(dapi)
    labels = segment_dapi_cells(enhanced)
    assert enhanced.dtype == np.uint16
    assert enhanced.max() == np.iinfo(np.uint16).max
    assert labels.dtype == np.uint32
    assert labels.max() == 2


def test_config_rejects_invalid_spot_bounds():
    try:
        PLAProcessingConfig(spot_min_size=10, spot_max_size=2)
    except ValueError as exc:
        assert "spot size" in str(exc)
    else:
        raise AssertionError("invalid bounds were accepted")
