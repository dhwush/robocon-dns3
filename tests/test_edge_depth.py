import numpy as np

from classify import _filter_edge_depth


def scene():
    color = np.zeros((30, 40, 3), np.uint8)
    color[5:25, 5:35] = (0, 0, 255)
    depth = np.full((30, 40), 4.0, np.float32)
    depth[5:25, 5:35] = 2.0
    return color, depth


def test_one_pixel_edge_strip_is_removed_without_changing_input():
    color, depth = scene()
    depth[5, 10:30] = 2.5
    before = depth.copy()
    cleaned = _filter_edge_depth(color, depth)
    assert np.isnan(cleaned[5, 10:30]).all()
    assert np.array_equal(depth, before)
    assert np.array_equal(cleaned[6:25, 5:35], depth[6:25, 5:35])
    assert np.array_equal(cleaned[:5], depth[:5])


def test_real_depth_step_and_small_variations_are_preserved():
    color, depth = scene()
    depth[5, 10:30] += 0.03
    assert np.array_equal(_filter_edge_depth(color, depth), depth)


def test_interior_outlier_is_outside_this_filter_scope():
    color, depth = scene()
    depth[12, 15] = 2.5
    assert np.array_equal(_filter_edge_depth(color, depth), depth)


def test_tiny_target_without_enough_support_is_preserved():
    color = np.zeros((10, 10, 3), np.uint8)
    color[4:6, 4:6] = (255, 255, 255)
    depth = np.full((10, 10), np.nan, np.float32)
    depth[4:6, 4:6] = [[2, 2.5], [2, 2]]
    assert np.array_equal(_filter_edge_depth(color, depth), depth, equal_nan=True)


def test_missing_depth_and_two_pixel_band_are_not_eroded():
    color, depth = scene()
    depth[5:7, 10:30] = 2.5
    cleaned = _filter_edge_depth(color, depth)
    assert np.array_equal(cleaned, depth)
    assert _filter_edge_depth(color, None) is None
