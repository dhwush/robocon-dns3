"""诊断必须反映真实取色，并保留可复查的原始帧。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from car_pose import Intrinsics
from classify import _faction
from field_map import DISPLAY
from tower_debug import paint_diagnostics, save_capture


@pytest.mark.parametrize('distance,expected', [(1., '红'), (2., '蓝')])
def test_debug_exposes_truncated_upper_color_without_changing_result(distance, expected):
    intr = Intrinsics(400, 500, 900, 900, 200, 250)
    color = np.zeros((500, 400, 3), np.uint8)
    color[120:210, 140:260] = (0, 0, 220)
    color[210:300, 140:260] = (220, 0, 0)
    color[300:480, 100:300] = 220
    white = np.zeros((500, 400), np.uint8)
    white[300:480, 100:300] = 255
    mask = np.full(white.shape, 255, np.uint8)
    diagnostic = {}
    plain = _faction(color, white, mask, distance, intr)
    assert _faction(color, white, mask, distance, intr, diagnostic) == plain == (expected, False)
    info = diagnostic['faction']
    assert info['result'] == expected
    assert info['cap_px'] == round(.20 * intr.fy / distance)
    assert (info['raw_red_pixels'] == 0) == (distance == 2.)
    assert info['blue_pixels'] >= info['need']


def test_capture_preserves_raw_color_invalid_depth_and_diagnostics(tmp_path):
    color = np.full((120, 160, 3), (12, 34, 56), np.uint8)
    original = color.copy()
    depth = np.full(color.shape[:2], np.nan, np.float32)
    depth[30:70, 40:80] = 1.234
    view = SimpleNamespace(conclusion=None, reason='测试', method=None,
                           measurements={'top_pixels': [[50, 40]], 'depth_height_cm': 70.},
                           region=None)
    views = {DISPLAY[0]: view}
    image = paint_diagnostics(color, depth, views)
    assert image.shape == (748, 1280, 3)
    stem = save_capture(tmp_path, color, depth, views, 0, {}, {'pitch_deg': 0.})
    with np.load(str(stem) + '.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['color'], original)
        np.testing.assert_array_equal(saved['depth'], depth)
        info = json.loads(str(saved['metadata']))
        assert info['depth_unit'] == 'm'
        assert info['color_format'] == 'BGR'
        assert info['views'][DISPLAY[0]]['reason'] == '测试'
    assert stem.with_suffix('.png').exists()
    np.testing.assert_array_equal(color, original)
    # 无深度、无识别结果时也能显示，不伪造深度。
    assert paint_diagnostics(color, None, None).shape == image.shape
