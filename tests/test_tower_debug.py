"""诊断必须反映真实取色，并保留可复查的原始帧。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from car_pose import Intrinsics
from classify import _faction
from field_map import DISPLAY
from tower_debug import paint_diagnostics, save_capture


def test_ssh_search_box_visible_without_classification():
    from view_field import paint_window
    color = np.zeros((400, 600, 3), np.uint8)
    region = np.array([[200, 200], [300, 220], [250, 300]], np.int32)
    view = SimpleNamespace(conclusion=None, region=region, piece=None,
                           method=None, measurements=None)
    views = {DISPLAY[0]: view}
    plain = paint_window(color, views, {}, [])
    boxed = paint_window(color, views, {}, [], show_boxes=True)
    assert not plain[250, 200].any()
    assert boxed[250, 200].any()
    assert not color.any()
    # 断流时不画历史框。
    assert not paint_window(color, None, {}, [], show_boxes=True).any()


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


def test_live_window_groups_targets_can_results_and_footer():
    from types import SimpleNamespace
    import numpy as np
    from car_pose import CarPose
    from classify import CellView
    from field_map import DISPLAY
    from motion import Action
    from view_field import live_panels, paint_window
    from vote import Vote
    name = DISPLAY[0]
    watch = SimpleNamespace(pitch=2., views={n: CellView('1',True,False,False,False,0.,2.,0.,100.) for n in DISPLAY},
        votes={n: Vote() for n in DISPLAY})
    watch.autonomy = SimpleNamespace(action=Action('pitch','',pitch=10.,cell=name,attempt=1),
        pitch_checks={n: [] for n in DISPLAY}, message='等待相机稳定', complete=False,
        heading_center=lambda name: 0.)
    panels = live_panels(watch,CarPose(0,0,0),'ssh',['光线：室内'])
    assert any('CAN' in row for row in panels['left'])
    assert any('车头朝向' in row for row in panels['left'])
    assert any('相机俯仰' in row for row in panels['left'])
    assert len(panels['right']) == 7
    assert all('号' not in row for row in panels['bottom'])
    assert '等待相机稳定' in panels['bottom']
    color = np.zeros((720,1280,3),np.uint8)
    result = paint_window(color,None,{},[],panels=panels)
    assert result.shape == color.shape
    assert result[:200,-240:].any()
    assert result[-100:].any()
    assert not result[400:500,400:700].any()
