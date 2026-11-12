"""3米截断不能把完整白盒变成窄板；远处和真实窄板仍需拒绝。"""
import numpy as np
import pytest

from car_pose import Intrinsics
from classify import _depth_on_cell
from field_map import tower_cells

INTR = Intrinsics(640, 480, 450, 450, 320, 240)
CELL = next(cell for cell in tower_cells() if cell.ground)


def cloud_for(width=.32, entirely_far=False, outside_extra=False):
    x, z = np.meshgrid(np.linspace(-width/2, width/2, 60), np.linspace(.10, .35, 40))
    x, z = x.ravel(), z.ravel()
    distances = 3. + .8 * x
    if entirely_far:
        distances += .3
    world = np.column_stack((CELL.x + x, np.full(x.shape, CELL.y), z))
    if outside_extra:
        world[distances > 3., 1] = CELL.y + 1.
    # 像素均在画面内，宽度/高度在本测试中由世界点和实际深度决定。
    return dict(world=world, z=distances,
                xs=np.tile(np.arange(290, 350), 40),
                ys=np.repeat(np.arange(200, 240), 60))


def reading(cloud):
    diagnostic = {}
    result = _depth_on_cell(CELL, cloud, np.full((480,640),255,np.uint8), INTR,
                            np.zeros((480,640,3),np.uint8), False,
                            diagnostic=diagnostic)
    return result, diagnostic


def test_crossing_three_meters_recovers_complete_tower_width():
    result, diagnostic = reading(cloud_for())
    assert result[0] == '1'
    recovery = diagnostic['depth_boundary_recovery']
    assert recovery['original_width_cm'] < 15
    assert 15 <= recovery['recovered_width_cm'] <= 65
    assert recovery['extra_points'] > 10


def test_far_points_alone_cannot_enable_depth_classification():
    result, diagnostic = reading(cloud_for(entirely_far=True))
    assert result[0] is None
    assert 'depth_boundary_recovery' not in diagnostic


def test_extra_points_outside_cell_cannot_repair_width():
    result, diagnostic = reading(cloud_for(outside_extra=True))
    assert result[0] is None
    assert 'depth_boundary_recovery' not in diagnostic
    assert diagnostic['depth_partial_at_boundary']


@pytest.mark.parametrize('near_boundary', [False, True])
def test_real_narrow_panel_still_has_no_depth_conclusion(near_boundary):
    cloud = cloud_for(width=.03)
    if not near_boundary:
        cloud['z'] -= 1.
    result, diagnostic = reading(cloud)
    assert result[0] is None
    assert 'depth_boundary_recovery' not in diagnostic
    # 边界附近仅允许独立合格彩色轮廓回退；其他位置仍硬拒绝。
    assert result[3] == (not near_boundary)


@pytest.mark.parametrize('upper,lower,expected', [((0,0,220),(220,0,0),'红'),
                                                 ((220,0,0),(0,0,220),'蓝')])
def test_boundary_recovery_keeps_upper_color_decision(upper, lower, expected):
    cloud = cloud_for()
    cloud['world'][:,2] = .1 + (cloud['world'][:,2] - .1) * (.6 / .25)
    color = np.zeros((480,640,3),np.uint8)
    color[170:185,290:350] = upper
    color[185:200,290:350] = lower
    diagnostic = {}
    result = _depth_on_cell(CELL, cloud, np.full((480,640),255,np.uint8), INTR,
                            color, False, diagnostic=diagnostic)
    assert result[0] == expected
    assert 'depth_boundary_recovery' in diagnostic
    assert diagnostic['faction']['result'] == expected
