"""红蓝顶块独立判定：不依赖白盒和深度，但必须匹配塔位和上下关系。"""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from car_pose import CarPose, Intrinsics, camera_frame
from classify import inspect_frame

M = runpy.run_path('tests/test_task3.py')


def merged_scene(upper=(0,0,220), lower=(220,0,0)):
    cell = M['_cell']('右侧地面远一格')
    offset, intr = M['OFFSET'], M['INTR']
    pose = M['_pose_for_camera'](cell.x, cell.y-4.2, 0., offset)
    color = np.full((intr.height,intr.width,3), 220, np.uint8)
    depth = np.full(color.shape[:2], 8., np.float32)
    fill = M['_fill_plane']
    fill(color,depth,pose,offset,0.,cell.y-.16,cell.base_z,cell.two_z,(220,220,220),cell.x,.175)
    fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z,cell.two_z+.1,lower,cell.x,.10)
    fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z+.1,cell.cap_z,upper,cell.x,.10)
    # 现有近距离绘制助手只写<=3米的深度，这里独立生成远处平面深度。
    origin,right,down,ahead = camera_frame(pose,offset,0.)
    rows,cols = np.indices(depth.shape)
    rays = ((cols-intr.cx)/intr.fx)[...,None]*right + ((rows-intr.cy)/intr.fy)[...,None]*down + ahead
    measured = (cell.y-.16-origin[1])/rays[:,:,1]
    world = origin + measured[:,:,None]*rays
    body = ((np.abs(world[:,:,0]-cell.x) <= .175)
            & (world[:,:,2] >= cell.base_z) & (world[:,:,2] <= cell.two_z))
    depth[body] = measured[body]
    return cell, pose, offset, intr, color, depth


@pytest.mark.parametrize('upper,lower,expected', [((0,0,220),(220,0,0),'红'), ((220,0,0),(0,0,220),'蓝')])
def test_far_tower_merged_into_white_wall(upper,lower,expected):
    cell,pose,offset,intr,color,depth = merged_scene(upper,lower)
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == expected
    assert view.method == 'RGB顶块'


def test_matching_rgb_cap_does_not_require_white_depth():
    cell,pose,offset,intr,color,depth = merged_scene()
    depth[:] = 8.
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == '红'
    assert view.method == 'RGB顶块'


def test_missing_depth_can_confirm_matching_rgb_cap():
    cell,pose,offset,intr,color,depth = merged_scene()
    view = inspect_frame(pose,offset,0.,intr,color,None)[cell.name]
    assert view.conclusion == '红'


@pytest.mark.parametrize('filename',['20261111_043851_560037','20261111_043853_583125'])
def test_recorded_far_tower_with_red_upper_band(filename):
    path = Path('debug_captures') / (filename+'.npz')
    if not path.exists():
        pytest.skip('本地实测帧未随代码分发')
    with np.load(path,allow_pickle=False) as frame:
        metadata=json.loads(str(frame['metadata']))
        view=inspect_frame(CarPose(**metadata['pose']),np.array(metadata['center_to_corner']),
                           metadata['pitch_deg'],Intrinsics(**metadata['intrinsics']),
                           frame['color'],frame['depth'],metadata['roll_deg'])[metadata['selected']]
    assert view.conclusion == '红'
    assert view.method == 'RGB顶块'


@pytest.mark.parametrize('kind', ['single', 'side_by_side', 'wrong_cell'])
def test_rgb_cap_rejects_unrelated_colors(kind):
    from classify import _rgb_cap_reading, _mask
    from car_pose import to_camera
    cell,pose,offset,intr,color,depth = merged_scene()
    color[:] = 0
    fill = M['_fill_plane']
    x = cell.x + (1.0 if kind == 'wrong_cell' else 0.)
    shift = -.40 if kind == 'wrong_height' else 0.
    if kind == 'side_by_side':
        for dx, bgr in [(-.05,(0,0,220)),(.05,(220,0,0))]:
            fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z,cell.cap_z,bgr,x+dx,.05)
    else:
        fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z+shift,cell.two_z+.1+shift,(220,0,0),x,.10)
        fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z+.1+shift,cell.cap_z+shift,
             (220,0,0) if kind == 'single' else (0,0,220),x,.10)
    mask = _mask(cell,pose,offset,0.,intr)
    z = float(to_camera(cell.center(cell.one_z),pose,offset,0.)[2])
    assert _rgb_cap_reading(cell,color,mask,pose,offset,0.,intr,z) is None


@pytest.mark.parametrize('upper,lower,expected', [((0,0,220),(220,0,0),'红'), ((220,0,0),(0,0,220),'蓝')])
def test_rgb_cap_with_no_visible_white_body(upper,lower,expected):
    cell,pose,offset,intr,color,depth = merged_scene(upper,lower)
    # 所有无色像素涂黑，模拟塔身被遮挡；只留下顶块。
    color[np.max(color,axis=2) == np.min(color,axis=2)] = 0
    view = inspect_frame(pose,offset,0.,intr,color,None)[cell.name]
    assert view.conclusion == expected
    assert view.method == 'RGB顶块'


@pytest.mark.parametrize('single', [False, True])
def test_rgb_diagnostic_reports_rejection_without_changing_result(single):
    from classify import _rgb_cap_reading, _mask
    from car_pose import to_camera
    cell, pose, offset, intr, color, depth = merged_scene()
    if single:
        color[(color[:, :, 2] > 120) & (color[:, :, 0] < 50)] = (220, 0, 0)
    mask = _mask(cell, pose, offset, 0., intr)
    z = float(to_camera(cell.center(cell.one_z), pose, offset, 0.)[2])
    info = {}
    plain = _rgb_cap_reading(cell, color, mask, pose, offset, 0., intr, z)
    measured = _rgb_cap_reading(cell, color, mask, pose, offset, 0., intr, z, diagnostic=info)
    assert (plain is None) == (measured is None) == single
    rgb = info['rgb_cap']
    target = max(rgb['checks'], key=lambda c: c['inside_pixels'])
    if single:
        assert any('红或蓝面积不足' in reason for reason in target['failures'])
        assert target['red_pixels'] == 0
    else:
        assert measured[0] == plain[0] == '红'
        assert not target['failures']
        assert abs(target['bottom_error_cm']) <= 12
        assert abs(target['top_error_cm']) <= 15
    from types import SimpleNamespace
    from tower_debug import paint_diagnostics
    from field_map import DISPLAY
    view = SimpleNamespace(conclusion=None if single else '红', reason=None,
                           region=mask_to_outline(mask), measurements=info)
    rendered = paint_diagnostics(color, depth, {cell.name: view}, DISPLAY.index(cell.name))
    assert rendered.shape == (748, 1280, 3)


def mask_to_outline(mask):
    import cv2
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]


@pytest.mark.parametrize('bgr',[(0,0,220),(220,0,0)])
def test_single_color_at_tower_top_does_not_block_two_layers(bgr):
    cell,pose,offset,intr,color,depth = merged_scene(bgr,(220,220,220))
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.measurements['white_reading']['result'] == '2'
    assert view.measurements['rgb_cap']['state'] == 'absent'
    assert view.conclusion == '2'
    assert not view.skip_vote
    assert not view.measurements.get('cap_blocks_white')


@pytest.mark.parametrize('kind',['none','low_color','tiny_noise','wrong_cell'])
def test_unrelated_color_does_not_block_plain_two_layers(kind):
    cell,pose,offset,intr,color,depth = merged_scene((220,220,220),(220,220,220))
    fill = M['_fill_plane']
    if kind == 'low_color':
        fill(color,depth,pose,offset,0.,cell.y-.16,cell.base_z+.10,
             cell.base_z+.20,(0,0,220),cell.x,.10)
    elif kind == 'tiny_noise':
        from car_pose import project
        u,v = np.round(project(cell.center(cell.cap_z-.05),pose,offset,0.,intr)).astype(int)
        color[v:v+2,u:u+2] = (0,0,220)
    elif kind == 'wrong_cell':
        fill(color,depth,pose,offset,0.,cell.y-.16,cell.two_z,cell.cap_z,
             (0,0,220),cell.x+1.,.10)
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == '2'
    assert view.measurements['rgb_cap']['state'] == 'absent'
    assert not view.measurements.get('cap_blocks_white')


@pytest.mark.parametrize('upper,lower,expected',[
    ((0,0,220),(220,0,0),'红'),((220,0,0),(0,0,220),'蓝')])
def test_confirmed_cap_overrides_simultaneous_white_two_layer_result(upper,lower,expected):
    cell,pose,offset,intr,color,depth = merged_scene(upper,lower)
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.measurements['white_reading']['result'] == '2'
    assert view.measurements['rgb_cap']['state'] == 'confirmed'
    assert view.conclusion == expected
    assert view.method == 'RGB顶块'
