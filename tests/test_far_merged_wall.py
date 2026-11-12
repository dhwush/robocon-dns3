"""白盒粘墙时用塔顶和格内深度，不能靠白色连通域或单独彩色贴纸认塔。"""
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
    assert view.method == '远处融合'
    assert view.measurements['far_support']['support_points'] >= 30


def test_color_on_distant_wall_without_own_white_depth_is_not_a_tower():
    cell,pose,offset,intr,color,depth = merged_scene()
    depth[:] = 8.
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion is None
    assert view.measurements['far_support']['support_points'] == 0


def test_missing_depth_cannot_confirm_merged_wall_tower():
    cell,pose,offset,intr,color,depth = merged_scene()
    view = inspect_frame(pose,offset,0.,intr,color,None)[cell.name]
    assert view.conclusion is None


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
    assert view.method == '远处融合'
