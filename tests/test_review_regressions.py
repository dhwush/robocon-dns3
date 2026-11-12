"""识别审查发现的误判与正确支撑面证据。"""
import runpy
import numpy as np
from car_pose import CarPose, closest_pose, camera_frame
from classify import inspect_frame, _faction, _empty_evidence
from report import FieldWatch
m = runpy.run_path('tests/test_task3.py')
INTR, OFFSET = m['INTR'], m['OFFSET']

def scene(top=.35, width=.16, bgr=(255,255,255)):
    cell=m['_cell']('最近右侧地面')
    pose=m['_pose_for_camera'](cell.x,cell.y-1.3,0.,OFFSET)
    color=m['_blank'](); depth=np.full(color.shape[:2],np.nan,np.float32)
    m['_fill_plane'](color,depth,pose,OFFSET,-8.,cell.y-.16,.02,top,bgr,cell.x,width)
    return cell,pose,color,depth

def test_dark_tower_is_unknown_without_depth():
    cell,pose,c,d=scene(bgr=(100,100,100))
    assert inspect_frame(pose,OFFSET,-8.,INTR,c,None)[cell.name].conclusion is None

def test_narrow_panel_is_not_a_layer():
    cell,pose,c,d=scene(width=.015)
    assert inspect_frame(pose,OFFSET,-8.,INTR,c,d)[cell.name].conclusion is None

def test_intermediate_height_is_not_two_layers():
    cell,pose,c,d=scene(top=.50)
    assert inspect_frame(pose,OFFSET,-8.,INTR,c,d)[cell.name].conclusion is None

def test_cap_noise_cannot_override_blue():
    c=np.zeros((720,1280,3),np.uint8); w=np.zeros((720,1280),np.uint8)
    w[400:600,500:700]=255; c[350:400,500:700]=(255,0,0)
    c[349,500:530]=(0,0,255)
    assert _faction(c,w,np.full(w.shape,255,np.uint8),2.,INTR)==('蓝',False)

def test_pose_requires_same_clock_and_fresh_sample():
    poses=[CarPose(0,0,0,1000),CarPose(1,0,0,1001)]
    assert closest_pose(poses,1.) is None
    assert closest_pose(poses,1002.) is None
    assert closest_pose(poses,1001.02)==poses[1]
    assert closest_pose(poses,float('nan')) is None

def test_slow_motion_cannot_lock():
    setup,pose,offset=m['_basic'](); watch=FieldWatch(setup,INTR)
    cell=m['_cell']('右侧地面远一格'); color=m['_blank']()
    m['_paint'](color,pose,0.,cell,.35,.35,0.,(255,255,255),offset=offset)
    for i in range(25):
        watch.update(CarPose(pose.x+.01*i,pose.y,pose.yaw),color)
    assert watch.votes[cell.name].locked is None

def test_empty_requires_observed_support_plane():
    cell,pose,c,d=scene(); pitch=-18.
    origin,right,down,ahead=camera_frame(pose,OFFSET,pitch)
    ys,xs=np.indices(d.shape)
    rays=((xs-INTR.cx)/INTR.fx)[...,None]*right+((ys-INTR.cy)/INTR.fy)[...,None]*down+ahead
    with np.errstate(divide='ignore',invalid='ignore'):
        d=((cell.base_z-origin[2])/rays[:,:,2]).astype(np.float32)
    assert _empty_evidence(cell,pose,OFFSET,pitch,INTR,d)
    assert not _empty_evidence(cell,pose,OFFSET,pitch,INTR,np.full(d.shape,np.nan))


def test_depth_rejection_has_specific_reason():
    from report import plain_state
    cell, pose, c, d = scene(width=.015)
    view = inspect_frame(pose, OFFSET, -8., INTR, c, d)[cell.name]
    assert view.conclusion is None
    assert '尺寸不符' in view.reason
    assert '尺寸不符' in plain_state(view, None, 'extend')
    assert '尺寸不符' in plain_state(view, None, 'basic')


def test_height_rejection_reports_measured_height():
    cell, pose, c, d = scene(top=.50)
    view = inspect_frame(pose, OFFSET, -8., INTR, c, d)[cell.name]
    assert view.conclusion is None
    assert '层高不符' in view.reason and '厘米' in view.reason


def test_sparse_depth_allows_valid_color():
    setup, pose, offset = m['_basic']()
    cell = m['_cell']('右侧地面远一格')
    color = m['_blank']()
    m['_paint'](
        color, pose, 0., cell, .35, .35, 0.,
        (255, 255, 255), offset=offset,
    )
    depth = np.full(color.shape[:2], np.nan, np.float32)
    view = inspect_frame(pose, offset, 0., INTR, color, depth)[cell.name]
    assert view.conclusion == '1'
