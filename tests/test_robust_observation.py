"""以射线与三维盒体求交生成场景，覆盖俯仰、侧视、多面和遮挡。"""
import math
import runpy

import numpy as np
import pytest

from car_pose import CarPose, Intrinsics, camera_frame, project, camera_position
from classify import inspect_frame, _z_at_cell, _pitch_target, _faction
from field_map import tower_cells

M = runpy.run_path('tests/test_task3.py')
INTR = Intrinsics(640,360, M['INTR'].fx/2, M['INTR'].fy/2,320,180)
OFFSET = M['OFFSET']
CELL = next(c for c in tower_cells() if c.name=='右侧地面远一格')


def boxes_scene(layers=1,distance=2.0,yaw=0.,pitch=0.,roll=0.,cap=False):
    angle = math.radians(yaw)
    pose = M['_pose_for_camera'](CELL.x+distance*math.sin(angle),
        CELL.y-distance*math.cos(angle),angle,OFFSET)
    origin,right,down,ahead = camera_frame(pose,OFFSET,pitch,roll)
    rows,cols = np.indices((INTR.height,INTR.width))
    rays = ((cols-INTR.cx)/INTR.fx)[...,None]*right + ((rows-INTR.cy)/INTR.fy)[...,None]*down + ahead
    color = np.full((INTR.height,INTR.width,3),160,np.uint8)
    depth = np.full(color.shape[:2],8.,np.float32)
    def render(x0,x1,y0,y1,z0,z1,ink):
        near = np.full(depth.shape,-np.inf)
        far = np.full(depth.shape,np.inf)
        for axis,lo,hi in [(0,x0,x1),(1,y0,y1),(2,z0,z1)]:
            direction = rays[...,axis]
            moving = np.abs(direction)>1e-9
            inv = np.zeros(depth.shape)
            inv[moving] = 1/direction[moving]
            a,b = (lo-origin[axis])*inv,(hi-origin[axis])*inv
            inside = lo<=origin[axis]<=hi
            low = np.where(moving,np.minimum(a,b),-np.inf if inside else np.inf)
            high = np.where(moving,np.maximum(a,b),np.inf if inside else -np.inf)
            near = np.maximum(near,low)
            far = np.minimum(far,high)
        hit = (far>=near)&(near>.3)&(near<depth)
        depth[hit]=near[hit]
        color[hit]=ink
        return hit
    for layer in range(layers):
        render(CELL.x-.175,CELL.x+.175,CELL.y-.175,CELL.y+.175,
               layer*.35,(layer+1)*.35, (220,220,220))
    if cap:
        for low,high,ink in [(.70,.80,(220,0,0)),(.80,.90,(0,0,220))]:
            render(CELL.x-.10,CELL.x+.10,CELL.y-.10,CELL.y+.10,low,high,ink)
    return pose,color,depth,render,rays


@pytest.mark.parametrize('pitch,roll,yaw', [(-12,0,0),(0,8,45),(10,-8,90)])
def test_world_height_projection_is_valid_when_viewing_from_side(pitch,roll,yaw):
    pose,_,_,_,_ = boxes_scene(yaw=yaw,pitch=pitch,roll=roll)
    pixel = project(CELL.center(CELL.one_z),pose,OFFSET,pitch,INTR,roll)
    assert _z_at_cell(CELL,*pixel,pose,OFFSET,pitch,INTR,roll) == pytest.approx(CELL.one_z,abs=1e-6)


@pytest.mark.parametrize('layers,distance,pitch,roll,yaw',[
    (1,1.5,-12,0,0),(2,1.8,4,8,45),
    (1,4.2,0,0,45),(2,4.2,5,-6,45),(1,4.2,-5,8,90)])
def test_cube_faces_stay_readable_across_pitch_distance_and_side_view(layers,distance,pitch,roll,yaw):
    pose,color,depth,_,_ = boxes_scene(layers,distance,yaw,pitch,roll)
    view = inspect_frame(pose,OFFSET,pitch,INTR,color,depth,roll)[CELL.name]
    assert view.conclusion == str(layers), (view.reason,view.measurements)


@pytest.mark.parametrize('pitch,roll,yaw',[(0,0,0),(8,12,45),(-8,-12,90)])
def test_complete_cap_order_uses_gravity_not_camera_rows(pitch,roll,yaw):
    pose,color,depth,_,_ = boxes_scene(2,2.5,yaw,pitch,roll,True)
    view = inspect_frame(pose,OFFSET,pitch,INTR,color,depth,roll)[CELL.name]
    assert view.conclusion == '红', (view.reason,view.measurements)


def test_foreground_at_intermediate_distance_blocks_far_layer_reading():
    pose,color,depth,render,_ = boxes_scene(2,4.2)
    render(CELL.x-.45,CELL.x+.45,CELL.y-1.1,CELL.y-.9,.10,.80,(100,100,100))
    view = inspect_frame(pose,OFFSET,0.,INTR,color,depth)[CELL.name]
    assert view.conclusion is None
    assert view.needs_parallax
    assert '前景遮挡' in view.reason


def test_single_visible_blue_side_cannot_decide_cap_faction():
    color = np.zeros((360,640,3),np.uint8)
    white = np.zeros((360,640),np.uint8)
    white[230:330,250:350]=255
    color[190:230,270:330]=(220,0,0)
    result = _faction(color,white,np.full_like(white,255),2.,INTR)
    assert result == (None,False)


@pytest.mark.parametrize('layers,pitch,roll',[ (1,0,0),(2,5,6)])
def test_missing_top_depth_on_multiple_faces_can_recover_visible_edge(layers,pitch,roll):
    pose,color,depth,_,rays = boxes_scene(layers,4.2,45,pitch,roll)
    origin = camera_position(pose,OFFSET,pitch)
    heights = origin[2]+depth*rays[...,2]
    depth[(depth<6)&(heights>layers*.35-.10)] = np.nan
    view = inspect_frame(pose,OFFSET,pitch,INTR,color,depth,roll)[CELL.name]
    assert view.conclusion == str(layers), (view.reason,view.measurements)
    assert view.measurements['far_white']['faces']>=2


def test_depth_only_middle_of_taller_box_is_not_a_one_layer_top():
    pose,color,depth,_,rays = boxes_scene(2,2.0)
    origin = camera_position(pose,OFFSET,0.)
    heights = origin[2]+depth*rays[...,2]
    depth[(depth<6)&(heights>.35)] = np.nan
    view = inspect_frame(pose,OFFSET,0.,INTR,color,depth)[CELL.name]
    assert view.conclusion != '1'


def test_cap_with_hsv_colors_missing_is_still_upper_occupancy():
    pose,color,depth,_,rays = boxes_scene(2,2.0,cap=True)
    origin = camera_position(pose,OFFSET,0.)
    heights = origin[2]+depth*rays[...,2]
    color[(depth<6)&(heights>.70)] = (65,65,65)
    view = inspect_frame(pose,OFFSET,0.,INTR,color,depth)[CELL.name]
    assert view.conclusion is None
    assert view.measurements.get('upper_object')
    assert view.measurements['cap_blocks_white']


def test_near_box_top_out_of_frame_cannot_be_mistaken_for_complete_layers():
    pose,color,depth,_,_ = boxes_scene(2,.75,pitch=-25)
    cropped = inspect_frame(pose,OFFSET,-25.,INTR,color,depth)[CELL.name]
    assert cropped.conclusion is None
    assert cropped.measurements.get('needs_standoff')
    pose,color,depth,_,_ = boxes_scene(2,1.5,pitch=0)
    visible = inspect_frame(pose,OFFSET,0.,INTR,color,depth)[CELL.name]
    assert visible.conclusion == '2'


def test_view_quality_prefers_clear_complete_view_to_too_close_or_far():
    from motion import observation_quality
    quality = []
    for distance in (.55,1.8,5.):
        pose,_,_,_,_ = boxes_scene(distance=distance)
        quality.append(observation_quality(CELL,pose,OFFSET,INTR))
    assert quality[1]>quality[0]
    assert quality[1]>quality[2]
