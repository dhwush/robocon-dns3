"""无彩色顶块的远处白盒，以及证据不足时的实际侧向观察。"""
import numpy as np
import pytest
import runpy

from car_pose import CarPose, camera_frame, camera_position
from classify import inspect_frame
from field_map import DISPLAY
from motion import plan_action, _different_view, _travel, _bypass_points, _path_clear

M = runpy.run_path('tests/test_task3.py')


def scene(layers, unknown_above=False):
    intr, offset = M['INTR'], M['OFFSET']
    cell = M['_cell']('右侧地面远一格')
    pose = M['_pose_for_camera'](cell.x,cell.y-4.2,0.,offset)
    color = np.full((intr.height,intr.width,3),220,np.uint8)
    depth = np.full(color.shape[:2],8.,np.float32)
    origin,right,down,ahead = camera_frame(pose,offset,0.)
    rows,cols = np.indices(depth.shape)
    rays = ((cols-intr.cx)/intr.fx)[...,None]*right + ((rows-intr.cy)/intr.fy)[...,None]*down + ahead
    measured = (cell.y-.16-origin[1])/rays[:,:,1]
    world = origin+measured[:,:,None]*rays
    top = cell.one_z if layers == 1 else cell.two_z
    body = ((np.abs(world[:,:,0]-cell.x)<=.175)
            &(world[:,:,2]>=cell.base_z)&(world[:,:,2]<=top))
    depth[body] = measured[body]
    if unknown_above:
        depth[world[:,:,2]>top+.02] = np.nan
    return cell,pose,offset,intr,color,depth


@pytest.mark.parametrize('layers',[1,2])
def test_plain_white_tower_merged_into_wall(layers):
    cell,pose,offset,intr,color,depth = scene(layers)
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == str(layers)
    assert view.method == '远处白盒'
    assert view.measurements['far_white']['upper_clear']
    assert not view.needs_parallax


@pytest.mark.parametrize('layers',[1,2])
def test_missing_upper_evidence_cannot_confirm_layer_count(layers):
    cell,pose,offset,intr,color,depth = scene(layers,True)
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion is None
    assert view.needs_parallax
    assert not view.measurements['far_white']['upper_clear']


def body_heights(pose, offset, intr, depth):
    origin,right,down,ahead = camera_frame(pose,offset,0.)
    rows,cols = np.indices(depth.shape)
    rays = ((cols-intr.cx)/intr.fx)[...,None]*right + ((rows-intr.cy)/intr.fy)[...,None]*down + ahead
    return (origin+depth[...,None]*rays)[...,2]


@pytest.mark.parametrize('layers',[1,2])
def test_far_white_with_only_top_strip_and_separated_background(layers):
    cell,pose,offset,intr,color,depth = scene(layers)
    top = cell.one_z if layers == 1 else cell.two_z
    z = body_heights(pose,offset,intr,depth)
    depth[(depth<6)&(z<top-.12)] = np.nan
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == str(layers)
    info = view.measurements['far_white']
    assert info['partial_body'] and info['top_edge_clear'] and info['upper_clear']
    assert info['height_span_cm'] < 18


@pytest.mark.parametrize('kind',['unknown_above','middle_of_two_layers','too_thin','wrong_top'])
def test_partial_white_cannot_invent_a_top_edge(kind):
    cell,pose,offset,intr,color,depth = scene(2 if kind == 'middle_of_two_layers' else 1)
    z = body_heights(pose,offset,intr,depth)
    top = cell.one_z
    if kind == 'wrong_top':
        top -= .12
        depth[(depth<6)&(z>top)] = 8.
    thickness = .03 if kind == 'too_thin' else .12
    depth[(depth<6)&(z<top-thickness)] = np.nan
    if kind == 'middle_of_two_layers':
        # 上层实际仍在，仅将其深度置为无效；无返回不能当作上方为空。
        depth[(depth<6)&(z>top)] = np.nan
    if kind == 'unknown_above':
        depth[depth>=6] = np.nan
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion is None
    assert view.needs_parallax


def test_separated_depth_fragments_of_same_white_face_are_combined():
    cell,pose,offset,intr,color,depth = scene(1)
    z = body_heights(pose,offset,intr,depth)
    # 8厘米的横向深度缺口，超出5像素闭运算的连接能力。
    depth[(depth<6)&(z>.12)&(z<.20)] = np.nan
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == '1'
    assert view.measurements['far_white']['height_span_cm'] > 25


def test_lower_fragment_on_different_depth_plane_is_not_combined():
    cell,pose,offset,intr,color,depth = scene(1)
    z = body_heights(pose,offset,intr,depth)
    body = depth<6
    depth[body&(z>.12)&(z<.23)] = np.nan
    depth[body&(z<=.12)] += .20
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion == '1'
    info = view.measurements['far_white']
    assert info['partial_body'] and info['height_span_cm'] < 18


def test_uniform_background_wall_is_not_a_partial_white_box():
    cell,pose,offset,intr,color,depth = scene(1)
    depth[:] = 8.
    view = inspect_frame(pose,offset,0.,intr,color,depth)[cell.name]
    assert view.conclusion is None


def test_lateral_request_selects_reachable_view_with_real_parallax():
    cell,pose,offset,intr,color,depth = scene(2,True)
    views = inspect_frame(pose,offset,0.,intr,color,depth)
    locked = {name:'0' for name in DISPLAY}
    locked[cell.name] = None
    action = plan_action(views,locked,pose,offset,0.,True,intr)
    assert action.kind == 'move'
    target = CarPose(action.final_x,action.final_y,pose.yaw)
    assert _different_view(cell,camera_position(pose,offset,0.),camera_position(target,offset,0.))
    trip = _travel(pose.x,pose.y,target.x,target.y,_bypass_points())
    assert trip is not None
    assert all(_path_clear(*a,*b) for a,b in zip(trip[1],trip[1][1:]))
    assert action.chase_to == 'fresh'
    assert cell.name in action.cells


def test_rotation_and_radial_approach_do_not_count_as_parallax():
    cell,pose,offset,intr,color,depth = scene(2)
    origin = camera_position(pose,offset,0.)
    assert not _different_view(cell,origin,origin)
    radial = (cell.x, origin[1]+1.)
    assert not _different_view(cell,origin,radial)


def test_verified_far_one_layer_stays_to_collect_votes():
    cell,pose,offset,intr,color,depth = scene(1)
    views = inspect_frame(pose,offset,0.,intr,color,depth)
    locked = {name:'0' for name in DISPLAY}
    locked[cell.name] = None
    action = plan_action(views,locked,pose,offset,views[cell.name].pitch_target_deg,True,intr)
    assert action.kind == 'wait'
    assert action.cell == cell.name


def test_unreachable_side_view_reports_unknown_instead_of_radial_fallback(monkeypatch):
    import motion
    cell,pose,offset,intr,color,depth = scene(2,True)
    views = inspect_frame(pose,offset,0.,intr,color,depth)
    locked = {name:'0' for name in DISPLAY}
    locked[cell.name] = None
    monkeypatch.setattr(motion,'_search',lambda *a,**kw: None)
    action = plan_action(views,locked,pose,offset,0.,True,intr)
    assert action.kind == 'stuck'
    assert '保持未知' in action.text


def test_moving_keeps_original_view_and_inflight_goal(monkeypatch):
    from report import FieldWatch
    from car_pose import Setup
    from motion import Action
    import report
    cell,pose,offset,intr,color,depth = scene(2,True)
    views = inspect_frame(pose,offset,0.,intr,color,depth)
    watch = FieldWatch(Setup(True,offset,0.,'', 'rad'),intr)
    monkeypatch.setattr(report,'inspect_frame',lambda *a,**kw: views)
    monkeypatch.setattr(watch,'_note_aim',lambda: None)
    watch.update(pose,color,depth)
    origin = watch.parallax_origins[cell.name]
    goal = Action('move','移动',x=pose.x+1.,y=pose.y,final_x=pose.x+1.,final_y=pose.y,
                  chase_to='fresh',cells=(cell.name,))
    watch._remember_goal(goal)
    watch.hold = goal
    moved = CarPose(pose.x+.3,pose.y,pose.yaw)
    watch.update(moved,color,depth)
    assert watch.parallax_origins[cell.name] == origin
    assert views[cell.name].measurements['parallax_origin'] == origin
    different = Action('move','另一落点',x=pose.x+2.,y=pose.y,final_x=pose.x+2.,final_y=pose.y,
                       chase_to='fresh',cells=(cell.name,))
    watch._remember_goal(different)
    assert watch.goal[cell.name][:2] == (goal.final_x,goal.final_y)
