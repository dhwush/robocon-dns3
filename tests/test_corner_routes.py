"""右前角定位、车身包络和绕障折点的回归测试。"""
import math
import numpy as np
from car_pose import CarPose, camera_position, body_center, turned_pose, load_setup
from motion import _path_clear, _parking_clear, _travel, _bypass_points, _fallback, _move_toward


def test_corner_camera_rotates_installation_offset():
    pose = CarPose(1., 2., math.pi/2)
    np.testing.assert_allclose(camera_position(pose, np.zeros(2), 0), [.95, 1.789, .675])


def test_in_place_turn_preserves_body_center():
    pose = CarPose(.8, -.3, .2)
    turned = turned_pose(pose, 2.)
    np.testing.assert_allclose(body_center(turned), body_center(pose))
    assert (turned.x, turned.y) != (pose.x, pose.y)


def test_zero_offset_setup_is_valid_after_alignment(tmp_path):
    path = tmp_path/'field.yaml'
    path.write_text('aligned: true\ncenter_to_corner: [0, 0]\n')
    assert load_setup(str(path)).ready


def test_body_and_corner_cut_are_blocked():
    assert not _parking_clear(-1.82, .90)
    assert not _path_clear(-1.816, .874, -1.795, .853)


def test_nonzero_heading_bypass_is_safe():
    yaw = .8
    extras = _bypass_points(yaw)
    trip = _travel(0., 0., 2.1, 3., extras, yaw)
    assert trip is not None
    assert all(_path_clear(*a, *b, yaw) for a,b in zip(trip[1],trip[1][1:]))


def test_near_waypoint_does_not_skip_blocked_segment():
    # 折点虽很近，若跨过它会切入膨胀障碍，必须仍指向折点。
    from motion import _navigation_boxes
    a,b,c,d = _navigation_boxes()[0]
    pose = CarPose(a-.01,c+.03,0.)
    waypoint = (a-.01,c-.01)
    target = (a+.3,c-.01)
    action = _move_toward(pose,*target,[(pose.x,pose.y),waypoint,target],None,())
    assert (action.x,action.y) == waypoint


def test_unreachable_fallback_does_not_import_missing_module():
    assert _fallback(CarPose(0,0,0),[],[]).kind == 'blocked'


def test_confirmed_corridor_limits_apply_to_parking_and_whole_paths():
    assert _parking_clear(-2.,1.4)
    assert not _parking_clear(-2.,2.05)
    assert _parking_clear(2.25,4.)
    assert not _parking_clear(2.25,4.7)
    assert not _parking_clear(0.,6.5)
    assert not _path_clear(-2.,1.4,-2.,2.05)
    assert not _path_clear(2.25,4.,2.25,4.7)
    assert _travel(-2.,1.4,-2.,2.05,_bypass_points()) is None


def test_corridor_limits_have_no_infinite_waypoints_or_outer_detour():
    extras = _bypass_points()
    assert all(math.isfinite(value) for point in extras for value in point)
    assert _travel(-20.,0.,-20.,3.,extras) is None
    assert _travel(20.,0.,20.,5.,extras) is None
    trip = _travel(-2.,1.4,2.25,4.,extras)
    assert trip is not None
    assert all(_path_clear(*a,*b) for a,b in zip(trip[1],trip[1][1:]))


def test_supplementary_stop_order_avoids_far_then_near_backtracking():
    from motion import _order_supplementary_stops,RouteStops,RouteStop,Look
    initial = RouteStop(0.,0.,(Look('2',0.,0.),),'初始点')
    far = RouteStop(2.7,0.,(Look('4',0.,0.),),'补充点1')
    near = RouteStop(1.,0.,(Look('1',0.,0.),),'补充点2')
    original = RouteStops([initial,far,near],observed=('known',),unobserved=('missing',))
    ordered,line = _order_supplementary_stops(original,[(0.,0.),(2.7,0.),(1.,0.)])
    assert [stop.x for stop in ordered] == [0.,1.,2.7]
    assert ordered[1].looks == near.looks
    assert ordered[2].looks == far.looks
    assert ordered.observed == original.observed
    assert ordered.unobserved == original.unobserved
    assert sum(math.dist(a,b) for a,b in zip(line,line[1:])) == 2.7


def test_rear_boundary_blocks_parking_and_routes_without_outer_detour():
    from field_map import DRIVING_MIN_Y
    from motion import _ground_blocked
    assert DRIVING_MIN_Y == -.6
    assert _parking_clear(0.,0.)
    assert not _parking_clear(0.,-.6)
    assert not _parking_clear(0.,-.7)
    assert not _path_clear(0.,0.,-0.,-.7)
    assert _travel(-20.,0.,-20.,-.7,_bypass_points()) is None
    assert _travel(20.,0.,20.,-.7,_bypass_points()) is None
    assert not _ground_blocked(-4.,-.8)  # 后方边界只限行，不遮挡视线。


def test_rear_boundary_includes_body_clearance_at_every_heading():
    from car_pose import BODY_HALF_M
    from field_map import DRIVING_MIN_Y
    from motion import BODY_CLEARANCE_M
    for yaw in (0.,.8,math.pi/2,math.pi):
        for margin,expected in ((-.01,False),(.01,True)):
            center_y = DRIVING_MIN_Y+BODY_CLEARANCE_M+margin
            pose = turned_pose(CarPose(0.,center_y+BODY_HALF_M,0.),yaw)
            assert _parking_clear(pose.x,pose.y,pose.yaw) is expected


def test_right_front_x_limits_align_with_right_edges_and_apply_at_all_yaws():
    from field_map import right_front_x_limits,tower_cells
    left,right = right_front_x_limits()
    assert math.isclose(left,-2.575)
    assert math.isclose(right,2.9)
    assert left == tower_cells()[0].bounds()[1]
    assert right == tower_cells()[5].bounds()[1]
    for yaw in (0.,.8,math.pi/2,math.pi,-math.pi/2):
        assert _parking_clear(left,.2,yaw)
        assert _parking_clear(right,.2,yaw)
        assert not _parking_clear(left-.001,.2,yaw)
        assert not _parking_clear(right+.001,.2,yaw)
        assert not _path_clear(0.,.2,left-.001,.2,yaw)
        assert not _path_clear(0.,.2,right+.001,.2,yaw)
    assert _travel(left-.1,.2,left-.1,.2,[]) is None
    assert all(left <= x <= right for x,y in _bypass_points())


def test_turn_cannot_cross_x_boundary_between_two_legal_endpoints():
    from field_map import right_front_x_limits
    from motion import _right_front_turn_clear
    left,right = right_front_x_limits()
    for x,yaw,target in ((right-.025,-math.pi/2,0.),(left+.025,math.pi/2,math.pi)):
        pose = CarPose(x,.2,yaw)
        end = turned_pose(pose,target)
        assert _parking_clear(pose.x,pose.y,pose.yaw)
        assert _parking_clear(end.x,end.y,end.yaw)
        assert not _right_front_turn_clear(pose,target)
    assert _right_front_turn_clear(CarPose(0.,.2,0.),math.pi)


def test_survey_rejects_goal_outside_confirmed_x_limits(monkeypatch):
    monkeypatch.setattr("motion._observable_at",lambda *args:False)
    from motion import survey,Action
    from field_map import DISPLAY,right_front_x_limits
    left,right = right_front_x_limits()
    for x in (left-.01,right+.01):
        def planner(*args):
            return Action('wait','',cells=(DISPLAY[0],),observation_x=x,observation_y=.2,
                          observation_yaw=0.,observation_pitch=0.)
        stops,line = survey(CarPose(0.,.2,0.),np.zeros(2),None,pending=[DISPLAY[0]],planner=planner)
        assert len(stops) == 1
        assert stops.unobserved == (DISPLAY[0],)
        assert line == [(0.,.2)]
