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
    trip = _travel(0., 0., 3.5, 4., extras, yaw)
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
