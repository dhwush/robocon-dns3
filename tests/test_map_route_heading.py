from map_test import route_position_key


def test_route_start_position_change_and_coordinate_noise():
    assert route_position_key(None) is None
    initial = route_position_key((0.005, 0.014))
    assert route_position_key((0.005000001, 0.014000001)) == initial
    assert route_position_key((0.105, 0.014)) != initial


def test_mode_buttons_hit_bounds():
    from map_test import mode_at
    assert mode_at(12, 40) == 'static'
    assert mode_at(200, 40) == 'ssh'
    assert mode_at(380, 40) == 'local'
    assert mode_at(12, 80) is None
    assert mode_at(550, 40) is None


def test_fixed_plan_survives_manual_heading_changes():
    import numpy as np
    from car_pose import load_intrinsics
    from map_test import ROOT, pose_for_optical, look_yaw_text, stop_lines, open_yaw
    from motion import survey
    intr = load_intrinsics(str(ROOT / 'config' / 'intrinsics.yaml'))
    stops, _ = survey(pose_for_optical(0.005, 0.014, 0), np.zeros(2), intr)
    before = [stop_lines(stops, i) for i in range(len(stops))]
    for text in ('63', '-45', '0'):
        assert open_yaw(text) is not None
        for i in range(len(stops)):
            look_yaw_text(stops, i, 0)
    assert [stop_lines(stops, i) for i in range(len(stops))] == before


def test_second_stop_keeps_turn_when_target_heading_is_zero():
    import math
    from map_test import select_route_stop, stop_lines
    from motion import RouteStop, Look
    stops = [RouteStop(0,0,(Look('1',63.,0.),)), RouteStop(2,1,(Look('6',0.,-1.),))]
    view = {'pinned':True,'arrived':0,'look':0}
    select_route_stop(view, stops, 1, math.radians(63))
    assert view['auto_yaw'] == '0.0'
    text = '\n'.join(stop_lines(stops,1,view['look'],view['arrival_yaw']))
    assert '需向右转 63 度' in text
    assert '朝向 向左 0 度' in text


def test_repeated_click_uses_previous_heading_and_wraps_short_turn():
    import math
    from map_test import select_route_stop, stop_lines
    from motion import RouteStop, Look
    stops = [RouteStop(0,0,(Look('1',170.,0.),Look('2',-170.,0.)))]
    view = {'pinned':True,'arrived':0,'look':0}
    select_route_stop(view,stops,0,math.radians(170))
    assert '需向左转 20 度' in '\n'.join(stop_lines(stops,0,view['look'],view['arrival_yaw']))
