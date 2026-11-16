from map_test import route_position_key


def test_route_start_position_change_and_coordinate_noise():
    assert route_position_key(None) is None
    initial = route_position_key((0.005, 0.014))
    assert route_position_key((0.005000001, 0.014000001)) == initial
    assert route_position_key((0.105, 0.014)) != initial


def test_mode_buttons_hit_bounds():
    from map_test import mode_at
    assert mode_at(12, 40) == 'static'
    assert mode_at(200, 40) == 'laptop'
    assert mode_at(380, 40) == 'host'
    assert mode_at(12, 80) is None
    assert mode_at(550, 40) is None


def test_fixed_plan_survives_manual_heading_changes():
    import numpy as np
    from car_pose import load_intrinsics
    from map_test import ROOT, pose_for_corner, look_yaw_text, stop_lines, open_yaw
    from motion import survey
    intr = load_intrinsics(str(ROOT / 'config' / 'intrinsics.yaml'))
    stops, _ = survey(pose_for_corner(0.005, 0.014, 0), np.zeros(2), intr)
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
    assert '向右转 63 度' in text
    assert '先朝正前方0°' in text


def test_repeated_click_uses_previous_heading_and_wraps_short_turn():
    import math
    from map_test import select_route_stop, stop_lines
    from motion import RouteStop, Look
    stops = [RouteStop(0,0,(Look('1',170.,0.),Look('2',-170.,0.)))]
    view = {'pinned':True,'arrived':0,'look':0}
    select_route_stop(view,stops,0,math.radians(170))
    assert '向左转 20 度' in '\n'.join(stop_lines(stops,0,view['look'],view['arrival_yaw']))


def test_history_migrates_optical_points_once(tmp_path):
    import math
    import numpy as np
    from car_pose import CarPose, camera_position
    from map_test import load_points, save_points
    path = tmp_path / 'points.txt'
    # 旧点在不同航向下均保存光心，读入后必须恢复同一个右前角。
    lines = []
    for degrees in (0, 63, 90, 180, -135):
        optical = camera_position(CarPose(1.2, .4, math.radians(degrees)), np.zeros(2), 0)
        lines.append(f'{optical[0]:.10f} {optical[1]:.10f} {degrees}')
    path.write_text('\n'.join(lines))
    points = load_points(path)
    for point in points:
        np.testing.assert_allclose((point.x, point.y), (1.2, .4), atol=1e-9)
    save_points(path, points)
    again = load_points(path)
    for point in again:
        assert (point.x, point.y) == (1.2, .4)
    assert [point.yaw_text for point in again] == [point.yaw_text for point in points]


def test_camera_entry_preserves_corner_in_all_modes(monkeypatch):
    import math
    from types import SimpleNamespace
    import map_test
    captured = []
    monkeypatch.setattr(map_test, '_live', lambda args, watch, pose: captured.append(pose))
    for mode in ('static', 'ssh', 'local'):
        for heading in (0, 63, 90, 180, -135):
            yaw = math.radians(heading)
            map_test._open_camera(1.2, .4, yaw, 0, SimpleNamespace(pose_source=mode))
            assert (captured[-1].x, captured[-1].y, captured[-1].yaw) == (1.2, .4, yaw)


def test_station_panel_gives_one_fixed_pose_without_scan_lists():
    from map_test import stop_lines
    from motion import RouteStop, Look
    text = '\n'.join(stop_lines([RouteStop(0,0,(Look('1',62.,-1.),))],0))
    assert '先朝左62°' in text
    assert '未看清时可尝试' not in text
    assert '基准上下' not in text
    assert '固定姿态' in text
    assert '相机低头到 1 度' in text
    assert '完成识别' not in text
    assert '每个补充点仅一组朝向和俯仰' in text
