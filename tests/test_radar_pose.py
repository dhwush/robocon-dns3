import math
from types import SimpleNamespace

import pytest

from car_pose import CarPose
from radar_pose import RadarPoseStream, decode_pose, normalize_source


def test_corner_output_is_used_without_mount_or_yaw_reset():
    p = decode_pose(dict(x=1.3, y=-.2, yaw=1.2, stamp_ns=1700000000123000000))
    assert (p.x, p.y, p.yaw) == (1.3, -.2, 1.2)
    assert p.stamp == pytest.approx(1700000000.123, abs=1e-6, rel=0.)


@pytest.mark.parametrize('payload', [dict(x=0,y=0,yaw=0), dict(x=math.nan,y=0,yaw=0,stamp_ns=1),
    dict(x=0,y=0,yaw=0,stamp_ns=0), dict(valid=False,x=0,y=0,yaw=0,stamp_ns=1)])
def test_invalid_pose_is_rejected(payload):
    with pytest.raises((ValueError, KeyError)):
        decode_pose(payload)


def test_match_rejects_frozen_data_and_time_mismatch(monkeypatch):
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream(start=False)
    stream.accept(CarPose(2,3,.4,99.95))
    assert stream.match(99.9) == CarPose(2,3,.4,99.95)
    assert stream.match(99.) is None
    stream.accept(CarPose(9,9,1.,99.95))  # 重复时间戳不产生新样本
    assert len(stream.samples) == 1
    assert stream.match().x == 2
    monkeypatch.setattr('radar_pose.time.time', lambda: 101.)
    assert stream.match() is None


def test_invalid_status_clears_previous_pose(monkeypatch):
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream(start=False)
    stream.accept(CarPose(2,3,.4,100.))
    stream.invalidate('定位失效')
    assert stream.match() is None
    assert stream.samples == __import__('collections').deque()


@pytest.mark.parametrize('source', ['ssh','local','static','radar','ros','laptop','host'])
def test_all_live_entry_aliases_use_radar(source):
    args = SimpleNamespace(pose_source=source, offline=False)
    normalize_source(args)
    assert args.pose_source == {'ssh': 'laptop', 'local': 'host'}.get(source, source)
    args.offline = True
    normalize_source(args)
    assert args.pose_source == 'static'


def test_once_does_not_use_manual_pose_when_radar_missing(monkeypatch):
    import view_field as vf
    from car_pose import Setup
    import numpy as np
    stream = SimpleNamespace(wait=lambda: None,close=lambda: None,error='离线')
    monkeypatch.setattr(vf, 'open_radar', lambda args: stream)
    watch = SimpleNamespace(setup=Setup(False,np.zeros(2),0.,'', 'rad'),
        votes={}, autonomy=None, stable_pose=CarPose(9,9,0), views={},
        update=lambda *args: pytest.fail('没有雷达时不允许计算'))
    lines = vf._once(SimpleNamespace(pose_source='radar'), watch, CarPose(9,9,0))
    assert 'x=--' in lines[0]
    assert watch.views is None


def test_once_uses_received_xy_yaw(monkeypatch):
    import view_field as vf
    from car_pose import Setup
    import numpy as np
    current = CarPose(2.1,-.8,1.3,100.)
    stream = SimpleNamespace(wait=lambda: current,close=lambda: None)
    monkeypatch.setattr(vf, 'open_radar', lambda args: stream)
    seen = []
    watch = SimpleNamespace(setup=Setup(False,np.zeros(2),0.,'', 'rad'),
        update=lambda pose,*args: seen.append(pose),lines=lambda mode: [])
    result = vf._once(SimpleNamespace(pose_source='radar',image='',mode='basic'), watch, CarPose(9,9,0))
    assert seen == [current]
    assert 'x=+2.100' in result[0] and 'y=-0.800' in result[0]


@pytest.mark.parametrize('mode', ['basic', 'extend', 'static'])
def test_all_modes_retain_radar_and_use_actual_pose(monkeypatch, mode):
    import view_field as vf
    current = CarPose(-1.4,2.6,.9,100.)
    stream = SimpleNamespace(wait=lambda: current, close=lambda: None)
    monkeypatch.setattr(vf, 'open_radar', lambda args: stream)
    seen = []
    monkeypatch.setattr('report.FieldWatch.update', lambda self, pose, *args: seen.append(pose))
    args = SimpleNamespace(mode=mode, pose_source='radar', offline=False, pitch=None,
        center_to_corner=None, aligned=False, topic='/vision/robot_pose', yaw_unit='rad',
        pose=(99.,99.,99.), image='', once=True)
    result = vf.run(args)
    assert args.pose_source == 'radar'
    assert seen == [current]
    assert 'x=-1.400' in result[0] and 'y=+2.600' in result[0]


def test_ros_message_preserves_corner_and_measurement_stamp(monkeypatch):
    from radar_pose import RosRadarPoseStream
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream(start=False)
    p = SimpleNamespace(position=SimpleNamespace(x=2.,y=-3.),
                        orientation=SimpleNamespace(x=0.,y=0.,z=math.sin(.4),w=math.cos(.4)))
    msg = SimpleNamespace(pose=SimpleNamespace(pose=p),
                          header=SimpleNamespace(stamp=SimpleNamespace(sec=100,nanosec=0)))
    RosRadarPoseStream._message(stream, msg)
    assert stream.match().x == 2.
    assert stream.match().y == -3.
    assert stream.match().yaw == pytest.approx(.8)
    msg.pose.pose.orientation.w = 0.
    RosRadarPoseStream._message(stream, msg)
    assert stream.match() is None


@pytest.mark.parametrize('source', ['laptop', 'host'])
def test_map_opens_from_radar_but_waits_for_b_before_planning(monkeypatch, source):
    import cv2
    import sys
    import numpy as np
    import map_test as m
    pose = CarPose(1.3,-.7,.8,100.)
    closed, planned, opened, drawn = [], [], [], []
    stream = SimpleNamespace(match=lambda *args: pose, close=lambda: closed.append(True))
    monkeypatch.setattr(m, 'open_radar', lambda args: stream)
    monkeypatch.setattr(sys, 'argv', ['map_test.py', '--pose-source', source])
    monkeypatch.setattr(m, 'load_points', lambda *args: [])
    monkeypatch.setattr(m, 'save_points', lambda *args: None)
    monkeypatch.setattr(m, 'survey', lambda actual,*args: (planned.append(actual) or [], []))
    monkeypatch.setattr(m, '_draw_map', lambda xy,yaw,*args,**kwargs: (drawn.append((xy,yaw)) or np.zeros((2,2,3),np.uint8)))
    monkeypatch.setattr(m, 'paint_modes', lambda board,*args: board)
    monkeypatch.setattr(m, '_open_camera', lambda x,y,yaw,*args: opened.append((x,y,yaw)))
    for name in ('namedWindow','setMouseCallback','destroyWindow','destroyAllWindows','imshow'):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    keys = iter([13, 27])
    monkeypatch.setattr(cv2, 'waitKeyEx', lambda *args: next(keys))
    m.main()
    assert planned == []  # 回车仅打开相机，B确认起点后才规划。
    assert opened == [(pose.x,pose.y,pose.yaw)]
    assert all(xy == (pose.x,pose.y) and yaw == pytest.approx(pose.yaw) for xy,yaw in drawn)
    assert closed == [True]


def test_http_reads_documented_json_and_original_timestamp(monkeypatch):
    import io
    import json
    from contextlib import closing
    monkeypatch.setattr('radar_pose.time.time', lambda: 100.)
    stream = RadarPoseStream('http://radar-host:8765/pose', start=False)
    calls = []
    def response(url, timeout):
        calls.append((url, timeout))
        stream.stop.set()
        return closing(io.BytesIO(json.dumps(dict(x=3.,y=4.,yaw=.5,stamp_ns=100000000000)).encode()))
    monkeypatch.setattr('radar_pose.urlopen', response)
    stream._run()
    assert calls == [('http://radar-host:8765/pose', .4)]
    assert stream.match() == CarPose(3.,4.,.5,100.)


@pytest.mark.parametrize('source', ['laptop', 'host'])
def test_device_modes_choose_ssh_or_local_corner_service(monkeypatch, source):
    import radar_pose as radar
    opened = []
    monkeypatch.setattr(radar, 'RadarPoseStream', lambda url, reference: opened.append((url, reference)))
    monkeypatch.setattr(radar, 'SshRadarPoseStream', lambda host, url, reference: opened.append((host, url, reference)))
    args = SimpleNamespace(pose_source=source, offline=False, ssh_host='robot@test-host')
    normalize_source(args)
    radar.open_radar(args)
    expected = ('http://127.0.0.1:8765/pose', 'corner')
    assert opened == [(('robot@test-host',) + expected) if source == 'laptop' else expected]


def test_frame_waits_briefly_for_next_5hz_measurement(monkeypatch):
    stream = RadarPoseStream(start=False)
    expected = CarPose(1,2,.3,100.)
    responses = iter([None, expected])
    monkeypatch.setattr(stream, 'match', lambda stamp: next(responses))
    assert stream.match_frame(100.) is expected
    monkeypatch.setattr(stream, 'match', lambda stamp: None)
    assert stream.match_frame(100., timeout=0.) is None


def test_ssh_stream_reads_original_pose_and_invalidates_disconnect(monkeypatch):
    import io
    import json
    import radar_pose as radar
    monkeypatch.setattr(radar.time, 'time', lambda: 100.)
    payload = dict(x=2.3, y=-1., yaw=.7, stamp_ns=100000000000)
    commands, accepted = [], []
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(payload)+'\n'), poll=lambda: 0)
    def spawn(command, **kwargs):
        commands.append(command)
        return process
    monkeypatch.setattr(radar.subprocess, 'Popen', spawn)
    original = radar.RadarPoseStream.accept
    def accept(self, pose):
        original(self, pose)
        accepted.append(pose)
    monkeypatch.setattr(radar.RadarPoseStream, 'accept', accept)
    stream = radar.SshRadarPoseStream('robot@test-host')
    stream.thread.join(timeout=1.)
    assert accepted == [CarPose(2.3, -1., .7, 100.)]
    assert commands[0][0] == 'ssh'
    assert commands[0][-2] == 'robot@test-host'
    assert '127.0.0.1:8765/pose' in commands[0][-1]
    assert stream.match() is None
    assert 'SSH' in stream.error
    stream.close()



def test_radar_text_distinguishes_sample_time_from_delay(monkeypatch):
    import radar_pose
    from car_pose import CarPose
    monkeypatch.setattr(radar_pose.time, 'time', lambda: 1794746759.289841)
    text = radar_pose.radar_pose_text(CarPose(0,0,0,1794746758.610841))
    assert '采样时刻=' in text
    assert '位姿延迟=679 ms' in text
    assert '测量时间=' not in text
    assert '1794746758' not in text


def test_raw_measurement_history_requires_latest_radar_to_remain_valid(monkeypatch):
    monkeypatch.setattr('radar_pose.time.time',lambda:100.)
    stream=RadarPoseStream(start=False)
    for stamp in (99.6,99.8,100.):
        stream.accept(CarPose(0,0,0,stamp))
    assert [p.stamp for p in stream.measurements_between(99.7,100.)] == [99.8,100.]
    stream.invalidate('断流')
    assert stream.measurements_between(99.,100.) == ()
