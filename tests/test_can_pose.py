import struct
from can_pose import PoseDecoder, align_can_pose
import math
from car_pose import CarPose


def test_alignment_maps_first_frame_to_known_field_pose():
    raw = CarPose(.606, -.696, math.radians(98.7), 12)
    field = CarPose(1.185, .254, 0)
    aligned = align_can_pose(raw, raw, field)
    assert (aligned.x, aligned.y, aligned.yaw, aligned.stamp) == (1.185, .254, 0, 12)


def test_alignment_rotates_motion_and_preserves_heading_change():
    raw_origin = CarPose(2, 3, math.pi / 2)
    field_origin = CarPose(1, .25, 0)
    # CAN 初始前方为 -X，走一米应变为场地 +Y；左转20度。
    result = align_can_pose(CarPose(1, 3, math.pi / 2 + math.radians(20), 15),
                            raw_origin, field_origin)
    assert result.x == pytest.approx(1)
    assert result.y == pytest.approx(1.25)
    assert result.yaw == pytest.approx(math.radians(20))
    assert result.stamp == 15


def test_alignment_wraps_yaw_and_supports_nonzero_start_heading():
    raw = CarPose(0, 0, math.radians(170))
    field = CarPose(-2.415, 3.254, math.radians(90))
    result = align_can_pose(CarPose(0, 0, math.radians(-170), 20), raw, field)
    assert result.yaw == pytest.approx(math.radians(110))
    assert (result.x, result.y) == (field.x, field.y)
import json
import subprocess
from types import SimpleNamespace
import pytest
from can_pose import CAN_SETUP


@pytest.mark.parametrize('bitrate,up,reconfigure', [(1000000, True, False), (500000, True, True), (1000000, False, True)])
def test_can_setup_uses_protocol_bitrate(monkeypatch, bitrate, up, reconfigure):
    calls = []
    link = {'flags': ['UP'] if up else [], 'linkinfo': {'info_kind': 'can',
            'info_data': {'bittiming': {'bitrate': bitrate}, 'state': 'ERROR-ACTIVE'}}}
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([link]))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('os.geteuid', lambda: 1000)
    namespace = {}
    exec(CAN_SETUP, namespace)
    namespace['setup_can']('can0')
    assert calls[1:] == ([
        ['sudo', '-n', 'ip', 'link', 'set', 'dev', 'can0', 'down'],
        ['sudo', '-n', 'ip', 'link', 'set', 'dev', 'can0', 'type', 'can', 'bitrate', '1000000'],
        ['sudo', '-n', 'ip', 'link', 'set', 'dev', 'can0', 'up'],
    ] if reconfigure else [])


def test_can_setup_stops_on_permission_failure(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) > 1:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps([{'linkinfo': {'info_kind': 'can'}}]))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('os.geteuid', lambda: 1000)
    namespace = {}
    exec(CAN_SETUP, namespace)
    with pytest.raises(RuntimeError, match='sudo -n permission'):
        namespace['setup_can']('can0')
    assert len(calls) == 2


def frame(ident, a, b):
    return f'{ident:X} {struct.pack("<ff", a, b).hex()}'


def test_decode_units_pairing_and_expiry():
    decoder = PoseDecoder()
    assert decoder.feed(frame(0xAB, 1, 0), 1) is None
    assert decoder.feed(frame(0xAA, 1200, -300), 1) is None
    pose = decoder.feed(frame(0xAB, 0.5, 0), 1.01)
    assert (pose.x, pose.y, pose.yaw) == (1.2, -0.3, 0.5)
    assert decoder.feed(frame(0xAB, 1, 0), 1.02) is None
    decoder.feed(frame(0xAA, 100, 200), 2)
    assert decoder.feed(frame(0xAB, 1, 0), 2.2) is None
    assert decoder.feed(frame(0xAA, float('nan'), 0), 3) is None
    assert decoder.feed('AA 00', 3) is None


def test_can_corner_pose_moves_projection_without_center_offset():
    import numpy as np
    from car_pose import Intrinsics, camera_position, project
    decoder = PoseDecoder()
    intr = Intrinsics(1280, 720, 600, 600, 640, 360)
    offset = np.zeros(2)
    target = np.array([0.0, 3.0, 0.35])
    def read(x, yaw):
        decoder.feed(frame(0xAA, x, 0), 1)
        return decoder.feed(frame(0xAB, yaw, 0), 1.01)
    start = read(0, 0)
    moved = read(100, 0)
    turned = read(0, 0.1)
    assert np.allclose(camera_position(start, offset, 0)[:2], [-0.211, 0.05])
    assert project(target, moved, offset, 0, intr)[0] < project(target, start, offset, 0, intr)[0]
    assert project(target, turned, offset, 0, intr)[0] > project(target, start, offset, 0, intr)[0]


def test_interactive_can_setup_allows_sudo_password(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps([{'linkinfo': {'info_kind': 'can'}}]))
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('os.geteuid', lambda: 1000)
    namespace = {}
    exec(CAN_SETUP, namespace)
    namespace['setup_can']('can0', interactive=True)
    assert len(calls) == 4
    for command, kwargs in calls[1:]:
        assert command[:2] == ['sudo', 'ip']
        assert '-n' not in command
        assert not kwargs.get('capture_output', False)
        assert 'stdin' not in kwargs


def test_ssh_setup_finishes_before_starting_can_reader(monkeypatch):
    import io
    import shlex
    from can_pose import CanPoseStream
    events = []
    def run(command, **kwargs):
        events.append(('setup', command, kwargs))
    def popen(command, **kwargs):
        events.append(('reader', command, kwargs))
        return SimpleNamespace(stdout=io.StringIO(''), poll=lambda: 0)
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    stream = CanPoseStream('ssh', 'can0', 'robot@example')
    stream.close()
    assert [e[0] for e in events] == ['setup', 'reader']
    setup, reader = events
    assert '-t' in setup[1]
    assert '-t' not in reader[1]
    assert setup[2] == {'check': True}
    remote = shlex.split(setup[1][-1])
    assert remote[:3] == ['python3', '-u', '-c']
    assert 'interactive=True' in remote[3]
    assert remote[4] == 'can0'
    assert reader[2]['stdout'] == subprocess.PIPE


def test_ssh_setup_failure_does_not_start_reader(monkeypatch):
    from can_pose import CanPoseStream
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    def popen(*args, **kwargs):
        raise AssertionError('初始化失败后不能启动读取进程')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='SSH CAN 初始化失败'):
        CanPoseStream('ssh')
