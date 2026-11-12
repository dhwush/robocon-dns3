import struct
from can_pose import PoseDecoder


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
