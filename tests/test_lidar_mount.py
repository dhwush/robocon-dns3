import math
import numpy as np
from car_pose import CarPose, lidar_pose_to_corner, LIDAR_FROM_CORNER


def test_lidar_installation_rotates_with_vehicle():
    for yaw in (0., math.pi / 2, -math.pi / 2, math.pi):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = LIDAR_FROM_CORNER[:2]
        raw = CarPose(1.2 + c*dx - s*dy, -.4 + s*dx + c*dy, yaw, 123.)
        corner = lidar_pose_to_corner(raw)
        np.testing.assert_allclose([corner.x, corner.y], [1.2, -.4])
        assert corner.yaw == yaw
        assert corner.stamp == 123.
