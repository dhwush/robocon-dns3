"""定位输入为小车右前角；相机安装偏移随车身航向旋转。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml


# 光心相对右前角：左侧 0.211 m，前方 0.05 m，上方 0.669 m。
BODY_SIDE_M = 0.35
BODY_HALF_M = BODY_SIDE_M / 2.0
CAMERA_HEIGHT_M = 0.669
OPTICAL_FROM_CORNER = np.array([-0.211, 0.05, CAMERA_HEIGHT_M], dtype=float)

# 静止时大约 1 cm 的抖动不重新计票。超过 2 cm 或 1 度才当成车被推过。
JITTER_M = 0.02
JITTER_YAW = np.deg2rad(1.0)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CarPose:
    x: float
    y: float
    yaw: float
    stamp: float = 0.0


@dataclass(frozen=True)
class Setup:
    aligned: bool
    center_to_corner: np.ndarray | None
    pitch_deg: float
    topic: str
    yaw_unit: str
    origin_basic: bool = False

    @property
    def ready(self) -> bool:
        # 基础用摆好的右前角；拓展确认右前角定位与场地对齐。
        if self.origin_basic:
            return self.center_to_corner is not None
        return bool(self.aligned) and self.center_to_corner is not None


def load_intrinsics(path: str) -> Intrinsics:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Intrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
    )


def load_setup(path: str) -> Setup:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    offset = raw.get("center_to_corner")
    if offset is None:
        corner = None
    else:
        corner = np.array([float(offset[0]), float(offset[1])], dtype=float)
    return Setup(
        aligned=bool(raw.get("aligned", False)),
        center_to_corner=corner,
        pitch_deg=float(raw.get("pitch_deg", 0.0)),
        topic=str(raw.get("topic") or ""),
        yaw_unit=str(raw.get("yaw_unit") or "rad"),
    )


# 基础把右前角摆在场地原点右侧 1.2 m，前后仍与场地原点对齐。车头朝前。
BASIC_CORNER_X = 1.2
BASIC_CORNER_Y = 0.0


def basic_origin_setup(pitch_deg: float) -> tuple[Setup, CarPose]:
    # 不订阅雷达，也不读取中心到右前角的测量。
    # 航向为 0 时，这段安装差在投影里抵消，位姿数字就是右前角在场地里的位置。
    # 这里的零向量只用来完成这次抵消，不是量到的尺寸，也不写入配置。
    setup = Setup(
        aligned=False,
        center_to_corner=np.zeros(2, dtype=float),
        pitch_deg=float(pitch_deg),
        topic="",
        yaw_unit="rad",
        origin_basic=True,
    )
    return setup, CarPose(BASIC_CORNER_X, BASIC_CORNER_Y, 0.0)


def yaw_to_rad(yaw: float, unit: str) -> float:
    if unit == "deg":
        return float(np.deg2rad(yaw))
    return float(yaw)


def closest_pose(samples: list[CarPose], stamp: float, max_delta: float = 0.15) -> CarPose | None:
    if not samples:
        return None
    match = min(samples, key=lambda pose: abs(pose.stamp - stamp))
    return match if abs(match.stamp - stamp) <= max_delta else None


def pose_moved(previous: CarPose | None, current: CarPose) -> bool:
    if previous is None:
        return False
    step = float(np.hypot(current.x - previous.x, current.y - previous.y))
    turn = abs(float(np.arctan2(np.sin(current.yaw - previous.yaw), np.cos(current.yaw - previous.yaw))))
    return bool(step > JITTER_M or turn > JITTER_YAW)


def _axes(yaw: float, pitch_deg: float, roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 航向为正是向左转。俯仰为正是抬头。横滚为正是右侧向下。相机 Y 轴朝下。
    yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
    right = np.array([yaw_c, yaw_s, 0.0], dtype=float)
    forward = np.array([-yaw_s, yaw_c, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    pitch = float(np.deg2rad(pitch_deg))
    ahead = np.cos(pitch) * forward + np.sin(pitch) * up
    down = np.sin(pitch) * forward - np.cos(pitch) * up
    roll = float(np.deg2rad(roll_deg))
    if abs(roll) < 1e-12:
        return right, down, ahead
    roll_c, roll_s = float(np.cos(roll)), float(np.sin(roll))
    rolled_right = roll_c * right + roll_s * down
    rolled_down = -roll_s * right + roll_c * down
    return rolled_right, rolled_down, ahead


def camera_position(pose: CarPose, center_to_corner: np.ndarray, pitch_deg: float) -> np.ndarray:
    # 保留 offset 参数以兼容调用方；定位点已是右前角，不再做中心补偿。
    yaw_c, yaw_s = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    dx, dy, dz = OPTICAL_FROM_CORNER
    return np.array([pose.x + yaw_c * dx - yaw_s * dy,
                     pose.y + yaw_s * dx + yaw_c * dy, dz], dtype=float)


def body_center(pose: CarPose) -> tuple[float, float]:
    c, s = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    return (pose.x - BODY_HALF_M * (c - s),
            pose.y - BODY_HALF_M * (s + c))


def turned_pose(pose: CarPose, yaw: float) -> CarPose:
    """绕车身几何中心原地转动；定位角点会随之移动。"""
    x, y = body_center(pose)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return CarPose(x + BODY_HALF_M * (c - s),
                   y + BODY_HALF_M * (s + c), yaw, pose.stamp)


def camera_frame(
    pose: CarPose, center_to_corner: np.ndarray, pitch_deg: float, roll_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 光心，以及相机系的右、下、前。深度点乘这三轴加回光心，就是场地上的点。
    return camera_position(pose, center_to_corner, pitch_deg), *_axes(pose.yaw, pitch_deg, roll_deg)


def to_camera(
    point: np.ndarray,
    pose: CarPose,
    center_to_corner: np.ndarray,
    pitch_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    origin, right, down, ahead = camera_frame(pose, center_to_corner, pitch_deg, roll_deg)
    delta = np.asarray(point, dtype=float) - origin
    return np.array([float(np.dot(delta, right)), float(np.dot(delta, down)), float(np.dot(delta, ahead))])


def project(
    point: np.ndarray,
    pose: CarPose,
    center_to_corner: np.ndarray,
    pitch_deg: float,
    intr: Intrinsics,
    roll_deg: float = 0.0,
) -> np.ndarray | None:
    cam = to_camera(point, pose, center_to_corner, pitch_deg, roll_deg)
    if cam[2] <= 0.2:
        return None
    return np.array([intr.fx * cam[0] / cam[2] + intr.cx, intr.fy * cam[1] / cam[2] + intr.cy], dtype=float)


def _gravity_down(gravity: np.ndarray | None) -> np.ndarray | None:
    if gravity is None:
        return None
    down = np.asarray(gravity, dtype=float)
    norm = float(np.linalg.norm(down))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return down / norm


def pitch_from_gravity(gravity: np.ndarray | None) -> float | None:
    # 彩色相机系里的重力方向。正数是抬头，单位度。相机 Z 向前，抬头时重力的前向分量是负的。
    down = _gravity_down(gravity)
    if down is None:
        return None
    return float(-np.degrees(np.arcsin(np.clip(float(down[2]), -1.0, 1.0))))


def roll_from_gravity(gravity: np.ndarray | None) -> float | None:
    # 和俯仰同一条重力。正数是右侧向下，单位度。
    down = _gravity_down(gravity)
    if down is None:
        return None
    return float(np.degrees(np.arctan2(float(down[0]), float(down[1]))))


def shifted_pose(pose: CarPose, lateral_m: float) -> CarPose:
    # 正数是向车的右侧平移。航向不变，高度仍在地面上。
    yaw_c, yaw_s = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    right = np.array([yaw_c, yaw_s], dtype=float)
    xy = np.array([pose.x, pose.y], dtype=float) + lateral_m * right
    return CarPose(float(xy[0]), float(xy[1]), pose.yaw, pose.stamp)


def bearing_deg(point: np.ndarray, pose: CarPose, center_to_corner: np.ndarray) -> float:
    # 地面上的方位。正数在车的右侧，身后的目标大于 90 度。不随俯仰改变。
    origin = camera_position(pose, center_to_corner, 0.0)
    dx = float(point[0] - origin[0])
    dy = float(point[1] - origin[1])
    ahead = -float(np.sin(pose.yaw)) * dx + float(np.cos(pose.yaw)) * dy
    right = float(np.cos(pose.yaw)) * dx + float(np.sin(pose.yaw)) * dy
    if abs(ahead) < 1e-9 and abs(right) < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(right, ahead)))

