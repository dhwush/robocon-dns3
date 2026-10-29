"""小车中心的车位换成相机光心。

雷达话题给出的是中心的 x、y 和航向。场地表以右前角为原点。
两者差一段安装尺寸，没量之前不投影。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml


# 光心相对右前角：左侧 0.211 m，前方 0.05 m，上方 0.70 m。
OPTICAL_FROM_CORNER = np.array([-0.211, 0.05, 0.70], dtype=float)

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
        # 基础把车摆在原点，不看雷达。拓展要确认对齐，并且安装尺寸已经量过。
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
        if float(np.linalg.norm(corner)) < 1e-6:
            corner = None
    return Setup(
        aligned=bool(raw.get("aligned", False)),
        center_to_corner=corner,
        pitch_deg=float(raw.get("pitch_deg", 0.0)),
        topic=str(raw.get("topic") or ""),
        yaw_unit=str(raw.get("yaw_unit") or "rad"),
    )


def basic_origin_setup(pitch_deg: float) -> tuple[Setup, CarPose]:
    # 人把右前角摆在原点，车头朝前。不订阅雷达，也不读取中心到右前角的测量。
    # 航向为 0 时，这段安装差在投影里抵消，光心就是已知的安装位置。
    # 这里的零向量只用来完成这次抵消，不是量到的尺寸，也不写入配置。
    setup = Setup(
        aligned=False,
        center_to_corner=np.zeros(2, dtype=float),
        pitch_deg=float(pitch_deg),
        topic="",
        yaw_unit="rad",
        origin_basic=True,
    )
    return setup, CarPose(0.0, 0.0, 0.0)


def yaw_to_rad(yaw: float, unit: str) -> float:
    if unit == "deg":
        return float(np.deg2rad(yaw))
    return float(yaw)


def closest_pose(samples: list[CarPose], stamp: float) -> CarPose | None:
    if not samples:
        return None
    return min(samples, key=lambda pose: abs(pose.stamp - stamp))


def pose_moved(previous: CarPose | None, current: CarPose) -> bool:
    if previous is None:
        return False
    step = float(np.hypot(current.x - previous.x, current.y - previous.y))
    turn = abs(float(np.arctan2(np.sin(current.yaw - previous.yaw), np.cos(current.yaw - previous.yaw))))
    return bool(step > JITTER_M or turn > JITTER_YAW)


def _axes(yaw: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 航向为正是向左转。俯仰为正是抬头。相机 Y 轴朝下。
    yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
    right = np.array([yaw_c, yaw_s, 0.0], dtype=float)
    forward = np.array([-yaw_s, yaw_c, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    pitch = float(np.deg2rad(pitch_deg))
    ahead = np.cos(pitch) * forward + np.sin(pitch) * up
    down = np.sin(pitch) * forward - np.cos(pitch) * up
    return right, down, ahead


def camera_position(pose: CarPose, center_to_corner: np.ndarray, pitch_deg: float) -> np.ndarray:
    # 雷达原点是起步时的小车中心。场地原点是起步时的右前角。
    # 起步、航向为 0 时，光心要落在右前角坐标系里的安装位置上。
    center = np.array([pose.x, pose.y, 0.0], dtype=float) - np.array(
        [center_to_corner[0], center_to_corner[1], 0.0], dtype=float
    )
    yaw_c, yaw_s = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    body = np.array(
        [
            center_to_corner[0] + OPTICAL_FROM_CORNER[0],
            center_to_corner[1] + OPTICAL_FROM_CORNER[1],
            OPTICAL_FROM_CORNER[2],
        ],
        dtype=float,
    )
    rotated = np.array(
        [
            yaw_c * body[0] - yaw_s * body[1],
            yaw_s * body[0] + yaw_c * body[1],
            body[2],
        ],
        dtype=float,
    )
    return center + rotated


def to_camera(point: np.ndarray, pose: CarPose, center_to_corner: np.ndarray, pitch_deg: float) -> np.ndarray:
    origin = camera_position(pose, center_to_corner, pitch_deg)
    right, down, ahead = _axes(pose.yaw, pitch_deg)
    delta = np.asarray(point, dtype=float) - origin
    return np.array([float(np.dot(delta, right)), float(np.dot(delta, down)), float(np.dot(delta, ahead))])


def project(point: np.ndarray, pose: CarPose, center_to_corner: np.ndarray, pitch_deg: float, intr: Intrinsics) -> np.ndarray | None:
    cam = to_camera(point, pose, center_to_corner, pitch_deg)
    if cam[2] <= 0.2:
        return None
    return np.array([intr.fx * cam[0] / cam[2] + intr.cx, intr.fy * cam[1] / cam[2] + intr.cy], dtype=float)


def shifted_pose(pose: CarPose, lateral_m: float) -> CarPose:
    # 正数是向车的右侧平移。航向不变，高度仍在地面上。
    yaw_c, yaw_s = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    right = np.array([yaw_c, yaw_s], dtype=float)
    xy = np.array([pose.x, pose.y], dtype=float) + lateral_m * right
    return CarPose(float(xy[0]), float(xy[1]), pose.yaw, pose.stamp)

