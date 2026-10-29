"""视线先于颜色。射线碰到更高的台，这一档高度就不能当结论。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from car_pose import CarPose, camera_position
from field_map import CELL_M, Cell, all_cells


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    name: str


def platform_boxes() -> tuple[Box, ...]:
    half = CELL_M / 2.0
    boxes = []
    for cell in all_cells():
        boxes.append(
            Box(
                cell.x - half,
                cell.x + half,
                cell.y - half,
                cell.y + half,
                0.0,
                cell.height,
                cell.name,
            )
        )
    return tuple(boxes)


def _hits(origin: np.ndarray, target: np.ndarray, box: Box) -> bool:
    # 只算还没走到目标点之前的那一段。贴着目标的末端不算挡住。
    direction = target - origin
    enter, leave = 1e-3, 1.0 - 1e-3
    for axis, lo, hi in (
        (0, box.x0, box.x1),
        (1, box.y0, box.y1),
        (2, box.z0, box.z1),
    ):
        if abs(float(direction[axis])) < 1e-9:
            if float(origin[axis]) < lo or float(origin[axis]) > hi:
                return False
            continue
        inv = 1.0 / float(direction[axis])
        near = (lo - float(origin[axis])) * inv
        far = (hi - float(origin[axis])) * inv
        if near > far:
            near, far = far, near
        enter = max(enter, near)
        leave = min(leave, far)
        if enter > leave:
            return False
    return True


def sees(
    pose: CarPose,
    center_to_corner: np.ndarray,
    pitch_deg: float,
    cell: Cell,
    height: float,
    boxes: tuple[Box, ...] | None = None,
) -> bool:
    origin = camera_position(pose, center_to_corner, pitch_deg)
    target = cell.center(height)
    solids = boxes if boxes is not None else platform_boxes()
    for box in solids:
        if box.name == cell.name:
            continue
        if _hits(origin, target, box):
            return False
    return True


def visible_height(
    pose: CarPose,
    center_to_corner: np.ndarray,
    pitch_deg: float,
    cell: Cell,
    z0: float,
    z1: float,
) -> float:
    # 从下往上每 2 cm 看一次。挡住的下半截不计入可见高度。
    boxes = platform_boxes()
    levels = np.arange(z0, z1 + 1e-6, 0.02)
    clear = [
        float(level)
        for level in levels
        if sees(pose, center_to_corner, pitch_deg, cell, float(level), boxes)
    ]
    if len(clear) < 2:
        return 0.0
    best = 0.0
    run = clear[0]
    previous = clear[0]
    for level in clear[1:]:
        if level - previous > 0.03:
            best = max(best, previous - run)
            run = level
        previous = level
    return max(best, previous - run)

