"""一帧里每一格的结论。对不上三档就保持未知，不挑一个最近的。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from car_pose import CarPose, Intrinsics, project, to_camera
from field_map import MARGIN_M, Cell, tower_cells
from sight import sees, visible_height


# 白：饱和度不超过 110，明度不低于 140。红蓝沿用塔顶阈值。
WHITE_S = 110
WHITE_V = 140
RED = ((0, 150, 120), (8, 255, 255))
RED_WRAP = ((172, 150, 120), (180, 255, 255))
BLUE = ((105, 150, 50), (130, 255, 255))

WIDTH_M = 0.35
WIDTH_TOL_M = 0.10
VIEW_FRACTION = 0.70
# 可见高度小于 8 cm 的那一档，当成没看见。
MIN_VISIBLE_M = 0.08
# 白盒 0.35 m，两层 0.70 m，塔顶再高 0.20 m。搜索框比塔顶再高出 0.10 m。
_TOP_MARGIN_M = 0.10


@dataclass
class CellView:
    conclusion: str | None
    in_view: bool
    needs_yaw: bool
    needs_pitch: bool
    blocked: bool
    bearing_deg: float
    pitch_target_deg: float
    white_frac: float
    mean_v: float
    region: np.ndarray | None = None
    piece: np.ndarray | None = None


def heights_match(measured: float, predicted: float) -> bool:
    if predicted < MIN_VISIBLE_M:
        return False
    return abs(measured - predicted) <= 0.20 * predicted + 1e-6


def _samples(cell: Cell) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    xs = np.linspace(x0, x1, 5)
    ys = np.linspace(y0, y1, 5)
    return [(float(x), float(y)) for x in xs for y in ys]


def _inside(pixel: np.ndarray | None, intr: Intrinsics) -> bool:
    if pixel is None:
        return False
    return 0.0 <= float(pixel[0]) < intr.width and 0.0 <= float(pixel[1]) < intr.height


def _bearing_deg(point: np.ndarray, pose: CarPose, offset: np.ndarray, pitch: float) -> float:
    cam = to_camera(point, pose, offset, pitch)
    return float(np.degrees(np.arctan2(cam[0], max(cam[2], 1e-6))))


def _pitch_target(cell: Cell, pose: CarPose, offset: np.ndarray) -> float:
    # 台面（或地面）到塔顶的中点，抬到相对水平的角度。这是要转到的角度。
    origin_z = 0.70
    mid = cell.center((cell.base_z + cell.cap_z) / 2.0)
    flat = np.array([mid[0], mid[1], origin_z], dtype=float)
    cam = to_camera(mid, pose, offset, 0.0)
    ahead = to_camera(flat, pose, offset, 0.0)
    horizontal = float(np.hypot(ahead[0], ahead[2]))
    return float(np.degrees(np.arctan2(mid[2] - origin_z, max(horizontal, 1e-6))))


def _mask(cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics) -> np.ndarray:
    shape = (intr.height, intr.width)
    pixels = []
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    low_z = cell.base_z
    high_z = cell.cap_z + _TOP_MARGIN_M
    for x, y, z in (
        (x0, y0, low_z),
        (x1, y0, low_z),
        (x1, y1, low_z),
        (x0, y1, low_z),
        (x0, y0, high_z),
        (x1, y0, high_z),
        (x1, y1, high_z),
        (x0, y1, high_z),
    ):
        pixel = project(np.array([x, y, z]), pose, offset, pitch, intr)
        if pixel is not None:
            pixels.append(pixel)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pixels) < 3:
        return mask
    hull = cv2.convexHull(np.round(np.stack(pixels)).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _view_flags(cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics) -> tuple[bool, bool, bool]:
    horizontal = 0
    vertical = 0
    total = 0
    for x, y in _samples(cell):
        total += 1
        low = project(np.array([x, y, cell.base_z]), pose, offset, pitch, intr)
        high = project(np.array([x, y, cell.cap_z]), pose, offset, pitch, intr)
        low_h = low is not None and 0.0 <= float(low[0]) < intr.width
        high_h = high is not None and 0.0 <= float(high[0]) < intr.width
        if low_h and high_h:
            horizontal += 1
        low_v = _inside(low, intr)
        high_v = _inside(high, intr)
        if low_v and high_v:
            vertical += 1
    if total == 0:
        return False, True, False
    horiz_ok = horizontal / total >= VIEW_FRACTION
    vert_ok = vertical / total >= VIEW_FRACTION
    return horiz_ok and vert_ok, not horiz_ok, horiz_ok and not vert_ok


def _white_mask(color: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S) & (hsv[:, :, 2] >= WHITE_V)
    return white.astype(np.uint8) * 255


def _color_count(hsv: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int], region: np.ndarray) -> int:
    picked = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    return int(np.count_nonzero((picked > 0) & region))


def _faction(color: np.ndarray, mask: np.ndarray, top: int, bottom: int, z_cam: float, intr: Intrinsics) -> str | None:
    cap_px = max(4, int(round(0.20 * intr.fy / max(z_cam, 0.3))))
    y0 = max(0, top - cap_px)
    y1 = min(color.shape[0], top + max(4, int(0.4 * max(bottom - top, 1))))
    region = np.zeros(mask.shape, dtype=bool)
    region[y0:y1, :] = mask[y0:y1, :] > 0
    if int(region.sum()) == 0:
        return None
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = _color_count(hsv, RED[0], RED[1], region) + _color_count(hsv, RED_WRAP[0], RED_WRAP[1], region)
    blue = _color_count(hsv, BLUE[0], BLUE[1], region)
    need = max(4, int(0.25 * cap_px))
    if red < need and blue < need:
        return None
    if red >= need and red > blue * 1.2:
        return "红"
    if blue >= need and blue > red * 1.2:
        return "蓝"
    return None


def _metric(pixels: float, z_cam: float, focal: float) -> float:
    return float(pixels) * z_cam / focal


def _depth_ok(depth: np.ndarray | None, ys: np.ndarray, xs: np.ndarray, z_cam: float) -> float:
    if depth is None or len(xs) == 0:
        return z_cam
    values = depth[ys, xs]
    good = np.isfinite(values) & (values >= 0.3) & (values <= 3.0)
    if int(good.sum()) < 10:
        return z_cam
    measured = float(np.median(values[good]))
    if abs(measured - z_cam) / max(z_cam, 1e-6) > 0.15:
        return z_cam
    return measured


def _outline(mask: np.ndarray) -> np.ndarray | None:
    if mask.size == 0 or int(mask.max()) == 0:
        return None
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    return max(found, key=cv2.contourArea)


def _classify_cell(
    cell: Cell,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None,
    depth: np.ndarray | None,
) -> CellView:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr)
    # 一层只露出不到 8 cm 时，这一档不能写成空，也不能写成一层。
    one_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.one_z)
    blocked = one_h < MIN_VISIBLE_M
    bearing = _bearing_deg(cell.center(cell.one_z), pose, offset, pitch)
    pitch_target = _pitch_target(cell, pose, offset)
    mask = _mask(cell, pose, offset, pitch, intr)
    region = _outline(mask)

    def pack(conclusion, frac: float, mean_v: float, piece=None) -> CellView:
        return CellView(
            conclusion,
            in_view,
            needs_yaw,
            needs_pitch,
            blocked,
            bearing,
            pitch_target,
            frac,
            mean_v,
            region,
            piece,
        )

    if color is None or not in_view:
        return pack(None, 0.0, 255.0)
    white = _white_mask(color)
    inside = (white > 0) & (mask > 0)
    area = int(np.count_nonzero(mask))
    frac = 0.0 if area == 0 else float(np.count_nonzero(inside)) / float(area)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mean_v = 255.0 if area == 0 else float(np.mean(hsv[:, :, 2][mask > 0]))
    cam = to_camera(cell.center(cell.one_z), pose, offset, pitch)
    z_cam = float(cam[2])
    if z_cam <= 0.2:
        return pack(None, 0.0, 255.0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    best = None
    best_width = None
    for index in range(1, count):
        x, y, w, h, _pixels = [int(stats[index, k]) for k in range(5)]
        if w < 2 or h < 2:
            continue
        if x <= 1 or y <= 1 or x + w >= intr.width - 1 or y + h >= intr.height - 1:
            continue
        component = labels == index
        ys, xs = np.nonzero(component)
        distance = _depth_ok(depth, ys, xs, z_cam)
        width_m = _metric(w, distance, intr.fx)
        if abs(width_m - WIDTH_M) > WIDTH_TOL_M:
            continue
        if best_width is None or abs(width_m - WIDTH_M) < abs(best_width - WIDTH_M):
            best = (x, y, w, h, distance, component)
            best_width = width_m
    if best is None:
        if blocked:
            return pack(None, frac, mean_v)
        return pack("0", frac, mean_v)
    _x, y, _w, h, distance, component = best
    piece = _outline(component.astype(np.uint8) * 255)
    height_m = _metric(h, distance, intr.fy)
    two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
    one_ok = (not blocked) and heights_match(height_m, one_h)
    two_ok = heights_match(height_m, two_h)
    if not one_ok and not two_ok:
        return pack(None, frac, mean_v)
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return pack("1", frac, mean_v, piece)
    faction = _faction(color, mask, y, y + h, distance, intr)
    state = faction if faction is not None else "2"
    return pack(state, frac, mean_v, piece)


def inspect_frame(
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> dict[str, CellView]:
    return {
        cell.name: _classify_cell(cell, pose, offset, pitch, intr, color, depth)
        for cell in tower_cells()
    }

