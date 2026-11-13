"""一帧里每一格的结论。对不上三档就保持未知。3 m 以内，白块不是实心直边也保持未知。

搜索框只决定白块属于哪一格：质心落在框里，宽度和高度量整块，不再先和框相与。
近处白块顶边或左右贴到画面边，或整块宽过 65 cm，彩色这条不收。
底边被切掉、顶边还在里面时，用车位把顶边换成离地高度，仍走彩色。
0.3 m 到 3 m 里，只要有足够的白点落在这一格，层数用最高一截的离地高度。
整块在画面里、彩色轮廓收不成直边时，这段对上了层高的深度仍然写。
近处两层的顶面常常没有深度。同一块白的顶边按车位换成离地高度，对上两层就写两层。
深度最高一截落在层高之间时保持未知；仅在顶边有证据时补判。
远处超过 3 m、没有深度时，框里的白先补上几像素宽的黑缝，再量整块的宽和高，不再要求直边。
3 m 以内仍要直边；轮廓不整齐时只靠深度写层数。
偏黄的地板从白掩膜里拿掉，白盒正面留着。层数仍看剩下的白块。
框里的白被切碎，或只剩不够 15 cm 的一截，同样保持未知，不写成空。
三层塔的阵营看贴在白盒顶上的像素带，从上往下认。和顶边隔开的红蓝不算。分不出先后就保持未知，不写成两层。
0.3 m 到 3 m 里，挡在格子近面前面、又落在这块塔的画面里的箱子和杂物不写成塔，也不写成空。
像方块、但深度不在台心 50 cm 里的白块同样丢掉。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from car_pose import CAMERA_HEIGHT_M, CarPose, Intrinsics, bearing_deg, camera_frame, project, to_camera
from field_map import MARGIN_M, Cell, tower_cells
from sight import sees, visible_height


# 白：饱和度不超过 110，明度不低于 118。6 号盒面偏暗的地方大约 122，125 收不进。
# 红的色相仍是 0–8 和 172–180，明度放到 110，罩住这张画面里大约 118 的红。
# 蓝的色相仍是 105–130，饱和度放到 110，罩住这张画面里大约 113 的蓝。
WHITE_S = 110
WHITE_V = 118
# 黄地面色相大约 16，亮处大约 12–19，饱和度大约 49，明度大约 135。
# 这块地上的阴影色相大约 19，大约 13–24，饱和度大约 27，明度大约 113。色相还是黄，靠明度分开。
# 饱和度达到 48 的黄才从白里拿掉，盒边大约 40 的黄留着。盒子底边连着格子外面的地板，所以整幅都拿。
FLOOR_H = (10, 26)
FLOOR_S = 48
RED = ((0, 150, 110), (8, 255, 255))
RED_WRAP = ((172, 150, 110), (180, 255, 255))
BLUE = ((105, 110, 50), (130, 255, 255))

WIDTH_M = 0.35
# 搜索框宽 60 cm。白块铺满框时，按格子中心的距离会量到 60 cm 出头。收到 65 cm，两块并排的 70 cm 仍然不要。
WIDTH_MIN_M = 0.15
WIDTH_MAX_M = 0.65
# 实心：白像素至少占外轮廓内部的七成五。线、空心框留不住。
FILL_MIN = 0.75
# 边比较直。先按短边的 8% 收成折线，折线离原轮廓也不超过这个距离。
# 白掩膜边上大约 2.5 像素的台阶和毛刺，这个数留给它。
# 四到七边直接留下：正面、看见顶面的六边、远处多出来的一个角、错开的台阶。
# 八边才比圆度，高于方块（π/4）和圆（1）的中点就丢掉。齿轮的角多于 8 个，也丢掉。
EDGE_SIMPLIFY = 0.08
CORNER_MIN = 4
CORNER_MAX = 8
_CIRCLE_LIMIT = (np.pi / 4.0 + 1.0) / 2.0
VIEW_FRACTION = 0.70
# 可见高度小于 8 cm 的那一档，当成没看见。
MIN_VISIBLE_M = 0.08
# 白盒 0.35 m，两层 0.70 m，塔顶再高 0.20 m。搜索框比塔顶再高出 0.10 m。
_TOP_MARGIN_M = 0.10
# 深度和视线一样，每 2 cm 看一截。相机深度只在大约 0.3 m 到 3 m 之间有数。
_DEPTH_SLICE_M = 0.02
_DEPTH_NEAR_M = 0.3
_DEPTH_FAR_M = 3.0
# 仅用于修复跨过3米边界、被硬截断的同一座塔；不扩大空点/遮挡判定范围。
_DEPTH_BOUNDARY_M = 0.30
_DEPTH_EDGE_M = 0.05
_DEPTH_MIN_POINTS = 10
# 远处黑缝大约几像素。补上这一道，两块并排的 70 cm 仍然合不到一起。
_FAR_SEAM_PX = 6
# 比这一格 50 cm 的近面再近 25 cm，才算挡在塔前面。塔自己的正面在近面里侧，不会被算进去。
_FRONT_M = 0.25
# 这些点占到搜索框的一成，或盖住一层顶到两层顶，这一眼就不认塔。
_OCCLUDE_FRAC = 0.10


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
    method: str | None = None
    skip_vote: bool = False
    reason: str | None = None
    measurements: dict | None = None
    needs_parallax: bool = False


def heights_match(measured: float, predicted: float) -> bool:
    # 五成。没有深度时用像素高度，近处抬头会把顶面算进去，两层会比 70 cm 高一截。
    if predicted < MIN_VISIBLE_M:
        return False
    return abs(measured - predicted) <= 0.50 * predicted + 1e-6


def heights_match_level(measured: float, predicted: float) -> bool:
    # 两成。离地高度对一层、两层的绝对档，间隔是 35 cm，两成以内两档不重叠。
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
    # 俯仰不参与。身后的格子按真实转角算，不再被夹成 90 度。
    del pitch
    return bearing_deg(point, pose, offset)


def _pitch_target(cell: Cell, pose: CarPose, offset: np.ndarray) -> float:
    # 一层顶到塔顶的中点，抬到相对水平的角度。这是要转到的角度。
    # 不把台面算进去。近处台面到塔顶整段对不齐彩色的垂直视场。
    origin_z = CAMERA_HEIGHT_M
    mid = cell.center((cell.one_z + cell.cap_z) / 2.0)
    flat = np.array([mid[0], mid[1], origin_z], dtype=float)
    cam = to_camera(mid, pose, offset, 0.0)
    ahead = to_camera(flat, pose, offset, 0.0)
    horizontal = float(np.hypot(ahead[0], ahead[2]))
    return float(np.degrees(np.arctan2(mid[2] - origin_z, max(horizontal, 1e-6))))


def _mask(cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0) -> np.ndarray:
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
        pixel = project(np.array([x, y, z]), pose, offset, pitch, intr, roll)
        if pixel is not None:
            pixels.append(pixel)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pixels) < 3:
        return mask
    hull = cv2.convexHull(np.round(np.stack(pixels)).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _view_flags(
    cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0
) -> tuple[bool, bool, bool]:
    # 左右仍要这块 50 cm 的大部分进画面。
    # 竖直只看一层顶和塔顶。近处从台面到搜索框顶已经塞满彩色的垂直视场，
    # 台面掉在画面下面不因此算没对准。
    horizontal = 0
    vertical = 0
    total = 0
    for x, y in _samples(cell):
        total += 1
        low = project(np.array([x, y, cell.one_z]), pose, offset, pitch, intr, roll)
        high = project(np.array([x, y, cell.cap_z]), pose, offset, pitch, intr, roll)
        low_h = low is not None and 0.0 <= float(low[0]) < intr.width
        high_h = high is not None and 0.0 <= float(high[0]) < intr.width
        if low_h and high_h:
            horizontal += 1
        if _inside(low, intr) and _inside(high, intr):
            vertical += 1
    if total == 0:
        return False, True, False
    horiz_ok = horizontal / total >= VIEW_FRACTION
    vert_ok = vertical / total >= VIEW_FRACTION
    return horiz_ok and vert_ok, not horiz_ok, horiz_ok and not vert_ok


def _split_white(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 第二项是从白里拿掉的地板。这一格的框里有这种黄，彩色结论才标成黄地。
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    floor = (
        (hsv[:, :, 0] >= FLOOR_H[0])
        & (hsv[:, :, 0] <= FLOOR_H[1])
        & (hsv[:, :, 1] >= FLOOR_S)
        & (hsv[:, :, 1] <= WHITE_S)
        & (hsv[:, :, 2] >= WHITE_V)
    )
    white = (hsv[:, :, 1] <= WHITE_S) & (hsv[:, :, 2] >= WHITE_V) & ~floor
    return white.astype(np.uint8) * 255, floor


def _white_mask(color: np.ndarray) -> np.ndarray:
    white, _floor = _split_white(color)
    return white


def _color_method(mask: np.ndarray, floor: np.ndarray | None) -> str:
    if floor is not None and mask.size and bool(np.any(floor & (mask > 0))):
        return "黄地"
    return "彩色"


def _band_above(white: np.ndarray, cell_mask: np.ndarray, cap_px: int) -> tuple[np.ndarray, np.ndarray]:
    # 每一列从这一列的白盒顶边往上 cap_px。再收进这一格的投影里。
    present = white > 0
    height = present.shape[0]
    rows = np.arange(height, dtype=np.int32)[:, None]
    tops = np.where(present, rows, height).min(axis=0)
    y0 = np.clip(tops - int(cap_px), 0, height)
    band = (rows >= y0) & (rows < tops) & (tops < height)
    if cell_mask.shape == band.shape:
        band &= cell_mask > 0
    return band, tops


def _faction(color: np.ndarray, white: np.ndarray, cell_mask: np.ndarray, z_cam: float, intr: Intrinsics, diagnostic: dict | None = None, linked_cap: bool = False) -> tuple[str | None, bool]:
    # 只认贴着白盒顶边的那条像素带，从上往下数。
    # 和顶边隔开的红蓝不算塔顶。红蓝从上面一起数够，第二项为真，调用处保持未知。
    cap_px = max(4, int(round(0.20 * intr.fy / max(float(z_cam), 0.3))))
    band, tops = _band_above(white, cell_mask, cap_px)
    if linked_cap:
        # 独立核验过完整顶块后，保留整块高度，防止透视或小黑缝截掉上色。
        cap_y, cap_x = np.nonzero(cell_mask)
        usable = tops[cap_x] < white.shape[0]
        if np.any(usable):
            cap_px = max(cap_px, int(np.max(tops[cap_x[usable]] - cap_y[usable])))
            band, tops = _band_above(white, cell_mask, cap_px)
    debug = None
    if diagnostic is not None:
        debug = {"distance_m": float(z_cam), "cap_px": cap_px,
                 "red_pixels": 0, "blue_pixels": 0, "result": None,
                 "status": "无取色带", "need": 0}
        diagnostic["faction"] = debug
        by, bx = np.nonzero(band)
        if len(bx):
            x0, x1 = int(bx.min()), int(bx.max()) + 1
            y0, y1 = int(by.min()), int(by.max()) + 1
            debug.update(bbox=(x0, y0, x1, y1),
                         band=band[y0:y1, x0:x1].copy(),
                         tops=tops[x0:x1].copy(),
                         status="带内没有红蓝像素")
    if not np.any(band):
        return None, False
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = (
        (cv2.inRange(hsv, np.array(RED[0], dtype=np.uint8), np.array(RED[1], dtype=np.uint8)) > 0)
        | (cv2.inRange(hsv, np.array(RED_WRAP[0], dtype=np.uint8), np.array(RED_WRAP[1], dtype=np.uint8)) > 0)
    ) & band
    blue = (cv2.inRange(hsv, np.array(BLUE[0], dtype=np.uint8), np.array(BLUE[1], dtype=np.uint8)) > 0) & band
    colored = red | blue
    if debug is not None:
        debug["raw_red_pixels"] = int(red.sum())
        debug["raw_blue_pixels"] = int(blue.sum())
        debug["red"] = red[y0:y1, x0:x1].copy()
        debug["blue"] = blue[y0:y1, x0:x1].copy()
    if not np.any(colored):
        return None, False
    # 顶边上大约 8% 的一条算贴着。膨胀只用来跨过这一条里的小缝，不伸出带子。
    touch = max(2, int(round(0.08 * cap_px)))
    rows = np.arange(band.shape[0], dtype=np.int32)[:, None]
    contact = band & (rows >= (tops - touch))
    radius = max(1, touch // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    grown = (cv2.dilate(colored.astype(np.uint8), kernel) > 0) & band
    _count, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    hit = np.unique(labels[contact])
    hit = hit[hit > 0]
    if debug is not None:
        debug["contact"] = contact[y0:y1, x0:x1].copy()
        debug["status"] = "红蓝未连接白盒顶边"
    if hit.size == 0:
        return None, False
    keep = (cell_mask > 0) & band if linked_cap else np.isin(labels, hit)
    # 颜色至少覆盖候选带的 3%，不能由一排噪点决定阵营。
    need = max(8, int(0.03 * np.count_nonzero(band)))
    for colored_part in (red, blue):
        total = int(np.count_nonzero(colored_part & keep))
        if total < need:
            colored_part[:] = False
    if debug is not None:
        debug.update(need=need, red_pixels=int((red & keep).sum()),
                     blue_pixels=int((blue & keep).sum()),
                     kept=(keep & (red | blue))[y0:y1, x0:x1].copy(),
                     status="有效面积不足或先后不明确")
    red_acc = np.cumsum(np.count_nonzero(red & keep, axis=1))
    blue_acc = np.cumsum(np.count_nonzero(blue & keep, axis=1))
    both = np.flatnonzero((red_acc >= need) & (blue_acc >= need))
    red_only = np.flatnonzero((red_acc >= need) & (blue_acc < need))
    blue_only = np.flatnonzero((blue_acc >= need) & (red_acc < need))
    found: list[tuple[int, str]] = []
    if both.size:
        found.append((int(both[0]), ""))
    if red_only.size:
        found.append((int(red_only[0]), "红"))
    if blue_only.size:
        found.append((int(blue_only[0]), "蓝"))
    if not found:
        return None, True
    _row, name = min(found)
    if debug is not None:
        debug["decision_row"] = _row
        debug["result"] = name or None
        debug["status"] = "已判定" if name else "同一行达阈值"
    if not name:
        return None, True
    return name, False


def _metric(pixels: float, z_cam: float, focal: float) -> float:
    return float(pixels) * z_cam / focal


def _z_at_cell(
    cell: Cell, u: float, v: float, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0
) -> float | None:
    # 这一像素的射线交到格子中心所在的横线上，读离地高度。距离用车位。
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    direction = ((u - intr.cx) / intr.fx) * right + ((v - intr.cy) / intr.fy) * down + ahead
    if abs(float(direction[1])) < 1e-6:
        return None
    t = (cell.y - float(origin[1])) / float(direction[1])
    if t <= 0.2:
        return None
    return float(origin[2] + t * float(direction[2]))


def _depth_in_range(z_cam: float) -> bool:
    return _DEPTH_NEAR_M <= z_cam <= _DEPTH_FAR_M


def _enter_depth(
    origin: np.ndarray,
    direction: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> np.ndarray:
    # 每条射线进入台心柱体时的相机深度。进不去的是 nan。
    count = len(direction)
    enter = np.full(count, -np.inf)
    leave = np.full(count, np.inf)
    valid = np.ones(count, dtype=bool)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        comp = direction[:, axis]
        parallel = np.abs(comp) < 1e-9
        valid &= ~(parallel & ((float(origin[axis]) < lo) | (float(origin[axis]) > hi)))
        inv = np.zeros(count, dtype=np.float64)
        moving = ~parallel
        inv[moving] = 1.0 / comp[moving]
        near = (lo - float(origin[axis])) * inv
        far = (hi - float(origin[axis])) * inv
        swap = near > far
        near_s = np.where(swap, far, near)
        far_s = np.where(swap, near, far)
        near_s = np.where(parallel, -np.inf, near_s)
        far_s = np.where(parallel, np.inf, far_s)
        enter = np.maximum(enter, near_s)
        leave = np.minimum(leave, far_s)
        valid &= enter <= leave + 1e-6
    valid &= leave >= 0.0
    return np.where(valid, np.maximum(enter, 0.0), np.nan)


def _tower_clutter(
    cell: Cell,
    depth: np.ndarray | None,
    mask: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float = 0.0,
) -> tuple[bool, bool]:
    # 第一项：搜索框里有东西比近面更近，挡住了这一格的塔。
    # 第二项：台心柱体里、台面 8 cm 以上有东西，不能把这一格写成空。
    if depth is None or depth.shape[:2] != mask.shape:
        return False, False
    ys, xs = np.nonzero(mask)
    if len(xs) < _DEPTH_MIN_POINTS:
        return False, False
    values = depth[ys, xs].astype(np.float64)
    good = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return False, False
    examined = float(len(xs))
    xs = xs[good]
    ys = ys[good]
    values = values[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    px = (xs.astype(np.float64) - intr.cx) / intr.fx
    py = (ys.astype(np.float64) - intr.cy) / intr.fy
    direction = px[:, None] * right + py[:, None] * down + ahead
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    entered = _enter_depth(origin, direction, x0, x1, y0, y1)
    front = np.isfinite(entered) & (values <= entered - _FRONT_M)
    world = origin + values[:, None] * direction
    risen = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
        & (world[:, 2] >= cell.base_z + MIN_VISIBLE_M)
        & (world[:, 2] <= cell.cap_z + _TOP_MARGIN_M)
    )
    holds = int(np.count_nonzero(risen)) >= _DEPTH_MIN_POINTS
    front_n = int(np.count_nonzero(front))
    covers = front_n / examined >= _OCCLUDE_FRAC
    if not covers and front_n >= _DEPTH_MIN_POINTS:
        low = project(cell.center(cell.one_z), pose, offset, pitch, intr, roll)
        high = project(cell.center(cell.two_z), pose, offset, pitch, intr, roll)
        if low is not None and high is not None:
            row0, row1 = sorted((float(low[1]), float(high[1])))
            span = row1 - row0
            band = front & (ys.astype(np.float64) >= row0) & (ys.astype(np.float64) <= row1)
            if span >= 4.0 and int(np.count_nonzero(band)) >= _DEPTH_MIN_POINTS:
                covers = float(ys[band].max() - ys[band].min()) >= 0.6 * span
    return covers, holds


def _depth_place(
    cell: Cell,
    depth: np.ndarray | None,
    ys: np.ndarray,
    xs: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    z_cam: float,
    roll: float = 0.0,
) -> tuple[str, float]:
    # 深度只把像素换成米。反投影落在这一格里才采用，不再拿和格子中心相差的成数否决。
    # 读不到就用格子的距离。点够了却不在这一格里，是挡在画面上的箱子，调用处丢掉。
    # 远处超过大约 3 m 常常没有深度，不因此把这块丢掉。
    if depth is None or len(xs) == 0:
        return "none", z_cam
    values = depth[ys, xs].astype(np.float64)
    good = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return "none", z_cam
    xs = xs[good]
    ys = ys[good]
    values = values[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    cam_x = (xs.astype(np.float64) - intr.cx) * values / intr.fx
    cam_y = (ys.astype(np.float64) - intr.cy) * values / intr.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + values[:, None] * ahead
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    inside = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
    )
    if int(inside.sum()) < _DEPTH_MIN_POINTS:
        return "out", float(np.median(values))
    return "in", float(np.median(values[inside]))


def _white_cloud(
    color: np.ndarray,
    depth: np.ndarray | None,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float = 0.0,
) -> dict | None:
    # 多保留3米后30厘米的白点，常规判定仍在3米内；仅在边界截断时尝试补全。
    if depth is None or depth.shape[:2] != color.shape[:2]:
        return None
    ys, xs = np.nonzero(_white_mask(color))
    if len(xs) < _DEPTH_MIN_POINTS:
        return None
    measured = depth[ys, xs].astype(np.float64)
    good = np.isfinite(measured) & (measured >= _DEPTH_NEAR_M) & (measured <= _DEPTH_FAR_M + _DEPTH_BOUNDARY_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return None
    xs = xs[good]
    ys = ys[good]
    measured = measured[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    cam_x = (xs.astype(np.float64) - intr.cx) * measured / intr.fx
    cam_y = (ys.astype(np.float64) - intr.cy) * measured / intr.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + measured[:, None] * ahead
    return {"xs": xs, "ys": ys, "z": measured, "world": world}


def _silhouette_top(
    cell: Cell,
    labels: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float,
) -> float | None:
    # 顶面经常没有深度，深度云的最高点就停在正面中间。
    # 和这些深度点连在同一块白上的顶边还在，用车位把顶边换成离地高度。
    if labels is None or len(xs) == 0:
        return None
    ids, counts = np.unique(labels[ys, xs], return_counts=True)
    best = None
    for index, count in zip(ids, counts):
        if int(index) <= 0 or int(count) < _DEPTH_MIN_POINTS:
            continue
        comp_y, comp_x = np.nonzero(labels == int(index))
        top_row = int(comp_y.min())
        if top_row <= 1 or top_row >= intr.height - 2:
            continue
        on_top = comp_y == top_row
        height = _z_at_cell(
            cell,
            float(np.median(comp_x[on_top])),
            float(top_row),
            pose,
            offset,
            pitch,
            intr,
            roll,
        )
        if height is None:
            continue
        if best is None or height > best:
            best = height
    return best



def _cap_from_depth(cell, world, xs, ys, distance, color, mask, intr, diagnostic=None):
    """整塔高点必须同时有白盒支撑和相接的红蓝顶块证据。"""
    support = np.abs(world[:, 2] - cell.two_z) <= 0.08
    if int(support.sum()) < 10:
        return None, "疑似带顶块：缺少两层白盒顶附近的有效白色深度"

    support_xy = world[support, :2]
    span = (
        np.percentile(support_xy, 95, axis=0)
        - np.percentile(support_xy, 5, axis=0)
    )
    if float(np.linalg.norm(span)) < 0.12:
        return None, "疑似带顶块：两层白盒支撑证据过窄"

    # 只用白盒高度范围构造顶边，排除整塔顶部的高点。
    body = world[:, 2] <= cell.two_z + 0.08
    white_body = np.zeros(mask.shape, np.uint8)
    white_body[ys[body], xs[body]] = 255

    z_cam = float(np.median(distance[support]))
    cap_px = max(
        4, int(round(0.20 * intr.fy / max(z_cam, 0.3)))
    )
    band, tops = _band_above(white_body, mask, cap_px)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = (
        (cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1])) > 0)
        | (cv2.inRange(
            hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1])
        ) > 0)
    )
    blue = cv2.inRange(
        hsv, np.array(BLUE[0]), np.array(BLUE[1])
    ) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ((red | blue) & band).astype(np.uint8),
        connectivity=8,
    )
    rows = np.arange(mask.shape[0])[:, None]
    contact = band & (
        rows >= tops - max(3, int(0.12 * cap_px))
    )

    accepted = np.zeros(mask.shape, np.uint8)
    candidates = 0

    for index in range(1, count):
        x, y, width, height, pixels = stats[index]
        part = labels == index
        width_m = _metric(float(width), z_cam, intr.fx)
        height_m = _metric(float(height), z_cam, intr.fy)

        if (
            pixels >= max(12, int(0.03 * cap_px**2))
            and 0.08 <= width_m <= 0.34
            and 0.04 <= height_m <= 0.28
            and np.any(part & contact)
        ):
            candidates += 1
            accepted[part] = 255

    if candidates != 1:
        return None, (
            "疑似带顶块：没有唯一且尺寸合理、与白盒相接的红蓝块"
        )

    faction, unclear = _faction(
        color, white_body, accepted, z_cam, intr, diagnostic
    )
    if unclear or faction is None:
        return None, "已见两层白盒和疑似顶块：红蓝阵营仍不明确"

    return faction, None


def _depth_on_cell(
    cell: Cell,
    cloud: dict | None,
    mask: np.ndarray,
    intr: Intrinsics,
    color: np.ndarray,
    blocked: bool,
    labels: np.ndarray | None = None,
    pose: CarPose | None = None,
    offset: np.ndarray | None = None,
    pitch: float = 0.0,
    roll: float = 0.0,
    diagnostic: dict | None = None,
) -> tuple[str | None, np.ndarray | None, bool, bool]:
    # 只留下落在这一格 50 cm 里的白点。最高那一截的离地高度对上某一档，才写层数。
    # 最高一截若只剩画面边上的点，顶被切掉了，不写。
    # 最后一项为真：看见塔顶红蓝，但分不出哪一色在上面。
    def fail(reason, piece=None, outside=False, veto=False):
        if diagnostic is not None:
            diagnostic["reason"] = reason
        return None, piece, outside, veto

    if cloud is None:
        return fail("深度不足：没有有效白点云")
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    world = cloud["world"]
    chosen = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
        & (world[:, 2] >= cell.base_z)
        & (world[:, 2] <= cell.cap_z + _TOP_MARGIN_M)
    )
    spatial = chosen.copy()
    chosen &= cloud["z"] <= _DEPTH_FAR_M
    if int(chosen.sum()) < _DEPTH_MIN_POINTS:
        return fail("深度不足：格内有效白点少于10个")
    xs = cloud["xs"][chosen]
    ys = cloud["ys"][chosen]
    distance = cloud["z"][chosen]
    world = world[chosen]
    risen = world[:, 2] >= cell.base_z + MIN_VISIBLE_M
    body = world[risen]
    # 深度不能绕过基本尺寸检查；用水平主轴跨度拒绝细杆、窄板。
    if len(body) < 30:
        return fail("深度不足：盒身有效点少于30个")
    def width_of(points):
        xy = points[:, :2] - np.median(points[:, :2], axis=0)
        _u, _s, axes = np.linalg.svd(xy, full_matrices=False)
        spans = np.percentile(xy @ axes.T, 95, axis=0) - np.percentile(xy @ axes.T, 5, axis=0)
        return float(spans.max())

    width = width_of(body)
    at_boundary = bool(np.any(distance >= _DEPTH_FAR_M - _DEPTH_EDGE_M))
    if width < WIDTH_MIN_M and at_boundary:
        # 只有原来已有格内近3米的支撑点，才考虑同一柱体里3–3.3米的点。
        # 远处整片墙、没有近侧支撑的点云不能靠这个入口变成塔。
        extended = spatial & (cloud["z"] <= _DEPTH_FAR_M + _DEPTH_BOUNDARY_M)
        extra = extended & (cloud["z"] > _DEPTH_FAR_M)
        if int(extra.sum()) >= _DEPTH_MIN_POINTS:
            extended_world = cloud["world"][extended]
            extended_risen = extended_world[:, 2] >= cell.base_z + MIN_VISIBLE_M
            extended_body = extended_world[extended_risen]
            recovered_width = width_of(extended_body)
            if WIDTH_MIN_M <= recovered_width <= WIDTH_MAX_M:
                world = extended_world
                xs, ys = cloud["xs"][extended], cloud["ys"][extended]
                distance = cloud["z"][extended]
                risen = extended_risen
                if diagnostic is not None:
                    diagnostic["depth_boundary_recovery"] = {
                        "original_width_cm": width * 100,
                        "recovered_width_cm": recovered_width * 100,
                        "extra_points": int(extra.sum()),
                    }
                width = recovered_width
    if diagnostic is not None:
        diagnostic["depth_width_cm"] = width * 100
    if not (WIDTH_MIN_M <= width <= WIDTH_MAX_M):
        partial = width < WIDTH_MIN_M and at_boundary
        if diagnostic is not None and partial:
            diagnostic["depth_partial_at_boundary"] = True
        return fail(
            f"尺寸不符：深度水平跨度{width*100:.0f}厘米，要求15至65厘米"
            + ("；接近3米边界，需完整彩色轮廓核验" if partial else ""),
            veto=not partial,
        )
    outside = int(np.count_nonzero((mask[ys, xs] == 0) & risen)) >= _DEPTH_MIN_POINTS
    step = _DEPTH_SLICE_M
    bins = np.floor((world[:, 2] - cell.base_z) / step).astype(int)
    counts = np.bincount(np.clip(bins, 0, None))
    filled = np.flatnonzero(counts >= _DEPTH_MIN_POINTS)
    if len(filled) == 0:
        return fail("深度不足：没有连续的高度切片", outside=outside)
    top = bins == int(filled[-1])
    # 最高一截贴着画面上沿或下沿，说明顶已经出画，量到的只是正面中间。
    cut = (ys <= 1) | (ys >= intr.height - 2)
    if bool(np.any(top & cut)):
        return fail("顶边出画：深度最高一段贴近画面边缘", outside=outside)
    top_z = float(np.median(world[top, 2]))
    if diagnostic is not None:
        diagnostic["top_pixels"] = np.column_stack(
            (xs[top], ys[top])
        )[::max(1, int(top.sum())//80)].tolist()
        diagnostic["depth_height_cm"] = top_z * 100
        diagnostic["height_source"] = "深度最高切片"
    if pose is not None and offset is not None:
        seen = _silhouette_top(cell, labels, xs, ys, pose, offset, pitch, intr, roll)
        # 彩色顶边对得上两层、深度却停在下面，用顶边。不要把一面墙的高点当成两层。
        if (
            seen is not None
            and seen > top_z
            and heights_match_level(seen, cell.two_z)
            and not heights_match_level(top_z, cell.two_z)
        ):
            top_z = seen
            if diagnostic is not None:
                diagnostic["height_source"] = "彩色顶边补高"
    one_ok = (not blocked) and heights_match_level(top_z, cell.one_z)
    two_ok = heights_match_level(top_z, cell.two_z)
    # 中间高度没有足够证据，保持未知。
    piece_mask = np.zeros(mask.shape, dtype=np.uint8)
    piece_mask[ys, xs] = 255
    piece = _outline(piece_mask)
    # 整塔高点与白盒层高分开判定。
    if top_z > cell.two_z + 0.08:
        if abs(top_z - cell.cap_z) > 0.06:
            return fail(
                f"整塔高度不符：{top_z*100:.1f}厘米，"
                f"带顶块允许{(cell.cap_z-.06)*100:.1f}"
                f"至{(cell.cap_z+.06)*100:.1f}厘米",
                piece, outside, True,
            )

        faction, cap_reason = _cap_from_depth(
            cell, world, xs, ys, distance, color, mask, intr, diagnostic
        )
        if faction is None:
            return fail(cap_reason, piece, outside, True)

        if diagnostic is not None:
            diagnostic["height_source"] = (
                "整塔高度＋两层白盒支撑＋红蓝顶块"
            )
            diagnostic["cap_verified"] = True

        return faction, piece, outside, False

    if not one_ok and not two_ok:
        return fail(
            f"层高不符：采用高度{top_z*100:.1f}厘米；一层允许{cell.one_z*80:.1f}至{cell.one_z*120:.1f}，两层允许{cell.two_z*80:.1f}至{cell.two_z*120:.1f}厘米；来源{diagnostic.get('height_source', '深度') if diagnostic is not None else '深度'}",
            piece, outside, True,
        )
    if one_ok and two_ok:
        use_two = abs(top_z - cell.two_z) <= abs(top_z - cell.one_z)
    else:
        use_two = two_ok
    if not use_two:
        return "1", piece, outside, False
    faction, unclear = _faction(color, piece_mask, mask, float(np.median(distance)), intr, diagnostic)
    if unclear:
        return fail("阵营不清：塔顶颜色不足或上下关系不明确", piece, outside, True)
    return (faction if faction is not None else "2"), piece, outside, False


def _outline(mask: np.ndarray) -> np.ndarray | None:
    if mask.size == 0 or int(mask.max()) == 0:
        return None
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    return max(found, key=cv2.contourArea)


def _max_edge_deviation(contour: np.ndarray, approx: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) > 400:
        points = points[:: max(1, len(points) // 400)]
    verts = approx.reshape(-1, 2).astype(np.float64)
    nearest = np.full(len(points), np.inf)
    count = len(verts)
    for index in range(count):
        start = verts[index]
        edge = verts[(index + 1) % count] - start
        length2 = float(edge @ edge)
        if length2 < 1e-6:
            dist = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
        else:
            along = np.clip((points - start) @ edge / length2, 0.0, 1.0)
            proj = start + along[:, None] * edge
            dist = np.hypot(points[:, 0] - proj[:, 0], points[:, 1] - proj[:, 1])
        nearest = np.minimum(nearest, dist)
    return float(nearest.max()) if len(nearest) else 0.0


def _close_small(component: np.ndarray) -> np.ndarray:
    # 补上短边 8% 以内的孔。线缆和标签留下的小洞不算空心，大的空心框仍然留不住。
    mask = component.astype(np.uint8) * 255
    _x, _y, width, height = cv2.boundingRect(mask)
    short = float(min(width, height))
    radius = max(1, int(round(EDGE_SIMPLIFY * short / 2.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0


def _block_shaped(component: np.ndarray) -> bool:
    closed = _close_small(component)
    contour = _outline(closed.astype(np.uint8) * 255)
    if contour is None or len(contour) < CORNER_MIN:
        return False
    area = float(cv2.contourArea(contour))
    if area < 4.0:
        return False
    if float(np.count_nonzero(closed)) < FILL_MIN * area:
        return False
    _x, _y, width, height = cv2.boundingRect(contour)
    short = float(min(width, height))
    if short < 2.0:
        return False
    approx = cv2.approxPolyDP(contour, max(2.0, EDGE_SIMPLIFY * short), True)
    if len(approx) < CORNER_MIN or len(approx) > CORNER_MAX:
        return False
    # 圆收成折线后仍接近圆。像素台阶会把原轮廓的圆度拉低，所以用折线来比。
    # 六边和七边是看见顶面的白盒。远处多出一个角仍留下，八边才比圆度。
    poly_peri = float(cv2.arcLength(approx, True))
    poly_area = float(cv2.contourArea(approx))
    circular = poly_peri >= 1e-6 and 4.0 * np.pi * poly_area / (poly_peri * poly_peri) > _CIRCLE_LIMIT
    if poly_peri < 1e-6 or (len(approx) > 7 and circular):
        return False
    if poly_area < 0.80 * area or area < 0.80 * poly_area:
        return False
    return _max_edge_deviation(contour, approx) <= max(2.5, EDGE_SIMPLIFY * short)


def _white_pieces(white: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, float, float]]]:
    # 整幅白掩膜上找块。框只在后面用来看质心落进哪一格。
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
    pieces = []
    for index in range(1, count):
        x, y, w, h, _pixels = [int(stats[index, k]) for k in range(5)]
        if w < 2 or h < 2:
            continue
        pieces.append((index, x, y, w, h, float(centroids[index, 0]), float(centroids[index, 1])))
    return labels, pieces


def _centroid_inside(mask: np.ndarray, cx: float, cy: float) -> bool:
    ix, iy = int(round(cx)), int(round(cy))
    height, width = mask.shape
    if not (0 <= ix < width and 0 <= iy < height):
        return False
    return bool(mask[iy, ix])


def _color_reading(
    cell: Cell,
    accepted: list[tuple],
    color: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    blocked: bool,
    one_h: float,
    two_h: float,
    diagnostic: dict | None = None,
) -> tuple[str, np.ndarray | None, float] | None:
    def fail(text):
        if diagnostic is not None:
            diagnostic["reason"] = text
        return None

    # 一格里多块错开的白合在一起，取从最低底到最高顶的整段。单块仍量它自己。
    union = np.zeros(mask.shape, dtype=np.uint8)
    distances = []
    pixel_tops = []
    pixel_bots = []
    world_tops = []
    top_y = None
    bot_y = None
    for _x, y, _w, h, distance, component, top_z in accepted:
        union[component] = 255
        distances.append(distance)
        if top_z is None:
            pixel_tops.append(y)
            pixel_bots.append(y + h)
        else:
            world_tops.append(top_z)
        top_y = y if top_y is None else min(top_y, y)
        bot_y = y + h if bot_y is None else max(bot_y, y + h)
    piece = _outline(union)
    distance = float(np.median(distances))
    if len(accepted) == 1 and accepted[0][6] is not None:
        height_m = float(accepted[0][6]) - cell.base_z
    elif pixel_tops and not world_tops:
        height_m = _metric(max(pixel_bots) - min(pixel_tops), distance, intr.fy)
    elif world_tops and not pixel_tops:
        height_m = max(world_tops) - cell.base_z
    else:
        span = _metric(max(pixel_bots) - min(pixel_tops), distance, intr.fy) if pixel_tops else 0.0
        lifted = (max(world_tops) - cell.base_z) if world_tops else 0.0
        height_m = max(span, lifted)
    one_ok = (not blocked) and heights_match(height_m, one_h)
    two_ok = heights_match(height_m, two_h)
    if not one_ok and not two_ok:
        return fail(
            f"彩色层高不符：可见高{height_m*100:.1f}厘米；"
            f"一层允许{one_h*50:.1f}至{one_h*150:.1f}，"
            f"两层允许{two_h*50:.1f}至{two_h*150:.1f}厘米"
        )
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return ("1", piece, distance)
    faction, unclear = _faction(color, union, mask, distance, intr, diagnostic)
    if unclear:
        return fail("阵营不清：颜色有效面积不足或上下关系不明确")
    return (faction if faction is not None else "2", piece, distance)


def _far_cap_reading(cell, color, white, depth, mask, pose, offset, pitch,
                     intr, z_cam, roll=0.0, diagnostic=None):
    """远处白盒粘墙时，用塔顶定位，再用格内深度分离白盒；不量整片白墙。"""
    info = {}
    if diagnostic is not None:
        diagnostic["far_support"] = info

    def fail(reason):
        info["reason"] = reason
        return None

    if depth is None or depth.shape != white.shape:
        return fail("远处粘连：缺少可核验白盒归属的深度")
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    colored = ((cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1]))
                | cv2.inRange(hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1]))
                | cv2.inRange(hsv, np.array(BLUE[0]), np.array(BLUE[1]))) > 0)
    # 只跨过红蓝之间很小的黑缝，不能把相隔的彩色杂物拼成顶块。
    radius = max(1, min(3, int(round(.015 * intr.fy / z_cam))))
    closed = cv2.morphologyEx(colored.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((2*radius+1, 2*radius+1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    candidates = []
    for index in range(1, count):
        x, y, w, h, pixels = stats[index]
        if pixels < 12 or x <= 1 or y <= 1 or x+w >= intr.width-1 or y+h >= intr.height-1:
            continue
        if not (.08 <= _metric(w, z_cam, intr.fx) <= .34
                and .14 <= _metric(h, z_cam, intr.fy) <= .28
                and pixels >= .45*w*h):
            continue
        part = labels == index
        if np.count_nonzero(part & (mask > 0)) < .9*pixels:
            continue
        # 顶块底边必须位于两层白盒顶，顶部也要符合整塔高度。
        py, px = np.nonzero(part)
        low = np.array([py[px == col].max()+1 for col in range(x,x+w)])
        high = np.array([py[px == col].min() for col in range(x,x+w)])
        bottom_z = [_z_at_cell(cell, col, row, pose, offset, pitch, intr, roll)
                    for col, row in zip(range(x,x+w),low)]
        top_z = [_z_at_cell(cell, col, row, pose, offset, pitch, intr, roll)
                 for col, row in zip(range(x,x+w),high)]
        if any(v is None for v in bottom_z+top_z):
            continue
        if abs(float(np.median(bottom_z))-cell.two_z) > .12:
            continue
        if abs(float(np.median(top_z))-cell.cap_z) > .15:
            continue
        # 白盒必须直接接在顶块下面；这个顶边由彩色块定位，不由整片白色的最高点定位。
        anchor = np.zeros(white.shape, np.uint8)
        contacts = 0
        for col, row in zip(range(x,x+w),low):
            end = min(intr.height, row+max(3, radius*2+1))
            hit = np.flatnonzero(white[row:end,col] > 0)
            if hit.size:
                anchor[row+int(hit[0]),col] = 255
                contacts += 1
        if contacts < .5*w:
            continue
        candidates.append((part, anchor, x, y, int(w), int(h)))
    info["cap_candidates"] = len(candidates)
    if len(candidates) != 1:
        return fail("远处粘连：没有唯一且位置/尺寸正确、下接白盒的塔顶")
    cap, anchor, x, y, w, h = candidates[0]

    # 深度可以超过3米，但须反投影落在这座塔的柱体内，白墙和前景不能当支撑。
    good = ((white > 0) & (mask > 0) & np.isfinite(depth)
            & (depth >= _DEPTH_NEAR_M))
    ys, xs = np.nonzero(good)
    values = depth[ys,xs].astype(np.float64)
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    x0,x1,y0,y1 = cell.search_bounds(MARGIN_M)
    own = ((world[:,0] >= x0) & (world[:,0] <= x1)
           & (world[:,1] >= y0) & (world[:,1] <= y1)
           & (world[:,2] >= cell.base_z+MIN_VISIBLE_M)
           & (world[:,2] <= cell.two_z+.08)
           & (xs >= x+w/2-.325*intr.fx/z_cam)
           & (xs <= x+w/2+.325*intr.fx/z_cam)
           & (ys >= y+h-3))
    support = world[own]
    info["support_points"] = len(support)
    if len(support) < 30:
        return fail("远处粘连：塔顶下方格内白盒深度不足30点")
    xy = support[:,:2]-np.median(support[:,:2],axis=0)
    _u,_s,axes = np.linalg.svd(xy,full_matrices=False)
    spans = np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    width_m = float(spans.max())
    low_z, high_z = np.percentile(support[:,2],[10,90])
    info.update(width_cm=width_m*100, height_span_cm=float(high_z-low_z)*100,
                top_height_cm=float(high_z)*100)
    if not (WIDTH_MIN_M <= width_m <= WIDTH_MAX_M):
        return fail("远处粘连：分离后的白盒宽度不符")
    if high_z-low_z < .20 or high_z < cell.two_z-.15:
        return fail("远处粘连：缺少足够的白盒高度支撑")
    faction, unclear = _faction(color, anchor, cap.astype(np.uint8)*255,
                                z_cam, intr, diagnostic, linked_cap=True)
    if unclear or faction is None:
        return fail("远处粘连：塔顶阵营不明确")
    piece = np.zeros(mask.shape,np.uint8)
    piece[ys[own],xs[own]] = 255
    info["reason"] = "塔顶定位＋格内深度分离白盒"
    info["support_pixels"] = np.column_stack((xs[own],ys[own]))[::max(1,int(own.sum())//300)].tolist()
    return faction, _outline(piece)


def _upper_space_clear(cell, top_z, pose, offset, pitch, intr, depth, roll=0.0):
    """多条射线穿过预期上层/顶块区域后才碰到背景，才能证明上方为空。"""
    total = clear = 0
    for level in np.arange(top_z+.08, cell.cap_z+.081, .06):
        for dx in (-.10, 0., .10):
            point = np.array([cell.x+dx, cell.y, level])
            pixel = project(point, pose, offset, pitch, intr, roll)
            total += 1
            if pixel is None:
                continue
            u,v = np.round(pixel).astype(int)
            if not (1 <= u < intr.width-1 and 1 <= v < intr.height-1):
                continue
            patch = depth[v-1:v+2,u-1:u+2]
            valid = np.isfinite(patch) & (patch >= _DEPTH_NEAR_M)
            expected = float(to_camera(point,pose,offset,pitch,roll)[2])
            # 台心纵深半宽30cm，另留10cm，须看见柱体后面的背景。
            if valid.sum() >= 7 and np.count_nonzero(valid & (patch > expected+.40)) >= 7:
                clear += 1
    return total >= 6 and clear / total >= .80


def _far_white_reading(cell, color, white, depth, mask, pose, offset, pitch,
                       intr, roll=0., diagnostic=None):
    """没有顶块锚点时，独立用格内白色深度及上方空域判断一/两层。"""
    info = {}
    if diagnostic is not None:
        diagnostic["far_white"] = info
    def fail(reason):
        info["reason"] = reason
        return None
    if depth is None or depth.shape != white.shape:
        return fail("白盒与背景未分离：缺少深度，需要侧向视角")
    ys,xs = np.nonzero((white > 0) & (mask > 0) & np.isfinite(depth)
                       & (depth >= _DEPTH_NEAR_M))
    if len(xs) < 30:
        return fail("白盒深度不足，需要侧向视角")
    values = depth[ys,xs].astype(float)
    origin,right,down,ahead = camera_frame(pose,offset,pitch,roll)
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    x0,x1,y0,y1 = cell.search_bounds(MARGIN_M)
    own = ((world[:,0]>=x0)&(world[:,0]<=x1)&(world[:,1]>=y0)&(world[:,1]<=y1)
           &(world[:,2]>=cell.base_z+MIN_VISIBLE_M)&(world[:,2]<=cell.cap_z+.10))
    if own.sum() < 30:
        return fail("格内白盒支撑不足，需要侧向视角")
    # 深度分离后才做连通域，白墙不会与白盒粘成整片；小孔只用于分组，不补造深度。
    points = np.zeros(white.shape,np.uint8)
    points[ys[own],xs[own]] = 1
    grouped = cv2.morphologyEx(points,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(grouped,8)
    if count < 2:
        return fail("没有完整白盒支撑，需要侧向视角")
    component = int(1+np.argmax(stats[1:,4]))
    own &= labels[ys,xs] == component
    support = world[own]
    info["support_points"] = len(support)
    if len(support)<30:
        return fail("白盒有效支撑不足，需要侧向视角")
    xy = support[:,:2]-np.median(support[:,:2],axis=0)
    _u,_s,axes = np.linalg.svd(xy,full_matrices=False)
    span = np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    width = float(span.max())
    low,top = np.percentile(support[:,2],[5,98])
    info.update(width_cm=width*100,top_height_cm=float(top)*100)
    if not WIDTH_MIN_M<=width<=WIDTH_MAX_M or top-low<.18:
        return fail("白盒尺寸或高度支撑不足，需要侧向视角")
    levels = [(abs(top-cell.one_z), "1", cell.one_z),
              (abs(top-cell.two_z), "2", cell.two_z)]
    error,state,level = min(levels)
    if error>.08:
        return fail("白盒顶高不确定，需要侧向视角")
    info["upper_clear"] = _upper_space_clear(cell,level,pose,offset,pitch,intr,depth,roll)
    if not info["upper_clear"]:
        return fail("未证实白盒上方为空：可能还有上层或顶块，需要侧向视角")
    piece = np.zeros(white.shape,np.uint8)
    piece[ys[own],xs[own]] = 255
    info["support_pixels"] = np.column_stack((xs[own],ys[own]))[::max(1,int(own.sum())//300)].tolist()
    info["reason"] = "格内白盒深度＋顶高＋上方空域"
    return state,_outline(piece)


def _far_reading(
    color: np.ndarray,
    white: np.ndarray,
    mask: np.ndarray,
    z_cam: float,
    intr: Intrinsics,
    blocked: bool,
    one_h: float,
    two_h: float,
    diagnostic: dict | None = None,
) -> tuple[str, np.ndarray | None] | None:
    def fail(text):
        if diagnostic is not None:
            diagnostic["reason"] = text
        return None

    # 只在超过深度范围、直边又收不上时用。先留下够宽的白，再补黑缝，量整块宽高。
    # 窄于 15 cm 的碎片不补到一起。圆和齿轮仍然不写。
    local = np.zeros(mask.shape, dtype=np.uint8)
    local[(white > 0) & (mask > 0)] = 255
    if int(local.max()) == 0:
        return fail("远处：搜索框内没有通过颜色阈值的白色")
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(local, connectivity=8)
    wide = np.zeros(mask.shape, dtype=np.uint8)
    found = False
    for index in range(1, count):
        _x, _y, w, h, pixels = [int(stats[index, k]) for k in range(5)]
        if pixels < 4 or w < 2 or h < 2:
            continue
        if not (WIDTH_MIN_M <= _metric(w, z_cam, intr.fx) <= WIDTH_MAX_M):
            continue
        wide[labels == index] = 255
        found = True
    if not found:
        return fail("远处：没有宽15至65厘米且至少2×2像素的白块")
    radius = _FAR_SEAM_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    closed = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, kernel)
    contour = _outline(closed)
    if contour is None:
        return fail("远处：补缝后没有有效轮廓")
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri < 1e-6 or 4.0 * np.pi * area / (peri * peri) > _CIRCLE_LIMIT:
        return fail("远处轮廓不符：周长无效或轮廓过圆")
    _x, _y, w, h = cv2.boundingRect(contour)
    if w < 2 or h < 2 or area < FILL_MIN * float(w * h):
        return fail(
            f"远处填充不足：{100*area/max(1,w*h):.1f}%，"
            f"要求至少75%；框{w}×{h}像素"
        )
    width_m = _metric(float(w), z_cam, intr.fx)
    if width_m < WIDTH_MIN_M or width_m > WIDTH_MAX_M:
        return fail(
            f"远处宽度不符：{width_m*100:.1f}厘米，要求15至65厘米"
        )
    height_m = _metric(float(h), z_cam, intr.fy)
    one_ok = (not blocked) and heights_match(height_m, one_h)
    two_ok = heights_match(height_m, two_h)
    piece = contour
    if not one_ok and not two_ok:
        return fail(
            f"彩色层高不符：可见高{height_m*100:.1f}厘米；"
            f"一层允许{one_h*50:.1f}至{one_h*150:.1f}，"
            f"两层允许{two_h*50:.1f}至{two_h*150:.1f}厘米"
        )
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return "1", piece
    faction, unclear = _faction(color, closed, mask, z_cam, intr, diagnostic)
    if unclear:
        return fail("阵营不清：颜色有效面积不足或上下关系不明确")
    return (faction if faction is not None else "2"), piece


def _classify_cell(
    cell: Cell,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None,
    depth: np.ndarray | None,
    cloud: dict | None,
    labels: np.ndarray | None = None,
    pieces: list | None = None,
    roll: float = 0.0,
    floor: np.ndarray | None = None,
) -> CellView:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr, roll)
    # 一层只露出不到 8 cm 时，这一档不能写成空，也不能写成一层。
    one_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.one_z)
    blocked = one_h < MIN_VISIBLE_M
    bearing = _bearing_deg(cell.center(cell.one_z), pose, offset, pitch)
    pitch_target = _pitch_target(cell, pose, offset)
    mask = _mask(cell, pose, offset, pitch, intr, roll)
    region = _outline(mask)
    reason = None
    diagnostic = {}

    def pack(conclusion, frac: float, mean_v: float, piece=None, method=None, hold=False, skip=False) -> CellView:
        # 深度已经量到、彩色又没把整块看全的格，先留在这里记满 15 帧。
        shown_reason = reason
        if conclusion is None and diagnostic.get("needs_parallax"):
            detail = diagnostic.get("far_white", {}).get("reason", "证据不足，需要侧向视角")
            shown_reason = detail + (("；" + reason) if reason else "")
        return CellView(
            conclusion,
            True if hold else in_view,
            False if hold else needs_yaw,
            False if hold else needs_pitch,
            blocked,
            bearing,
            pitch_target,
            frac,
            mean_v,
            region,
            piece,
            method,
            skip,
            shown_reason if conclusion is None else None,
            diagnostic,
            needs_parallax=conclusion is None and diagnostic.get("needs_parallax", False),
        )

    if color is None:
        reason = "等待有效画面或曝光稳定"
        return pack(None, 0.0, 255.0)
    white = _white_mask(color)
    inside = (white > 0) & (mask > 0)
    area = int(np.count_nonzero(mask))
    frac = 0.0 if area == 0 else float(np.count_nonzero(inside)) / float(area)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mean_v = 255.0 if area == 0 else float(np.mean(hsv[:, :, 2][mask > 0]))
    cam = to_camera(cell.center(cell.one_z), pose, offset, pitch, roll)
    z_cam = float(cam[2])
    color_usable = in_view and z_cam > 0.2
    clipped = False
    overwide = False
    missed_shape = False
    missed_width = False
    covered = False
    occludes, holds = (False, False)
    if depth is not None and _depth_in_range(z_cam):
        occludes, holds = _tower_clutter(cell, depth, mask, pose, offset, pitch, intr, roll)
    if occludes:
        # 箱子或杂物挡在这块塔的画面里。不写成层数，也不写成空，这一帧不计票。
        reason = "前景遮挡：近处杂物盖住建塔区域"
        return pack(None, frac, mean_v, skip=True)
    if z_cam > _DEPTH_FAR_M and color_usable:
        far_cap = _far_cap_reading(cell, color, white, depth, mask, pose, offset,
                                   pitch, intr, z_cam, roll, diagnostic)
        if far_cap is not None:
            state, piece = far_cap
            return pack(state, frac, mean_v, piece, "远处融合")
        far_white = _far_white_reading(cell,color,white,depth,mask,pose,offset,pitch,
                                       intr,roll,diagnostic)
        if far_white is not None:
            state,piece = far_white
            return pack(state,frac,mean_v,piece,"远处白盒")
        if frac >= .05:
            diagnostic["needs_parallax"] = True
    accepted: list[tuple] = []
    if z_cam > 0.2 and labels is not None:
        for index, x, y, w, h, cx, cy in pieces or []:
            if not _centroid_inside(mask, cx, cy):
                continue
            component = labels == index
            ys, xs = np.nonzero(component)
            sits, own = _depth_place(cell, depth, ys, xs, pose, offset, pitch, intr, z_cam, roll)
            if _depth_in_range(z_cam) and sits == "out":
                # 质心落在塔的框里，深度却不在台心。大小又像一块白盒时，就是挡在前面的箱子。
                width_own = _metric(w, own, intr.fx)
                if WIDTH_MIN_M <= width_own <= WIDTH_MAX_M and _block_shaped(component):
                    covered = True
                continue
            touches_side = x <= 1 or x + w >= intr.width - 1
            touches_top = y <= 1
            touches_bottom = y + h >= intr.height - 1
            if _metric(w, z_cam, intr.fx) > WIDTH_MAX_M:
                overwide = True
            # 顶或左右出画就量不准。底边出画时顶边还在，层高改由顶边来算。
            if touches_side or touches_top or overwide:
                clipped = True
                continue
            if not color_usable:
                continue
            distance = own if sits == "in" else z_cam
            width_m = _metric(w, distance, intr.fx)
            if width_m < WIDTH_MIN_M:
                # 阴影、线缆、标签切出来的段。看见了白，不能写成空。
                missed_width = True
                continue
            if width_m > WIDTH_MAX_M:
                overwide = True
                continue
            if not _block_shaped(component):
                missed_shape = True
                continue
            top_z = None
            if touches_bottom:
                top_rows = ys == int(ys.min())
                top_z = _z_at_cell(
                    cell,
                    float(np.median(xs[top_rows])),
                    float(ys.min()),
                    pose,
                    offset,
                    pitch,
                    intr,
                    roll,
                )
                if top_z is None:
                    clipped = True
                    continue
            accepted.append((x, y, w, h, distance, component, top_z))
    depth_state, depth_piece, saw_outside, cap_unclear = _depth_on_cell(
        cell, cloud, mask, intr, color, blocked, labels, pose, offset, pitch, roll,
        diagnostic=diagnostic,
    )
    reason = diagnostic.get("reason")
    if cap_unclear:
        return pack(None, frac, mean_v, depth_piece)
    if diagnostic.get("depth_partial_at_boundary"):
        # 不把不完整深度当成反证，但必须有独立通过尺寸/直边检查的彩色白盒。
        if not accepted:
            return pack(None, frac, mean_v, depth_piece)
        diagnostic["depth_fallback_reason"] = reason
    # 白点落在这一格 50 cm 里，最高一截对上一层或两层，就写层数。
    # 整块在画面里、和木板粘在一起收不成直边时，也用这段深度。
    if depth is not None and depth_state is not None:
        return pack(depth_state, frac, mean_v, depth_piece, "深度", hold=not accepted)
    if accepted:
        two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
        reading = _color_reading(cell, accepted, color, mask, intr, blocked, one_h, two_h, diagnostic)
        if reading is None:
            reason = diagnostic.get("reason", "彩色判定未通过")
            return pack(None, frac, mean_v)
        state, piece, _distance = reading
        return pack(state, frac, mean_v, piece, _color_method(mask, floor))
    if (
        z_cam > _DEPTH_FAR_M
        and color_usable
        and (missed_shape or missed_width)
        and not overwide
        and not clipped
    ):
        two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
        far = _far_reading(color, white, mask, z_cam, intr, blocked, one_h, two_h, diagnostic)
        if far is None:
            reason = diagnostic.get("reason", "远处彩色未通过")
            return pack(None, frac, mean_v)
        state, piece = far
        return pack(state, frac, mean_v, piece, "远处")
    if depth is not None and (clipped or overwide or saw_outside):
        reason = (
            "尺寸不符：白块宽度超过65厘米" if overwide else
            "白块出画：顶边或左右边被切掉" if clipped else
            "投影不符：白点落在搜索框外"
        )
        return pack(None, frac, mean_v)
    if not color_usable or blocked or missed_shape or missed_width:
        reason = (
            "不在画面里，要先转向" if needs_yaw else
            "高度不在画面里，要改俯仰" if needs_pitch else
            "被高台遮挡：一层可见高度不足8厘米" if blocked else
            "轮廓不符：白块破碎、被遮挡或直边检查未通过" if missed_shape else
            "尺寸不符：白块宽度不足15厘米" if missed_width else
            "距离过近：彩色估距不足20厘米"
        )
        return pack(None, frac, mean_v)
    if covered or holds:
        reason = "前面有东西挡住了或格内有物体：不能确认为空点"
        # 框里有一块像塔的箱子，或柱体里已经有东西。没有合格白盒也不能写成空。
        return pack(None, frac, mean_v, skip=True)
    if not _empty_evidence(cell, pose, offset, pitch, intr, depth, roll):
        reason = ((reason + "；") if reason else "") + (
            "空点未证实：支撑面未完整入画、深度不足或高度不符"
        )
        return pack(None, frac, mean_v)
    return pack("0", frac, mean_v, method=_color_method(mask, floor))


def _empty_evidence(cell, pose, offset, pitch, intr, depth, roll=0.0) -> bool:
    """在台面投影内部看到覆盖充分的空支撑面，才确认空点。"""
    if depth is None or depth.shape != (intr.height, intr.width):
        return False
    x0, x1, y0, y1 = cell.search_bounds(0.0)
    verts = [project(np.array([x, y, cell.base_z]), pose, offset, pitch, intr, roll)
             for x, y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
    if any(v is None or not _inside(v, intr) for v in verts):
        return False
    patch = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(patch, np.round(verts).astype(np.int32), 255)
    ys, xs = np.nonzero(patch)
    if len(xs) < 30:
        return False
    values = depth[ys, xs]
    valid = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if valid.mean() < 0.80:
        return False
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    xs, ys, values = xs[valid], ys[valid], values[valid]
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    return bool(np.mean(np.abs(world[:,2]-cell.base_z) <= 0.03) >= 0.95)



def _filter_edge_depth(color: np.ndarray, depth: np.ndarray | None) -> np.ndarray | None:
    """只剔除颜色边界内一像素带中、被同类邻域多数否定的深度；不填补深度。"""
    if depth is None or depth.shape != color.shape[:2]:
        return depth
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    kinds = np.zeros(depth.shape, np.uint8)
    kinds[_white_mask(color) > 0] = 1
    red = (cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1]))
           | cv2.inRange(hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1]))) > 0
    kinds[red] = 2
    kinds[cv2.inRange(hsv, np.array(BLUE[0]), np.array(BLUE[1])) > 0] = 3
    edge = np.zeros(depth.shape, bool)
    for kind in (1, 2, 3):
        part = (kinds == kind).astype(np.uint8)
        inner = cv2.erode(part, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        edge |= (part > 0) & (inner == 0)
    valid = np.isfinite(depth) & (depth >= _DEPTH_NEAR_M)
    ys, xs = np.nonzero(edge & valid)
    if not len(xs):
        return depth
    padded = np.pad(depth.astype(np.float32), 2, constant_values=np.nan)
    labels = np.pad(kinds, 2)
    neighbors = []
    for dy in range(5):
        for dx in range(5):
            if dy == 2 and dx == 2:
                continue
            values = padded[ys + dy, xs + dx]
            same = labels[ys + dy, xs + dx] == kinds[ys, xs]
            neighbors.append(np.where(same & np.isfinite(values)
                                      & (values >= _DEPTH_NEAR_M), values, np.nan))
    samples = np.stack(neighbors, axis=1)
    enough = np.sum(np.isfinite(samples), axis=1) >= 8
    if not enough.any():
        return depth
    samples = samples[enough]
    ys, xs = ys[enough], xs[enough]
    middle = np.nanmedian(samples, axis=1)
    # 至少相差8厘米，远处按距离的3%提高门槛；只拒绝明显跳变。
    tolerance = np.maximum(0.08, middle * 0.03)
    coherent = np.sum(np.abs(samples - middle[:, None]) <= tolerance[:, None], axis=1)
    majority = coherent >= 0.65 * np.sum(np.isfinite(samples), axis=1)
    reject = majority & (np.abs(depth[ys, xs] - middle) > tolerance)
    if not reject.any():
        return depth
    cleaned = depth.astype(np.float32, copy=True)
    cleaned[ys[reject], xs[reject]] = np.nan
    return cleaned


def inspect_frame(
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    roll: float = 0.0,
) -> dict[str, CellView]:
    labels = None
    pieces = None
    floor = None
    if color is not None:
        depth = _filter_edge_depth(color, depth)
        white, floor = _split_white(color)
        labels, pieces = _white_pieces(white)
    cloud = None if color is None else _white_cloud(color, depth, pose, offset, pitch, intr, roll)
    return {
        cell.name: _classify_cell(
            cell, pose, offset, pitch, intr, color, depth, cloud, labels, pieces, roll, floor
        )
        for cell in tower_cells()
    }
