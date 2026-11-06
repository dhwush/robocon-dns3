"""一帧里每一格的结论。对不上三档就保持未知。没有深度时，白块不是实心直边也保持未知。

搜索框只决定白块属于哪一格：质心落在框里，宽度和高度量整块，不再先和框相与。
近处白块顶边或左右贴到画面边，或整块宽过 65 cm，彩色这条不收。
底边被切掉、顶边还在里面时，用车位把顶边换成离地高度，仍走彩色。
0.3 m 到 3 m 里，只要有足够的白点落在这一格，层数用最高一截的离地高度。
整块在画面里、彩色轮廓收不成直边时，这段对上了层高的深度仍然写。
远处没有深度时，用整块的像素高度。彩色上对不上形状的，保持未知，不写成空。
框里的白被切碎，或只剩不够 15 cm 的一截，同样保持未知，不写成空。
三层塔的阵营看贴在白盒顶上的像素带，从上往下认。和顶边隔开的红蓝不算。分不出先后就保持未知，不写成两层。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from car_pose import CarPose, Intrinsics, bearing_deg, camera_frame, project, to_camera
from field_map import MARGIN_M, Cell, tower_cells
from sight import sees, visible_height


# 白：饱和度不超过 110，明度不低于 125。背光的白盒正面大约 131，140 收不进。
# 红的色相仍是 0–8 和 172–180，明度放到 110，罩住这张画面里大约 118 的红。
# 蓝的色相仍是 105–130，饱和度放到 110，罩住这张画面里大约 113 的蓝。
WHITE_S = 110
WHITE_V = 125
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
_DEPTH_MIN_POINTS = 10


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
    origin_z = 0.70
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


def _white_mask(color: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S) & (hsv[:, :, 2] >= WHITE_V)
    return white.astype(np.uint8) * 255


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


def _faction(color: np.ndarray, white: np.ndarray, cell_mask: np.ndarray, z_cam: float, intr: Intrinsics) -> tuple[str | None, bool]:
    # 只认贴着白盒顶边的那条像素带，从上往下数。
    # 和顶边隔开的红蓝不算塔顶。红蓝从上面一起数够，第二项为真，调用处保持未知。
    cap_px = max(4, int(round(0.20 * intr.fy / max(float(z_cam), 0.3))))
    band, tops = _band_above(white, cell_mask, cap_px)
    if not np.any(band):
        return None, False
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = (
        (cv2.inRange(hsv, np.array(RED[0], dtype=np.uint8), np.array(RED[1], dtype=np.uint8)) > 0)
        | (cv2.inRange(hsv, np.array(RED_WRAP[0], dtype=np.uint8), np.array(RED_WRAP[1], dtype=np.uint8)) > 0)
    ) & band
    blue = (cv2.inRange(hsv, np.array(BLUE[0], dtype=np.uint8), np.array(BLUE[1], dtype=np.uint8)) > 0) & band
    colored = red | blue
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
    if hit.size == 0:
        return None, False
    keep = np.isin(labels, hit)
    need = max(4, int(0.25 * cap_px))
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
        return None, False
    _row, name = min(found)
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


def _depth_in_cell(
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
) -> float:
    # 深度只把像素换成米。反投影落在这一格里才采用，不再拿和格子中心相差的成数否决。
    # 读不到，或点不在这一格，就用格子的距离。远处超过大约 3 m 常常没有深度，不因此把这块丢掉。
    if depth is None or len(xs) == 0:
        return z_cam
    values = depth[ys, xs].astype(np.float64)
    good = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return z_cam
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
        return z_cam
    return float(np.median(values[inside]))


def _white_cloud(
    color: np.ndarray,
    depth: np.ndarray | None,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float = 0.0,
) -> dict | None:
    # 白点的深度变到场地坐标。比 0.3 m 更近或远过 3 m 的没有数，不拿来补。
    if depth is None or depth.shape[:2] != color.shape[:2]:
        return None
    ys, xs = np.nonzero(_white_mask(color))
    if len(xs) < _DEPTH_MIN_POINTS:
        return None
    measured = depth[ys, xs].astype(np.float64)
    good = np.isfinite(measured) & (measured >= _DEPTH_NEAR_M) & (measured <= _DEPTH_FAR_M)
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


def _depth_on_cell(
    cell: Cell,
    cloud: dict | None,
    mask: np.ndarray,
    intr: Intrinsics,
    color: np.ndarray,
    blocked: bool,
) -> tuple[str | None, np.ndarray | None, bool, bool]:
    # 只留下落在这一格 50 cm 里的白点。最高那一截的离地高度对上某一档，才写层数。
    # 最高一截若只剩画面边上的点，顶被切掉了，不写。
    # 最后一项为真：看见塔顶红蓝，但分不出哪一色在上面。
    if cloud is None:
        return None, None, False, False
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
    if int(chosen.sum()) < _DEPTH_MIN_POINTS:
        return None, None, False, False
    xs = cloud["xs"][chosen]
    ys = cloud["ys"][chosen]
    distance = cloud["z"][chosen]
    world = world[chosen]
    risen = world[:, 2] >= cell.base_z + MIN_VISIBLE_M
    outside = int(np.count_nonzero((mask[ys, xs] == 0) & risen)) >= _DEPTH_MIN_POINTS
    step = _DEPTH_SLICE_M
    bins = np.floor((world[:, 2] - cell.base_z) / step).astype(int)
    counts = np.bincount(np.clip(bins, 0, None))
    filled = np.flatnonzero(counts >= _DEPTH_MIN_POINTS)
    if len(filled) == 0:
        return None, None, outside, False
    top = bins == int(filled[-1])
    # 最高一截贴着画面上沿或下沿，说明顶已经出画，量到的只是正面中间。
    cut = (ys <= 1) | (ys >= intr.height - 2)
    if bool(np.any(top & cut)):
        return None, None, outside, False
    top_z = float(np.median(world[top, 2]))
    one_ok = (not blocked) and heights_match_level(top_z, cell.one_z)
    two_ok = heights_match_level(top_z, cell.two_z)
    piece_mask = np.zeros(mask.shape, dtype=np.uint8)
    piece_mask[ys, xs] = 255
    piece = _outline(piece_mask)
    if not one_ok and not two_ok:
        return None, piece, outside, False
    if one_ok and two_ok:
        use_two = abs(top_z - cell.two_z) <= abs(top_z - cell.one_z)
    else:
        use_two = two_ok
    if not use_two:
        return "1", piece, outside, False
    faction, unclear = _faction(color, piece_mask, mask, float(np.median(distance)), intr)
    if unclear:
        return None, piece, outside, True
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
) -> tuple[str, np.ndarray | None, float] | None:
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
        return None
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return ("1", piece, distance)
    faction, unclear = _faction(color, union, mask, distance, intr)
    if unclear:
        return None
    return (faction if faction is not None else "2", piece, distance)


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
) -> CellView:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr, roll)
    # 一层只露出不到 8 cm 时，这一档不能写成空，也不能写成一层。
    one_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.one_z)
    blocked = one_h < MIN_VISIBLE_M
    bearing = _bearing_deg(cell.center(cell.one_z), pose, offset, pitch)
    pitch_target = _pitch_target(cell, pose, offset)
    mask = _mask(cell, pose, offset, pitch, intr, roll)
    region = _outline(mask)

    def pack(conclusion, frac: float, mean_v: float, piece=None, method=None, hold=False) -> CellView:
        # 深度已经量到、彩色又没把整块看全的格，先留在这里记满 15 帧。
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
        )

    if color is None:
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
    accepted: list[tuple] = []
    if z_cam > 0.2 and labels is not None:
        for index, x, y, w, h, cx, cy in pieces or []:
            if not _centroid_inside(mask, cx, cy):
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
            component = labels == index
            ys, xs = np.nonzero(component)
            distance = _depth_in_cell(cell, depth, ys, xs, pose, offset, pitch, intr, z_cam, roll)
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
        cell, cloud, mask, intr, color, blocked
    )
    if cap_unclear:
        return pack(None, frac, mean_v, depth_piece)
    # 白点落在这一格 50 cm 里，最高一截对上一层或两层，就写层数。
    # 整块在画面里、和木板粘在一起收不成直边时，也用这段深度。
    if depth is not None and depth_state is not None:
        return pack(depth_state, frac, mean_v, depth_piece, "深度", hold=not accepted)
    if accepted:
        two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
        reading = _color_reading(cell, accepted, color, mask, intr, blocked, one_h, two_h)
        if reading is None:
            return pack(None, frac, mean_v)
        state, piece, _distance = reading
        return pack(state, frac, mean_v, piece, "彩色")
    if depth is not None and (clipped or overwide or saw_outside):
        return pack(None, frac, mean_v)
    if not color_usable or blocked or missed_shape or missed_width:
        return pack(None, frac, mean_v)
    return pack("0", frac, mean_v, method="彩色")


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
    if color is not None:
        labels, pieces = _white_pieces(_white_mask(color))
    cloud = None if color is None else _white_cloud(color, depth, pose, offset, pitch, intr, roll)
    return {
        cell.name: _classify_cell(
            cell, pose, offset, pitch, intr, color, depth, cloud, labels, pieces, roll
        )
        for cell in tower_cells()
    }

