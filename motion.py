"""拓展指令的下一步。

旋转和俯仰不限次数。每一格先转到画面正中，再把一层顶和塔顶放到画面中段，
认满再换下一格。认不出就换一个俯仰或偏一个角度重看。
挡住、竖直装不下、超出深度范围，或白墙粘连缺少空间证据时，才平移。
白墙粘连时要求实际侧向基线与观察方向变化，原地转头不算换侧向视角。
定位点为右前角，车身 35 cm 见方。落点绕开车身膨胀后的障碍，
在采样候选点内按总平移路程最短排序；无安全落点时保持原位。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from car_pose import BODY_HALF_M, turned_pose, CAMERA_HEIGHT_M, CarPose, Intrinsics, bearing_deg, camera_position
from classify import MIN_VISIBLE_M, CellView, _pitch_target, _view_flags
from field_map import TOWER_NO, all_cells, ground_towers, tower_cells
from sight import visible_height


# 正中：方位和目标俯仰都进到这个范围里，才开始记这一格。
CENTER_DEG = 5.0
PITCH_TOL_DEG = 5.0
# 每个航向都测基准、抬高8度和降低8度；共21个姿态。
YAW_TOL_DEG = 3.0
SCAN_YAWS = (0.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0)
SCAN_PITCHES = (0.0, 8.0, -8.0)
GOAL_BEARING = tuple(yaw for yaw in SCAN_YAWS for _ in SCAN_PITCHES)
PITCH_OFFSET = SCAN_PITCHES * len(SCAN_YAWS)
MAX_TRIES = len(GOAL_BEARING)
# 深度相机大约 0.3 m 到 3 m。落点要落在这段里。
DEPTH_NEAR_M = 0.3
DEPTH_FAR_M = 3.0
# 一层顶和塔顶占竖直视场的比例。留出边上的空，避免贴着画面边缘认。
VERTICAL_MARGIN = 0.84
ARRIVE_M = 0.15
MOVE_ARRIVE_M = 0.02
PARALLAX_ANGLE_DEG = 12.0
PARALLAX_BASELINE_M = 0.30


@dataclass(frozen=True)
class Look:
    number: str
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class RouteStop:
    x: float
    y: float
    looks: tuple[Look, ...]
    kind: str = "兜底"


@dataclass(frozen=True)
class Action:
    kind: str
    text: str
    pitch: float | None = None
    yaw: float | None = None
    x: float | None = None
    y: float | None = None
    cell: str | None = None
    attempt: int = 0
    final_x: float | None = None
    final_y: float | None = None
    chase_to: str | None = None
    cells: tuple[str, ...] = ()


def heading_words(degrees: float) -> str:
    """场地绝对朝向，避免把目标朝向误读成增量转角。"""
    value = (degrees + 180.0) % 360.0 - 180.0
    if abs(value) < .5:
        return "正前方0°"
    return f"{'左' if value > 0 else '右'}{abs(value):.0f}°"


def format_yaw(bearing_deg_value: float) -> str:
    # 正的方位在车的右侧。度数是相对现在的车头还要转的量。
    degrees = max(1, int(round(abs(bearing_deg_value))))
    if bearing_deg_value >= 0.0:
        return f"向右转 {degrees} 度"
    return f"向左转 {degrees} 度"


def format_pitch(target_deg: float) -> str:
    degrees = int(round(target_deg))
    if degrees >= 0:
        return f"抬头到 {degrees} 度"
    return f"低头到 {abs(degrees)} 度"


def format_slide(lateral_m: float) -> str:
    meters = max(0.1, math.ceil(abs(lateral_m) * 10.0 - 1e-9) / 10.0)
    text = f"{meters:.1f}"
    if lateral_m >= 0.0:
        return f"向右平移 {text} 米"
    return f"向左平移 {text} 米"


def _wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _cells() -> dict:
    return {cell.name: cell for cell in tower_cells()}


def _slant(cell, origin: np.ndarray) -> float:
    mid = cell.center((cell.one_z + cell.cap_z) / 2.0)
    return float(np.linalg.norm(mid - origin))


def aimed_pose(pose: CarPose, offset: np.ndarray, cell) -> CarPose:
    # 平移不改航向。这里只是算出转正以后的航向，用来判断那一格能不能看全。
    faced = pose
    for _ in range(20):
        turn = bearing_deg(cell.center(cell.one_z), faced, offset)
        if abs(turn) < 0.01:
            break
        faced = turned_pose(pose, faced.yaw - math.radians(turn))
    return faced


def place_done(locked: dict[str, str | None], pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> bool:
    # 这一站转正还能看清的格子都锁定了，才离开去下一站。
    for cell in tower_cells():
        if locked.get(cell.name):
            continue
        if fits_when_aimed(cell, pose, offset, intr):
            return False
    return True


def observation_poses(cell, pose: CarPose, offset: np.ndarray, intr: Intrinsics, allow_rgb=False):
    """与原地扫描相同的姿态集合；索引对应执行器的attempt。"""
    centered = aimed_pose(pose, offset, cell)
    for index, (goal, delta_pitch) in enumerate(zip(GOAL_BEARING, PITCH_OFFSET)):
        faced = turned_pose(pose, centered.yaw + math.radians(goal))
        # 相机不在旋转中心，偏航时需重新求目标方位。
        for _ in range(10):
            error = _wrap(bearing_deg(cell.center(cell.one_z), faced, offset)-goal)
            if abs(error) < .01:
                break
            faced = turned_pose(pose, faced.yaw-math.radians(error))
        pitch = _pitch_target(cell, faced, offset) + delta_pitch
        if not _parking_clear(faced.x, faced.y, faced.yaw):
            continue
        if not _sees(cell, faced, offset, pitch, intr):
            continue
        slant = _slant(cell, camera_position(faced, offset, 0.0))
        if slant > DEPTH_FAR_M and allow_rgb:
            # 远处只承诺顶块有足够像素可试读，不把它当作已识别整塔。
            if (.20*intr.fx/slant >= 12 and .20*intr.fy/slant >= 16
                    and visible_height(faced, offset, pitch, cell, cell.two_z, cell.cap_z) >= .14):
                yield index, faced, pitch
        elif DEPTH_NEAR_M <= slant <= DEPTH_FAR_M:
            if visible_height(faced, offset, pitch, cell, cell.base_z, cell.one_z) >= MIN_VISIBLE_M:
                yield index, faced, pitch


def fits_when_aimed(cell, pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> bool:
    return next(observation_poses(cell, pose, offset, intr), None) is not None


def fits_rgb_when_aimed(cell, pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> bool:
    """当前点是否值得先尝试远处RGB；不用于证明空点或无顶块塔。"""
    return next(observation_poses(cell, pose, offset, intr, allow_rgb=True), None) is not None


def _near_visited(x: float, y: float, visited: list[tuple[float, float]]) -> bool:
    return any(math.hypot(x - vx, y - vy) < 0.25 for vx, vy in visited)


def _obstacles() -> tuple[tuple[float, float, float, float], ...]:
    # 场地不变。采样时直接用这十五块的边，不再每次重造格子。
    boxes = getattr(_obstacles, "boxes", None)
    if boxes is None:
        boxes = tuple(cell.bounds(0.0) for cell in list(all_cells()) + list(ground_towers()))
        _obstacles.boxes = boxes
    return boxes


def _ground_blocked(x: float, y: float) -> bool:
    # 原始障碍点检查；车身避障另用 _parking_clear / _path_clear。
    point_x, point_y = float(x), float(y)
    for x0, x1, y0, y1 in _obstacles():
        if x0 <= point_x <= x1 and y0 <= point_y <= y1:
            return True
    return False


# 使用外接圆保证平移和原地环顾均有空间；保守，不宣称连续空间最短。
BODY_CLEARANCE_M = math.sqrt(2.0) * BODY_HALF_M + 0.02


def _navigation_boxes(yaw: float = 0.0):
    # 把车中心的膨胀障碍转换到右前角定位坐标。
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = BODY_HALF_M * (c - s), BODY_HALF_M * (s + c)
    r = BODY_CLEARANCE_M
    return tuple((a-r+dx, b+r+dx, c0-r+dy, d+r+dy)
                 for a, b, c0, d in _obstacles())


def _parking_clear(x: float, y: float, yaw: float = 0.0) -> bool:
    return not any(a <= x <= b and c <= y <= d
                   for a, b, c, d in _navigation_boxes(yaw))


def _path_clear(x0: float, y0: float, x1: float, y1: float, yaw: float = 0.0) -> bool:
    # 精确线段/矩形相交，包含端点；不允许从碰撞状态盲目穿出。
    for a, b, c, d in _navigation_boxes(yaw):
        enter, leave = 0.0, 1.0
        for origin, delta, low, high in ((x0, x1-x0, a, b), (y0, y1-y0, c, d)):
            if abs(delta) < 1e-12:
                if origin < low or origin > high:
                    break
            else:
                lo, hi = sorted(((low-origin)/delta, (high-origin)/delta))
                enter, leave = max(enter, lo), min(leave, hi)
                if enter > leave:
                    break
        else:
            return False
    return True



def _vertical_span(cell, distance_h: float) -> float:
    high = math.atan2(cell.cap_z - CAMERA_HEIGHT_M, max(distance_h, 1e-6))
    low = math.atan2(cell.one_z - CAMERA_HEIGHT_M, max(distance_h, 1e-6))
    return abs(high - low)


def _distances(cell, intr: Intrinsics) -> list[float]:
    limit = 2.0 * math.atan((intr.height * 0.5) / intr.fy) * VERTICAL_MARGIN
    found = []
    for dist in np.arange(0.45, DEPTH_FAR_M + 1e-6, 0.15):
        if _vertical_span(cell, float(dist)) >= limit:
            continue
        mid_z = (cell.one_z + cell.cap_z) / 2.0
        slant = math.hypot(float(dist), mid_z - CAMERA_HEIGHT_M)
        if slant < DEPTH_NEAR_M or slant > DEPTH_FAR_M:
            continue
        found.append(float(dist))
    if not found:
        return []
    return sorted({found[0], found[len(found) // 2], found[-1]})


def _pose_for_camera(pose: CarPose, offset: np.ndarray, cam_x: float, cam_y: float) -> CarPose:
    current = camera_position(pose, offset, 0.0)
    return CarPose(
        pose.x + cam_x - float(current[0]),
        pose.y + cam_y - float(current[1]),
        pose.yaw,
        pose.stamp,
    )


def _move_text(pose: CarPose, target: CarPose) -> str | None:
    dx = target.x - pose.x
    dy = target.y - pose.y
    yaw_c, yaw_s = math.cos(pose.yaw), math.sin(pose.yaw)
    lateral = dx * yaw_c + dy * yaw_s
    forward = dx * (-yaw_s) + dy * yaw_c
    parts = []
    if abs(forward) >= 0.005:
        meters = max(0.01, round(abs(forward), 2))
        parts.append(f"{'向前' if forward >= 0.0 else '向后'}平移 {meters:.2f} 米")
    if abs(lateral) >= 0.005:
        meters = max(0.01, round(abs(lateral), 2))
        parts.append(f"{'向右' if lateral >= 0.0 else '向左'}平移 {meters:.2f} 米")
    if not parts:
        return None
    return ("同时" if len(parts) > 1 else "") + "，".join(parts)


def _blocks() -> tuple[tuple[float, float, float, float], ...]:
    # 十二块绿台连成一块。三块木块另外算。绕行从外角外面过去。
    cells = list(all_cells())
    left = min(cell.bounds(0.0)[0] for cell in cells)
    right = max(cell.bounds(0.0)[1] for cell in cells)
    near = min(cell.bounds(0.0)[2] for cell in cells)
    far = max(cell.bounds(0.0)[3] for cell in cells)
    blocks = [(left, right, near, far)]
    blocks.extend(cell.bounds(0.0) for cell in ground_towers())
    return tuple(blocks)


def _bypass_points(yaw: float = 0.0) -> list[tuple[float, float]]:
    gap = 0.001
    points = []
    for left, right, near, far in _navigation_boxes(yaw):
        for x, y in (
            (left - gap, near - gap),
            (right + gap, near - gap),
            (left - gap, far + gap),
            (right + gap, far + gap),
        ):
            if _parking_clear(x, y, yaw):
                points.append((x, y))
    return points


def _travel(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    extras: list[tuple[float, float]],
    yaw: float = 0.0,
) -> tuple[float, list[tuple[float, float]]] | None:
    # 直线能过就直走。过不去就从台子外角绕。返回总路程，以及从起点到终点的折线。
    if math.hypot(x1 - x0, y1 - y0) < 1e-6:
        return 0.0, [(x0, y0)]
    if _path_clear(x0, y0, x1, y1, yaw):
        return math.hypot(x1 - x0, y1 - y0), [(x0, y0), (x1, y1)]
    nodes = [(x0, y0), (x1, y1), *extras]
    count = len(nodes)
    dist = [1e9] * count
    dist[0] = 0.0
    prev = [-1] * count
    used = [False] * count
    weights: dict[tuple[int, int], float] = {}

    def link(left: int, right: int) -> float | None:
        key = (left, right) if left < right else (right, left)
        if key in weights:
            cached = weights[key]
            return None if cached < 0 else cached
        ax, ay = nodes[left]
        bx, by = nodes[right]
        if not _path_clear(ax, ay, bx, by, yaw):
            weights[key] = -1.0
            return None
        length = math.hypot(bx - ax, by - ay)
        weights[key] = length
        return length

    for _ in range(count):
        current = None
        best_dist = 1e9
        for index in range(count):
            if not used[index] and dist[index] < best_dist:
                best_dist = dist[index]
                current = index
        if current is None or current == 1:
            break
        used[current] = True
        for other in range(count):
            if used[other]:
                continue
            weight = link(current, other)
            if weight is None:
                continue
            total = dist[current] + weight
            if total < dist[other]:
                dist[other] = total
                prev[other] = current
    if dist[1] >= 1e9:
        return None
    node = 1
    chain = [1]
    while prev[node] != -1:
        node = prev[node]
        chain.append(node)
    chain.reverse()
    return dist[1], [nodes[index] for index in chain]


def _fallback(pose: CarPose, need: list[str], visited: list[tuple[float, float]]) -> Action:
    # 无安全路线时不盲目回原点，也不依赖缺失的固定路线模块。
    return Action("blocked", "没有安全可达的观察点，请检查车位或障碍；保持原位")


def _looks(pose: CarPose, offset: np.ndarray, intr: Intrinsics, names: list[str] | tuple[str, ...]) -> tuple[Look, ...]:
    # Look保留每塔一个基准姿态，供地图展开偏航/俯仰列表；
    # 任务顺序按该塔最近的可用姿态计算，避免界面重复展开全部姿态。
    cells = _cells()
    remaining = []
    for name in names:
        cell = cells[name]
        poses = tuple(observation_poses(cell, pose, offset, intr, allow_rgb=True))
        if not poses:
            continue
        centered = aimed_pose(pose, offset, cell)
        base = Look(TOWER_NO[name], math.degrees(centered.yaw),
                    _pitch_target(cell, centered, offset))
        remaining.append((base, poses))
    ordered = []
    yaw = math.degrees(pose.yaw)
    while remaining:
        task, first = min(((task, min(task[1], key=lambda p: abs(_wrap(math.degrees(p[1].yaw)-yaw))))
                           for task in remaining),
                          key=lambda pair: abs(_wrap(math.degrees(pair[1][1].yaw)-yaw)))
        remaining.remove(task)
        ordered.append(task[0])
        yaw = math.degrees(first[1].yaw)
    return tuple(ordered)


def _different_view(cell, origin, candidate):
    """转头不能产生视差；目标方位至少变化12度且侧向基线至少30厘米。"""
    before = np.array(origin[:2],dtype=float)-np.array([cell.x,cell.y])
    after = np.array(candidate[:2],dtype=float)-np.array([cell.x,cell.y])
    distance = float(np.linalg.norm(before))
    other = float(np.linalg.norm(after))
    if min(distance,other)<1e-6:
        return False
    angle = math.degrees(math.acos(float(np.clip(before@after/(distance*other),-1,1))))
    delta = np.array(candidate[:2])-np.array(origin[:2])
    lateral = abs(float(before[0]*delta[1]-before[1]*delta[0]))/distance
    return angle>=PARALLAX_ANGLE_DEG and lateral>=PARALLAX_BASELINE_M


def _search(
    pose: CarPose,
    offset: np.ndarray,
    intr: Intrinsics,
    need: list[str],
    visited: list[tuple[float, float]],
    parallax: dict | None = None,
) -> tuple[list[tuple[float, float, tuple[str, ...]]], list[tuple[float, float]]] | None:
    # 落点按总路程最短排。返回各停点新看清的格，以及从当前车位出发的折线。
    cells = _cells()
    names = list(need)
    if not names:
        return [], [(pose.x, pose.y)]
    full = (1 << len(names)) - 1
    extras = _bypass_points(pose.yaw)
    found: list[tuple[float, float, int]] = []
    for name in names:
        cell = cells[name]
        for dist in _distances(cell, intr):
            for step in range(8):
                angle = step * math.pi / 4.0
                candidate = _pose_for_camera(
                    pose,
                    offset,
                    cell.x + dist * math.cos(angle),
                    cell.y + dist * math.sin(angle),
                )
                if math.hypot(candidate.x - pose.x, candidate.y - pose.y) < ARRIVE_M:
                    continue
                if not _parking_clear(candidate.x, candidate.y, pose.yaw):
                    continue
                cam = camera_position(candidate, offset, 0.0)
                if _ground_blocked(float(cam[0]), float(cam[1])):
                    continue
                if _near_visited(candidate.x, candidate.y, visited):
                    continue
                mask = 0
                for index, other in enumerate(names):
                    if fits_when_aimed(cells[other], candidate, offset, intr):
                        if parallax and other in parallax and not _different_view(cells[other],parallax[other],cam):
                            continue
                        mask |= 1 << index
                if mask:
                    found.append((candidate.x, candidate.y, mask))
    if not found:
        return None
    kept: list[tuple[float, float, int]] = []
    for x, y, mask in found:
        same = None
        for index, (ox, oy, old) in enumerate(kept):
            if mask == old and math.hypot(x - ox, y - oy) < 0.02:
                same = index
                break
        if same is None:
            kept.append((x, y, mask))
    legs: list[tuple[float, list[tuple[float, float]]]] = []
    usable: list[tuple[float, float, int]] = []
    for x, y, mask in kept:
        trip = _travel(pose.x, pose.y, x, y, extras, pose.yaw)
        if trip is None:
            continue
        legs.append(trip)
        usable.append((x, y, mask))
    if not usable:
        return None
    count = len(usable)
    pair: dict[tuple[int, int], float] = {}

    def between(left: int, right: int) -> float | None:
        key = (left, right) if left < right else (right, left)
        if key in pair:
            cached = pair[key]
            return None if cached < 0 else cached
        ax, ay, _mask = usable[left]
        bx, by, _other = usable[right]
        trip = _travel(ax, ay, bx, by, extras, pose.yaw)
        if trip is None:
            pair[key] = -1.0
            return None
        pair[key] = trip[0]
        return trip[0]

    cost_at: list[dict[int, float]] = [dict() for _ in range(count)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    for index, (_x, _y, mask) in enumerate(usable):
        cost_at[index][mask] = legs[index][0]
        parent[(index, mask)] = None
    done: tuple[float, int, int] | None = None
    for _round in range(count):
        grew = False
        for left in range(count):
            for mask, cost in list(cost_at[left].items()):
                if mask == full:
                    if done is None or cost < done[0]:
                        done = (cost, left, mask)
                    continue
                for right in range(count):
                    if right == left:
                        continue
                    added = usable[right][2]
                    if added & ~mask == 0:
                        continue
                    step = between(left, right)
                    if step is None:
                        continue
                    nxt = mask | added
                    total = cost + step
                    old = cost_at[right].get(nxt)
                    if old is None or total + 1e-6 < old:
                        cost_at[right][nxt] = total
                        parent[(right, nxt)] = (left, mask)
                        grew = True
        if not grew:
            break
    if done is None:
        ranked = None
        for index in range(count):
            for mask_now, cost in cost_at[index].items():
                rank = (bin(mask_now).count("1"), -cost)
                if ranked is None or rank > ranked[0]:
                    ranked = (rank, index, mask_now)
        if ranked is None:
            return None
        _rank, index, mask = ranked
    else:
        _cost, index, mask = done
    order = [(index, mask)]
    while parent.get((index, mask)) is not None:
        index, mask = parent[(index, mask)]
        order.append((index, mask))
    order.reverse()
    line = [(pose.x, pose.y)]
    stops: list[tuple[float, float, tuple[str, ...]]] = []
    seen = 0
    cursor = (pose.x, pose.y)
    for stop_index, stop_mask in order:
        x, y, _covered = usable[stop_index]
        trip = _travel(cursor[0], cursor[1], x, y, extras, pose.yaw)
        if trip is None:
            return None
        line.extend(trip[1][1:])
        fresh = stop_mask & ~seen
        seen = stop_mask
        named = tuple(names[bit] for bit in range(len(names)) if fresh & (1 << bit))
        stops.append((x, y, named))
        cursor = (x, y)
    return stops, line


def _move_toward(
    pose: CarPose,
    x: float,
    y: float,
    line: list[tuple[float, float]],
    chase_to: str | None,
    cells: tuple[str, ...],
) -> Action | None:
    # 指令先指到下一折点。final 是这次要去认的落点。
    turn = line[1] if len(line) > 1 else (x, y)
    if len(line) > 2 and _path_clear(pose.x, pose.y, line[2][0], line[2][1], pose.yaw):
        turn = line[2]
    elif len(line) <= 2 and math.hypot(turn[0] - pose.x, turn[1] - pose.y) < MOVE_ARRIVE_M:
        turn = (x, y)
    target = CarPose(turn[0], turn[1], pose.yaw, pose.stamp)
    text = _move_text(pose, target)
    if text is None:
        return None
    return Action(
        "move",
        text,
        x=target.x,
        y=target.y,
        final_x=x,
        final_y=y,
        chase_to=chase_to,
        cells=cells,
    )


def _point_used(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    return any(math.hypot(x - px, y - py) < 0.25 for px, py in points)


def _return_visit(
    pose: CarPose,
    offset: np.ndarray,
    intr: Intrinsics,
    need: list[str],
    visited: list[tuple[float, float]],
    used: dict[str, list[tuple[float, float]]],
    parallax: dict | None = None,
) -> Action | None:
    # 已到过、而且这格转正还能看清的点，取路程最短的一个。当前点和用过的点跳过。
    if not need or not visited:
        return None
    cells = _cells()
    extras = _bypass_points(pose.yaw)
    best = None
    for x, y in visited:
        if math.hypot(x - pose.x, y - pose.y) < ARRIVE_M:
            continue
        stood = CarPose(x, y, pose.yaw, pose.stamp)
        useful = []
        for name in need:
            if _point_used(x, y, used.get(name, [])):
                continue
            if fits_when_aimed(cells[name], stood, offset, intr):
                cam = camera_position(stood,offset,0.)
                if parallax and name in parallax and not _different_view(cells[name],parallax[name],cam):
                    continue
                useful.append(name)
        if not useful:
            continue
        trip = _travel(pose.x, pose.y, x, y, extras, pose.yaw)
        if trip is None:
            continue
        if best is None or trip[0] < best[0]:
            best = (trip[0], x, y, trip[1], tuple(useful))
    if best is None:
        return None
    _cost, x, y, line, useful = best
    return _move_toward(pose, x, y, line, "revisit", useful)


def _leg_cost(pose: CarPose, action: Action | None) -> float | None:
    # 比的是绕开方台之后的路程，不是新旧。
    if action is None or action.final_x is None or action.final_y is None:
        return None
    trip = _travel(pose.x, pose.y, action.final_x, action.final_y, _bypass_points(pose.yaw), pose.yaw)
    if trip is None:
        return None
    return trip[0]


def _relocate(
    pose: CarPose,
    offset: np.ndarray,
    intr: Intrinsics,
    need: list[str],
    visited: list[tuple[float, float]],
    chase_to: str | None = None,
    cells: tuple[str, ...] = (),
    parallax: dict | None = None,
) -> Action:
    # 能不能定，按转到正中、俯仰对准之后算。总路程短的优先。原点不参与比较。
    planned = _search(pose, offset, intr, need, visited, parallax)
    if planned is None:
        if parallax:
            return Action("stuck", "没有可达且有足够侧向视差的观察点，保持未知")
        return _fallback(pose, need, visited)
    stops, line = planned
    if not stops or len(line) < 2:
        return _fallback(pose, need, visited)
    x, y, _named = stops[0]
    turn = line[1] if len(line) > 1 else (x, y)
    if len(line) > 2 and _path_clear(pose.x, pose.y, line[2][0], line[2][1], pose.yaw):
        turn = line[2]
    elif len(line) <= 2 and math.hypot(turn[0] - pose.x, turn[1] - pose.y) < MOVE_ARRIVE_M:
        turn = (x, y)
    # 这一步能直达落点时，落点本身要能看清。折点只是绕台，到了再算。
    if abs(turn[0] - x) < 1e-6 and abs(turn[1] - y) < 1e-6:
        stood = CarPose(x, y, pose.yaw, pose.stamp)
        if not any(fits_when_aimed(_cells()[name], stood, offset, intr) for name in need):
            return _fallback(pose, need, visited)
    action = _move_toward(pose, x, y, line, chase_to, tuple(_named) if parallax else cells)
    if action is None:
        if parallax:
            return Action("stuck", "没有可达且有足够侧向视差的观察点，保持未知")
        return _fallback(pose, need, visited)
    if parallax:
        action = replace(action,text="换侧向视角，"+action.text)
    return action


def primary_route(pose, offset, intr):
    """起点A、沿途两次停稳复看、1.4米内近点B；不预先承诺远处塔型。"""
    cells = _cells()
    def targets(stood):
        return tuple(n for n,c in cells.items() if fits_rgb_when_aimed(c,stood,offset,intr))
    start = RouteStop(pose.x,pose.y,_looks(pose,offset,intr,targets(pose)), 'A起点')
    best = None
    for radius in (1.0,1.4):
        for degree in range(0,360,45):
            angle = math.radians(degree)
            x,y = pose.x+radius*math.cos(angle),pose.y+radius*math.sin(angle)
            if not _parking_clear(x,y,pose.yaw):
                continue
            trip = _travel(pose.x,pose.y,x,y,[],pose.yaw)
            if trip is None or trip[0] > 1.8:
                continue
            stood = CarPose(x,y,pose.yaw)
            names = targets(stood)
            near = sum(fits_when_aimed(cells[n],stood,offset,intr) for n in names)
            if not names or not near:
                continue
            rank = (-len(names),-near,trip[0])
            if best is None or rank < best[0]:
                best = rank,stood,names,trip[1],trip[0]
    if best is None:
        return [start], [(pose.x,pose.y)]
    _,stood,names,line,length = best
    stops = [start]
    # 沿规划折线插入停靠点，禁止把绕障折线直接连成穿台捷径。
    for fraction in (1/3,2/3):
        remaining = length*fraction
        point = None
        for first,second in zip(line,line[1:]):
            span = math.dist(first,second)
            if remaining <= span and span > 0:
                t = remaining/span
                point = (first[0]+t*(second[0]-first[0]),first[1]+t*(second[1]-first[1]))
                break
            remaining -= span
        if point is None or not _parking_clear(*point,pose.yaw):
            continue
        paused = CarPose(*point,pose.yaw)
        observed = targets(paused)
        if observed:
            stops.append(RouteStop(*point,_looks(paused,offset,intr,observed),'沿途复看'))
    stops.append(RouteStop(stood.x,stood.y,_looks(stood,offset,intr,names),'B近点'))
    return stops,line


def survey(pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> tuple[list[RouteStop], list[tuple[float, float]]]:
    # 地图先展示A→B主线；兜底由实时未确认任务决定，不画成必须走完的远端环路。
    return primary_route(pose,offset,intr)


def _sees(cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics) -> bool:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr)
    return bool(in_view and not needs_yaw and not needs_pitch)


def _turn_action(pose: CarPose, cell_name: str, turn: float, attempt: int) -> Action:
    if not _parking_clear(pose.x, pose.y, pose.yaw):
        return Action("blocked", "当前车位没有足够的安全旋转空间，请检查定位；保持原位")
    turn = _wrap(turn)
    shown = max(1, int(round(abs(turn))))
    signed = float(shown if turn >= 0.0 else -shown)
    return Action(
        "yaw",
        format_yaw(signed),
        yaw=pose.yaw - math.radians(signed),
        cell=cell_name,
        attempt=attempt,
    )


def _pitch_action(cell_name: str, target: float, attempt: int) -> Action:
    shown = float(int(round(target)))
    return Action("pitch", format_pitch(shown), pitch=shown, cell=cell_name, attempt=attempt)


def _aim_cell(
    name: str,
    view: CellView,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    have_color: bool,
    intr: Intrinsics | None,
    attempt_index: int,
) -> Action | None:
    # 按剩余姿态依次尝试；出画的姿态跳过，不回到已经尝试的角度。
    cells = _cells()
    cell = cells[name]
    order = range(attempt_index, MAX_TRIES)
    for index in order:
        goal = GOAL_BEARING[index]
        turn = _wrap(view.bearing_deg - goal)
        goal_pitch = view.pitch_target_deg + PITCH_OFFSET[index]
        faced = turned_pose(pose, pose.yaw - math.radians(turn))
        if intr is not None and not _sees(cell, faced, offset, goal_pitch, intr):
            continue
        if abs(turn) > YAW_TOL_DEG:
            return _turn_action(pose, name, turn, index)
        if abs(goal_pitch - pitch) > PITCH_TOL_DEG:
            if intr is not None and not _sees(cell, pose, offset, goal_pitch, intr):
                continue
            return _pitch_action(name, goal_pitch, index)
        if index == 0 and (view.needs_pitch or not view.in_view):
            return None
        if have_color:
            return Action("wait", "", cell=name, attempt=index)
        return Action("idle", "", cell=name, attempt=index)
    return None


def plan_action(
    views: dict[str, CellView],
    locked: dict[str, str | None],
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    have_color: bool,
    intr: Intrinsics | None = None,
    tries: dict[str, int] | None = None,
    visited: list[tuple[float, float]] | None = None,
    recheck: dict[str, bool] | None = None,
    used: dict[str, list[tuple[float, float]]] | None = None,
    chase: dict[str, str] | None = None,
) -> Action | None:
    names = [cell.name for cell in tower_cells()]
    doubt = recheck or {}
    spent = used or {}
    stage = chase or {}
    done = {name for name in names if stage.get(name) == "done"}
    if all((locked.get(name) and not doubt.get(name)) or name in done for name in names):
        return Action("stop", "停止")
    pending = [
        name
        for name in names
        if name not in done and (not locked.get(name) or doubt.get(name))
    ]
    attempt = tries or {}
    been = visited or []
    cells = _cells()
    origin = camera_position(pose,offset,0.)
    parallax = {
        name: (views[name].measurements or {}).get("parallax_origin",origin[:2])
        for name in pending if views[name].needs_parallax
    }

    def attempt_of(name: str) -> int:
        return int(attempt.get(name, 0))

    def can_aim(name: str) -> bool:
        # 原地姿态用完才换位；需要视差也先尝试剩余姿态。
        if attempt_of(name) >= MAX_TRIES:
            return False
        if views[name].method == "远处白盒" and views[name].conclusion in ("1", "2"):
            return True
        if doubt.get(name):
            if intr is None:
                return not views[name].blocked
            return fits_when_aimed(cells[name], pose, offset, intr) or views[name].conclusion in (
                "2",
                "红",
                "蓝",
            )
        if views[name].conclusion in ("2", "红", "蓝"):
            return True
        if intr is None:
            return not views[name].blocked
        return fits_when_aimed(cells[name], pose, offset, intr)

    missed: list[str] = []
    waiting_color: list[str] = []
    for _ in pending:
        aimable = [
            name
            for name in pending
            if can_aim(name) and name not in missed and name not in waiting_color
        ]
        if not aimable:
            break
        facing = []
        for name in aimable:
            goal = GOAL_BEARING[min(attempt_of(name), MAX_TRIES - 1)]
            if abs(_wrap(views[name].bearing_deg - goal)) <= CENTER_DEG:
                facing.append(name)
        target = min(facing or aimable, key=lambda name: abs(views[name].bearing_deg))
        action = _aim_cell(
            target,
            views[target],
            pose,
            offset,
            pitch,
            have_color,
            intr,
            attempt_of(target),
        )
        if action is None:
            missed.append(target)
            continue
        if action.kind == "idle":
            waiting_color.append(target)
            continue
        return action
    if intr is None:
        return None
    need = [name for name in pending if name in missed or not can_aim(name)]
    if not need:
        return None
    again = []
    rest = []
    for name in need:
        should = fits_when_aimed(cells[name], pose, offset, intr)
        # 这个角度理应看见却没稳定结果，或和旧结论冲突，才换个地方重看。
        if doubt.get(name) or (should and not locked.get(name)):
            again.append(name)
        else:
            rest.append(name)
    if again:
        # 已到过的点和新找的落点放在一起，只留路程更短的那个。
        requested = {name: parallax[name] for name in again if name in parallax}
        back = _return_visit(pose, offset, intr, again, been, spent, requested)
        found = _relocate(pose, offset, intr, again, been, "fresh", tuple(again), requested)
        back_cost = _leg_cost(pose, back)
        new_cost = _leg_cost(pose, found)
        if back is not None and back_cost is not None and (new_cost is None or back_cost < new_cost):
            return back
        if found is not None:
            return found
        if back is not None:
            return back
    if not rest:
        return None
    requested = {name: parallax[name] for name in rest if name in parallax}
    return _relocate(pose, offset, intr, rest, been,
                     "fresh" if requested else None, tuple(rest) if requested else (), requested)


def next_action(
    views: dict[str, CellView],
    locked: dict[str, str | None],
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    have_color: bool,
    intr: Intrinsics | None = None,
    tries: dict[str, int] | None = None,
    visited: list[tuple[float, float]] | None = None,
    recheck: dict[str, bool] | None = None,
    used: dict[str, list[tuple[float, float]]] | None = None,
    chase: dict[str, str] | None = None,
) -> str | None:
    action = plan_action(
        views, locked, pose, offset, pitch, have_color, intr, tries, visited, recheck, used, chase
    )
    if action is None or action.kind == "wait":
        return None
    return action.text


def nearest_observation(pose, offset, intr, need, exhausted=None, references=None):
    """按每座可覆盖塔的绕障路程选点，同分优先姿态丰富、转向少的点。"""
    exhausted, references = exhausted or {}, references or {}
    cells = _cells()
    candidates = set()
    # 加密当前位置附近，避免仅有绕塔大半径候选时过度移动。
    for radius in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0, 1.2):
        for degrees in range(0, 360, 15):
            angle = math.radians(degrees)
            candidates.add((pose.x+radius*math.cos(angle), pose.y+radius*math.sin(angle)))
    for name in need:
        cell = cells[name]
        for distance in _distances(cell, intr):
            for degrees in range(0, 360, 15):
                angle = math.radians(degrees)
                stood = _pose_for_camera(pose, offset, cell.x+distance*math.cos(angle),
                                         cell.y+distance*math.sin(angle))
                candidates.add((stood.x, stood.y))
    extras = _bypass_points(pose.yaw)
    best = None
    for x, y in sorted(candidates, key=lambda p: math.hypot(p[0]-pose.x, p[1]-pose.y)):
        direct = math.hypot(x-pose.x, y-pose.y)
        if best is not None and direct/max(1, len(need)) > best[0]+1e-6:
            break
        if direct < MOVE_ARRIVE_M or not _parking_clear(x, y, pose.yaw):
            continue
        stood = CarPose(x, y, pose.yaw, pose.stamp)
        cam = camera_position(stood, offset, 0.)
        if _ground_blocked(float(cam[0]), float(cam[1])):
            continue
        profiles = {}
        for name in need:
            if any(math.hypot(x-ex, y-ey)<0.08 for ex,ey in exhausted.get(name, [])):
                continue
            if name in references and not _different_view(cells[name], references[name], cam):
                continue
            poses = tuple(observation_poses(cells[name], stood, offset, intr))
            if poses:
                profiles[name] = poses
        useful = tuple(profiles)
        if not useful:
            continue
        trip = _travel(pose.x, pose.y, x, y, extras, pose.yaw)
        if trip is None:
            continue
        # 路程/覆盖塔数避免只贪最近单塔；姿态数量只作同成本时的偏好。
        turn = min(abs(_wrap(math.degrees(faced.yaw-pose.yaw)))
                   for poses in profiles.values() for _, faced, _ in poses)
        rank = (trip[0]/len(useful), -len(useful),
                -sum(len(poses) for poses in profiles.values()), turn, trip[0])
        if best is None or rank < best[:5]:
            best = (*rank, x, y, trip[1], useful)
    if best is None:
        return Action("blocked", "没有安全有效的观察点，保留待确认任务")
    _, _, _, _, _, x, y, line, useful = best
    return _move_toward(pose, x, y, line, "fresh", useful) or Action("blocked", "保持原位等待有效观察条件")
