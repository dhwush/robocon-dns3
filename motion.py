"""拓展指令的下一步。

旋转和俯仰不限次数。每一格先转到画面正中，再把一层顶和塔顶放到画面中段，
认满再换下一格。认不出就换一个俯仰或偏一个角度重看。
只有挡住、竖直装不下、或超出深度范围，才平移。
落点从当前车位绕开方台，走完还没定的格，总路程取最短。找不到落点才回原点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from car_pose import CarPose, Intrinsics, bearing_deg, camera_position
from classify import MIN_VISIBLE_M, CellView, _pitch_target, _view_flags
from field_map import TOWER_NO, all_cells, ground_towers, tower_cells
from sight import visible_height


# 正中：方位和目标俯仰都进到这个范围里，才开始记这一格。
CENTER_DEG = 5.0
PITCH_TOL_DEG = 1.0
# 正中看了还没有结论，就换俯仰、再偏一个角度。五个视角轮流看，不因此改去平移。
MAX_TRIES = 5
PITCH_OFFSET = (0.0, 8.0, -8.0, 0.0, 0.0)
GOAL_BEARING = (0.0, 0.0, 0.0, 15.0, -15.0)
# 深度相机大约 0.3 m 到 3 m。落点要落在这段里。
DEPTH_NEAR_M = 0.3
DEPTH_FAR_M = 3.0
# 一层顶和塔顶占竖直视场的比例。留出边上的空，避免贴着画面边缘认。
VERTICAL_MARGIN = 0.84
ARRIVE_M = 0.15
PATH_STEP_M = 0.05


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
    turn = bearing_deg(cell.center(cell.one_z), pose, offset)
    return CarPose(pose.x, pose.y, pose.yaw - math.radians(turn), pose.stamp)


def place_done(locked: dict[str, str | None], pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> bool:
    # 这一站转正还能看清的格子都锁定了，才离开去下一站。
    for cell in tower_cells():
        if locked.get(cell.name):
            continue
        if fits_when_aimed(cell, pose, offset, intr):
            return False
    return True


def fits_when_aimed(cell, pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> bool:
    # 转到正中、俯仰对准一层顶和塔顶的中点之后，还要看得见、让得开、落在深度里。
    faced = aimed_pose(pose, offset, cell)
    pitch = _pitch_target(cell, faced, offset)
    in_view, needs_yaw, needs_pitch = _view_flags(cell, faced, offset, pitch, intr)
    if not in_view or needs_yaw or needs_pitch:
        return False
    if visible_height(faced, offset, pitch, cell, cell.base_z, cell.one_z) < MIN_VISIBLE_M:
        return False
    slant = _slant(cell, camera_position(faced, offset, 0.0))
    return DEPTH_NEAR_M <= slant <= DEPTH_FAR_M


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
    # 十二块绿台和 1、5、6 号木块都不能停。车身尺寸还没量，先按光心和车位中心挡。
    point_x, point_y = float(x), float(y)
    for x0, x1, y0, y1 in _obstacles():
        if x0 <= point_x <= x1 and y0 <= point_y <= y1:
            return True
    return False


def _path_clear(x0: float, y0: float, x1: float, y1: float) -> bool:
    # 起点已经贴在台边上时，允许先离开。中途再压上台或木块就不走。
    # 同一段直线只查一次。绕行时外角之间的边会反复用到。
    key = (float(x0), float(y0), float(x1), float(y1))
    cached = _path_clear.cache.get(key)
    if cached is not None:
        return cached
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(math.ceil(dist / PATH_STEP_M)))
    clear = True
    for index in range(1, steps + 1):
        t = index / steps
        if _ground_blocked(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
            clear = False
            break
    _path_clear.cache[key] = clear
    return clear


_path_clear.cache = {}


def _vertical_span(cell, distance_h: float) -> float:
    high = math.atan2(cell.cap_z - 0.70, max(distance_h, 1e-6))
    low = math.atan2(cell.one_z - 0.70, max(distance_h, 1e-6))
    return abs(high - low)


def _distances(cell, intr: Intrinsics) -> list[float]:
    limit = 2.0 * math.atan((intr.height * 0.5) / intr.fy) * VERTICAL_MARGIN
    found = []
    for dist in np.arange(0.45, DEPTH_FAR_M + 1e-6, 0.15):
        if _vertical_span(cell, float(dist)) >= limit:
            continue
        mid_z = (cell.one_z + cell.cap_z) / 2.0
        slant = math.hypot(float(dist), mid_z - 0.70)
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
    if abs(forward) >= 0.05:
        meters = max(0.1, round(abs(forward) * 10.0) / 10.0)
        parts.append(f"{'向前' if forward >= 0.0 else '向后'}平移 {meters:.1f} 米")
    if abs(lateral) >= 0.05:
        meters = max(0.1, round(abs(lateral) * 10.0) / 10.0)
        parts.append(f"{'向右' if lateral >= 0.0 else '向左'}平移 {meters:.1f} 米")
    if not parts:
        return None
    return "，".join(parts)


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


def _bypass_points() -> list[tuple[float, float]]:
    gap = 0.08
    points = []
    for left, right, near, far in _blocks():
        for x, y in (
            (left - gap, near - gap),
            (right + gap, near - gap),
            (left - gap, far + gap),
            (right + gap, far + gap),
        ):
            if not _ground_blocked(x, y):
                points.append((x, y))
    return points


def _travel(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    extras: list[tuple[float, float]],
) -> tuple[float, list[tuple[float, float]]] | None:
    # 直线能过就直走。过不去就从台子外角绕。返回总路程，以及从起点到终点的折线。
    if math.hypot(x1 - x0, y1 - y0) < 1e-6:
        return 0.0, [(x0, y0)]
    if _path_clear(x0, y0, x1, y1):
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
        if not _path_clear(ax, ay, bx, by):
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
    # 采样圈上没有落点时，才走事先定好的右前角。
    from route import fixed_move

    return fixed_move(pose, need, visited)


def _looks(pose: CarPose, offset: np.ndarray, intr: Intrinsics, names: list[str] | tuple[str, ...]) -> tuple[Look, ...]:
    # 每一格转到正中的航向，俯仰对准一层顶和塔顶的中点。
    cells = _cells()
    rows = []
    for name in names:
        cell = cells[name]
        faced = aimed_pose(pose, offset, cell)
        pitch = _pitch_target(cell, faced, offset)
        rows.append(Look(TOWER_NO[name], float(np.degrees(faced.yaw)), float(pitch)))
    rows.sort(key=lambda item: item.number)
    return tuple(rows)


def _search(
    pose: CarPose,
    offset: np.ndarray,
    intr: Intrinsics,
    need: list[str],
    visited: list[tuple[float, float]],
) -> tuple[list[tuple[float, float, tuple[str, ...]]], list[tuple[float, float]]] | None:
    # 落点按总路程最短排。返回各停点新看清的格，以及从当前车位出发的折线。
    cells = _cells()
    names = list(need)
    if not names:
        return [], [(pose.x, pose.y)]
    full = (1 << len(names)) - 1
    extras = _bypass_points()
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
                if _ground_blocked(candidate.x, candidate.y):
                    continue
                cam = camera_position(candidate, offset, 0.0)
                if _ground_blocked(float(cam[0]), float(cam[1])):
                    continue
                if _near_visited(candidate.x, candidate.y, visited):
                    continue
                mask = 0
                for index, other in enumerate(names):
                    if fits_when_aimed(cells[other], candidate, offset, intr):
                        mask |= 1 << index
                if mask:
                    found.append((candidate.x, candidate.y, mask))
    if not found:
        return None
    kept: list[tuple[float, float, int]] = []
    for x, y, mask in found:
        same = None
        for index, (ox, oy, old) in enumerate(kept):
            if math.hypot(x - ox, y - oy) < 0.2:
                same = index
                if bin(mask).count("1") > bin(old).count("1"):
                    kept[index] = (x, y, mask)
                break
        if same is None:
            kept.append((x, y, mask))
    legs: list[tuple[float, list[tuple[float, float]]]] = []
    usable: list[tuple[float, float, int]] = []
    for x, y, mask in kept:
        trip = _travel(pose.x, pose.y, x, y, extras)
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
        trip = _travel(ax, ay, bx, by, extras)
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
        trip = _travel(cursor[0], cursor[1], x, y, extras)
        if trip is None:
            return None
        line.extend(trip[1][1:])
        fresh = stop_mask & ~seen
        seen = stop_mask
        named = tuple(names[bit] for bit in range(len(names)) if fresh & (1 << bit))
        stops.append((x, y, named))
        cursor = (x, y)
    return stops, line


def _relocate(
    pose: CarPose,
    offset: np.ndarray,
    intr: Intrinsics,
    need: list[str],
    visited: list[tuple[float, float]],
) -> Action:
    # 能不能定，按转到正中、俯仰对准之后算。总路程短的优先。原点不参与比较。
    planned = _search(pose, offset, intr, need, visited)
    if planned is None:
        return _fallback(pose, need, visited)
    stops, line = planned
    if not stops or len(line) < 2:
        return _fallback(pose, need, visited)
    x, y, _named = stops[0]
    turn = line[1]
    if math.hypot(turn[0] - pose.x, turn[1] - pose.y) < ARRIVE_M and len(line) > 2:
        turn = line[2]
    elif math.hypot(turn[0] - pose.x, turn[1] - pose.y) < ARRIVE_M:
        turn = (x, y)
    # 这一步能直达落点时，落点本身要能看清。折点只是绕台，到了再算。
    target = CarPose(turn[0], turn[1], pose.yaw, pose.stamp)
    if abs(turn[0] - x) < 1e-6 and abs(turn[1] - y) < 1e-6:
        cells = _cells()
        if not any(fits_when_aimed(cells[name], target, offset, intr) for name in need):
            return _fallback(pose, need, visited)
    text = _move_text(pose, target)
    if text is None:
        return _fallback(pose, need, visited)
    return Action("move", text, x=target.x, y=target.y)


def survey(pose: CarPose, offset: np.ndarray, intr: Intrinsics) -> tuple[list[RouteStop], list[tuple[float, float]]]:
    # 起点先记下已经能看清的格。剩下的按最短路线排停点。折线是车位，不是光心。
    here = []
    need = []
    for cell in tower_cells():
        if fits_when_aimed(cell, pose, offset, intr):
            here.append(cell.name)
        else:
            need.append(cell.name)
    start = RouteStop(pose.x, pose.y, _looks(pose, offset, intr, here))
    if not need:
        return [start], [(pose.x, pose.y)]
    planned = _search(pose, offset, intr, need, [])
    if planned is None:
        return [start], [(pose.x, pose.y)]
    stops, line = planned
    route = [start]
    for x, y, named in stops:
        stood = CarPose(x, y, pose.yaw, pose.stamp)
        route.append(RouteStop(x, y, _looks(stood, offset, intr, named)))
    return route, line


def _sees(cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics) -> bool:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr)
    return bool(in_view and not needs_yaw and not needs_pitch)


def _turn_action(pose: CarPose, cell_name: str, turn: float, attempt: int) -> Action:
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
    # 五个视角轮流用。某一档会把格子推出画面就跳过，转完仍从正中接着看，不改去平移。
    cells = _cells()
    cell = cells[name]
    start = attempt_index % MAX_TRIES
    order = list(range(start, MAX_TRIES)) + list(range(start))
    for index in order:
        goal = GOAL_BEARING[index]
        if abs(_wrap(view.bearing_deg - goal)) > CENTER_DEG:
            return _turn_action(pose, name, view.bearing_deg - goal, index)
        goal_pitch = view.pitch_target_deg + PITCH_OFFSET[index]
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
) -> Action | None:
    names = [cell.name for cell in tower_cells()]
    if all(locked.get(name) for name in names):
        return Action("stop", "停止")
    pending = [name for name in names if not locked.get(name)]
    attempt = tries or {}
    been = visited or []
    cells = _cells()

    def attempt_of(name: str) -> int:
        return int(attempt.get(name, 0))

    def can_aim(name: str) -> bool:
        # 已经看见两层或塔顶，继续转、继续俯仰，把这一格锁上，不为它平移。
        if views[name].conclusion in ("2", "红", "蓝"):
            return True
        if intr is None:
            return not views[name].blocked and attempt_of(name) < MAX_TRIES
        # 转到正中、俯仰对准还能看清时，视角用完也接着看，不因此平移。
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
    return _relocate(pose, offset, intr, need, been)


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
) -> str | None:
    action = plan_action(views, locked, pose, offset, pitch, have_color, intr, tries, visited)
    if action is None or action.kind == "wait":
        return None
    return action.text
