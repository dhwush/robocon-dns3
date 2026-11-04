"""拓展指令的下一步。当前朝向里能看清的先记下，再改俯仰，然后转向，被挡住才平移。"""

from __future__ import annotations

import math

from car_pose import CarPose, camera_position, shifted_pose
from classify import MIN_VISIBLE_M, CellView
from field_map import all_cells, tower_cells
from sight import visible_height


def format_yaw(bearing_deg: float) -> str:
    # 正的方位在车的右侧。度数是相对现在的车头还要转的量。
    degrees = max(1, int(round(abs(bearing_deg))))
    if bearing_deg >= 0.0:
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


def _on_platform(position: np.ndarray) -> bool:
    x, y = float(position[0]), float(position[1])
    for cell in all_cells():
        x0, x1, y0, y1 = cell.bounds(0.0)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def slide_for(
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    blocked_names: set[str],
) -> float | None:
    # 在左右方向上找最短的一步，让开挡住的台，车仍停在地面上。
    cells = {cell.name: cell for cell in tower_cells()}
    wanted = [cells[name] for name in blocked_names]
    if not wanted:
        return None
    best: tuple[int, float, float] | None = None
    for direction in (1.0, -1.0):
        for step in range(1, 81):
            lateral = direction * step * 0.05
            moved = shifted_pose(pose, lateral)
            if _on_platform(camera_position(moved, offset, pitch)):
                continue
            cleared = sum(
                1
                for cell in wanted
                if visible_height(moved, offset, pitch, cell, cell.base_z, cell.one_z) >= MIN_VISIBLE_M
            )
            if cleared == 0:
                continue
            rank = (cleared, -abs(lateral))
            if best is None or rank > (best[0], best[1]):
                best = (cleared, -abs(lateral), lateral)
    if best is None:
        return None
    return best[2]


def next_action(
    views: dict[str, CellView],
    locked: dict[str, str | None],
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    have_color: bool,
) -> str | None:
    names = [cell.name for cell in tower_cells()]
    if all(locked.get(name) for name in names):
        return "停止"
    pending = [name for name in names if not locked.get(name)]
    if have_color:
        # 这一眼已经能看清的格，先攒够 5 帧再开走，避免还没写下就离开。
        readable = [
            name
            for name in pending
            if views[name].in_view and not views[name].needs_yaw and not views[name].needs_pitch and not views[name].blocked
        ]
        if readable:
            return None
    # 左右已经在画面里、只差高度的，先改俯仰，再为左右出画的格子转向。
    pitch_names = [
        name
        for name in pending
        if views[name].needs_pitch and abs(views[name].pitch_target_deg - pitch) > 1.0
    ]
    if pitch_names:
        target = pitch_names[0]
        return format_pitch(views[target].pitch_target_deg)
    yaw_names = [name for name in pending if views[name].needs_yaw]
    if yaw_names:
        target = min(yaw_names, key=lambda name: abs(views[name].bearing_deg))
        return format_yaw(views[target].bearing_deg)
    blocked = [name for name in pending if views[name].blocked and views[name].conclusion is None]
    if not blocked:
        return None
    lateral = slide_for(pose, offset, pitch, set(blocked))
    if lateral is None:
        return None
    return format_slide(lateral)

