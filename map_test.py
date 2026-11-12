"""不订雷达，在平面图上点光心，再开相机认塔。

python map_test.py

点下去的是相机光心在地面上的位置。点下去就画出从这里看完六格的最短路线。
再点路线上的圆点，当作车已经到了这一点，光心改到这一站的坐标，窗口列出朝向和俯仰。
航向在这个窗口里输入，单位是度，正数向左转。没输入时按正前方。改变航向只更新当前朝向，不重新规划路线；选择新的起步光心位置才重画路线。
点住某一站之后，航向直接用这一站已经算好的角度，再点同一站换下一格。路线不动。回车就在这一站的坐标打开相机，退格仍可改数字。刚打开时俯仰用 config/field.yaml 里的角度。第一次读到加速度计时框不动，之后俯仰变多少，框就跟着变多少。光心高度为 0.669 m（66.9 cm）。
摄像头窗口最上面写框正在用的俯仰。
关掉相机窗口后回到平面图，刚才的路线还在。再点一个新的光心才换路线。之前打开过相机的点留在窗口里，按数字或点一下那个点就能直接选用。q 或 Esc 退出。
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from car_pose import OPTICAL_FROM_CORNER, CarPose, Setup, camera_position, load_intrinsics, load_setup
from field_map import TOWER_NO, all_cells, tower_cells
from motion import format_pitch, survey
from report import FieldWatch
from view_field import ROOT, _live


# 黑色地面比最外一圈方块再多出 3 m。相机深度大约到 3 m，观测点可以点到这个范围。
_OPEN_M = 3.0
SCALE = 100
# 离边不到 1 cm 才吸。图上大约一个像素，避免光标离得远也跟着跳。
SNAP_M = 0.01
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_POINTS_PATH = ROOT / "config" / "map_points.txt"
_POINT_LIMIT = 9
_COLORS = {
    0.20: (20, 70, 20),
    0.40: (40, 140, 50),
    0.60: (150, 200, 150),
}


def pose_for_optical(x: float, y: float, yaw: float) -> CarPose:
    # 由光心反推右前角定位点，安装偏移随航向旋转。
    yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
    body = OPTICAL_FROM_CORNER
    rotated = np.array(
        [yaw_c * body[0] - yaw_s * body[1], yaw_s * body[0] + yaw_c * body[1]],
        dtype=float,
    )
    origin = np.array([x, y], dtype=float) - rotated
    return CarPose(float(origin[0]), float(origin[1]), float(yaw))


def _outer_limits() -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for cell in list(all_cells()) + list(tower_cells()):
        x0, x1, y0, y1 = cell.bounds(0.0)
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    return min(xs) - _OPEN_M, max(xs) + _OPEN_M, min(ys) - _OPEN_M, max(ys) + _OPEN_M


X_MIN, X_MAX, Y_MIN, Y_MAX = _outer_limits()


def _pixel(x: float, y: float) -> tuple[int, int]:
    return int(round((x - X_MIN) * SCALE)), int(round((Y_MAX - y) * SCALE))


def _world(u: int, v: int) -> tuple[float, float]:
    return X_MIN + u / SCALE, Y_MAX - v / SCALE


def optical_of(x: float, y: float, yaw: float) -> tuple[float, float]:
    # 路线算的是车位。图上的点和点击都是光心，按同一航向换过去。
    cam = camera_position(CarPose(x, y, yaw), np.zeros(2), 0.0)
    return float(cam[0]), float(cam[1])


def hit_route(points: list[tuple[float, float]], u: int, v: int) -> int | None:
    # 点在圆点上，14 个像素以内，当作车已经到达这一站。
    chosen = None
    gap = 14.0
    for index, (x, y) in enumerate(points):
        su, sv = _pixel(x, y)
        distance = float(np.hypot(su - u, sv - v))
        if distance <= gap:
            gap = distance
            chosen = index
    return chosen


def camera_point(
    marker: tuple[float, float] | None,
    route_points: list[tuple[float, float]],
    arrived: int | None,
    pinned: bool,
) -> tuple[float, float] | None:
    # 点住某一站时，回车用这一站的光心。没点住就用起步时点下的那个。
    if pinned and isinstance(arrived, int) and 0 <= arrived < len(route_points):
        return route_points[arrived]
    return marker


def look_yaw_text(stops: list, arrived: int, look_index: int) -> str | None:
    # 选中的站用已经算好的朝向。正数向左，和窗口里打的航向同一个方向。
    if not 0 <= arrived < len(stops):
        return None
    looks = stops[arrived].looks
    if not looks:
        return None
    return f"{looks[look_index % len(looks)].yaw_deg:.1f}"


def open_yaw(yaw_text: str) -> float | None:
    # 没打数字时按正前方。打到一半，比如只有负号，先不开。
    if not yaw_text:
        return 0.0
    return _yaw_from_text(yaw_text)


def select_route_stop(view: dict, stops: list, index: int, current_yaw: float) -> None:
    """保存点击前的航向，目标航向更新后仍能显示本次需要转的角度。"""
    view["look"] = int(view["look"]) + 1 if view.get("pinned") and view.get("arrived") == index else 0
    view["arrived"] = index
    view["pinned"] = True
    view["arrival_yaw"] = current_yaw
    view["auto_yaw"] = look_yaw_text(stops, index, int(view["look"]))


def stop_lines(stops: list, index: int, chosen: int = 0, arrival_yaw: float | None = None) -> list[str]:
    stop = stops[index]
    lines = [f"停 {index + 1} / {len(stops)}  已到达"]
    if not stop.looks:
        lines.append("这一站没有新的格子")
    for look_index, look in enumerate(stop.looks):
        side = "向左" if look.yaw_deg >= 0.0 else "向右"
        current = "  当前" if stop.looks and look_index == chosen % len(stop.looks) else ""
        lines.append(
            f"{look.number}  朝向 {side} {abs(look.yaw_deg):.0f} 度   {format_pitch(look.pitch_deg)}{current}"
        )
    if arrival_yaw is not None and stop.looks:
        look = stop.looks[chosen % len(stop.looks)]
        turn = (look.yaw_deg - math.degrees(arrival_yaw) + 180.0) % 360.0 - 180.0
        if abs(turn) < 0.5:
            lines.append(f"看 {look.number}  航向已对准，无需旋转")
        else:
            side = "向左" if turn > 0 else "向右"
            lines.append(f"看 {look.number}  需{side}转 {abs(turn):.0f} 度（相对点击前航向）")
    lines.append("完成识别")
    seen = {look.number for item in stops for look in item.looks}
    if index == len(stops) - 1 and seen >= set("123456"):
        lines.append("六格到此看完，停止")
    return lines


def _edge_lines() -> tuple[tuple[float, ...], tuple[float, ...]]:
    # 1.2 m 的台，以及台上和地面上那块 50 cm 方块。吸的是边所在的整条直线。
    xs: list[float] = []
    ys: list[float] = []
    seen: set[tuple[float, float, float, float]] = set()
    boxes = [cell.bounds(0.0) for cell in all_cells()]
    boxes.extend(cell.search_bounds(0.0) for cell in tower_cells())
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        x0, x1, y0, y1 = box
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    return tuple(xs), tuple(ys)


def _snap_axis(value: float, lines: tuple[float, ...]) -> float:
    snapped = value
    gap = SNAP_M
    for line in lines:
        distance = abs(value - line)
        if distance <= gap:
            gap = distance
            snapped = line
    return snapped


def _snap_cursor(x: float, y: float) -> tuple[float, float]:
    xs, ys = _edge_lines()
    return _snap_axis(x, xs), _snap_axis(y, ys)


def _canvas() -> np.ndarray:
    width = int(round((X_MAX - X_MIN) * SCALE))
    height = int(round((Y_MAX - Y_MIN) * SCALE))
    board = np.full((height, width, 3), 32, dtype=np.uint8)
    return board


def _block_name(cell) -> str:
    height_cm = int(round(cell.height * 100))
    return f"{height_cm} cm 绿色方块地面"


def nearest_green_block(x: float, y: float):
    # 距离量到绿色方块地面上最近的一点。人站在这块 1.2 m 的台上就是 0。
    # 十二块绿台都算，不用台上那块 50 cm 建塔区，也不算木头上的 1、5、6。
    chosen = None
    gap = None
    point = (x, y)
    for cell in all_cells():
        x0, x1, y0, y1 = cell.bounds(0.0)
        near_x = min(max(x, x0), x1)
        near_y = min(max(y, y0), y1)
        distance = float(np.hypot(x - near_x, y - near_y))
        if gap is None or distance < gap:
            chosen = cell
            gap = distance
            point = (near_x, near_y)
    return chosen, float(gap), point


def _cursor_text(x: float, y: float) -> str:
    cell, distance, _point = nearest_green_block(x, y)
    return (
        f"相对原点  x {x:+.3f} m    y {y:+.3f} m"
        f"    距最近的{_block_name(cell)} {distance:.3f} m"
    )


def _yaw_from_text(text: str) -> float | None:
    if not text or text in {"-", ".", "-."}:
        return None
    return float(np.deg2rad(float(text)))


def _append_yaw(text: str, char: str) -> str:
    if char == "-":
        return text if text else "-"
    if char == ".":
        return text if "." in text else text + "."
    if char.isdigit():
        return text + char
    return text


@dataclass(frozen=True)
class UsedPoint:
    x: float
    y: float
    yaw_text: str


def _same_used(one: UsedPoint, other: UsedPoint) -> bool:
    # 同一处、同一航向只留一条。差 1 mm 或航向对不上，就当成另一个点。
    left = _yaw_from_text(one.yaw_text)
    right = _yaw_from_text(other.yaw_text)
    if left is None or right is None:
        return False
    return abs(one.x - other.x) < 1e-3 and abs(one.y - other.y) < 1e-3 and abs(left - right) < 1e-6


def remember_point(points: list[UsedPoint], x: float, y: float, yaw_text: str) -> list[UsedPoint]:
    # 最新的排在最前，按 1 就是刚用过的。最多留 9 个，对应数字键。
    fresh = UsedPoint(float(x), float(y), yaw_text)
    kept = [point for point in points if not _same_used(point, fresh)]
    return [fresh, *kept][:_POINT_LIMIT]


def load_points(path: Path) -> list[UsedPoint]:
    if not path.is_file():
        return []
    points: list[UsedPoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if _yaw_from_text(parts[2]) is None:
            continue
        points.append(UsedPoint(x, y, parts[2]))
    return _trim_loaded(points)


def _trim_loaded(points: list[UsedPoint]) -> list[UsedPoint]:
    kept: list[UsedPoint] = []
    for point in points:
        if any(_same_used(point, older) for older in kept):
            continue
        kept.append(point)
        if len(kept) >= _POINT_LIMIT:
            break
    return kept


def save_points(path: Path, points: list[UsedPoint]) -> None:
    lines = [f"{point.x:.3f} {point.y:.3f} {point.yaw_text}" for point in points]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def point_at(points: list[UsedPoint], action: str) -> UsedPoint | None:
    if len(action) != 1 or not action.isdigit() or action == "0":
        return None
    index = int(action) - 1
    if index >= len(points):
        return None
    return points[index]


def _hit_used(points: list[UsedPoint], u: int, v: int, radius: float = 8.0) -> UsedPoint | None:
    chosen = None
    gap = radius
    for point in points:
        px, py = _pixel(point.x, point.y)
        distance = float(np.hypot(px - u, py - v))
        if distance <= gap:
            chosen = point
            gap = distance
    return chosen


def _history_lines(points: list[UsedPoint]) -> list[str]:
    if not points:
        return []
    lines = ["之前的点，按数字或点一下直接选用："]
    for index, point in enumerate(points, start=1):
        lines.append(
            f"{index}  x {point.x:+.3f} m  y {point.y:+.3f} m  航向 {point.yaw_text}"
        )
    return lines


def _yaw_key(key: int) -> str:
    # Qt 把回车和退格编成大于 255 的键值。窗口还在刷新时才能收到这些键。
    if key < 0:
        return ""
    if key in (10, 13, 0x01000004, 0x01000005):
        return "ok"
    if key in (8, 127, 0x01000003):
        return "back"
    if key in (27, 0x01000000):
        return "esc"
    if key > 255:
        return ""
    if key == ord("q"):
        return "quit"
    char = chr(key)
    if char in "0123456789.-":
        return char
    return ""


def _draw_map(
    marker: tuple[float, float] | None,
    yaw: float | None,
    pitch: float,
    hover: tuple[float, float] | None = None,
    yaw_text: str = "",
    used: list[UsedPoint] | None = None,
    route_line: list[tuple[float, float]] | None = None,
    route_stops: list[tuple[float, float]] | None = None,
    arrived: int | None = None,
    arrived_lines: list[str] | None = None,
) -> np.ndarray:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    board = _canvas()
    for cell in all_cells():
        x0, x1, y0, y1 = cell.bounds(0.0)
        u0, v0 = _pixel(x0, y1)
        u1, v1 = _pixel(x1, y0)
        color = _COLORS.get(round(cell.height, 2), (80, 80, 80))
        cv2.rectangle(board, (u0, v0), (u1, v1), color, -1)
        cv2.rectangle(board, (u0, v0), (u1, v1), (20, 20, 20), 1)
    font = ImageFont.truetype(_FONT, 16)
    panel = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    for cell in all_cells():
        u, v = _pixel(cell.x, cell.y)
        if cell.tower:
            x0, x1, y0, y1 = cell.search_bounds(0.0)
            u0, v0 = _pixel(x0, y1)
            u1, v1 = _pixel(x1, y0)
            draw.rectangle((u0, v0, u1, v1), outline=(255, 255, 255))
            draw.text((u - 6, v - 10), TOWER_NO[cell.name], font=font, fill=(255, 255, 255))
        else:
            draw.text((u - 14, v - 10), f"{int(round(cell.height * 100))}", font=font, fill=(30, 30, 30))
    for cell in tower_cells():
        if not cell.ground:
            continue
        # 1、5、6 是木头地面上的 50 cm 方块，按各自的边长画，不画成 1.2 m 的台。
        x0, x1, y0, y1 = cell.bounds(0.0)
        u0, v0 = _pixel(x0, y1)
        u1, v1 = _pixel(x1, y0)
        draw.rectangle((u0, v0, u1, v1), fill=(176, 122, 62), outline=(90, 58, 24))
        u, v = _pixel(cell.x, cell.y)
        draw.text((u - 6, v - 10), TOWER_NO[cell.name], font=font, fill=(255, 255, 255))
    for index, point in enumerate(used or [], start=1):
        u, v = _pixel(point.x, point.y)
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(80, 180, 255), width=2)
        draw.text((u + 8, v - 18), str(index), font=font, fill=(80, 180, 255))
    ox, oy = _pixel(0.0, 0.0)
    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=(255, 80, 80))
    draw.text((ox + 8, oy - 8), "原点", font=font, fill=(255, 180, 180))
    if route_line and len(route_line) >= 2:
        draw.line([_pixel(x, y) for x, y in route_line], fill=(255, 140, 0), width=3)
    for index, point in enumerate(route_stops or [], start=1):
        u, v = _pixel(*point)
        if arrived == index - 1:
            draw.ellipse((u - 8, v - 8, u + 8, v + 8), fill=(255, 140, 0), outline=(255, 230, 160))
        else:
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline=(255, 140, 0), width=2)
        draw.text((u + 10, v - 10), str(index), font=font, fill=(255, 190, 90))
    if marker is not None:
        u, v = _pixel(*marker)
        draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(255, 220, 0), width=2)
        if yaw is not None:
            yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
            ahead = (marker[0] - yaw_s * 0.7, marker[1] + yaw_c * 0.7)
            draw.line((u, v, *_pixel(*ahead)), fill=(255, 220, 0), width=2)
        draw.text((u + 8, v + 4), "光心", font=font, fill=(255, 220, 0))
    if marker is None:
        hint = f"点一下光心，靠近方块边会吸住。然后在窗口里输入航向，正数向左转。俯仰 {pitch:.0f} 度。q 退出"
        if used:
            hint = f"点新的光心，或按 1 到 {len(used)} 选用之前的点。俯仰 {pitch:.0f} 度。q 退出"
    else:
        hint = "点路线上的圆点，光心和航向改到这一站算出的角度。再点同一站换下一格。回车打开相机。q 退出"
    draw.text((8, 8), hint, font=font, fill=(255, 255, 255))
    text_y = 84
    for row in arrived_lines or []:
        draw.text((8, text_y), row, font=font, fill=(255, 210, 120))
        text_y += 20
    history = _history_lines(used or [])
    line_h = 20
    y = board.shape[0] - 8 - line_h * len(history)
    for row in history:
        draw.text((8, y), row, font=font, fill=(180, 220, 255))
        y += line_h
    if hover is not None:
        hx, hy = hover
        label = _cursor_text(hx, hy)
        draw.text((560, 34), label, font=font, fill=(255, 220, 80))
        _cell, _distance, near = nearest_green_block(hx, hy)
        u, v = _pixel(hx, hy)
        draw.line((u, v, *_pixel(*near)), fill=(255, 220, 80), width=1)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(255, 220, 80))
        text_x = u + 10 if u < board.shape[1] - 520 else max(8, u - 510)
        text_y = v - 22 if v > 56 else v + 8
        draw.text((text_x, text_y), label, font=font, fill=(255, 220, 80))
    board[:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return board


def _open_camera(x: float, y: float, yaw: float, pitch: float, args=None) -> None:
    pose = pose_for_optical(x, y, yaw)
    offset = np.zeros(2, dtype=float)
    optical = camera_position(pose, offset, pitch)
    print(f"光心 x={optical[0]:.3f} m  y={optical[1]:.3f} m  高 {optical[2]:.3f} m  航向 {np.degrees(yaw):.1f} 度")
    setup = Setup(False, offset, pitch, "", "rad", origin_basic=True)
    watch = FieldWatch(setup, load_intrinsics(str(ROOT / "config" / "intrinsics.yaml")))
    watch.pitch = pitch
    args = args or SimpleNamespace(mode="extend", pose_source="ssh")
    if args.pose_source in ("ssh", "local"):
        measured = load_setup(str(ROOT / "config" / "field.yaml"))
        # 此入口按用户约定：CAN 已是场地右前角位姿，无中心补偿。
        measured = replace(measured, aligned=True, center_to_corner=np.zeros(2), origin_basic=False)
        watch = FieldWatch(measured, watch.intr)
        watch.pitch = pitch
    _live(args, watch, pose)


def route_position_key(marker):
    """路线只在起步光心的位置改变时重算，转头不会改变既定观察任务。"""
    return None if marker is None else (round(marker[0], 4), round(marker[1], 4))


MODE_BUTTONS = (("static", "手动离线"), ("ssh", "笔记本 SSH"), ("local", "主机 CAN"))


def mode_at(u, v):
    if 32 <= v <= 58:
        for index, (source, _) in enumerate(MODE_BUTTONS):
            if 8 + index * 180 <= u < 180 + index * 180:
                return source
    return None


def paint_modes(board, source):
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    panel = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(_FONT, 16)
    for index, (value, label) in enumerate(MODE_BUTTONS):
        left = 8 + index * 180
        draw.rectangle((left, 32, left + 172, 58), fill=(30, 110, 70) if value == source else (60, 60, 60))
        draw.text((left + 8, 34), label, font=font, fill="white")
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def main() -> None:
    import cv2

    import argparse
    parser = argparse.ArgumentParser(description="地图与实时 CAN 相机测试")
    parser.add_argument("--pose-source", choices=("ssh", "local", "static"), default="static")
    parser.add_argument("--verify-motion", action="store_true", help="慢速移动验证识别框，CAN 直接作为右前角场地位姿")
    parser.add_argument("--ssh-host", default="volleyball@192.168.3.194")
    parser.add_argument("--can-interface", default="can0")
    args = parser.parse_args()
    args.mode = "extend"
    pitch = load_setup(str(ROOT / "config" / "field.yaml")).pitch_deg
    intrinsics = load_intrinsics(str(ROOT / "config" / "intrinsics.yaml"))
    picked: dict[str, tuple[float, float] | None] = {"xy": None}
    chosen: dict[str, UsedPoint | None] = {"point": None}
    hover: dict[str, tuple[float, float] | None] = {"xy": None}
    view: dict[str, object] = {"stops": [], "arrived": None, "pinned": False, "look": 0, "auto_yaw": None}
    used = load_points(_POINTS_PATH)
    marker: tuple[float, float] | None = None
    yaw_text = ""
    route_key = None
    route_stops: list = []
    route_line: list[tuple[float, float]] = []
    route_points: list[tuple[float, float]] = []

    def on_mouse(event, u, v, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            source = mode_at(u, v)
            if source is not None:
                args.pose_source = source
                return
        if 32 <= v <= 58:
            return
        xy = _snap_cursor(*_world(u, v))
        if event == cv2.EVENT_MOUSEMOVE:
            hover["xy"] = xy
        elif event == cv2.EVENT_LBUTTONDOWN:
            arrived = hit_route(view["stops"], u, v)
            if arrived is not None:
                select_route_stop(view, route_stops, arrived, open_yaw(yaw_text) or 0.0)
                return
            hit = _hit_used(used, u, v)
            if hit is not None:
                chosen["point"] = hit
            else:
                picked["xy"] = xy

    # 窗口名必须是英文。这版 OpenCV 用 Qt 时按 Latin-1 找窗口，中文名绑不上鼠标。
    window = "field"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        if view["auto_yaw"] is not None:
            yaw_text = str(view["auto_yaw"])
            view["auto_yaw"] = None
        if chosen["point"] is not None:
            hit = chosen["point"]
            chosen["point"] = None
            marker = (hit.x, hit.y)
            yaw_text = hit.yaw_text
            view["pinned"] = False
        elif picked["xy"] is not None:
            marker = picked["xy"]
            picked["xy"] = None
            view["pinned"] = False
        yaw = _yaw_from_text(yaw_text) if marker is not None else None
        heading = 0.0 if yaw is None else yaw
        key = route_position_key(marker)
        # 航向是执行观察动作时的实际朝向，不能使已经规划的路线重新生成。
        if not view["pinned"] and key != route_key:
            route_key = key
            view["look"] = 0
            view["arrived"] = 0 if key is not None else None
            if marker is None:
                route_stops = []
                route_line = []
                route_points = []
            else:
                pose = pose_for_optical(marker[0], marker[1], heading)
                route_stops, line = survey(pose, np.zeros(2, dtype=float), intrinsics)
                route_line = [optical_of(x, y, heading) for x, y in line]
                route_points = [optical_of(stop.x, stop.y, heading) for stop in route_stops]
            view["stops"] = route_points
        arrived = view["arrived"]
        where = camera_point(
            marker,
            route_points,
            arrived if isinstance(arrived, int) else None,
            bool(view["pinned"]),
        )
        lines = []
        if route_stops and isinstance(arrived, int) and 0 <= arrived < len(route_stops):
            lines = stop_lines(route_stops, arrived, int(view["look"]) if view["pinned"] else 0, view.get("arrival_yaw") if view["pinned"] else heading)
        if where is not None:
            lines.insert(0, f"光心  x {where[0]:+.3f} m  y {where[1]:+.3f} m  航向 {yaw_text or '0'}")
        shown = where if view["pinned"] else marker
        shown_yaw = open_yaw(yaw_text) if view["pinned"] else (heading if marker is not None else None)
        cv2.imshow(
            window,
            paint_modes(_draw_map(
                shown,
                shown_yaw,
                pitch,
                hover["xy"],
                yaw_text,
                used,
                route_line,
                route_points,
                arrived if isinstance(arrived, int) else None,
                lines,
            ), args.pose_source),
        )
        action = _yaw_key(cv2.waitKeyEx(30))
        if marker is None:
            if action in ("quit", "esc"):
                break
            earlier = point_at(used, action)
            if earlier is not None:
                marker = (earlier.x, earlier.y)
                yaw_text = earlier.yaw_text
            continue
        if action == "ok":
            opening = open_yaw(yaw_text)
            if where is None or opening is None:
                continue
            used = remember_point(used, where[0], where[1], yaw_text or "0")
            save_points(_POINTS_PATH, used)
            cv2.destroyWindow(window)
            selected_pitch = pitch
            target_lines = []
            if isinstance(arrived, int) and 0 <= arrived < len(route_stops):
                looks = route_stops[arrived].looks
                if looks:
                    selected_pitch = looks[int(view["look"]) % len(looks)].pitch_deg
                target_lines = stop_lines(route_stops, arrived, int(view["look"]), view.get("arrival_yaw"))
            # 固定计划与测试位姿分别传递，实时定位只移动地图标记。
            args.fixed_target_lines = target_lines
            live_heading = {"yaw": opening}
            def select_live_target(event, u, v, _flags, _param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                index = hit_route(route_points, u, v)
                if index is None:
                    return
                select_route_stop(view, route_stops, index, live_heading["yaw"])
                view["auto_yaw"] = None
                args.fixed_target_lines = stop_lines(route_stops, index, int(view["look"]), live_heading["yaw"])
            def live_map(current):
                live_heading["yaw"] = current.yaw
                optical = camera_position(current, np.zeros(2), 0)
                status = [f"实时右前角 x={current.x:+.3f} y={current.y:+.3f} yaw={np.degrees(current.yaw):+.1f}°"]
                cv2.imshow("route", _draw_map(tuple(optical[:2]), current.yaw, selected_pitch,
                    route_line=route_line, route_stops=route_points,
                    arrived=view["arrived"], arrived_lines=status + args.fixed_target_lines))
                cv2.setMouseCallback("route", select_live_target)
            args.live_map = live_map
            _open_camera(where[0], where[1], opening, selected_pitch, args)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window, on_mouse)
            continue
        if action in ("quit", "esc"):
            break
        if action == "back":
            yaw_text = yaw_text[:-1]
        elif action:
            yaw_text = _append_yaw(yaw_text, action)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
