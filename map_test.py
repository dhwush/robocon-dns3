"""不订雷达，在平面图上点光心，再开相机认塔。

python map_test.py

点下去的是相机光心在地面上的位置。航向在这个窗口里输入，单位是度，正数向左转。
回车打开相机，退格改数字。俯仰用 config/field.yaml 里的角度，光心高度仍是 0.70 m。
关掉相机窗口后回到平面图，可以再点下一个位置。q 或 Esc 退出。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from car_pose import OPTICAL_FROM_CORNER, CarPose, Setup, camera_position, load_intrinsics, load_setup
from field_map import TOWER_NO, all_cells, tower_cells
from report import FieldWatch
from view_field import ROOT, _live


# 平面图范围盖住场地和左侧那块木头方，也盖住场地近边前面那一段。
X_MIN = -3.8
X_MAX = 3.6
Y_MIN = -0.8
Y_MAX = 6.4
SCALE = 100
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_COLORS = {
    0.20: (20, 70, 20),
    0.40: (40, 140, 50),
    0.60: (150, 200, 150),
}


def pose_for_optical(x: float, y: float, yaw: float) -> CarPose:
    # 安装差按 0 处理，位姿数字不再代表右前角。反推出来的车位要让光心落在点的位置上。
    yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
    body = OPTICAL_FROM_CORNER
    rotated = np.array(
        [yaw_c * body[0] - yaw_s * body[1], yaw_s * body[0] + yaw_c * body[1]],
        dtype=float,
    )
    origin = np.array([x, y], dtype=float) - rotated
    return CarPose(float(origin[0]), float(origin[1]), float(yaw))


def _pixel(x: float, y: float) -> tuple[int, int]:
    return int(round((x - X_MIN) * SCALE)), int(round((Y_MAX - y) * SCALE))


def _world(u: int, v: int) -> tuple[float, float]:
    return X_MIN + u / SCALE, Y_MAX - v / SCALE


def _canvas() -> np.ndarray:
    width = int(round((X_MAX - X_MIN) * SCALE))
    height = int(round((Y_MAX - Y_MIN) * SCALE))
    board = np.full((height, width, 3), 32, dtype=np.uint8)
    return board


def _cursor_text(x: float, y: float) -> str:
    return f"相对原点  x {x:+.3f} m    y {y:+.3f} m"


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
    ox, oy = _pixel(0.0, 0.0)
    draw.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=(255, 80, 80))
    draw.text((ox + 8, oy - 8), "原点", font=font, fill=(255, 180, 180))
    if marker is not None:
        u, v = _pixel(*marker)
        draw.ellipse((u - 6, v - 6, u + 6, v + 6), outline=(255, 220, 0), width=2)
        if yaw is not None:
            yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
            ahead = (marker[0] - yaw_s * 0.7, marker[1] + yaw_c * 0.7)
            draw.line((u, v, *_pixel(*ahead)), fill=(255, 220, 0), width=2)
        draw.text((u + 8, v + 4), "光心", font=font, fill=(255, 220, 0))
    if marker is None:
        hint = f"点一下光心。然后在窗口里输入航向，正数向左转。俯仰 {pitch:.0f} 度。q 退出"
    else:
        hint = f"航向 {yaw_text} 度，正数向左转。回车打开相机，退格修改。俯仰 {pitch:.0f} 度。q 退出"
    draw.text((8, 8), hint, font=font, fill=(255, 255, 255))
    if hover is not None:
        hx, hy = hover
        label = _cursor_text(hx, hy)
        draw.text((8, 30), label, font=font, fill=(255, 220, 80))
        u, v = _pixel(hx, hy)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(255, 220, 80))
        text_x = u + 10 if u < board.shape[1] - 340 else max(8, u - 330)
        text_y = v - 22 if v > 56 else v + 8
        draw.text((text_x, text_y), label, font=font, fill=(255, 220, 80))
    board[:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return board


def _open_camera(x: float, y: float, yaw: float, pitch: float) -> None:
    pose = pose_for_optical(x, y, yaw)
    offset = np.zeros(2, dtype=float)
    optical = camera_position(pose, offset, pitch)
    print(f"光心 x={optical[0]:.3f} m  y={optical[1]:.3f} m  高 {optical[2]:.2f} m  航向 {np.degrees(yaw):.1f} 度")
    setup = Setup(False, offset, pitch, "", "rad", origin_basic=True)
    watch = FieldWatch(setup, load_intrinsics(str(ROOT / "config" / "intrinsics.yaml")))
    watch.pitch = pitch
    _live(SimpleNamespace(mode="extend"), watch, pose)


def main() -> None:
    import cv2

    pitch = load_setup(str(ROOT / "config" / "field.yaml")).pitch_deg
    picked: dict[str, tuple[float, float] | None] = {"xy": None}
    hover: dict[str, tuple[float, float] | None] = {"xy": None}
    marker: tuple[float, float] | None = None
    yaw_text = ""

    def on_mouse(event, u, v, _flags, _param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            hover["xy"] = _world(u, v)
        elif event == cv2.EVENT_LBUTTONDOWN:
            picked["xy"] = _world(u, v)

    # 窗口名必须是英文。这版 OpenCV 用 Qt 时按 Latin-1 找窗口，中文名绑不上鼠标。
    window = "field"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        if picked["xy"] is not None:
            marker = picked["xy"]
            picked["xy"] = None
        yaw = _yaw_from_text(yaw_text) if marker is not None else None
        cv2.imshow(window, _draw_map(marker, yaw, pitch, hover["xy"], yaw_text))
        action = _yaw_key(cv2.waitKeyEx(30))
        if marker is None:
            if action in ("quit", "esc"):
                break
            continue
        if action == "ok":
            if yaw is None:
                continue
            cv2.destroyWindow(window)
            _open_camera(marker[0], marker[1], yaw, pitch)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window, on_mouse)
            marker = None
            yaw_text = ""
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
