"""合成画面。不打开相机，也不订阅雷达。"""

from __future__ import annotations

import numpy as np

from car_pose import CarPose, Setup, camera_position, closest_pose, load_intrinsics, pose_moved
from map_test import (
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    _append_yaw,
    _cursor_text,
    _snap_cursor,
    _yaw_from_text,
    _yaw_key,
    UsedPoint,
    load_points,
    nearest_green_block,
    point_at,
    pose_for_corner,
    remember_point,
    save_points,
)
from classify import heights_match, inspect_frame
from field_map import MARGIN_M, all_cells, search_contains, tower_cells
from report import FieldWatch, step_light
from sight import sees, visible_height
from view_field import run
from vote import LOCK_FRAMES


INTR = load_intrinsics("config/intrinsics.yaml")
OFFSET = np.array([0.25, -0.15], dtype=float)
START = CarPose(0.0, 0.0, 0.0)


def _setup(aligned: bool = True) -> Setup:
    return Setup(aligned, OFFSET if aligned else None, 0.0, "", "rad")


def _cell(name: str):
    return next(cell for cell in tower_cells() if cell.name == name)


def _spot(pose, pitch, cell, width_m, height_m, z0, offset=None):
    from car_pose import project, to_camera

    used = OFFSET if offset is None else offset
    center = cell.center(z0 + height_m / 2.0)
    pixel = project(center, pose, used, pitch, INTR)
    assert pixel is not None
    distance = float(to_camera(center, pose, used, pitch)[2])
    width = max(2, int(round(width_m * INTR.fx / distance)))
    height = max(2, int(round(height_m * INTR.fy / distance)))
    return int(round(pixel[0])), int(round(pixel[1])), width, height


def _paint(color, pose, pitch, cell, width_m, height_m, z0, bgr, offset=None):
    import cv2

    u, v, width, height = _spot(pose, pitch, cell, width_m, height_m, z0, offset)
    cv2.rectangle(
        color,
        (u - width // 2, v - height // 2),
        (u + width // 2, v + height // 2),
        bgr,
        thickness=-1,
    )


def _paint_disk(color, pose, pitch, cell, width_m, offset=None):
    import cv2

    u, v, width, height = _spot(pose, pitch, cell, width_m, width_m, 0.0, offset)
    cv2.ellipse(color, (u, v), (width // 2, height // 2), 0.0, 0.0, 360.0, (255, 255, 255), thickness=-1)


def _paint_hex(color, pose, pitch, cell, width_m, offset=None):
    import cv2

    u, v, width, _height = _spot(pose, pitch, cell, width_m, width_m, 0.0, offset)
    radius = width / 2.0
    points = []
    for index in range(6):
        angle = -np.pi / 2.0 + index * np.pi / 3.0
        points.append([
            int(round(u + np.cos(angle) * radius)),
            int(round(v + np.sin(angle) * radius)),
        ])
    cv2.fillPoly(color, [np.array(points, dtype=np.int32)], (255, 255, 255))


def _paint_gear(color, pose, pitch, cell, width_m, offset=None):
    import cv2

    u, v, width, height = _spot(pose, pitch, cell, width_m, width_m, 0.0, offset)
    teeth = []
    for index in range(24):
        angle = 2.0 * np.pi * index / 24.0
        scale = 1.0 if index % 2 == 0 else 0.55
        teeth.append([
            int(round(u + np.cos(angle) * width / 2.0 * scale)),
            int(round(v + np.sin(angle) * height / 2.0 * scale)),
        ])
    cv2.fillPoly(color, [np.array(teeth, dtype=np.int32)], (255, 255, 255))


def _blank():
    return np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)


def test_start_camera_stays_on_the_known_optical_center():
    from car_pose import camera_position

    place = camera_position(START, OFFSET, 0.0)
    assert abs(place[0] + 0.211) < 1e-6
    assert abs(place[1] - 0.05) < 1e-6
    assert abs(place[2] - 0.669) < 1e-6


def _basic():
    from car_pose import basic_origin_setup

    setup, pose = basic_origin_setup(0.0)
    return setup, pose, setup.center_to_corner


def test_third_row_hides_one_layer_and_shows_two():
    _, pose, offset = _basic()
    cell = _cell("第三行左侧")
    assert sees(pose, offset, 0.0, cell, 0.55) is False
    assert sees(pose, offset, 0.0, cell, 0.90) is True
    assert visible_height(pose, offset, 0.0, cell, 0.20, 0.55) < 0.08
    assert visible_height(pose, offset, 0.0, cell, 0.20, 0.90) > 0.2
    near = _cell("最近一行右侧")
    assert sees(pose, offset, 0.0, near, near.one_z) is True


def test_platform_tower_is_the_center_half_metre():
    from field_map import TOWER_NO

    assert [TOWER_NO[name] for name in (
        "最近左侧地面",
        "最近一行左侧",
        "第三行左侧",
        "最近一行右侧",
        "最近右侧地面",
        "右侧地面远一格",
    )] == ["1", "2", "3", "4", "5", "6"]
    cell = _cell("最近一行左侧")
    assert cell.size == 1.2 and cell.foot == 0.50
    x0, x1, y0, y1 = cell.search_bounds(0.0)
    assert abs(x1 - x0 - 0.50) < 1e-9 and abs(y1 - y0 - 0.50) < 1e-9
    assert abs((x0 + x1) / 2.0 - cell.x) < 1e-9
    assert search_contains(cell, np.array([cell.x + 0.29, cell.y]), MARGIN_M)
    assert not search_contains(cell, np.array([cell.x + 0.40, cell.y]), MARGIN_M)
    assert _cell("最近左侧地面").foot == 0.50


def test_layer_bands_keep_a_stretched_stack_on_its_own_level():
    assert heights_match(0.35 * 1.5, 0.35)
    assert heights_match(0.35 * 0.5, 0.35)
    assert heights_match(0.70 * 1.5, 0.70)
    assert heights_match(0.70 * 0.5, 0.70)
    assert not heights_match(0.70 * 0.8, 0.35)


def test_wider_white_face_still_counts_until_two_boxes():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.62, 0.35, 0.0, (255, 255, 255), offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "1"
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.98, 0.0, (255, 255, 255), offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "2"
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.70, 0.35, 0.0, (255, 255, 255), offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion is None


def test_yellow_floor_stays_out_of_the_white_box():
    # 黄地面色相大约 16、饱和度大约 49。白盒正面饱和度大约 19。地板铺满时白盒仍按一层写。
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    floor = (109, 123, 135)
    warm = (118, 125, 128)
    from report import plain_state
    from view_field import mark_text

    color = np.full((INTR.height, INTR.width, 3), floor, dtype=np.uint8)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion is None
    assert views[cell.name].method is None
    _paint(color, pose, 0.0, cell, 0.35, 0.35, 0.0, warm, offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "1"
    assert views[cell.name].method == "黄地"
    assert mark_text("6", "1", views[cell.name].method) == "6 一层·黄地"
    assert plain_state(views[cell.name], None, "extend").endswith("，黄地")


def test_far_notch_uses_size_without_straight_edges():
    # 超过 3 m 的黑缝把轮廓咬开。近处仍不写，远处按整块宽高写成两层。
    from classify import _block_shaped

    def _notch(color, pose, offset, cell):
        _paint(color, pose, 0.0, cell, 0.35, 0.70, 0.0, (255, 255, 255), offset=offset)
        white = np.all(color == 255, axis=2)
        ys, xs = np.nonzero(white)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        depth = max(4, int(0.45 * (x1 - x0)))
        thick = max(3, int(0.06 * (y1 - y0)))
        for frac in (0.25, 0.50, 0.75):
            top = y0 + int((y1 - y0) * frac)
            color[top : top + thick, x0 : x0 + depth] = 0
        assert not _block_shaped(np.all(color == 255, axis=2))

    _, pose, offset = _basic()
    near = _cell("最近一行右侧")
    color = _blank()
    _notch(color, pose, offset, near)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[near.name].conclusion is None
    far = _cell("右侧地面远一格")
    color = _blank()
    _notch(color, pose, offset, far)
    view = inspect_frame(pose, offset, 0.0, INTR, color, None)[far.name]
    assert view.conclusion == "2"
    assert view.method == "远处"


def test_near_white_is_one_and_outside_white_is_ignored():
    setup, pose, offset = _basic()
    color = _blank()
    _paint(color, pose, 0.0, _cell("右侧地面远一格"), 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    color[0:80, 0:80] = 255
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views["右侧地面远一格"].conclusion == "1"
    assert views["右侧地面远一格"].method == "彩色"
    assert views["第三行左侧"].conclusion is None
    assert setup.origin_basic is True


def test_near_platform_stays_readable_when_the_base_drops_out():
    # 4 号正对基础车位。台面到搜索框顶塞不满彩色垂直视场，平视时台面在画面下面。
    # 一层顶和塔顶还在，就要能用彩色写下，不把这一格留成只能靠深度。
    _, pose, offset = _basic()
    cell = _cell("最近一行右侧")
    color = _blank()
    depth = np.full((INTR.height, INTR.width), np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, offset, 0.0, cell, 0.20, 0.55)
    view = inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name]
    assert view.in_view
    assert view.conclusion == "1"
    assert view.method == "彩色"
    color = _blank()
    _fill_face(color, depth, pose, offset, 8.0, cell, 0.20, 0.90)
    raised = inspect_frame(pose, offset, 8.0, INTR, color, None)[cell.name]
    assert raised.in_view
    assert raised.conclusion == "2"
    assert raised.method == "彩色"
    color = _blank()
    _fill_face(color, depth, pose, offset, 0.0, cell, 0.20, 0.90)
    white = np.all(color == 255, axis=2)
    top = int(np.nonzero(white)[0].min())
    cols = np.nonzero(white[top])[0]
    color[max(0, top - 30) : top, int(cols.min()) : int(cols.max())] = (0, 0, 255)
    capped = inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name]
    assert capped.conclusion == "红"
    assert capped.method == "彩色"
    blank = inspect_frame(pose, offset, 0.0, INTR, _blank(), None)
    assert blank[cell.name].conclusion is None
    assert blank["第三行左侧"].in_view and blank["右侧地面远一格"].in_view
    for name in ("最近左侧地面", "最近一行左侧", "最近右侧地面"):
        assert blank[name].needs_yaw
        assert not blank[name].in_view


def test_peeking_two_layers_is_not_called_one():
    _, pose, offset = _basic()
    cell = _cell("第三行左侧")
    height = visible_height(pose, offset, 0.0, cell, 0.20, 0.90)
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, height, 0.90 - height, (255, 255, 255), offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "2"


def test_red_cap_uses_the_upper_side():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.70, 0.0, (255, 255, 255), offset=offset)
    _paint(color, pose, 0.0, cell, 0.20, 0.20, 0.70, (0, 0, 255), offset=offset)
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "红"


def test_upper_band_is_the_faction_and_a_tie_stays_unknown():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")

    def stacked(lower_bgr, upper_bgr):
        color = _blank()
        _paint(color, pose, 0.0, cell, 0.35, 0.70, 0.0, (255, 255, 255), offset=offset)
        _paint(color, pose, 0.0, cell, 0.20, 0.10, 0.70, lower_bgr, offset=offset)
        _paint(color, pose, 0.0, cell, 0.20, 0.10, 0.80, upper_bgr, offset=offset)
        return inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion

    assert stacked((255, 0, 0), (0, 0, 255)) == "红"
    assert stacked((0, 0, 255), (255, 0, 0)) == "蓝"
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.70, 0.0, (255, 255, 255), offset=offset)
    _paint(color, pose, 0.0, cell, 0.20, 0.20, 0.70, (0, 0, 255), offset=offset)
    red = np.all(color == (0, 0, 255), axis=2)
    rows, cols = np.nonzero(red)
    right = cols >= np.median(cols)
    color[rows[right], cols[right]] = (255, 0, 0)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None


def _paint_separated_red(color, pose, offset, pitch, cell, z_top):
    from car_pose import to_camera

    white = np.all(color == 255, axis=2)
    top = int(np.nonzero(white)[0].min())
    cols = np.nonzero(white[top])[0]
    distance = float(to_camera(cell.center(z_top), pose, offset, pitch)[2])
    cap_px = max(4, int(round(0.20 * INTR.fy / max(distance, 0.3))))
    color[max(0, top - cap_px) : max(0, top - cap_px // 2), int(cols.min()) : int(cols.max())] = (0, 0, 255)


def test_color_separated_from_the_white_is_not_a_cap():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.70, 0.0, (255, 255, 255), offset=offset)
    _paint_separated_red(color, pose, offset, 0.0, cell, 0.70)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion == "2"


def _shift_white(color, du, dv):
    white = np.all(color == 255, axis=2)
    ys, xs = np.nonzero(white)
    color[ys, xs] = 0
    nys = ys + int(dv)
    nxs = xs + int(du)
    keep = (nys >= 0) & (nys < color.shape[0]) & (nxs >= 0) & (nxs < color.shape[1])
    color[nys[keep], nxs[keep]] = (255, 255, 255)


def test_shifted_block_is_measured_whole_while_its_center_stays_in_the_cell():
    # 框切过白盒时，质心还在格里就按整块的宽高认，不按被框切剩的那一截。
    import cv2

    from classify import _mask

    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    _u, v, width, _height = _spot(pose, 0.0, cell, 0.35, 0.35, 0.0, offset)
    mask = _mask(cell, pose, offset, 0.0, INTR)
    cols = np.nonzero(mask[v])[0]
    assert len(cols) > width
    du = int(cols.max() - width // 4 - _u)
    _shift_white(color, du, 0)
    view = inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name]
    _x, _y, piece_w, _piece_h = cv2.boundingRect(view.piece)
    assert view.conclusion == "1"
    assert view.method == "彩色"
    assert abs(piece_w - width) <= 2


def test_offset_whites_take_the_higher_slice():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    lower = _blank()
    _paint(lower, pose, 0.0, cell, 0.28, 0.28, 0.0, (255, 255, 255), offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, lower, None)[cell.name].conclusion == "1"
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.28, 0.28, 0.0, (255, 255, 255), offset=offset)
    _paint(color, pose, 0.0, cell, 0.28, 0.28, 0.42, (255, 255, 255), offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion == "2"


def test_curved_white_stays_unknown_and_a_straight_block_still_counts():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint_disk(color, pose, 0.0, cell, 0.35, offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None
    color = _blank()
    _paint_gear(color, pose, 0.0, cell, 0.35, offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion == "1"
    assert inspect_frame(pose, offset, 0.0, INTR, _blank(), None)[cell.name].conclusion is None
    color = _blank()
    _paint_hex(color, pose, 0.0, cell, 0.35, offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion == "1"


def test_distant_hex_with_a_burr_still_counts_as_one_layer():
    import cv2

    from classify import EDGE_SIMPLIFY, _close_small, _outline

    _, _, offset = _basic()
    cell = _cell("最近右侧地面")
    # 35 cm 的盒子投成大约 30 像素，和远处看见顶面的一层一样大。
    distance = 0.35 * float(INTR.fx) / 30.0
    pose = _pose_for_camera(cell.x, cell.y - distance, 0.0, offset)
    color = _blank()
    _paint_hex(color, pose, 0.0, cell, 0.35, offset=offset)
    white = np.all(color == 255, axis=2)
    ys, xs = np.nonzero(white)
    x = int(xs.max())
    y = int(np.median(ys[xs == x]))
    cv2.circle(color, (min(color.shape[1] - 1, x + 2), y), 2, (255, 255, 255), thickness=-1)
    white = np.all(color == 255, axis=2)
    closed = _close_small(white)
    contour = _outline(closed.astype(np.uint8) * 255)
    _x, _y, width, height = cv2.boundingRect(contour)
    short = float(min(width, height))
    assert 24 <= short <= 40
    approx = cv2.approxPolyDP(contour, max(2.0, EDGE_SIMPLIFY * short), True)
    assert len(approx) == 7
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion == "1"
    color = _blank()
    _paint_disk(color, pose, 0.0, cell, 0.35, offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None
    color = _blank()
    _paint_gear(color, pose, 0.0, cell, 0.35, offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None


def _pose_for_camera(cam_x, cam_y, yaw, offset):
    from car_pose import OPTICAL_FROM_CORNER

    yaw_c, yaw_s = float(np.cos(yaw)), float(np.sin(yaw))
    body = np.array(
        [offset[0] + OPTICAL_FROM_CORNER[0], offset[1] + OPTICAL_FROM_CORNER[1]],
        dtype=float,
    )
    rotated = np.array([yaw_c * body[0] - yaw_s * body[1], yaw_s * body[0] + yaw_c * body[1]])
    return CarPose(
        float(cam_x + offset[0] - rotated[0]),
        float(cam_y + offset[1] - rotated[1]),
        float(yaw),
    )


def _fill_face(color, depth, pose, offset, pitch, cell, z0, z_top, x_half=0.16):
    # 在格子近侧的正面上铺白，并按这个平面写下深度。出画的角用投影能落下来的点。
    import cv2

    from car_pose import camera_frame, project

    origin, right, down, ahead = camera_frame(pose, offset, pitch)
    y_face = cell.y - 0.16
    pixels = []
    for x in (cell.x - x_half, cell.x + x_half):
        for z in (z0, z_top):
            pixel = project(np.array([x, y_face, z]), pose, offset, pitch, INTR)
            if pixel is not None:
                pixels.append(pixel)
    assert len(pixels) >= 3
    hull = cv2.convexHull(np.round(np.stack(pixels)).astype(np.int32))
    cv2.fillConvexPoly(color, hull, (255, 255, 255))
    ys, xs = np.nonzero(np.all(color == 255, axis=2))
    denom = ((xs - INTR.cx) / INTR.fx) * right[1] + ((ys - INTR.cy) / INTR.fy) * down[1] + ahead[1]
    measured = (y_face - origin[1]) / denom
    cam_x = (xs - INTR.cx) * measured / INTR.fx
    cam_y = (ys - INTR.cy) * measured / INTR.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + measured[:, None] * ahead
    ok = (measured >= 0.3) & (measured <= 3.0)
    ok &= np.abs(world[:, 1] - y_face) < 0.05
    ok &= (world[:, 2] >= z0 - 0.02) & (world[:, 2] <= z_top + 0.02)
    depth[ys[ok], xs[ok]] = measured[ok].astype(np.float32)


def _fill_plane(color, depth, pose, offset, pitch, y_face, z0, z_top, bgr, x_center, x_half=0.16):
    # 在指定的竖直面上铺一块颜色，并写下这个平面的深度。用来放挡在塔前面的箱子。
    import cv2

    from car_pose import camera_frame, project

    origin, right, down, ahead = camera_frame(pose, offset, pitch)
    pixels = []
    for x in (x_center - x_half, x_center + x_half):
        for z in (z0, z_top):
            pixel = project(np.array([x, y_face, z]), pose, offset, pitch, INTR)
            if pixel is not None:
                pixels.append(pixel)
    assert len(pixels) >= 3
    hull = cv2.convexHull(np.round(np.stack(pixels)).astype(np.int32))
    cv2.fillConvexPoly(color, hull, bgr)
    picked = np.all(color == bgr, axis=2)
    ys, xs = np.nonzero(picked)
    denom = ((xs - INTR.cx) / INTR.fx) * right[1] + ((ys - INTR.cy) / INTR.fy) * down[1] + ahead[1]
    measured = (y_face - origin[1]) / denom
    cam_x = (xs - INTR.cx) * measured / INTR.fx
    cam_y = (ys - INTR.cy) * measured / INTR.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + measured[:, None] * ahead
    ok = (measured >= 0.3) & (measured <= 3.0) & (np.abs(denom) > 1e-6)
    ok &= np.abs(world[:, 1] - y_face) < 0.05
    ok &= (world[:, 2] >= z0 - 0.02) & (world[:, 2] <= z_top + 0.02)
    depth[ys[ok], xs[ok]] = measured[ok].astype(np.float32)


def test_box_in_front_of_the_tower_is_not_a_layer():
    from car_pose import Setup
    from report import FieldWatch

    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 1.3, 0.0, OFFSET)
    pitch = -8.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.02, 0.35)
    tower = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert tower.conclusion == "1"
    assert tower.skip_vote is False
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_plane(color, depth, pose, OFFSET, pitch, cell.y - 0.75, 0.05, 0.45, (255, 255, 255), cell.x)
    view = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert view.conclusion is None
    assert view.skip_vote
    assert view.blocked is False
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_plane(color, depth, pose, OFFSET, pitch, cell.y - 0.16, 0.10, 0.45, (40, 70, 120), cell.x, 0.18)
    piled = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert piled.conclusion is None
    assert piled.skip_vote
    watch = FieldWatch(Setup(True, OFFSET, pitch, "", "rad"), INTR)
    face = _blank()
    face_depth = np.full(face.shape[:2], np.nan, dtype=np.float32)
    _fill_face(face, face_depth, pose, OFFSET, pitch, cell, 0.02, 0.35)
    for _ in range(10):
        watch.update(pose, face, face_depth)
    assert watch.votes[cell.name].count == 5
    watch.update(pose, color, depth)
    assert watch.votes[cell.name].count == 5
    assert watch.votes[cell.name].locked is None
    assert "前面有东西挡住了" in "\n".join(watch.window_lines("extend"))


def test_depth_outside_range_does_not_cancel_color():
    _, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    views = inspect_frame(pose, offset, 0.0, INTR, color, depth)
    assert views[cell.name].conclusion == "1"
    depth[:] = 1.0
    views = inspect_frame(pose, offset, 0.0, INTR, color, depth)
    assert views[cell.name].conclusion == "1"
    assert views[cell.name].method == "彩色"


def test_close_face_uses_depth_and_names_it():
    from view_field import mark_text

    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 0.55, 0.0, OFFSET)
    pitch = -25.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.02, 0.35)
    missed = inspect_frame(pose, OFFSET, pitch, INTR, color, None)
    assert missed[cell.name].conclusion is None
    found = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)
    assert found[cell.name].conclusion == "1"
    assert found[cell.name].method == "深度"
    assert mark_text("5", found[cell.name].conclusion, found[cell.name].method) == "5 一层·深度"


def _drop_depth_above(depth, pose, offset, pitch, z_max):
    from car_pose import camera_frame

    origin, right, down, ahead = camera_frame(pose, offset, pitch)
    ys, xs = np.nonzero(np.isfinite(depth))
    values = depth[ys, xs].astype(np.float64)
    cam_x = (xs - INTR.cx) * values / INTR.fx
    cam_y = (ys - INTR.cy) * values / INTR.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + values[:, None] * ahead
    drop = world[:, 2] > z_max
    depth[ys[drop], xs[drop]] = np.nan


def test_missing_top_depth_on_a_near_two_layer_still_counts():
    # 顶面没有深度时，最高一截会停在一层和两层的容差中间，或只量到下层。仍写两层。
    _, pose, offset = _basic()
    cell = _cell("最近一行右侧")
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, offset, 0.0, cell, 0.20, 0.70)
    gap = inspect_frame(pose, offset, 0.0, INTR, color, depth)[cell.name]
    assert gap.conclusion is None
    assert gap.method is None
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, offset, 0.0, cell, 0.20, 0.90)
    _drop_depth_above(depth, pose, offset, 0.0, 0.60)
    lifted = inspect_frame(pose, offset, 0.0, INTR, color, depth)[cell.name]
    assert lifted.conclusion == "2"
    assert lifted.method == "深度"
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, offset, 0.0, cell, 0.20, 0.55)
    one = inspect_frame(pose, offset, 0.0, INTR, color, depth)[cell.name]
    assert one.conclusion == "1"


def test_jagged_face_still_uses_depth_when_the_top_matches():
    # 两层白盒整块在画面里，轮廓和木板粘成锯齿。深度对上 70 cm 就写两层。
    import cv2

    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 0.8, 0.0, OFFSET)
    pitch = -15.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.35, 0.70)
    white = np.all(color == 255, axis=2)
    ys, xs = np.nonzero(white)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    for index, x in enumerate(range(x0, x1, max(1, (x1 - x0) // 8))):
        reach = 36 if index % 2 == 0 else 18
        cv2.line(color, (x, y0), (x, max(0, y0 - reach)), (255, 255, 255), 2)
    assert inspect_frame(pose, OFFSET, pitch, INTR, color, None)[cell.name].conclusion is None
    view = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert view.conclusion == "2"
    assert view.method == "深度"


def test_cut_off_top_stays_unknown():
    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 0.8, 0.0, OFFSET)
    pitch = -30.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.2, 0.9)
    view = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert view.conclusion is None
    assert view.method is None


def test_depth_sees_the_cap_color():
    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 0.8, 0.0, OFFSET)
    pitch = -15.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.2, 0.70)
    white = np.all(color == 255, axis=2)
    rows = np.nonzero(white)[0]
    top = int(rows.min())
    cols = np.nonzero(white[top])[0]
    color[max(0, top - 30) : top, int(cols.min()) : int(cols.max())] = (0, 0, 255)
    view = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert view.conclusion == "红"
    assert view.method == "深度"


def test_depth_ignores_color_separated_from_the_white():
    cell = _cell("最近右侧地面")
    pose = _pose_for_camera(cell.x, cell.y - 0.8, 0.0, OFFSET)
    pitch = -15.0
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.2, 0.70)
    _paint_separated_red(color, pose, OFFSET, pitch, cell, 0.70)
    view = inspect_frame(pose, OFFSET, pitch, INTR, color, depth)[cell.name]
    assert view.conclusion == "2"
    assert view.method == "深度"


def test_depth_lock_stays_out_of_the_judge_line():
    from car_pose import Setup
    from report import FieldWatch

    cell = _cell("最近右侧地面")
    pitch = -25.0
    pose = _pose_for_camera(cell.x, cell.y - 0.55, 0.0, OFFSET)
    color = _blank()
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    _fill_face(color, depth, pose, OFFSET, pitch, cell, 0.02, 0.35)
    watch = FieldWatch(Setup(True, OFFSET, pitch, "", "rad"), INTR)
    for _ in range(LOCK_FRAMES + 4):
        watch.update(pose, color, depth)
    assert watch.votes[cell.name].locked is None
    watch.update(pose, color, depth)
    lines = watch.lines("extend")
    assert f"{cell.name} 1" in lines
    assert all("深度" not in line for line in lines)
    assert "5：一层白盒，深度" in watch.window_lines("extend")


def test_narrow_white_stays_unknown_and_does_not_keep_a_wrong_empty():
    setup, pose, offset = _basic()
    cell = _cell("右侧地面远一格")
    short = _blank()
    _paint(short, pose, 0.0, cell, 0.10, 0.35, 0.0, (255, 255, 255), offset=offset)
    assert inspect_frame(pose, offset, 0.0, INTR, short, None)[cell.name].conclusion is None
    color = _blank()
    _paint(color, pose, 0.0, cell, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    u, v, width, height = _spot(pose, 0.0, cell, 0.35, 0.35, 0.0, offset)
    x0, y0 = u - width // 2, v - height // 2
    gap = max(3, width // 20)
    for center in (x0 + width // 3, x0 + 2 * width // 3):
        color[y0 : y0 + height, center - gap // 2 : center + gap // 2 + 1] = 0
    assert inspect_frame(pose, offset, 0.0, INTR, color, None)[cell.name].conclusion is None
    watch = FieldWatch(setup, INTR)
    for _ in range(LOCK_FRAMES + 5):
        watch.update(pose, _blank())
    assert watch.votes[cell.name].locked is None
    # 模拟此前由有效空台面证据锁定的旧状态，检查冲突重看。
    watch.votes[cell.name].locked = "0"
    one = _blank()
    _paint(one, pose, 0.0, cell, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for _ in range(LOCK_FRAMES - 1):
        watch.update(pose, one)
        watch.update(pose, short)
    assert watch.votes[cell.name].locked == "0"
    watch.update(pose, one)
    vote = watch.votes[cell.name]
    assert vote.locked == "0"
    assert vote.recheck
    vote.ready = True
    for _ in range(LOCK_FRAMES + 5):
        vote.see("1")
    assert vote.locked == "1"
    assert vote.recheck is False
    for name, other in watch.votes.items():
        if name != cell.name:
            other.locked = "0"
            other.recheck = False
    assert watch.lines("extend")[-1] == "6 1"
    watch.update(pose, _blank())
    assert watch.votes[cell.name].locked == "1"


def test_frames_lock_and_one_frame_does_not_unlock():
    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    color = _blank()
    _paint(color, pose, 0.0, _cell("右侧地面远一格"), 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for _ in range(LOCK_FRAMES + 4):
        watch.update(pose, color)
    assert watch.votes["右侧地面远一格"].locked is None
    watch.update(pose, color)
    assert watch.votes["右侧地面远一格"].locked == "1"
    watch.update(pose, _blank())
    assert watch.votes["右侧地面远一格"].locked == "1"


def test_jitter_does_not_reset_the_count():
    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    color = _blank()
    _paint(color, pose, 0.0, _cell("右侧地面远一格"), 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for index in range(LOCK_FRAMES + 5):
        shifted = CarPose(pose.x + 0.01 * (index % 2), pose.y, pose.yaw)
        assert pose_moved(pose, shifted) is False or index == 0
        watch.update(shifted, color)
    assert watch.votes["右侧地面远一格"].locked == "1"


def test_extend_without_alignment_prints_only_the_warning():
    watch = FieldWatch(_setup(False), INTR)
    watch.update(START, _blank())
    assert watch.lines("extend") == ["车位还没对齐"]


def _field_args(**overrides):
    values = {
        "mode": "basic",
        "image": "",
        "pose": [0.0, 0.0, 0.0],
        "pitch": None,
        "aligned": False,
        "center_to_corner": None,
        "topic": "",
        "yaw_unit": "",
        "once": True,
        "pose_is_default": True,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_basic_field_at_origin_does_not_use_radar():
    from car_pose import basic_origin_setup, camera_position

    setup, pose = basic_origin_setup(0.0)
    place = camera_position(pose, setup.center_to_corner, 0.0)
    assert abs(pose.x - 1.2) < 1e-6
    assert abs(pose.y) < 1e-6
    assert abs(pose.yaw) < 1e-6
    assert abs(place[0] - (1.2 - 0.211)) < 1e-6
    assert abs(place[1] - 0.05) < 1e-6
    assert abs(place[2] - 0.669) < 1e-6
    assert setup.topic == ""
    assert setup.aligned is False
    watch = FieldWatch(setup, INTR)
    color = _blank()
    cell = _cell("右侧地面远一格")
    _paint(
        color,
        pose,
        0.0,
        cell,
        0.35,
        0.35,
        0.0,
        (255, 255, 255),
        offset=setup.center_to_corner,
    )
    for _ in range(LOCK_FRAMES + 5):
        watch.update(pose, color)
    lines = watch.lines("basic")
    assert "右侧地面远一格 1" in lines
    assert "第三行左侧 0" not in lines
    assert "车位还没对齐" not in lines
    assert all("平移" not in line and "度" not in line for line in lines)
    assert run(_field_args(mode="basic", topic="/radar/odom")) == []


def test_extend_field_still_waits_for_radar():
    lines = run(_field_args(mode="extend"))
    assert lines == ["车位还没对齐"]


def test_basic_and_extend_commands_print_different_lines():
    watch = FieldWatch(_setup(), INTR)
    watch.update(CarPose(0.0, 0.0, 1.2), None)
    basic = "\n".join(watch.lines("basic"))
    extend = watch.lines("extend")
    assert "平移" not in basic
    assert "向左" not in basic and "向右" not in basic
    assert "抬头" not in basic and "低头" not in basic
    assert len(extend) == 7
    # 定位改为右前角后，此场景先向右转，对准下一格。
    assert extend[-1] == "向右转 35 度"
    assert "度" in extend[-1]
    assert "还没看清" in "\n".join(extend)


def test_pitch_comes_before_yaw_after_the_current_view_is_read():
    from classify import CellView
    from motion import next_action

    def view(**kwargs):
        fields = dict(
            conclusion=None,
            in_view=False,
            needs_yaw=False,
            needs_pitch=False,
            blocked=False,
            bearing_deg=0.0,
            pitch_target_deg=0.0,
            white_frac=0.0,
            mean_v=255.0,
        )
        fields.update(kwargs)
        return CellView(**fields)

    names = [cell.name for cell in tower_cells()]
    locked = {name: "0" for name in names}
    views = {name: view() for name in names}
    open_name, pitch_name, yaw_name = names[0], names[3], names[1]
    locked[open_name] = None
    locked[pitch_name] = None
    locked[yaw_name] = None
    views[open_name] = view(in_view=True, conclusion="1")
    views[pitch_name] = view(needs_pitch=True, pitch_target_deg=12.0)
    views[yaw_name] = view(needs_yaw=True, bearing_deg=-38.0)
    pose = CarPose(0.0, 0.0, 0.0)
    assert next_action(views, locked, pose, OFFSET, 0.0, True) is None
    locked[open_name] = "1"
    assert next_action(views, locked, pose, OFFSET, 0.0, True) == "抬头到 12 度"
    locked[pitch_name] = "0"
    assert next_action(views, locked, pose, OFFSET, 0.0, True) == "向左转 38 度"


def test_remaining_view_stays_put_and_exhausted_view_moves():
    from classify import CellView
    from motion import MAX_TRIES, _bypass_points, _ground_blocked, _travel, fits_when_aimed, next_action, plan_action

    pose = CarPose(0.0, 0.0, 0.0)
    names = [cell.name for cell in tower_cells()]
    target = names[0]
    cell = _cell(target)
    assert fits_when_aimed(cell, pose, OFFSET, INTR)
    locked = {name: "0" for name in names}
    locked[target] = None
    views = {
        name: CellView(None, False, False, False, False, 0.0, 0.0, 0.0, 255.0) for name in names
    }
    views[target] = CellView(None, False, True, False, False, 40.0, -6.0, 0.0, 255.0)
    text = next_action(views, locked, pose, OFFSET, 0.0, True, INTR, {target: MAX_TRIES - 1})
    assert text is not None
    assert "平移" not in text
    assert "度" in text
    fresh = plan_action(views, locked, pose, OFFSET, 0.0, True, INTR, {target: MAX_TRIES})
    assert fresh is not None and fresh.kind == "move" and fresh.final_x is not None and fresh.final_y is not None
    new_trip = _travel(pose.x, pose.y, fresh.final_x, fresh.final_y, _bypass_points())
    assert new_trip is not None
    near = far = None
    for x in np.linspace(-2.0, 1.5, 15):
        for y in np.linspace(-1.5, 0.6, 10):
            if _ground_blocked(float(x), float(y)):
                continue
            stood = CarPose(float(x), float(y), 0.0)
            if not fits_when_aimed(cell, stood, OFFSET, INTR):
                continue
            trip = _travel(pose.x, pose.y, float(x), float(y), _bypass_points())
            if trip is None:
                continue
            if trip[0] + 0.3 < new_trip[0]:
                near = (float(x), float(y))
            elif trip[0] > new_trip[0] + 0.3:
                far = (float(x), float(y))
    assert near is not None and far is not None
    closer = plan_action(views, locked, pose, OFFSET, 0.0, True, INTR, {target: MAX_TRIES}, [near])
    assert closer is not None and closer.final_x is not None
    assert abs(closer.final_x - near[0]) < 1e-6 and abs(closer.final_y - near[1]) < 1e-6
    farther = plan_action(views, locked, pose, OFFSET, 0.0, True, INTR, {target: MAX_TRIES}, [far])
    assert farther is not None and farther.final_x is not None
    assert abs(farther.final_x - fresh.final_x) < 1e-6
    assert abs(farther.final_y - fresh.final_y) < 1e-6


def test_blocked_cell_asks_for_one_slide_when_the_view_is_open():
    watch = FieldWatch(_setup(), INTR)
    watch.update(START, None)
    lines = watch.lines("extend")
    assert lines[-1].startswith("向")
    assert "度" in lines[-1]


def test_visible_cells_lock_before_the_slide():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    for _ in range(LOCK_FRAMES + 4):
        watch.update(START, color)
    assert all(not line.endswith("米") for line in watch.lines("extend"))
    watch.update(START, color)
    assert "度" in watch.lines("extend")[-1]


def test_done_place_names_the_next_move_after_recognition():
    from motion import fits_when_aimed

    watch = FieldWatch(_setup(), INTR)
    watch.update(START, None)
    for cell in tower_cells():
        if fits_when_aimed(cell, START, OFFSET, INTR):
            watch.votes[cell.name].locked = "0"
    lines = watch.lines("extend")
    assert "完成识别" in lines
    assert "平移" in lines[-1]
    assert lines.index("完成识别") < len(lines) - 1


def test_all_locked_says_stop():
    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    color = _blank()
    near = _cell("右侧地面远一格")
    _paint(color, pose, 0.0, near, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for _ in range(LOCK_FRAMES + 5):
        watch.update(pose, color)
    for name, vote in watch.votes.items():
        if vote.locked is None:
            vote.locked = "0"
    lines = watch.lines("extend")
    assert "完成识别" in lines
    assert "停止" in lines
    assert lines[lines.index("停止") + 1 :] == ["总结", "1 0", "2 0", "3 0", "4 0", "5 0", "6 1"]
    basic = watch.lines("basic")
    assert all("平移" not in line and "度" not in line for line in basic)
    assert any(line.startswith("右侧地面远一格 1") for line in basic)


def test_window_explains_the_tower_and_keeps_a_chosen_light():
    from car_pose import basic_origin_setup
    from report import light_from_key, light_hint
    from view_field import paint_window

    assert light_from_key(ord("1")) == "bright"
    assert light_from_key(ord("5")) == "dark"
    assert light_from_key(ord("q")) is None
    assert "1很亮" in light_hint("daylight")
    assert "5很暗" in light_hint("daylight")
    setup, pose = basic_origin_setup(0.0)
    watch = FieldWatch(setup, INTR)
    color = _blank()
    cell = _cell("右侧地面远一格")
    _paint(
        color,
        pose,
        0.0,
        cell,
        0.35,
        0.35,
        0.0,
        (255, 255, 255),
        offset=setup.center_to_corner,
    )
    seen = inspect_frame(pose, setup.center_to_corner, 0.0, INTR, color, None)
    assert seen["右侧地面远一格"].region is not None
    assert seen["右侧地面远一格"].piece is not None
    for _ in range(LOCK_FRAMES + 5):
        watch.update(pose, color)
    text = "\n".join(watch.window_lines("basic"))
    assert "6：一层白盒" in text
    assert "按 1很亮" in text
    assert "平移" not in text
    assert "右侧地面远一格 1" in watch.lines("basic")
    assert "第三行左侧 0" not in watch.lines("basic")
    watch.choose_light("dark")
    flooded = np.full_like(color, 255)
    watch.update(pose, flooded)
    assert watch.light == "dark"
    board = paint_window(
        color,
        watch.views,
        {name: vote.locked for name, vote in watch.votes.items()},
        watch.window_lines("basic"),
    )
    assert board.shape == color.shape
    turned = FieldWatch(_setup(), INTR)
    turned.update(START, None)
    extend = turned.window_lines("extend")
    assert any("转向" in line for line in extend)
    assert "度" in extend[-1]


def test_light_steps_once_and_then_waits():
    level, wait = step_light("daylight", 0.0, 10.0, 0)
    assert level == "indoor"
    assert wait == 5
    level, wait = step_light(level, 0.0, 10.0, wait)
    assert level == "indoor"
    assert wait == 4
    level, wait = step_light("daylight", 0.9, 200.0, 0)
    assert level == "bright"


def test_origin_sits_outside_the_near_forty_platform():
    cells = {cell.name: cell for cell in tower_cells()}
    platform = next(cell for cell in all_cells() if cell.row == 3 and cell.col == 1)
    x0, _x1, y0, _y1 = platform.bounds(0.0)
    assert platform.height == 0.40 and platform.tower is False
    assert abs(x0 - (-0.615)) < 1e-9
    assert abs(y0 - 0.854) < 1e-9
    four = _cell("最近一行左侧")
    three = _cell("最近一行右侧")
    one = cells["最近左侧地面"]
    five = cells["最近右侧地面"]
    six = cells["右侧地面远一格"]
    assert one.ground and five.ground and six.ground
    assert one.size == five.size == six.size == 0.50
    one_x0, one_x1, one_y0, one_y1 = one.bounds(0.0)
    five_x0, five_x1, five_y0, five_y1 = five.bounds(0.0)
    six_x0, six_x1, six_y0, six_y1 = six.bounds(0.0)
    four_x0, _, four_y0, _ = four.bounds(0.0)
    _, three_x1, three_y0, _ = three.bounds(0.0)
    sixty = next(cell for cell in all_cells() if cell.row == 1 and cell.col == 2)
    _, sixty_x1, sixty_y0, _ = sixty.bounds(0.0)
    assert sixty.height == 0.60
    assert abs(one_x1 - (four_x0 - 0.76)) < 1e-9 and abs(one_y0 - (four_y0 + 0.10)) < 1e-9
    assert abs(one_y1 - one_y0 - 0.50) < 1e-9 and abs(one_x1 - one_x0 - 0.50) < 1e-9
    assert abs(five_x0 - (three_x1 + 0.535)) < 1e-9 and abs(five_y0 - (three_y0 + 0.385)) < 1e-9
    assert abs(five_x1 - five_x0 - 0.50) < 1e-9 and abs(five_y1 - five_y0 - 0.50) < 1e-9
    assert abs(six_x0 - (sixty_x1 + 0.615)) < 1e-9 and abs(six_y0 - (sixty_y0 + 0.46)) < 1e-9
    assert abs(six_x1 - six_x0 - 0.50) < 1e-9 and abs(six_y1 - six_y0 - 0.50) < 1e-9
    for spot in (one, five, six):
        sx0, sx1, sy0, sy1 = spot.bounds(0.0)
        for platform in all_cells():
            px0, px1, py0, py1 = platform.bounds(0.0)
            assert sx1 <= px0 or px1 <= sx0 or sy1 <= py0 or py1 <= sy0
    assert one.one_z == 0.35 and one.two_z == 0.70 and one.cap_z == 0.90
    assert four.one_z == 0.55 and four.two_z == 0.90 and four.cap_z == 1.10
    old = next(cell for cell in all_cells() if cell.row == 0 and cell.col == 1)
    assert old.tower is False and old.height == 0.20
    assert abs(old.x - (-0.015)) < 1e-6 and abs(old.y - 5.054) < 1e-6


def test_yaw_is_typed_in_the_map_window():
    text = ""
    for char in "30":
        assert _yaw_key(ord(char)) == char
        text = _append_yaw(text, char)
    assert abs(np.degrees(_yaw_from_text(text)) - 30.0) < 1e-9
    text = _append_yaw(text, _yaw_key(ord(".")))
    text = _append_yaw(text, ".")
    assert text == "30."
    text = _append_yaw("-", "-")
    assert text == "-"
    assert _yaw_from_text(text) is None
    assert _yaw_key(13) == "ok"
    assert _yaw_key(0x01000004) == "ok"
    assert _yaw_key(0x01000003) == "back"
    assert _yaw_key(27) == "esc"


def test_used_points_can_be_chosen_again():
    from map_test import _hit_used, _pixel

    first = remember_point([], 1.2, 0.4, "30")
    second = remember_point(first, -0.5, 1.0, "-15")
    assert second[0] == UsedPoint(-0.5, 1.0, "-15")
    assert second[1] == UsedPoint(1.2, 0.4, "30")
    # 同一处、同一航向再记一次，只留最新的那条。
    again = remember_point(second, 1.2004, 0.4004, "30.0")
    assert again[0].yaw_text == "30.0"
    assert len(again) == 2
    # 航向不同就另外留着。
    turned = remember_point(again, 1.2, 0.4, "40")
    assert len(turned) == 3
    crowded = []
    for index in range(12):
        crowded = remember_point(crowded, float(index), 0.0, "0")
    assert len(crowded) == 9
    assert crowded[0].x == 11.0
    assert point_at(second, "1") == second[0]
    assert point_at(second, "2") == second[1]
    assert point_at(second, "3") is None
    assert point_at(second, "0") is None
    u, v = _pixel(1.2, 0.4)
    assert _hit_used(second, u, v) == second[1]
    assert _hit_used(second, u + 30, v) is None
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "map_points.txt"
        save_points(path, second)
        assert load_points(path) == second
        path.write_text("# coordinates: right_front_corner\n坏行\n1.2 0.4\n0 1 -\n0.2 0.3 10\n", encoding="utf-8")
        assert load_points(path) == [UsedPoint(0.2, 0.3, "10")]


def test_cursor_snaps_onto_a_nearby_block_edge():
    from field_map import ORIGIN_X, ORIGIN_Y

    # 最近一行左缘。点在方块外面，仍吸到这条边的延长线上。
    x, y = _snap_cursor(ORIGIN_X + 0.008, -0.2)
    assert x == ORIGIN_X
    assert y == -0.2
    # 拐角两轴一起吸。
    x, y = _snap_cursor(ORIGIN_X - 0.006, ORIGIN_Y + 0.006)
    assert x == ORIGIN_X and y == ORIGIN_Y
    # 台心 50 cm 的边也吸。最近一行左侧格心 x = -1.215，右缘是 -0.965。
    x, y = _snap_cursor(-0.965 + 0.004, 1.30)
    assert abs(x - (-0.965)) < 1e-9
    assert y == 1.30
    # 离开 1 cm 就不吸。
    x, y = _snap_cursor(ORIGIN_X + 0.02, ORIGIN_Y + 0.02)
    assert abs(x - (ORIGIN_X + 0.02)) < 1e-9
    assert abs(y - (ORIGIN_Y + 0.02)) < 1e-9


def test_cursor_distance_uses_the_nearest_green_block():
    base = _cell("最近一行左侧")
    _x0, _x1, y0, _y1 = base.bounds(0.0)
    cell, distance, point = nearest_green_block(base.x, y0 - 1.0)
    assert cell == base
    assert abs(distance - 1.0) < 1e-9
    assert point == (base.x, y0)
    cell, distance, _point = nearest_green_block(base.x, base.y)
    assert distance == 0.0
    assert "20 cm 绿色方块地面 0.000 m" in _cursor_text(base.x, base.y)
    # 站在 40 cm 绿台上，距离是 0，不再量到旁边的建塔区。
    middle = next(cell for cell in all_cells() if cell.row == 3 and cell.col == 1)
    cell, distance, _point = nearest_green_block(middle.x, middle.y)
    assert cell == middle and distance == 0.0
    assert "40 cm 绿色方块地面 0.000 m" in _cursor_text(middle.x, middle.y)
    # 最远正中那块 20 cm 绿台不建塔，光心在它北面时仍要算上这块地面。
    plain = next(cell for cell in all_cells() if cell.row == 0 and cell.col == 1)
    _x0, _x1, _y0, far_y = plain.bounds(0.0)
    cell, distance, _point = nearest_green_block(plain.x, far_y + 1.0)
    assert cell == plain and abs(distance - 1.0) < 1e-9
    assert "20 cm 绿色方块地面" in _cursor_text(plain.x, far_y + 1.0)


def test_black_ground_extends_three_meters_past_the_blocks():
    xs = []
    ys = []
    for cell in list(all_cells()) + list(tower_cells()):
        x0, x1, y0, y1 = cell.bounds(0.0)
        xs.extend((x0, x1))
        ys.extend((y0, y1))
    assert abs(X_MIN - (min(xs) - 3.0)) < 1e-9
    assert abs(X_MAX - (max(xs) + 3.0)) < 1e-9
    assert abs(Y_MIN - (min(ys) - 3.0)) < 1e-9
    assert abs(Y_MAX - (max(ys) + 3.0)) < 1e-9


def test_blocked_cell_moves_to_a_clear_landing():
    from car_pose import bearing_deg, camera_position
    from classify import CellView
    from motion import _ground_blocked, fits_when_aimed, plan_action

    def view(**kwargs):
        fields = dict(
            conclusion=None,
            in_view=False,
            needs_yaw=False,
            needs_pitch=False,
            blocked=False,
            bearing_deg=0.0,
            pitch_target_deg=0.0,
            white_frac=0.0,
            mean_v=255.0,
        )
        fields.update(kwargs)
        return CellView(**fields)

    names = [cell.name for cell in tower_cells()]
    six = "右侧地面远一格"
    locked = {name: "0" for name in names}
    locked[six] = None
    views = {name: view(in_view=True, conclusion="0") for name in names}
    views[six] = view(blocked=True, in_view=True)
    pose = CarPose(0.0, 0.0, 0.0)
    action = plan_action(views, locked, pose, OFFSET, 0.0, True, INTR)
    assert action is not None and action.kind == "move"
    assert "平移" in action.text
    landed = CarPose(action.x, action.y, pose.yaw)
    cam = camera_position(landed, OFFSET, 0.0)
    assert not _ground_blocked(float(cam[0]), float(cam[1]))
    assert not _ground_blocked(landed.x, landed.y)
    # action.x/y 可能是绕障折点；最终观察点必须能看清。
    final = CarPose(action.final_x, action.final_y, pose.yaw)
    assert fits_when_aimed(_cell(six), final, OFFSET, INTR)
    # 身后的格子按真实转角转，不再被夹成 90 度。
    behind = CarPose(0.0, 0.0, np.pi)
    assert abs(bearing_deg(_cell(six).center(0.35), behind, OFFSET)) > 90.0


def test_too_close_to_fit_asks_for_a_move():
    from classify import inspect_frame
    from motion import plan_action

    # 旧点位虽未压到台，35 cm 车身的旋转包络已碰台，必须保持原位。
    pose = CarPose(1.185 + 0.211 - 0.25, 1.454 - 0.25 - 0.5 - 0.05 + 0.15, 0.0)
    views = inspect_frame(pose, OFFSET, 0.0, INTR, None)
    names = [cell.name for cell in tower_cells()]
    locked = {name: "0" for name in names}
    locked["最近一行右侧"] = None
    action = plan_action(views, locked, pose, OFFSET, 0.0, True, INTR)
    assert action is not None and action.kind == "blocked"
    assert "保持原位" in action.text


def test_command_stays_until_the_move_is_finished():
    watch = FieldWatch(_setup(), INTR)
    watch.update(START, None)
    first = watch.lines("extend")[-1]
    second = watch.lines("extend")[-1]
    assert first == second
    assert "度" in first or "平移" in first


def test_map_route_lists_heading_and_pitch_when_a_stop_is_clicked():
    from map_test import hit_route, stop_lines, _pixel
    from motion import survey

    pose = pose_for_corner(0.0, 0.0, 0.0)
    stops, line = survey(pose, np.zeros(2), INTR)
    assert stops[0].x == pose.x and stops[0].y == pose.y
    assert line[0] == (pose.x, pose.y)
    points = [(stop.x, stop.y) for stop in stops]
    assert points[0] == (0.0, 0.0)
    u, v = _pixel(*points[0])
    assert hit_route(points, u, v) == 0
    text = " ".join(stop_lines(stops, 0))
    assert "已到达" in text
    assert "完成识别" in text
    if stops[0].looks:
        assert "朝向" in text and "度" in text
    covered = {look.number for stop in stops for look in stop.looks}
    assert covered
    if covered >= set("123456"):
        assert "停止" in " ".join(stop_lines(stops, len(stops) - 1))


def test_selecting_a_stop_uses_its_calculated_heading():
    from map_test import look_yaw_text, stop_lines
    from motion import Look, RouteStop

    stops = [
        RouteStop(0.0, 0.0, (Look("2", 60.0, 3.0), Look("4", -28.4, 6.0))),
        RouteStop(3.0, 0.2, (Look("6", -0.4, -2.0),)),
    ]
    assert look_yaw_text(stops, 1, 0) == "-0.4"
    assert look_yaw_text(stops, 0, 1) == "-28.4"
    text = " ".join(stop_lines(stops, 0, 1))
    assert "4  朝向 向右 28 度" in text
    assert "当前" in text.split("4  朝向")[1]


def test_pinned_stop_keeps_its_coordinates_for_the_camera():
    from map_test import corner_point, open_yaw

    start = (0.2, -0.4)
    second = (3.165, 0.284)
    assert corner_point(start, [start, second], 1, True) == second
    assert corner_point(start, [start, second], 1, False) == start
    assert open_yaw("") == 0.0
    assert open_yaw("-") is None
    assert abs(open_yaw("-12") + np.deg2rad(12.0)) < 1e-9


def test_map_click_lands_on_the_right_front_corner():
    yaw = np.deg2rad(30.0)
    pose = pose_for_corner(1.2, 0.4, yaw)
    optical = camera_position(pose, np.zeros(2), 0.0)
    assert np.allclose([pose.x, pose.y], [1.2, 0.4])
    assert not np.allclose(optical[:2], [pose.x, pose.y])
    straight = pose_for_corner(0.5, 1.0, 0.0)
    assert np.allclose([straight.x, straight.y], [0.5, 1.0])


def test_live_pitch_reads_look_up_as_positive():
    from car_pose import pitch_from_gravity, project, roll_from_gravity
    from report import live_pitch_line

    assert pitch_from_gravity(None) is None
    assert roll_from_gravity(None) is None
    assert abs(pitch_from_gravity(np.array([0.0, 1.0, 0.0]))) < 1e-6
    assert abs(roll_from_gravity(np.array([0.0, 1.0, 0.0]))) < 1e-6
    look_up = np.array([0.0, np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0))])
    look_down = np.array([0.0, np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
    assert abs(pitch_from_gravity(look_up) - 30.0) < 1e-6
    assert abs(pitch_from_gravity(look_down) + 30.0) < 1e-6
    right_down = np.deg2rad(20.0)
    rolled = np.array([np.sin(right_down), np.cos(right_down), 0.0])
    assert abs(roll_from_gravity(rolled) - 20.0) < 1e-6
    _, pose, offset = _basic()
    point = _cell("右侧地面远一格").center(0.35)
    flat = project(point, pose, offset, 0.0, INTR)
    tilted = project(point, pose, offset, 0.0, INTR, 20.0)
    assert flat is not None and tilted is not None
    assert tilted[1] < flat[1] - 1.0
    assert live_pitch_line(None) == "俯仰读不出来"
    assert live_pitch_line(12.34) == "俯仰 抬头 12.3 度"
    assert live_pitch_line(-3.0) == "俯仰 低头 3.0 度"
    assert live_pitch_line(0.0) == "俯仰 0.0 度"
    from report import required_pitch_line, window_pitch_lines

    assert required_pitch_line(None) is None
    assert required_pitch_line(12.0) == "要求 抬头到 12 度"
    assert required_pitch_line(-6.0) == "要求 低头到 6 度"
    assert window_pitch_lines(-6.0, 2.0) == ["俯仰 低头 6.0 度", "要求 抬头到 2 度"]
    assert window_pitch_lines(None, None) == ["俯仰读不出来"]


def test_search_box_follows_pitch_change_and_keeps_the_tower():
    from car_pose import project
    from classify import _mask

    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    watch.pitch = -8.0
    watch.follow_pitch(None)
    assert watch.pitch == -8.0
    watch.follow_pitch(-3.0)
    assert watch.pitch == -3.0
    watch.follow_pitch(-13.0)
    assert watch.pitch == -13.0
    watch.follow_pitch(None)
    assert watch.pitch == -13.0
    cell = _cell("右侧地面远一格")
    base = 0.0
    used = 10.0

    def center_in(pitch: float) -> tuple[float, bool]:
        mask = _mask(cell, pose, offset, pitch, INTR)
        pix = project(cell.center(cell.one_z), pose, offset, pitch, INTR)
        assert pix is not None
        u, v = int(round(float(pix[0]))), int(round(float(pix[1])))
        assert 0 <= u < INTR.width and 0 <= v < INTR.height
        return float(pix[1]), bool(mask[v, u] > 0)

    flat_v, flat_in = center_in(base)
    moved_v, moved_in = center_in(used)
    assert flat_in and moved_in
    assert abs(moved_v - flat_v) > 5.0


def test_closest_pose_uses_the_nearest_stamp():
    samples = [CarPose(0.0, 0.0, 0.0, 1.0), CarPose(1.0, 0.0, 0.0, 1.2)]
    assert closest_pose(samples, 1.15).x == 1.0


def test_cli_basic_once_does_not_ask_the_referee_to_move():
    lines = run(
        type(
            "Args",
            (),
            {
                "mode": "basic",
                "image": "",
                "pose": [0.0, 0.0, 0.0],
                "pitch": 0.0,
                "aligned": True,
                "center_to_corner": [0.25, -0.15],
                "topic": "",
                "yaw_unit": "rad",
                "once": True,
                "pose_is_default": False,
            },
        )()
    )
    assert lines == []
