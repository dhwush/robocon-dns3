"""合成画面。不打开相机，也不订阅雷达。"""

from __future__ import annotations

import numpy as np

from car_pose import CarPose, Setup, camera_position, closest_pose, load_intrinsics, pose_moved
from map_test import _append_yaw, _snap_cursor, _yaw_from_text, _yaw_key, pose_for_optical
from classify import heights_match, inspect_frame
from field_map import MARGIN_M, all_cells, search_contains, tower_cells
from report import FieldWatch, step_light
from sight import sees, visible_height
from view_field import run


INTR = load_intrinsics("config/intrinsics.yaml")
OFFSET = np.array([0.25, -0.15], dtype=float)
START = CarPose(0.0, 0.0, 0.0)


def _setup(aligned: bool = True) -> Setup:
    return Setup(aligned, OFFSET if aligned else None, 0.0, "", "rad")


def _cell(name: str):
    return next(cell for cell in tower_cells() if cell.name == name)


def _paint(color, pose, pitch, cell, width_m, height_m, z0, bgr, offset=None):
    import cv2

    from car_pose import project, to_camera

    used = OFFSET if offset is None else offset
    center = cell.center(z0 + height_m / 2.0)
    pixel = project(center, pose, used, pitch, INTR)
    assert pixel is not None
    distance = float(to_camera(center, pose, used, pitch)[2])
    width = max(2, int(round(width_m * INTR.fx / distance)))
    height = max(2, int(round(height_m * INTR.fy / distance)))
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    cv2.rectangle(
        color,
        (u - width // 2, v - height // 2),
        (u + width // 2, v + height // 2),
        bgr,
        thickness=-1,
    )


def _blank():
    return np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)


def test_start_camera_stays_on_the_known_optical_center():
    from car_pose import camera_position

    place = camera_position(START, OFFSET, 0.0)
    assert abs(place[0] + 0.211) < 1e-6
    assert abs(place[1] - 0.05) < 1e-6
    assert abs(place[2] - 0.70) < 1e-6


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


def test_twenty_percent_range_does_not_swap_layers():
    assert heights_match(0.35 * 1.2, 0.35)
    assert heights_match(0.35 * 0.8, 0.35)
    assert not heights_match(0.35 * 1.2, 0.70)
    assert heights_match(0.70 * 0.8, 0.70)
    assert not heights_match(0.70 * 0.8, 0.35)


def test_near_white_is_one_and_outside_white_is_ignored():
    setup, pose, offset = _basic()
    color = _blank()
    _paint(color, pose, 0.0, _cell("右侧地面远一格"), 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    color[0:80, 0:80] = 255
    views = inspect_frame(pose, offset, 0.0, INTR, color, None)
    assert views["右侧地面远一格"].conclusion == "1"
    assert views["第三行左侧"].conclusion is None
    assert setup.origin_basic is True


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


def test_five_frames_lock_and_one_frame_does_not_unlock():
    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    color = _blank()
    _paint(color, pose, 0.0, _cell("右侧地面远一格"), 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for _ in range(4):
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
    for index in range(5):
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
    assert abs(place[2] - 0.70) < 1e-6
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
    for _ in range(5):
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
    assert extend[-1].startswith("向")
    assert "度" in extend[-1]
    assert "还没看清" in "\n".join(extend)


def test_blocked_cell_asks_for_one_slide_when_the_view_is_open():
    watch = FieldWatch(_setup(), INTR)
    watch.update(START, None)
    lines = watch.lines("extend")
    assert lines[-1].startswith("向")
    assert "度" in lines[-1]


def test_visible_cells_lock_before_the_slide():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    for _ in range(4):
        watch.update(START, color)
    assert all(not line.endswith("米") for line in watch.lines("extend"))
    watch.update(START, color)
    assert "度" in watch.lines("extend")[-1]


def test_all_locked_says_stop():
    setup, pose, offset = _basic()
    watch = FieldWatch(setup, INTR)
    color = _blank()
    near = _cell("右侧地面远一格")
    _paint(color, pose, 0.0, near, 0.35, 0.35, 0.0, (255, 255, 255), offset=offset)
    for _ in range(5):
        watch.update(pose, color)
    for name, vote in watch.votes.items():
        if vote.locked is None:
            vote.locked = "0"
    lines = watch.lines("extend")
    assert lines[-1] == "停止"
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
    for _ in range(5):
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


def test_map_click_lands_on_the_optical_center():
    yaw = np.deg2rad(30.0)
    pose = pose_for_optical(1.2, 0.4, yaw)
    optical = camera_position(pose, np.zeros(2), 0.0)
    assert np.allclose(optical, [1.2, 0.4, 0.70])
    straight = pose_for_optical(0.5, 1.0, 0.0)
    assert np.allclose([straight.x, straight.y], [0.5 + 0.211, 1.0 - 0.05])


def test_live_pitch_reads_look_up_as_positive():
    from car_pose import pitch_from_gravity
    from report import live_pitch_line

    assert pitch_from_gravity(None) is None
    assert abs(pitch_from_gravity(np.array([0.0, 1.0, 0.0]))) < 1e-6
    look_up = np.array([0.0, np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0))])
    look_down = np.array([0.0, np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
    assert abs(pitch_from_gravity(look_up) - 30.0) < 1e-6
    assert abs(pitch_from_gravity(look_down) + 30.0) < 1e-6
    assert live_pitch_line(None) == "俯仰读不出来"
    assert live_pitch_line(12.34) == "俯仰 抬头 12.3 度"
    assert live_pitch_line(-3.0) == "俯仰 低头 3.0 度"
    assert live_pitch_line(0.0) == "俯仰 0.0 度"


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

