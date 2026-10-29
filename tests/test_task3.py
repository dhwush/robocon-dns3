"""合成画面。不打开相机，也不订阅雷达。"""

from __future__ import annotations

import numpy as np

from car_pose import CarPose, Setup, closest_pose, load_intrinsics, pose_moved
from classify import heights_match, inspect_frame
from field_map import MARGIN_M, search_contains, tower_cells
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


def test_far_left_hides_one_layer_and_shows_two():
    cell = _cell("最远一行左侧")
    assert sees(START, OFFSET, 0.0, cell, 0.55) is False
    assert sees(START, OFFSET, 0.0, cell, 0.90) is True
    assert visible_height(START, OFFSET, 0.0, cell, 0.20, 0.55) < 0.08
    assert visible_height(START, OFFSET, 0.0, cell, 0.20, 0.90) > 0.2
    near = _cell("最近一行正中")
    assert sees(START, OFFSET, 0.0, near, 0.55) is True


def test_margin_covers_five_centimetres_and_not_forty():
    cell = _cell("最近一行正中")
    edge = np.array([cell.x + 0.60, cell.y])
    assert search_contains(cell, edge + np.array([0.05, 0.0]), MARGIN_M)
    assert not search_contains(cell, edge + np.array([0.40, 0.0]), MARGIN_M)


def test_twenty_percent_range_does_not_swap_layers():
    assert heights_match(0.35 * 1.2, 0.35)
    assert heights_match(0.35 * 0.8, 0.35)
    assert not heights_match(0.35 * 1.2, 0.70)
    assert heights_match(0.70 * 0.8, 0.70)
    assert not heights_match(0.70 * 0.8, 0.35)


def test_near_white_is_one_and_outside_white_is_ignored():
    color = _blank()
    _paint(color, START, 0.0, _cell("最近一行正中"), 0.35, 0.35, 0.20, (255, 255, 255))
    color[0:80, 0:80] = 255
    views = inspect_frame(START, OFFSET, 0.0, INTR, color, None)
    assert views["最近一行正中"].conclusion == "1"
    assert views["最远一行左侧"].conclusion is None


def test_peeking_two_layers_is_not_called_one():
    cell = _cell("最远一行左侧")
    height = visible_height(START, OFFSET, 0.0, cell, 0.20, 0.90)
    color = _blank()
    _paint(color, START, 0.0, cell, 0.35, height, 0.90 - height, (255, 255, 255))
    views = inspect_frame(START, OFFSET, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "2"


def test_red_cap_uses_the_upper_side():
    cell = _cell("最近一行正中")
    color = _blank()
    _paint(color, START, 0.0, cell, 0.35, 0.70, 0.20, (255, 255, 255))
    _paint(color, START, 0.0, cell, 0.20, 0.20, 0.90, (0, 0, 255))
    views = inspect_frame(START, OFFSET, 0.0, INTR, color, None)
    assert views[cell.name].conclusion == "红"


def test_depth_outside_range_does_not_cancel_color():
    cell = _cell("最近一行正中")
    color = _blank()
    _paint(color, START, 0.0, cell, 0.35, 0.35, 0.20, (255, 255, 255))
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    views = inspect_frame(START, OFFSET, 0.0, INTR, color, depth)
    assert views[cell.name].conclusion == "1"
    depth[:] = 1.0
    views = inspect_frame(START, OFFSET, 0.0, INTR, color, depth)
    assert views[cell.name].conclusion == "1"


def test_five_frames_lock_and_one_frame_does_not_unlock():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    _paint(color, START, 0.0, _cell("最近一行正中"), 0.35, 0.35, 0.20, (255, 255, 255))
    for _ in range(4):
        watch.update(START, color)
    assert watch.votes["最近一行正中"].locked is None
    watch.update(START, color)
    assert watch.votes["最近一行正中"].locked == "1"
    watch.update(START, _blank())
    assert watch.votes["最近一行正中"].locked == "1"


def test_jitter_does_not_reset_the_count():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    _paint(color, START, 0.0, _cell("最近一行正中"), 0.35, 0.35, 0.20, (255, 255, 255))
    for index in range(5):
        pose = CarPose(0.01 * (index % 2), 0.0, 0.0)
        assert pose_moved(START, pose) is False or index == 0
        watch.update(pose, color)
    assert watch.votes["最近一行正中"].locked == "1"


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
    assert abs(place[0] + 0.211) < 1e-6
    assert abs(place[1] - 0.05) < 1e-6
    assert abs(place[2] - 0.70) < 1e-6
    assert setup.topic == ""
    assert setup.aligned is False
    watch = FieldWatch(setup, INTR)
    color = _blank()
    _paint(
        color,
        pose,
        0.0,
        _cell("最近一行正中"),
        0.35,
        0.35,
        0.20,
        (255, 255, 255),
        offset=setup.center_to_corner,
    )
    for _ in range(5):
        watch.update(pose, color)
    lines = watch.lines("basic")
    assert "最近一行正中 1" in lines
    assert "从近往远第二行右侧 0" in lines
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
    assert len(extend) == 5
    assert extend[-1].startswith("向")
    assert "度" in extend[-1]
    assert "还没看清" in "\n".join(extend)


def test_blocked_cell_asks_for_one_slide_when_the_view_is_open():
    watch = FieldWatch(_setup(), INTR)
    watch.update(START, None)
    lines = watch.lines("extend")
    slides = [line for line in lines if line.endswith("米")]
    assert slides == [lines[-1]]
    assert lines[-1].startswith("向")


def test_visible_cells_lock_before_the_slide():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    for _ in range(4):
        watch.update(START, color)
    assert all(not line.endswith("米") for line in watch.lines("extend"))
    watch.update(START, color)
    assert watch.lines("extend")[-1].endswith("米")


def test_all_locked_says_stop():
    watch = FieldWatch(_setup(), INTR)
    color = _blank()
    near = _cell("最近一行正中")
    _paint(color, START, 0.0, near, 0.35, 0.35, 0.20, (255, 255, 255))
    for _ in range(5):
        watch.update(START, color)
    for name, vote in watch.votes.items():
        if vote.locked is None:
            vote.locked = "0"
    lines = watch.lines("extend")
    assert lines[-1] == "停止"
    basic = watch.lines("basic")
    assert all("平移" not in line and "度" not in line for line in basic)
    assert any(line.startswith("最近一行正中 1") for line in basic)


def test_light_steps_once_and_then_waits():
    level, wait = step_light("daylight", 0.0, 10.0, 0)
    assert level == "indoor"
    assert wait == 5
    level, wait = step_light(level, 0.0, 10.0, wait)
    assert level == "indoor"
    assert wait == 4
    level, wait = step_light("daylight", 0.9, 200.0, 0)
    assert level == "bright"


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

