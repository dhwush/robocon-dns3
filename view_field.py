"""任务三的两条实地指令，启动方式不同。

基础：人把车摆在原点，车头朝前。不订雷达。
python view_field.py basic

拓展：订阅雷达给的小车中心 x、y 和航向，再告诉裁判下一步。
python view_field.py extend
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from car_pose import (
    CarPose,
    basic_origin_setup,
    closest_pose,
    load_intrinsics,
    load_setup,
    yaw_to_rad,
)
from report import LEVEL_LOCK, FieldWatch


ROOT = Path(__file__).resolve().parent


def _setup_from_args(args):
    setup = load_setup(str(ROOT / "config" / "field.yaml"))
    offset = setup.center_to_corner
    if args.center_to_corner is not None:
        offset = np.array(args.center_to_corner, dtype=float)
    aligned = setup.aligned or args.aligned
    pitch = setup.pitch_deg if args.pitch is None else float(args.pitch)
    return setup.__class__(
        aligned=aligned,
        center_to_corner=offset,
        pitch_deg=pitch,
        topic=args.topic or setup.topic,
        yaw_unit=args.yaw_unit or setup.yaw_unit,
    )


def _pose_from_args(args, yaw_unit: str) -> CarPose:
    x, y, yaw = args.pose
    return CarPose(float(x), float(y), yaw_to_rad(float(yaw), yaw_unit), stamp=0.0)


def _read_image(path: str):
    import cv2

    color = cv2.imread(path, cv2.IMREAD_COLOR)
    if color is None:
        raise SystemExit(f"读不到画面：{path}")
    depth = np.full(color.shape[:2], np.nan, dtype=np.float32)
    return color, depth


def _subscribe(topic: str, yaw_unit: str) -> CarPose | None:
    # 只订阅一次，不给雷达下命令。
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
    except ImportError:
        return None
    if not topic:
        return None
    if not rclpy.ok():
        rclpy.init()
    node = Node("field_pose_once")
    holder: dict[str, CarPose] = {}

    def _on_msg(msg) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny, cosy))
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        holder["pose"] = CarPose(
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
            stamp,
        )

    node.create_subscription(Odometry, topic, _on_msg, 10)
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
        if "pose" in holder:
            break
    node.destroy_node()
    return holder.get("pose")


def _listen(topic: str, yaw_unit: str):
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
    except ImportError:
        return None
    if not rclpy.ok():
        rclpy.init()
    node = Node("field_pose")
    samples: list[CarPose] = []

    def _on_msg(msg) -> None:
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny, cosy))
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        samples.append(
            CarPose(
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw,
                stamp,
            )
        )
        if len(samples) > 50:
            del samples[:-50]

    node.create_subscription(Odometry, topic, _on_msg, 10)
    return node, samples


def _show(lines: list[str], color) -> None:
    import cv2

    canvas = color if color is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    board = canvas.copy()
    for index, line in enumerate(lines):
        cv2.putText(
            board,
            line,
            (24, 40 + 36 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("建塔点", board)
    cv2.waitKey(1)


def _apply_light(sensor, rs, level: str) -> None:
    lock = LEVEL_LOCK[level]
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0)
    for option, value in (
        (rs.option.exposure, lock["exposure"]),
        (rs.option.gain, lock["gain"]),
        (rs.option.brightness, lock["brightness"]),
    ):
        if not sensor.supports(option):
            continue
        span = sensor.get_option_range(option)
        sensor.set_option(option, float(np.clip(value, span.min, span.max)))


def _live(args, watch: FieldWatch, pose: CarPose) -> list[str]:
    import cv2

    try:
        import pyrealsense2 as rs
    except ImportError:
        lines = watch.lines(args.mode)
        print("\n".join(lines))
        print("没有接上相机")
        return lines
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    align = rs.align(rs.stream.color)
    sensor = profile.get_device().first_color_sensor()
    listener = _listen(watch.setup.topic, watch.setup.yaw_unit) if watch.setup.topic else None
    previous = ""
    light = watch.light
    try:
        _apply_light(sensor, rs, light)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
            stamp = float(color_frame.get_timestamp()) * 1e-3
            current = pose
            if listener is not None:
                import rclpy

                rclpy.spin_once(listener[0], timeout_sec=0.0)
                matched = closest_pose(listener[1], stamp)
                if matched is not None:
                    current = matched
            watch.update(current, color, depth)
            if watch.light != light:
                light = watch.light
                _apply_light(sensor, rs, light)
            lines = watch.lines(args.mode)
            text = "\n".join(lines)
            if text != previous:
                print(text)
                previous = text
            _show(lines, color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return lines
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if listener is not None:
            listener[0].destroy_node()


def _once(args, watch: FieldWatch, pose: CarPose) -> list[str]:
    color = depth = None
    if args.image:
        color, depth = _read_image(args.image)
    watch.update(pose, color, depth)
    lines = watch.lines(args.mode)
    text = "\n".join(lines)
    if text:
        print(text)
    return lines


def run_basic(args) -> list[str]:
    # 实地基础：右前角在原点，航向沿场地向前。俯仰用支架上人设的角度。
    loaded = load_setup(str(ROOT / "config" / "field.yaml"))
    pitch = loaded.pitch_deg if args.pitch is None else float(args.pitch)
    setup, pose = basic_origin_setup(pitch)
    intr = load_intrinsics(str(ROOT / "config" / "intrinsics.yaml"))
    watch = FieldWatch(setup, intr)
    watch.pitch = setup.pitch_deg
    if args.image or args.once:
        return _once(args, watch, pose)
    return _live(args, watch, pose)


def run_extend(args) -> list[str]:
    # 实地拓展：车位来自雷达话题。没对齐或安装差还没量，只提示，不投影。
    setup = _setup_from_args(args)
    intr = load_intrinsics(str(ROOT / "config" / "intrinsics.yaml"))
    watch = FieldWatch(setup, intr)
    watch.pitch = setup.pitch_deg
    pose = _pose_from_args(args, setup.yaw_unit)
    if args.image or args.once:
        if args.pose_is_default and setup.topic:
            listened = _subscribe(setup.topic, setup.yaw_unit)
            if listened is not None:
                pose = listened
        return _once(args, watch, pose)
    if args.pose_is_default and setup.topic:
        listened = _subscribe(setup.topic, setup.yaw_unit)
        if listened is not None:
            pose = listened
    return _live(args, watch, pose)


def run(args) -> list[str]:
    if args.mode == "basic":
        return run_basic(args)
    return run_extend(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="识别四块建塔格的状态")
    parser.add_argument("mode", choices=("basic", "extend"))
    parser.add_argument("--image", default="")
    parser.add_argument("--pose", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "YAW"))
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--center-to-corner", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--topic", default="")
    parser.add_argument("--yaw-unit", choices=("rad", "deg"), default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.pose_is_default = list(args.pose) == [0.0, 0.0, 0.0]
    run(args)


if __name__ == "__main__":
    main()

