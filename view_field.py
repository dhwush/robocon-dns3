"""任务三的两条实地指令，启动方式不同。

基础：人把右前角摆在场地原点右侧 1.2 米，前后对齐，车头朝前。不订雷达。
python view_field.py basic

拓展：订阅雷达给的小车中心 x、y 和航向，再告诉裁判下一步。
python view_field.py extend

按 d 开关颜色/深度诊断，n 切换诊断塔，s 保存原始帧和诊断图到 debug_captures。
打开画面后按 1 到 5 换光线：很亮、白天、室内、偏暗、很暗。按 q 或 Esc 退出。
画面最上面写加速度计的实时俯仰，正数是抬头，框用这个角度投影。
读不到就沿用上一帧。下一行写这一步要求的俯仰。左右转先不改。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from car_pose import (
    CarPose,
    basic_origin_setup,
    closest_pose,
    load_intrinsics,
    load_setup,
    pitch_from_gravity,
    roll_from_gravity,
    yaw_to_rad,
)
from field_map import TOWER_NO
from report import LEVEL_LOCK, FieldWatch, light_from_key, window_pitch_lines


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


_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_SIZE = 18
_MARK = {
    "0": (180, 180, 180),
    "1": (255, 255, 0),
    "2": (255, 0, 255),
    "红": (0, 0, 255),
    "蓝": (255, 80, 0),
}
_SHORT = {
    "0": "空",
    "1": "一层",
    "2": "两层",
    "红": "红顶",
    "蓝": "蓝顶",
}


def mark_text(number: str, state: str | None, method: str | None) -> str | None:
    # 画面上的短词。方法只标在这一眼实际用的那条后面，裁判短句不带这个字。
    word = _SHORT.get(state or "")
    if not word:
        return None
    if method:
        return f"{number} {word}·{method}"
    return f"{number} {word}"


def _anchor(contour: np.ndarray) -> tuple[int, int]:
    import cv2

    moment = cv2.moments(contour)
    if moment["m00"] <= 0:
        x, y, width, height = cv2.boundingRect(contour)
        return x + width // 2, y + height // 2
    return int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"])


def _wrap(draw, text: str, font, width: int) -> list[str]:
    if draw.textlength(text, font=font) <= width:
        return [text]
    rows: list[str] = []
    buf = ""
    for char in text:
        trial = buf + char
        if buf and draw.textlength(trial, font=font) > width:
            rows.append(buf)
            buf = char
        else:
            buf = trial
    if buf:
        rows.append(buf)
    return rows


def paint_window(color, views, locked: dict, texts: list[str]):
    # 格子轮廓标出在看哪一块。认到白盒后再沿白盒描一圈，并在旁边写短词。
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    board = color.copy() if color is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
    tags: list[tuple[int, int, str, tuple[int, int, int]]] = []
    if views:
        for name, view in views.items():
            state = locked.get(name) or view.conclusion
            bgr = _MARK.get(state, (0, 220, 255))
            if view.region is not None:
                cv2.drawContours(board, [view.region], -1, bgr, 2, cv2.LINE_AA)
            if view.piece is not None:
                cv2.drawContours(board, [view.piece], -1, bgr, 2, cv2.LINE_AA)
            measure = getattr(view, "measurements", None) or {}
            points = measure.get("top_pixels", [])
            for x, y in points:
                cv2.circle(
                    board, (int(x), int(y)), 2, (255, 0, 255), -1
                )
            if points:
                x, y = points[len(points)//2]
                height = measure.get("depth_height_cm", 0)
                tags.append((
                    int(x)+6, max(0,int(y)-18),
                    f"{TOWER_NO[name]} 深度取样 {height:.1f}cm",
                    (255,0,255),
                ))
            anchor = view.piece if view.piece is not None else view.region
            if anchor is None:
                continue
            u, v = _anchor(anchor)
            cv2.circle(board, (u, v), 6, bgr, -1, cv2.LINE_AA)
            method = view.method if view.conclusion == state else None
            word = mark_text(TOWER_NO[name], state, method)
            if word:
                tags.append((u + 10, max(0, v - 18), word, bgr))
    font = ImageFont.truetype(_FONT, _FONT_SIZE)
    panel = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    limit = max(board.shape[1] - 16, 40)
    y = 6
    for line in texts:
        for row in _wrap(draw, line, font, limit):
            draw.text((8, y), row, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
            y += _FONT_SIZE + 4
    for u, v, text, bgr in tags:
        rgb = (bgr[2], bgr[1], bgr[0])
        draw.text((u, v), text, font=font, fill=rgb, stroke_width=1, stroke_fill=(0, 0, 0))
    board[:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return board


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


def _gravity_from_accel(specific: np.ndarray, rotation: np.ndarray) -> np.ndarray | None:
    # 加速度计静止时指向“上”。旋到彩色相机系后取反，得到相机系里的重力方向。
    up = rotation @ np.asarray(specific, dtype=float)
    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or not 8.3 <= norm <= 11.3:
        return None
    return -up / norm


def _accel_rotation(profile, rs) -> np.ndarray | None:
    try:
        motion = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
        color = profile.get_stream(rs.stream.color)
        ext = motion.get_extrinsics_to(color)
        return np.array(ext.rotation, dtype=float).reshape(3, 3)
    except Exception:
        return None


def _open_camera(rs):
    # 彩色和深度照旧。加速度计单独进来，用来读实时俯仰。没有加速度计就只开画面。
    import threading

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel)
    # 要求 SDK 提供与 ROS 系统时钟相同的全局时间；不使用设备启动时间冒充。
    align = rs.align(rs.stream.color)
    lock = threading.Lock()
    ready = threading.Event()
    scale = {"v": 0.001}
    accel = {"rotation": None, "gravity": None}
    pending: dict = {"color": None, "depth": None, "gravity": None, "stamp": 0.0, "received": 0.0}

    def on_frame(frame) -> None:
        try:
            if frame.is_motion_frame():
                rotation = accel["rotation"]
                if rotation is None or frame.profile.stream_type() != rs.stream.accel:
                    return
                data = frame.as_motion_frame().get_motion_data()
                specific = np.array([data.x, data.y, data.z], dtype=float)
                gravity = _gravity_from_accel(specific, rotation)
                if gravity is None:
                    return
                with lock:
                    accel["gravity"] = gravity
                return
            if not frame.is_frameset():
                return
            aligned = align.process(frame.as_frameset())
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale["v"]
            domain = color_frame.get_frame_timestamp_domain()
            stamp = (float(color_frame.get_timestamp()) * 1e-3
                     if domain in (rs.timestamp_domain.system_time, rs.timestamp_domain.global_time)
                     else float("nan"))
            with lock:
                gravity = None if accel["gravity"] is None else accel["gravity"].copy()
                pending["color"] = color
                pending["depth"] = depth
                pending["gravity"] = gravity
                pending["stamp"] = stamp
                pending["received"] = time.monotonic()
            ready.set()
        except Exception:
            return

    try:
        profile = pipeline.start(config, on_frame)
    except Exception as exc:
        if "No device connected" in str(exc):
            raise
        try:
            pipeline.stop()
        except Exception:
            pass
        pipeline = rs.pipeline()
        config.disable_stream(rs.stream.accel)
        profile = pipeline.start(config, on_frame)
    for device_sensor in profile.get_device().query_sensors():
        if device_sensor.supports(rs.option.global_time_enabled):
            device_sensor.set_option(rs.option.global_time_enabled, 1)
    scale["v"] = float(profile.get_device().first_depth_sensor().get_depth_scale())
    accel["rotation"] = _accel_rotation(profile, rs)
    sensor = profile.get_device().first_color_sensor()
    return pipeline, sensor, ready, lock, pending


def _live(args, watch: FieldWatch, pose: CarPose) -> list[str]:
    import cv2

    try:
        import pyrealsense2 as rs
    except ImportError:
        lines = watch.lines(args.mode)
        print("\n".join(lines))
        print("没有接上相机")
        return lines
    source = getattr(args, "pose_source", "ssh")
    verify_motion = getattr(args, "verify_motion", False)
    if verify_motion and source in ("ssh", "local"):
        from dataclasses import replace
        # 验证模式明确采用 CAN 为场地右前角位姿，不附加中心偏移。
        watch.setup = replace(watch.setup, aligned=True, center_to_corner=np.zeros(2), origin_basic=False)
    if source in ("ssh", "local") and watch.setup.origin_basic:
        setup = load_setup(str(ROOT / "config" / "field.yaml"))
        if not setup.ready:
            raise SystemExit("使用 CAN 前请在 config/field.yaml 确认右前角定位已对齐，并设置 aligned: true")
        watch.setup = setup
    pipeline, sensor, ready, lock, pending = _open_camera(rs)
    source = getattr(args, "pose_source", "ssh")
    can_stream = None
    try:
        if source in ("ssh", "local"):
            from can_pose import CanPoseStream
            can_stream = CanPoseStream(source, getattr(args, "can_interface", "can0"),
                                       getattr(args, "ssh_host", "volleyball@192.168.3.194"))
        listener = _listen(watch.setup.topic, watch.setup.yaw_unit) if source == "ros" and watch.setup.topic else None
    except Exception:
        pipeline.stop()
        raise
    previous = ""
    initial_can_pose = None
    light = watch.light
    exposure_wait = 5
    debug_visible = False
    debug_selected = 0
    from tower_debug import paint_diagnostics, save_capture
    try:
        _apply_light(sensor, rs, light)
        while True:
            if not ready.wait(0.05):
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return watch.lines(args.mode)
                continue
            ready.clear()
            with lock:
                color = pending["color"]
                depth = pending["depth"]
                gravity = None if pending["gravity"] is None else pending["gravity"].copy()
                stamp = float(pending["stamp"])
                received = float(pending["received"])
            if color is None or depth is None:
                continue
            measured = pitch_from_gravity(gravity)
            rolled = roll_from_gravity(gravity)
            watch.follow_pitch(measured)
            if rolled is not None:
                watch.roll = rolled
            current = pose
            if can_stream is not None:
                matched = can_stream.match(received)
                if matched is None:
                    for vote in watch.votes.values():
                        vote.on_move()
                    watch.stable_pose = None
                    watch.views = None
                    message = "等待 CAN 位姿 / 数据超时，暂停识别"
                    cv2.imshow("建塔点", paint_window(color, None, {}, [message]))
                    if previous != message:
                        print(message)
                        previous = message
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return []
                    if can_stream.process.poll() is not None:
                        raise RuntimeError("CAN 读取进程已退出，请检查 SSH 和 CAN 接口")
                    continue
                current = matched
                if initial_can_pose is None:
                    initial_can_pose = matched
            elif listener is not None:
                import rclpy

                rclpy.spin_once(listener[0], timeout_sec=0.0)
                matched = closest_pose(listener[1], stamp)
                if matched is None:
                    for vote in watch.votes.values():
                        vote.on_move()
                    watch.stable_pose = None
                    cv2.imshow("建塔点", color)
                    print("车位与画面时间不匹配，暂停识别")
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        return watch.lines(args.mode)
                    continue
                current = matched
            elif not watch.setup.origin_basic:
                print("没有实时车位，暂停识别")
                return watch.lines(args.mode)
            live_map = getattr(args, "live_map", None)
            if live_map is not None:
                live_map(current)
            if exposure_wait > 0:
                exposure_wait -= 1
                watch.update(current, None, None)
                for vote in watch.votes.values():
                    vote.on_move()
            else:
                watch.update(current, color, depth)
            if watch.light != light:
                light = watch.light
                _apply_light(sensor, rs, light)
                exposure_wait = 5
            fixed_targets = getattr(args, "fixed_target_lines", [])
            result_mode = "basic" if fixed_targets else args.mode
            lines = watch.lines(result_mode)
            shown = watch.window_lines(result_mode)
            text = "\n".join(shown)
            if text != previous:
                print(text)
                previous = text
            locked = {name: vote.locked for name, vote in watch.votes.items()}
            shown_pitch = None if measured is None else watch.pitch
            fixed_targets = getattr(args, "fixed_target_lines", [])
            painted = window_pitch_lines(shown_pitch, None if fixed_targets else watch.required_pitch()) + shown
            if fixed_targets:
                painted = window_pitch_lines(shown_pitch, None) + ["固定计划（目标角度保持）："] + fixed_targets + shown
            painted.append(f"位姿 {source}: x={current.x:.3f} m y={current.y:.3f} m 旋转={np.degrees(current.yaw):.1f}°")
            if verify_motion and initial_can_pose is not None:
                dx = current.x - initial_can_pose.x
                dy = current.y - initial_can_pose.y
                turn = np.degrees(np.arctan2(np.sin(current.yaw - initial_can_pose.yaw),
                                             np.cos(current.yaw - initial_can_pose.yaw)))
                painted.append(f"慢速验证：Δx={dx:+.3f} m Δy={dy:+.3f} m Δyaw={turn:+.1f}°")
                painted.append(f"CAN 接收年龄 {1000 * (time.monotonic() - current.stamp):.0f} ms；观察框是否保持贴合建塔点")
            painted.append("d 颜色/深度诊断，n 切换塔，s 保存原始帧")
            cv2.imshow("建塔点", paint_window(color, watch.views, locked, painted))
            if debug_visible:
                cv2.imshow("颜色与深度诊断", paint_diagnostics(color, depth, watch.views, debug_selected, locked))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return lines
            if key == ord("d"):
                debug_visible = not debug_visible
                if not debug_visible:
                    cv2.destroyWindow("颜色与深度诊断")
            elif key == ord("n"):
                debug_selected = (debug_selected + 1) % len(watch.votes)
            elif key == ord("s"):
                try:
                    saved = save_capture(ROOT / "debug_captures", color, depth,
                                         watch.views, debug_selected, locked, {
                        "stamp": stamp, "pose": vars(current),
                        "pitch_deg": watch.pitch, "roll_deg": watch.roll,
                        "intrinsics": vars(watch.intr),
                        "center_to_corner": watch.setup.center_to_corner,
                        "light": light, "exposure_wait": exposure_wait,
                    })
                    print(f"已保存诊断：{saved}.npz / .png")
                except (OSError, ValueError) as error:
                    print(f"保存诊断失败：{error}")
            picked = light_from_key(key)
            if picked is not None:
                watch.choose_light(picked)
                light = watch.light
                _apply_light(sensor, rs, light)
                exposure_wait = 5
    finally:
        pipeline.stop()
        if can_stream is not None:
            can_stream.close()
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
    # 实地基础：右前角在场地原点右侧 1.2 m，前后对齐，车头朝前。俯仰用支架上人设的角度。
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
        if getattr(args, "pose_source", "ssh") == "ros" and args.pose_is_default and setup.topic:
            listened = _subscribe(setup.topic, setup.yaw_unit)
            if listened is not None:
                pose = listened
        return _once(args, watch, pose)
    if getattr(args, "pose_source", "ssh") == "ros" and args.pose_is_default and setup.topic:
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
    parser.add_argument("--pose-source", choices=("ssh", "local", "ros", "static"), default="ssh")
    parser.add_argument("--verify-motion", action="store_true", help="慢速验证：直接采用 CAN 右前角场地位姿并显示相对变化")
    parser.add_argument("--ssh-host", default="volleyball@192.168.3.194")
    parser.add_argument("--can-interface", default="can0")
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

