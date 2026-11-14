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
    Intrinsics,
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


def can_pose_text(pose: CarPose | None, source: str) -> str:
    label = "SSH CAN" if source == "ssh" else "CAN"
    if pose is None:
        return f"{label} 等待位姿 / 数据超时：x=-- m  y=-- m  yaw=--°"
    label += " 已校准场地位姿"
    age = max(0.0, time.monotonic() - pose.stamp) * 1000
    return (f"{label} 当前右前角：x={pose.x:+.3f} m  y={pose.y:+.3f} m  "
            f"yaw={np.degrees(pose.yaw):+.1f}°  数据年龄={age:.0f} ms")


def live_panels(watch, pose, source, extras=()):
    """窗口三分区：操作目标/CAN、结果、其余状态。"""
    from motion import GOAL_BEARING, MAX_TRIES, heading_words, format_pitch
    left = [can_pose_text(pose, source), f"当前相机：{format_pitch(watch.pitch)}"]
    right = ["识别结果"]
    words = {'0': '空', '1': '一层', '2': '两层', '红': '红顶', '蓝': '蓝顶'}
    for name in sorted(watch.votes, key=lambda n: TOWER_NO[n]):
        vote = watch.votes[name]
        current = watch.views[name].conclusion
        status = '待复核' if vote.recheck else '已确认' if vote.locked is not None else '检测中'
        value = vote.locked if vote.locked is not None else current
        right.append(f"{TOWER_NO[name]}号  {words.get(value, '未知')}  {status}")
    survey = watch.autonomy
    action = None if survey is None else survey.action
    if action is not None and action.cell and action.kind != 'move':
        view = watch.views[action.cell]
        center = np.degrees(pose.yaw) - view.bearing_deg
        headings = [heading_words(center+d) for d in dict.fromkeys(GOAL_BEARING)]
        left += [f"当前点：观察{TOWER_NO[action.cell]}号 · 第{action.attempt+1}/{MAX_TRIES}档",
                 '车头朝向：' + ' / '.join(headings[:4]),
                 '                 ' + ' / '.join(headings[4:]),
                 '相机俯仰：' + ' / '.join(format_pitch(view.pitch_target_deg+d) for d in (0.,8.,-8.))]
        from motion import PITCH_OFFSET
        yaw = action.yaw if action.yaw is not None else pose.yaw
        target_pitch = view.pitch_target_deg + PITCH_OFFSET[action.attempt]
        left.append(f"本档目标：{heading_words(np.degrees(yaw))} · {format_pitch(target_pitch)}")
        required = survey.required_checks(action.cell) if hasattr(survey,'required_checks') else 2
        left.append(f"俯仰复核：{len(survey.pitch_checks[action.cell])}/{required}档" )
    elif action is not None and action.kind == 'move':
        left.append(f"下一点：x={action.x:+.2f}m y={action.y:+.2f}m")
    else:
        left.append('当前点：等待观察任务' if survey is None or not survey.complete else '当前点：测量完成')
    route_hint = []
    if survey is not None and getattr(survey,'primary_stops',None):
        route_hint = [f"主线 A→沿途复看→B近点（{min(survey.primary_index,len(survey.primary_stops))}/{len(survey.primary_stops)}）；远塔确认即跳过远端行程"]
    bottom = route_hint + ([survey.message] if survey is not None else []) + list(extras)
    bottom += ["朝向以场地前方为0°；近处两档、远处三档一致才确认；主线看够再去兜底",
               "1–5 光线 | d 诊断 | n 切换塔 | s 保存 | q 退出"]
    return dict(left=left, right=right, bottom=bottom)


def paint_window(color, views, locked: dict, texts: list[str], show_boxes: bool = False, panels=None):
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
                if show_boxes:
                    # 搜索区域由当前位姿投影得到；不依赖分类成功或投票锁定。
                    x, y, width, height = cv2.boundingRect(view.region)
                    cv2.rectangle(board, (x, y), (x + width - 1, y + height - 1), bgr, 2)
                    tags.append((max(0, x), max(0, y - 24),
                                 f"{TOWER_NO[name]} 搜索框", bgr))
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
    # 标记先画，说明文字直接叠加到画面，不绘制背景底板。
    for u, v, text, bgr in tags:
        rgb = (bgr[2], bgr[1], bgr[0])
        draw.text((u, v), text, font=font, fill=rgb, stroke_width=1, stroke_fill=(0, 0, 0))
    if panels is None:
        panels = dict(left=texts, right=[], bottom=[])
    width, height = panel.size
    gap = 8
    right_width = min(240, width//3)
    left_width = width-right_width-3*gap
    line_height = _FONT_SIZE+4

    def block(lines, x, y, available, max_height, bottom=False):
        rows = [row for line in lines for row in _wrap(draw, line, font, max(20,available-16))]
        rows = rows[:max(0, (max_height-8)//line_height)]
        if not rows:
            return
        box_height = len(rows)*line_height+8
        if bottom:
            y = height-gap-box_height
        for row in rows:
            draw.text((x+8,y+4), row, font=font, fill=(245,245,245),
                      stroke_width=1, stroke_fill=(0,0,0))
            y += line_height

    bottom_lines = panels.get('bottom', [])
    bottom_rows = sum(len(_wrap(draw,line,font,max(20,width-32))) for line in bottom_lines)
    footer_height = min(height//3, bottom_rows*line_height+8) if bottom_rows else 0
    top_height = height-footer_height-3*gap
    block(panels.get('left', []), gap, gap, left_width, top_height)
    block(panels.get('right', []), width-right_width-gap, gap, right_width, top_height)
    block(bottom_lines, gap, 0, width-2*gap, footer_height, bottom=True)
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


def _camera_calibration(profile, rs):
    """读取实际启动的彩色流标定；对齐到彩色的深度也使用这组内参。"""
    raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = Intrinsics(int(raw.width), int(raw.height), float(raw.fx), float(raw.fy),
                      float(raw.ppx), float(raw.ppy))
    values = (intr.fx, intr.fy, intr.cx, intr.cy)
    if (intr.width <= 0 or intr.height <= 0 or not np.all(np.isfinite(values))
            or intr.fx <= 0 or intr.fy <= 0):
        raise ValueError("相机返回的彩色内参无效")
    device = profile.get_device()
    metadata = {
        "serial_number": device.get_info(rs.camera_info.serial_number),
        "distortion_model": str(raw.model),
        "distortion_coefficients": [float(value) for value in raw.coeffs],
        "intrinsics_source": "realsense_color_stream",
    }
    return intr, metadata


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
    try:
        pending["intrinsics"], pending["camera"] = _camera_calibration(profile, rs)
    except Exception:
        pipeline.stop()
        raise
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
    if source in ("ssh", "local"):
        # CAN 两种入口起始实际朝向固定为场地0度，观测角度只是后续转向目标。
        pose = CarPose(pose.x, pose.y, 0.0, pose.stamp)
    verify_motion = getattr(args, "verify_motion", False)
    if source in ("ssh", "local"):
        from dataclasses import replace
        # 第一组 CAN 对应传入的已知场地右前角位姿，统一变换后再投影。
        watch.setup = replace(watch.setup, aligned=True, center_to_corner=np.zeros(2), origin_basic=False)
    previous_session = getattr(args, "survey_session", None)
    if args.mode == "static":
        pass  # 独立静态模式不启动自主移动流程。
    elif previous_session is None:
        watch.enable_autonomy()
        args.survey_session = watch.autonomy
        args.survey_report = ROOT / "debug_captures" / ("survey_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    else:
        watch.votes = previous_session.watch.votes
        watch.autonomy = previous_session
        previous_session.watch = watch
        previous_session.pause()
    report_saved_at = 0.0
    if watch.autonomy is not None:
        print(f"自主识别记录：{args.survey_report}")
    pipeline, sensor, ready, lock, camera_frames = _open_camera(rs)
    watch.intr = camera_frames["intrinsics"]
    camera = camera_frames["camera"]
    print(f"相机序列号：{camera['serial_number']}；内参来源：当前彩色流")
    print(f"内参：{watch.intr.width}×{watch.intr.height} "
          f"fx={watch.intr.fx:.6f} fy={watch.intr.fy:.6f} "
          f"cx={watch.intr.cx:.6f} cy={watch.intr.cy:.6f}")
    print(f"畸变：{camera['distortion_model']} {camera['distortion_coefficients']}")
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
    raw_can_origin = None
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
                color = camera_frames["color"]
                depth = camera_frames["depth"]
                gravity = None if camera_frames["gravity"] is None else camera_frames["gravity"].copy()
                stamp = float(camera_frames["stamp"])
                received = float(camera_frames["received"])
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
                    watch.autonomy.pause()
                    watch.stable_pose = None
                    watch.views = None
                    message = "等待 CAN 位姿 / 数据超时，暂停识别"
                    cv2.imshow("建塔点", paint_window(color, None, {}, [can_pose_text(None, source), message]))
                    live_map = getattr(args, "live_map", None)
                    if live_map is not None:
                        live_map(None)
                    if previous != message:
                        print(message)
                        previous = message
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        return []
                    if can_stream.process.poll() is not None:
                        raise RuntimeError("CAN 读取进程已退出，请检查 SSH 和 CAN 接口")
                    continue
                from can_pose import align_can_pose
                if raw_can_origin is None:
                    raw_can_origin = matched
                    print(f"CAN 起始校准：原始 {vars(matched)} → 场地 {vars(pose)}")
                current = align_can_pose(matched, raw_can_origin, pose)
                if initial_can_pose is None:
                    initial_can_pose = current
            elif listener is not None:
                import rclpy

                rclpy.spin_once(listener[0], timeout_sec=0.0)
                matched = closest_pose(listener[1], stamp)
                if matched is None:
                    for vote in watch.votes.values():
                        vote.on_move()
                    watch.autonomy.pause()
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
            if watch.autonomy is not None:
                args.autonomous_lines = watch.autonomy.lines()
                args.autonomous_route = watch.autonomy.route_display()
                confirmed = [TOWER_NO[n] for n,v in watch.votes.items() if v.locked is not None and not v.recheck]
                pending_towers = [TOWER_NO[n] for n,v in watch.votes.items() if v.locked is None or v.recheck]
                args.autonomous_map_lines = [watch.autonomy.message,
                    "已确认塔：" + ('、'.join(confirmed) or '无'),
                    "待测塔：" + ('、'.join(pending_towers) or '无；测量完成'),
                    "A起点→沿途复看→B近点；主线用尽才去远处兜底"]
                if time.monotonic() - report_saved_at >= 1.0:
                    try:
                        watch.autonomy.save(args.survey_report)
                    except OSError as error:
                        print(f"保存自主识别记录失败：{error}")
                    report_saved_at = time.monotonic()
            fixed_targets = getattr(args, "fixed_target_lines", [])
            result_mode = args.mode
            lines = watch.lines(result_mode)
            shown = watch.window_lines(result_mode)
            text = "\n".join(shown)
            if text != previous:
                print(text)
                previous = text
            locked = {name: vote.locked for name, vote in watch.votes.items()}
            shown_pitch = None if measured is None else watch.pitch
            fixed_targets = getattr(args, "fixed_target_lines", [])
            painted = window_pitch_lines(shown_pitch, watch.required_pitch()) + shown
            if fixed_targets and watch.autonomy is None:
                painted = window_pitch_lines(shown_pitch, None) + ["固定计划（目标角度保持）："] + fixed_targets + shown
            painted.append(f"位姿 {source}: x={current.x:.3f} m y={current.y:.3f} m 旋转={np.degrees(current.yaw):.1f}°")
            if source == "ssh":
                painted.insert(0, can_pose_text(current, source))
                painted.append("SSH CAN 已收到位姿；矩形为投影搜索框，轮廓为识别区域")
            if verify_motion and initial_can_pose is not None:
                dx = current.x - initial_can_pose.x
                dy = current.y - initial_can_pose.y
                turn = np.degrees(np.arctan2(np.sin(current.yaw - initial_can_pose.yaw),
                                             np.cos(current.yaw - initial_can_pose.yaw)))
                painted.append(f"慢速验证：Δx={dx:+.3f} m Δy={dy:+.3f} m Δyaw={turn:+.1f}°")
                painted.append(f"CAN 接收年龄 {1000 * (time.monotonic() - current.stamp):.0f} ms；观察框是否保持贴合建塔点")
            painted.append("d 颜色/深度诊断，n 切换塔，s 保存原始帧")
            cv2.imshow("建塔点", paint_window(color, watch.views, locked, painted,
                                             show_boxes=source == "ssh",
                                             panels=None if args.mode == "static" else live_panels(watch, current, source,
                                                 [shown[0]] + ([line for line in painted if line.startswith(("慢速验证", "CAN 接收年龄"))]))))
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
                        "camera": camera,
                        "can_calibration": None if raw_can_origin is None else {
                            "raw_origin": vars(raw_can_origin), "field_origin": vars(pose),
                            "raw_pose": vars(matched),
                        },
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
        if watch.autonomy is not None:
            try:
                watch.autonomy.save(args.survey_report)
            except OSError as error:
                print(f"保存自主识别记录失败：{error}")
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
    if args.mode == "static":
        from static_test import run_static
        return run_static(args)
    if args.mode == "basic":
        return run_basic(args)
    return run_extend(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="识别四块建塔格的状态")
    parser.add_argument("mode", choices=("basic", "extend", "static"))
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
