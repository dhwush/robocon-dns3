"""基础和拓展的窗口文字。认塔相同，印出来的字不同。"""

from __future__ import annotations

import math

from car_pose import CarPose, Intrinsics, Setup, pose_moved
from classify import inspect_frame
from field_map import DISPLAY, TOWER_NO
from motion import MOVE_ARRIVE_M, ARRIVE_M, MAX_TRIES, format_pitch, place_done, plan_action
from vote import Vote


LEVELS = ("bright", "daylight", "indoor", "dim", "dark")
LIGHT_WORDS = {
    "bright": "很亮",
    "daylight": "白天",
    "indoor": "室内",
    "dim": "偏暗",
    "dark": "很暗",
}
STATE_WORDS = {
    "0": "空的，没有白盒",
    "1": "一层白盒",
    "2": "两层白盒",
    "红": "两层白盒，顶上是红的",
    "蓝": "两层白盒，顶上是蓝的",
}
LEVEL_LOCK = {
    "bright": {"exposure": 25, "gain": 0, "brightness": -16},
    "daylight": {"exposure": 50, "gain": 16, "brightness": 0},
    "indoor": {"exposure": 150, "gain": 28, "brightness": 0},
    "dim": {"exposure": 400, "gain": 48, "brightness": 12},
    "dark": {"exposure": 1200, "gain": 72, "brightness": 24},
}


def light_hint(level: str) -> str:
    # 数字和档位顺序一致，和任务二的测试窗口一样。
    current = LIGHT_WORDS[level]
    exposure = float(LEVEL_LOCK[level]["exposure"])
    choices = "  ".join(f"{index}{LIGHT_WORDS[name]}" for index, name in enumerate(LEVELS, start=1))
    return f"光线 {current}，曝光 {exposure:.0f}。按 {choices}。q 退出"


def live_pitch_line(degrees: float | None) -> str:
    # 画面上的实时俯仰。正数是抬头，和投影用的角度同一个方向。
    if degrees is None:
        return "俯仰读不出来"
    shown = round(float(degrees), 1)
    if shown > 0.0:
        return f"俯仰 抬头 {shown:.1f} 度"
    if shown < 0.0:
        return f"俯仰 低头 {abs(shown):.1f} 度"
    return "俯仰 0.0 度"


def required_pitch_line(degrees: float | None) -> str | None:
    # 这一步要转到的俯仰，写在实时俯仰下面。没有目标就不写。
    if degrees is None:
        return None
    return f"要求 {format_pitch(float(degrees))}"


def window_pitch_lines(current: float | None, required: float | None) -> list[str]:
    rows = [live_pitch_line(current)]
    asked = required_pitch_line(required)
    if asked is not None:
        rows.append(asked)
    return rows


def light_from_key(key: int) -> str | None:
    index = int(key) - ord("1")
    if 0 <= index < len(LEVELS):
        return LEVELS[index]
    return None


def summary_lines(locked: dict[str, str | None]) -> list[str]:
    # 六格都走完以后，按窗口编号把最终结果再列一遍。没定下来的仍是还没看清。
    return ["总结"] + [f"{TOWER_NO[name]} {locked[name] or '还没看清'}" for name in DISPLAY]


def plain_state(view, locked: str | None, mode: str) -> str:
    if mode == "basic":
        if locked and view.in_view:
            text = STATE_WORDS[locked]
        elif view.blocked and not locked:
            text = "被挡住了，这一眼不写结论"
        elif view.skip_vote and view.conclusion is None and not locked:
            text = "前面有东西挡住了，还不能确定"
        else:
            text = "还没写下"
    elif locked:
        text = STATE_WORDS[locked]
    elif not view.in_view and view.needs_yaw:
        text = "不在画面里，要先转向"
    elif view.needs_pitch:
        text = "高度不在画面里，要改俯仰"
    elif view.blocked and view.conclusion is None:
        text = "被前面的台挡住了，还不能确定"
    elif view.skip_vote and view.conclusion is None:
        text = "前面有东西挡住了，还不能确定"
    elif view.conclusion is None:
        text = "还没看清"
    else:
        text = f"这一眼像是{STATE_WORDS[view.conclusion]}，还要再看几帧"
    # 只调整窗口说明，裁判输出和计票规则保持原样。
    if not locked:
        detail = getattr(view, "reason", None)
        if (
            view.conclusion is None and detail
            and not (view.needs_yaw or view.needs_pitch)
        ):
            text = detail
        elif detail and detail.startswith("等待相机稳定"):
            text = detail
        elif mode == "basic" and view.conclusion is not None:
            text = f"这一眼像是{STATE_WORDS[view.conclusion]}，等待连续帧锁定"
    shown = locked or view.conclusion
    if view.method == "深度" and view.conclusion is not None and shown == view.conclusion:
        return text + "，深度"
    if view.method == "黄地" and view.conclusion is not None and shown == view.conclusion:
        return text + "，黄地"
    if view.method == "远处" and view.conclusion is not None and shown == view.conclusion:
        return text + "，远处"
    return text


def step_light(level: str, white_frac: float, mean_v: float, wait: int) -> tuple[str, int]:
    # 调过一档之后停 5 帧，不每帧改曝光。空的绿台不是欠曝，不因此把曝光拉高。
    if wait > 0:
        return level, wait - 1
    index = LEVELS.index(level)
    if white_frac > 0.70 and index > 0:
        return LEVELS[index - 1], 5
    if white_frac < 0.01 and mean_v < 40.0 and index < len(LEVELS) - 1:
        return LEVELS[index + 1], 5
    return level, 0


class FieldWatch:
    def __init__(self, setup: Setup, intr: Intrinsics):
        self.setup = setup
        self.intr = intr
        self.pitch = float(setup.pitch_deg)
        self.roll = 0.0
        self.votes = {name: Vote() for name in DISPLAY}
        self.last_pose: CarPose | None = None
        self.light = "daylight"
        self.light_wait = 0
        self.light_hand = False
        self.views = None
        self.have_color = False
        self.tries = {name: 0 for name in DISPLAY}
        self.barren = {name: 0 for name in DISPLAY}
        self.visited: list[tuple[float, float]] = []
        self.used = {name: [] for name in DISPLAY}
        self.chase = {name: "local" for name in DISPLAY}
        self.goal: dict[str, tuple[float, float, str]] = {}
        self.hold = None
        self.stable_pose = None
        self.stable_pitch = self.pitch
        self.stable_frames = 0
        self.parallax_origins = {}

    def follow_pitch(self, measured: float | None) -> None:
        # 每帧用加速度计旋到彩色系之后的角度。读不到就留在上一帧。
        if measured is None:
            return
        self.pitch = float(measured)

    def required_pitch(self) -> float | None:
        # 正在执行的俯仰指令优先。否则用这一步那一格要对准的角度。
        if (
            self.hold is not None
            and self.hold.kind == "pitch"
            and self.hold.pitch is not None
            and self.last_pose is not None
            and not self._arrived(self.hold)
        ):
            return float(self.hold.pitch)
        if not self.setup.ready or self.views is None or self.last_pose is None or self.setup.center_to_corner is None:
            return None
        action = plan_action(
            self.views,
            {name: vote.locked for name, vote in self.votes.items()},
            self.last_pose,
            self.setup.center_to_corner,
            self.pitch,
            self.have_color,
            self.intr,
            self.tries,
            self.visited,
            {name: vote.recheck for name, vote in self.votes.items()},
            self.used,
            self.chase,
        )
        if action is None or action.cell is None:
            return None
        if action.kind == "pitch" and action.pitch is not None:
            return float(action.pitch)
        view = self.views.get(action.cell)
        if view is None:
            return None
        return float(view.pitch_target_deg)

    def choose_light(self, level: str) -> None:
        # 人按过数字键之后，不再自动改档，免得和手上的选择对着干。
        if level not in LEVELS:
            return
        self.light = level
        self.light_hand = True
        self.light_wait = 0

    def update(self, pose: CarPose, color=None, depth=None) -> None:
        if not self.setup.ready or self.setup.center_to_corner is None:
            self.views = None
            return
        if self.last_pose is not None and pose_moved(self.last_pose, pose):
            for vote in self.votes.values():
                vote.on_move()
            moved = math.hypot(pose.x - self.last_pose.x, pose.y - self.last_pose.y)
            if moved > 0.02:
                # 换了地方，五个视角重新看。重看落到预定的点上，才记成旧点或新点。
                self._note_arrival(pose)
                self.visited.append((self.last_pose.x, self.last_pose.y))
                self.tries = {name: 0 for name in DISPLAY}
            self.barren = {name: 0 for name in DISPLAY}
        if (self.stable_pose is None or pose_moved(self.stable_pose, pose)
                or abs(self.pitch - self.stable_pitch) > 1.0):
            self.stable_pose = pose
            self.stable_pitch = self.pitch
            self.stable_frames = 0
            for vote in self.votes.values():
                vote.on_move()
        else:
            self.stable_frames += 1
        self.last_pose = pose
        self.have_color = color is not None
        self.views = inspect_frame(
            pose,
            self.setup.center_to_corner,
            self.pitch,
            self.intr,
            color,
            depth,
            self.roll,
        )
        # 请求侧向观察时记住原始光心，不让移动途中的新帧不断移动视差基准。
        if color is not None:
            from car_pose import camera_position
            for name, view in self.views.items():
                if view.needs_parallax:
                    origin = camera_position(pose,self.setup.center_to_corner,0.)
                    self.parallax_origins.setdefault(name,tuple(origin[:2]))
                if name in self.parallax_origins:
                    if view.measurements is None:
                        view.measurements = {}
                    view.measurements["parallax_origin"] = self.parallax_origins[name]
                if self.votes[name].locked is not None and not self.votes[name].recheck:
                    self.parallax_origins.pop(name,None)
        if color is not None and self.stable_frames < 5:
            for view in self.views.values():
                if view.conclusion is None and view.reason:
                    continue
                view.reason = f"等待相机稳定：{self.stable_frames}/5帧"
        frac = max((view.white_frac for view in self.views.values()), default=0.0)
        mean_v = min((view.mean_v for view in self.views.values()), default=255.0)
        if color is not None and not self.light_hand:
            self.light, self.light_wait = step_light(self.light, frac, mean_v, self.light_wait)
        if color is not None and self.stable_frames >= 5:
            for name, view in self.views.items():
                if view.skip_vote:
                    continue
                vote = self.votes[name]
                was = vote.recheck
                vote.see(view.conclusion)
                if vote.recheck and not was:
                    # 冲突出现在当前视角，剩下的俯仰和偏角先用完，车先不走。
                    self.tries[name] = min(MAX_TRIES, self.tries[name] + 1)
                    vote.ready = False
            self._note_aim()

    def lines(self, mode: str) -> list[str]:
        if not self.setup.ready or self.views is None or self.last_pose is None or self.setup.center_to_corner is None:
            return ["车位还没对齐"]
        locked = {name: vote.locked for name, vote in self.votes.items()}
        if mode == "basic":
            rows = []
            for name in DISPLAY:
                if locked[name] and self.views[name].in_view:
                    rows.append(f"{name} {locked[name]}")
            return rows
        rows = [f"{name} {locked[name] or '还没看清'}" for name in DISPLAY]
        self._abandon()
        action = plan_action(
            self.views,
            locked,
            self.last_pose,
            self.setup.center_to_corner,
            self.pitch,
            self.have_color,
            self.intr,
            self.tries,
            self.visited,
            {name: vote.recheck for name, vote in self.votes.items()},
            self.used,
            self.chase,
        )
        self._sync_attempt(action)
        text = self._held(action)
        doubting = any(vote.recheck for vote in self.votes.values())
        if (
            text
            and not doubting
            and place_done(locked, self.last_pose, self.setup.center_to_corner, self.intr)
            and (text == "停止" or "平移" in text)
        ):
            rows.append("完成识别")
        if text:
            rows.append(text)
        if text == "停止" and all(self._settled(name) for name in DISPLAY):
            rows.extend(summary_lines(locked))
        return rows

    def window_lines(self, mode: str) -> list[str]:
        official = self.lines(mode)
        hint = light_hint(self.light)
        if official == ["车位还没对齐"]:
            return ["车位还没对齐，先不要认塔。", hint]
        rows = [hint]
        if self.views is None:
            return rows + official
        locked = {name: vote.locked for name, vote in self.votes.items()}
        for name in sorted(DISPLAY, key=lambda item: TOWER_NO[item]):
            view, vote = self.views[name], self.votes[name]
            text = plain_state(view, locked[name], mode)
            if (
                locked[name] is None and view.conclusion is not None
                and self.stable_frames >= 5
            ):
                text += f"；连续{vote.count}/15帧"
            if vote.recheck:
                text += f"；待重看，旧结果暂留；新结果{vote.count}/15帧"
            rows.append(f"{TOWER_NO[name]}：{text}")
        if mode == "extend" and len(official) > len(DISPLAY):
            rows.extend(official[len(DISPLAY) :])
        return rows

    def _sync_attempt(self, action) -> None:
        if action is None or action.cell is None:
            return
        if action.kind in ("pitch", "yaw", "wait") and action.attempt > self.tries.get(action.cell, 0):
            self.tries[action.cell] = action.attempt

    def _settled(self, name: str) -> bool:
        vote = self.votes[name]
        if self.chase[name] == "done":
            return True
        return vote.locked is not None and not vote.recheck

    def _plan(self, have_color: bool):
        return plan_action(
            self.views,
            {name: vote.locked for name, vote in self.votes.items()},
            self.last_pose,
            self.setup.center_to_corner,
            self.pitch,
            have_color,
            self.intr,
            self.tries,
            self.visited,
            {name: vote.recheck for name, vote in self.votes.items()},
            self.used,
            self.chase,
        )

    def _abandon(self) -> None:
        # 已经为这一格挪过一次，视角再用完仍没有稳定结论，就留下旧结果或还没看清。
        if self.views is None or self.last_pose is None or self.setup.center_to_corner is None:
            return
        for name, vote in self.votes.items():
            if self.chase[name] not in ("revisit", "fresh") or self.tries[name] < MAX_TRIES:
                continue
            if vote.locked is not None and not vote.recheck:
                continue
            self.chase[name] = "done"
            vote.recheck = False
            vote.ready = False

    def _note_arrival(self, pose: CarPose) -> None:
        if self.last_pose is None:
            return
        for name in list(self.goal):
            final_x, final_y, stage = self.goal[name]
            if math.hypot(pose.x - final_x, pose.y - final_y) > ARRIVE_M:
                continue
            self.used[name].append((self.last_pose.x, self.last_pose.y))
            self.chase[name] = stage
            self.goal.pop(name)

    def _remember_goal(self, action) -> None:
        if self.hold is not None and self.hold.kind == "move" and not self._arrived(self.hold):
            return
        if action is None or action.kind != "move" or action.chase_to is None:
            return
        if action.final_x is None or action.final_y is None:
            return
        for name in action.cells:
            self.goal[name] = (float(action.final_x), float(action.final_y), action.chase_to)

    def _note_aim(self) -> None:
        if self.views is None or self.last_pose is None or self.setup.center_to_corner is None:
            return
        self._abandon()
        action = self._plan(True)
        self._sync_attempt(action)
        self._remember_goal(action)
        if action is not None and action.cell is not None and self.votes[action.cell].recheck:
            # 只有已经对上这个视角、停下来看的时候，才把新结论往上计。
            self.votes[action.cell].ready = action.kind == "wait" and self.tries[action.cell] < MAX_TRIES
        if action is None or action.kind != "wait" or action.cell is None:
            return
        if self.views[action.cell].conclusion is not None:
            self.barren[action.cell] = 0
            return
        self.barren[action.cell] += 1
        if self.barren[action.cell] < 30:
            return
        self.tries[action.cell] = min(MAX_TRIES, self.tries[action.cell] + 1)
        self.votes[action.cell].ready = False
        self.barren[action.cell] = 0

    def _arrived(self, hold) -> bool:
        pose = self.last_pose
        if pose is None:
            return False
        if hold.kind == "pitch":
            return abs(self.pitch - float(hold.pitch)) <= 1.0
        if hold.kind == "yaw":
            turn = math.atan2(math.sin(pose.yaw - float(hold.yaw)), math.cos(pose.yaw - float(hold.yaw)))
            return abs(math.degrees(turn)) <= 1.0
        if hold.kind == "move":
            return math.hypot(pose.x - float(hold.x), pose.y - float(hold.y)) <= MOVE_ARRIVE_M
        return True

    def _held(self, action) -> str | None:
        if action is None or action.kind in ("wait", "idle"):
            self.hold = None
            return None
        if action.kind in ("stop", "stuck", "blocked"):
            self.hold = None
            return action.text
        if self.hold is not None and not self._arrived(self.hold):
            return self.hold.text
        self.hold = action
        self._remember_goal(action)
        return action.text

