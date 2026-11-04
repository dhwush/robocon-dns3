"""基础和拓展的窗口文字。认塔相同，印出来的字不同。"""

from __future__ import annotations

from car_pose import CarPose, Intrinsics, Setup, pose_moved
from classify import inspect_frame
from field_map import DISPLAY, TOWER_NO
from motion import next_action
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


def light_from_key(key: int) -> str | None:
    index = int(key) - ord("1")
    if 0 <= index < len(LEVELS):
        return LEVELS[index]
    return None


def plain_state(view, locked: str | None, mode: str) -> str:
    if mode == "basic":
        if locked and view.in_view:
            text = STATE_WORDS[locked]
        elif view.blocked and not locked:
            text = "被挡住了，这一眼不写结论"
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
    elif view.conclusion is None:
        text = "还没看清"
    else:
        text = f"这一眼像是{STATE_WORDS[view.conclusion]}，还要再看几帧"
    shown = locked or view.conclusion
    if view.method == "深度" and view.conclusion is not None and shown == view.conclusion:
        return text + "，深度"
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
        if pose_moved(self.last_pose, pose):
            for vote in self.votes.values():
                vote.on_move()
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
        frac = max((view.white_frac for view in self.views.values()), default=0.0)
        mean_v = min((view.mean_v for view in self.views.values()), default=255.0)
        if color is not None and not self.light_hand:
            self.light, self.light_wait = step_light(self.light, frac, mean_v, self.light_wait)
        if color is not None:
            for name, view in self.views.items():
                self.votes[name].see(view.conclusion)

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
        action = next_action(
            self.views,
            locked,
            self.last_pose,
            self.setup.center_to_corner,
            self.pitch,
            self.have_color,
        )
        if action:
            rows.append(action)
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
            rows.append(f"{TOWER_NO[name]}：{plain_state(self.views[name], locked[name], mode)}")
        if mode == "extend" and len(official) == len(DISPLAY) + 1:
            rows.append(official[-1])
        return rows

