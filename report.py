"""基础和拓展的窗口文字。认塔相同，印出来的字不同。"""

from __future__ import annotations

from car_pose import CarPose, Intrinsics, Setup, pose_moved
from classify import inspect_frame
from field_map import DISPLAY
from motion import next_action
from vote import Vote


LEVELS = ("bright", "daylight", "indoor", "dim", "dark")
LEVEL_LOCK = {
    "bright": {"exposure": 25, "gain": 0, "brightness": -16},
    "daylight": {"exposure": 50, "gain": 16, "brightness": 0},
    "indoor": {"exposure": 150, "gain": 28, "brightness": 0},
    "dim": {"exposure": 400, "gain": 48, "brightness": 12},
    "dark": {"exposure": 1200, "gain": 72, "brightness": 24},
}


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
        self.votes = {name: Vote() for name in DISPLAY}
        self.last_pose: CarPose | None = None
        self.light = "daylight"
        self.light_wait = 0
        self.views = None
        self.have_color = False

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
        )
        frac = max((view.white_frac for view in self.views.values()), default=0.0)
        mean_v = min((view.mean_v for view in self.views.values()), default=255.0)
        if color is not None:
            self.light, self.light_wait = step_light(self.light, frac, mean_v, self.light_wait)
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

