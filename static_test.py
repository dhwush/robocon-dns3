"""任务三基础静态测试：python static_test.py [--pitch DEG]。"""
from __future__ import annotations

import numpy as np

from car_pose import CarPose, Setup, load_intrinsics, load_setup
from field_map import TOWER_NO
from report import FieldWatch
from motion import format_pitch


class StaticFieldWatch(FieldWatch):
    """只改变俯仰，保留原有稳定等待和连续计票，不调用移动规划。"""

    def _note_aim(self):
        # 静态复核只需在目标俯仰稳定后重新采样，不要求转车。
        for name in self.targets():
            vote = self.votes[name]
            if vote.recheck:
                vote.ready = abs(self.pitch - self.views[name].pitch_target_deg) <= 5.0

    def targets(self):
        if self.views is None:
            return []
        return sorted((name for name, view in self.views.items() if not view.needs_yaw),
                      key=lambda name: TOWER_NO[name])

    def required_pitch(self):
        for name in self.targets():
            vote = self.votes[name]
            if vote.locked is None or vote.recheck:
                return float(self.views[name].pitch_target_deg)
        return self.pitch

    def lines(self, mode="static"):
        rows = ["静态基础测试：右前角原点 x=0 y=0，航向0°；保持车身不动"]
        names = self.targets()
        for name in names:
            vote = self.votes[name]
            value = vote.locked if not vote.recheck else None
            rows.append(f"{TOWER_NO[name]}号 {value if value is not None else '还没看清'}")
        remaining = sum(self.votes[name].locked is None or self.votes[name].recheck for name in names)
        rows.append(f"当前朝向范围内{len(names)}塔，已确认{len(names)-remaining}，未确认{remaining}；验收最多错1塔（须核对真值）")
        if names:
            rows.append(f"要求 {format_pitch(self.required_pitch())}" if remaining else "当前朝向范围内识别完成，保持俯仰")
        return rows

    def window_lines(self, mode="static"):
        return self.lines(mode) + super().window_lines("basic")


def run_static(args):
    from view_field import ROOT, _live, _once
    loaded = load_setup(str(ROOT / "config" / "field.yaml"))
    pitch = loaded.pitch_deg if args.pitch is None else float(args.pitch)
    setup = Setup(False, np.zeros(2), pitch, "", "rad", origin_basic=True)
    watch = StaticFieldWatch(setup, load_intrinsics(str(ROOT / "config" / "intrinsics.yaml")))
    args.pose_source = "static"
    args.mode = "static"
    pose = CarPose(0., 0., 0.)
    return _once(args, watch, pose) if args.image or args.once else _live(args, watch, pose)


if __name__ == "__main__":
    import sys
    from view_field import main
    sys.argv.insert(1, "static")
    main()
