"""任务三基础静态测试：python static_test.py [--pitch DEG]。"""
from __future__ import annotations

import numpy as np

from car_pose import CarPose, Setup, load_intrinsics, load_setup
from field_map import TOWER_NO
from report import FieldWatch


class StaticFieldWatch(FieldWatch):
    """固定车身和相机，只识别当前视场，不调用动作规划。"""

    def _note_aim(self):
        for name in self.targets():
            if self.votes[name].recheck:
                self.votes[name].ready = True

    def targets(self):
        if self.views is None:
            return []
        return sorted((name for name, view in self.views.items() if view.in_view),
                      key=lambda name: TOWER_NO[name])

    def required_pitch(self):
        return None

    def lines(self, mode="static"):
        rows = ["静态识别：使用当前右前角位姿；停止！保持车身和相机不动"]
        names = self.targets()
        for name in names:
            vote = self.votes[name]
            value = vote.locked if not vote.recheck else None
            rows.append(f"{TOWER_NO[name]}号 {value if value is not None else '还没看清'}")
        remaining = sum(self.votes[name].locked is None or self.votes[name].recheck for name in names)
        rows.append(f"当前朝向范围内{len(names)}塔，已确认{len(names)-remaining}，未确认{remaining}；按新版规则记录结果，不设通过门槛")
        if names:
            rows.append("保持当前位置和相机姿态，继续观察" if remaining else "当前视场识别完成，停止并保持")
        return rows

    def window_lines(self, mode="static"):
        return self.lines(mode) + super().window_lines("basic")


def run_static(args):
    from view_field import ROOT, _live, _once
    from radar_pose import normalize_source
    normalize_source(args)
    loaded = load_setup(str(ROOT / "config" / "field.yaml"))
    pitch = loaded.pitch_deg if args.pitch is None else float(args.pitch)
    setup = Setup(False, np.zeros(2), pitch, "", "rad", origin_basic=True)
    watch = StaticFieldWatch(setup, load_intrinsics(str(ROOT / "config" / "intrinsics.yaml")))
    # 静态只限制识别策略，不再把实时定位强制改成固定零位姿。
    if getattr(args, "offline", False):
        args.pose_source = "static"
    args.mode = "static"
    pose = CarPose(0., 0., 0.)
    return _once(args, watch, pose) if args.image or args.once else _live(args, watch, pose)


if __name__ == "__main__":
    import sys
    from view_field import main
    sys.argv.insert(1, "static")
    main()
