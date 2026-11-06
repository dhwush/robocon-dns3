"""连续 15 帧同一结论才锁定。锁定后，还没看清的帧跳过。另有 15 帧别的结论才改锁。"""

from __future__ import annotations

from dataclasses import dataclass


LOCK_FRAMES = 15


@dataclass
class Vote:
    locked: str | None = None
    pending: str | None = None
    count: int = 0
    disagree: int = 0

    def see(self, conclusion: str | None) -> None:
        if self.locked is not None:
            # 没看清既不是赞成，也不抹掉已经攒下的反对。和锁上相同的一帧才清零。
            if conclusion is None:
                return
            if conclusion == self.locked:
                self.disagree = 0
                return
            self.disagree += 1
            if self.disagree >= LOCK_FRAMES:
                self.locked = conclusion
                self.pending = conclusion
                self.count = LOCK_FRAMES
                self.disagree = 0
            return
        if conclusion is None:
            self.pending = None
            self.count = 0
            return
        if conclusion == self.pending:
            self.count += 1
        else:
            self.pending = conclusion
            self.count = 1
        if self.count >= LOCK_FRAMES:
            self.locked = conclusion

    def on_move(self) -> None:
        # 车被推走之后，没锁定的格重新计票。已经锁定的不在这里打开。
        if self.locked is None:
            self.pending = None
            self.count = 0
            self.disagree = 0

