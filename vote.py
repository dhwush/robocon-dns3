"""最近 20 帧内同一结论达到 15 帧才锁定，未知可间断，明确冲突重新计。锁定后再有 15 帧不同结论，先标成待重看，不立刻改锁。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


LOCK_FRAMES = 15
WINDOW_FRAMES = 20


@dataclass
class Vote:
    locked: str | None = None
    pending: str | None = None
    count: int = 0
    disagree: int = 0
    recheck: bool = False
    ready: bool = False
    _history: deque[str | None] = field(
        default_factory=lambda: deque(maxlen=WINDOW_FRAMES), repr=False
    )

    def see(self, conclusion: str | None) -> None:
        if self.recheck:
            # 换视角或换点到位之前不计。到位后的稳定结论才覆盖旧结果。
            if not self.ready:
                return
            self._count_fresh(conclusion)
            if self.pending is not None and self.count >= LOCK_FRAMES:
                self.locked = self.pending
                self.recheck = False
                self.ready = False
                self.disagree = 0
            return
        if self.locked is not None:
            # 没看清既不是赞成，也不抹掉已经攒下的反对。和锁上相同的一帧才清零。
            if conclusion is None:
                return
            if conclusion == self.locked:
                self.disagree = 0
                return
            self.disagree += 1
            if self.disagree >= LOCK_FRAMES:
                self.recheck = True
                self.ready = False
                self._history.clear()
                self.pending = None
                self.count = 0
                self.disagree = 0
            return
        self._count_fresh(conclusion)
        if conclusion is not None and self.count >= LOCK_FRAMES:
            self.locked = conclusion

    def _count_fresh(self, conclusion: str | None) -> None:
        # 未知占用窗口但不立即清零；明确冲突开始一轮新证据。
        if conclusion is not None and conclusion != self.pending:
            self._history.clear()
            self.pending = conclusion
        self._history.append(conclusion)
        self.count = sum(item == self.pending for item in self._history) if self.pending else 0
        if self.count == 0:
            self.pending = None

    def on_move(self) -> None:
        # 车被推走之后，没锁定的格重新计票。待重看的旧结论留着，计票从头来。
        self._history.clear()
        self.pending = None
        self.count = 0
        self.disagree = 0
        self.ready = False
        if self.locked is None:
            self.recheck = False

