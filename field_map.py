"""场地表。格子位置以小车右前角为原点，不从画面里重找。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 近处左侧外角。纵深向 +Y，横向向 +X。
ORIGIN_X = -0.211
ORIGIN_Y = 3.05
CELL_M = 1.2
# 白盒可以不居中，搜索比台面再宽 5 cm，用来盖住车位 5 cm 以内的误差。
MARGIN_M = 0.05

# 从远到近、从左到右。单位米。
_HEIGHTS = (
    (0.20, 0.40, 0.20),
    (0.40, 0.60, 0.40),
    (0.60, 0.40, 0.20),
    (0.40, 0.20, 0.40),
)

_TOWER_NAMES = {
    (0, 0): "最远一行左侧",
    (0, 2): "最远一行右侧",
    (2, 2): "从近往远第二行右侧",
    (3, 1): "最近一行正中",
}

# 拓展指令按从近到远列。
DISPLAY = (
    "最近一行正中",
    "从近往远第二行右侧",
    "最远一行右侧",
    "最远一行左侧",
)


@dataclass(frozen=True)
class Cell:
    name: str
    row: int
    col: int
    x: float
    y: float
    height: float
    tower: bool

    def center(self, z: float) -> np.ndarray:
        return np.array([self.x, self.y, z], dtype=float)

    def bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        half = CELL_M / 2.0 + margin
        return (self.x - half, self.x + half, self.y - half, self.y + half)


def _center(row: int, col: int) -> tuple[float, float]:
    # 第 0 行在最远处。近边是 ORIGIN_Y，远边再往前 4 格。
    x = ORIGIN_X + (col + 0.5) * CELL_M
    y = ORIGIN_Y + (3 - row + 0.5) * CELL_M
    return x, y


def all_cells() -> tuple[Cell, ...]:
    cells = []
    for row, heights in enumerate(_HEIGHTS):
        for col, height in enumerate(heights):
            x, y = _center(row, col)
            name = _TOWER_NAMES.get((row, col), "")
            cells.append(Cell(name, row, col, x, y, float(height), name != ""))
    return tuple(cells)


def tower_cells() -> tuple[Cell, ...]:
    by_name = {cell.name: cell for cell in all_cells() if cell.tower}
    return tuple(by_name[name] for name in DISPLAY)


def search_contains(cell: Cell, xy: np.ndarray, margin: float = MARGIN_M) -> bool:
    x0, x1, y0, y1 = cell.bounds(margin)
    x, y = float(xy[0]), float(xy[1])
    return x0 <= x <= x1 and y0 <= y <= y1

