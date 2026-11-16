"""场地表。格子位置以小车右前角为原点，不从画面里重找。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 近处左侧外角。纵深向 +Y，横向向 +X。
# 原点是起步时的右前角。它在最近一行正中那块 40 cm 台的近边外侧 0.854 m，
# 相对这块台靠近车的右上角（图上远边的右角，换到车的坐标后是近边的左角）向左 0.615 m。
# 车在原先图的远侧起步，头朝场地，所以原先的最远一行现在是最近一行，左右对调。
ORIGIN_X = -1.815
ORIGIN_Y = 0.854
CELL_M = 1.2
# 用户确认的走廊纵深上限；图上方为+Y。整车避让余量由导航层膨胀。
LEFT_CORRIDOR_MAX_Y = 1.854
RIGHT_CORRIDOR_MAX_Y = ORIGIN_Y + 3 * CELL_M
# 全场后方边界：原点后60 cm，整车不得越线。
DRIVING_MIN_Y = -0.60


def driving_exclusions():
    """仅用于行驶约束，不作为视觉遮挡物。无外侧可绕过的有限边界。"""
    inf = float('inf')
    return ((-inf, ORIGIN_X, LEFT_CORRIDOR_MAX_Y, inf),
            (-inf, inf, RIGHT_CORRIDOR_MAX_Y, inf),
            (-inf, inf, -inf, DRIVING_MIN_Y))


# 1、5、6 直接放在木头地面上，每块 50 cm 见方，不跟 1.2 m 的台齐边。
GROUND_M = 0.50
# 2、3、4 号的塔只占 20 cm 台正中的 50 cm 见方。白盒和塔顶都不超出这块。
FOOT_M = 0.50
# 搜索比塔的范围再宽 5 cm，用来盖住车位 5 cm 以内的误差。
MARGIN_M = 0.05

# 从远到近、从左到右。单位米。车在最近一行外侧，原来图上的左右已经对调。
_HEIGHTS = (
    (0.40, 0.20, 0.40),
    (0.20, 0.40, 0.60),
    (0.40, 0.60, 0.40),
    (0.20, 0.40, 0.20),
)

_TOWER_NAMES = {
    (1, 0): "第三行左侧",
    (3, 0): "最近一行左侧",
    (3, 2): "最近一行右侧",
}

# 按窗口编号 1 到 6 列。1、5、6 在地面上，2、3、4 在 20 cm 塔基的台心。
DISPLAY = (
    "最近左侧地面",
    "最近一行左侧",
    "第三行左侧",
    "最近一行右侧",
    "最近右侧地面",
    "右侧地面远一格",
)

TOWER_NO = {name: str(index) for index, name in enumerate(DISPLAY, start=1)}


@dataclass(frozen=True)
class Cell:
    name: str
    row: int
    col: int
    x: float
    y: float
    height: float
    tower: bool
    ground: bool = False
    size: float = CELL_M

    @property
    def base_z(self) -> float:
        # 地面建塔点从 0 起算。塔基上的点从台面起算。
        return 0.0 if self.ground else float(self.height)

    @property
    def one_z(self) -> float:
        return 0.35 if self.ground else 0.55

    @property
    def two_z(self) -> float:
        return 0.70 if self.ground else 0.90

    @property
    def cap_z(self) -> float:
        return 0.90 if self.ground else 1.10

    def center(self, z: float) -> np.ndarray:
        return np.array([self.x, self.y, z], dtype=float)

    @property
    def foot(self) -> float:
        # 塔实际占据的水平方块。地面木块整块都是。20 cm 台上只有台心 50 cm。
        return FOOT_M if self.tower else self.size

    def bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        half = self.size / 2.0 + margin
        return (self.x - half, self.x + half, self.y - half, self.y + half)

    def search_bounds(self, margin: float = 0.0) -> tuple[float, float, float, float]:
        half = self.foot / 2.0 + margin
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


def _ground(name: str, row: int, col: int, x0: float, x1: float, y0: float, y1: float) -> Cell:
    return Cell(
        name,
        row,
        col,
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        0.0,
        True,
        True,
        x1 - x0,
    )


def ground_towers() -> tuple[Cell, ...]:
    # 量的时候那张平面图左右、上下都是反的。图上向左是车的右（+X），图上向下是车的前（+Y）。
    # 图上的上沿是靠近车的一边。三块都从靠近参考台的那个角往外伸 50 cm，中间留出量到的缝。
    three_x1 = ORIGIN_X + 3 * CELL_M
    four_x0 = ORIGIN_X
    near_y0 = ORIGIN_Y
    # 从近往远第三行右侧那块 60 cm 台。图上它在从下往上第二行的最左边。
    sixty_x1 = ORIGIN_X + 3 * CELL_M
    sixty_y0 = ORIGIN_Y + 2 * CELL_M
    five_x0 = three_x1 + 0.535
    five_y0 = near_y0 + 0.385
    one_x1 = four_x0 - 0.76
    one_y0 = near_y0 + 0.10
    six_x0 = sixty_x1 + 0.615
    six_y0 = sixty_y0 + 0.46
    return (
        _ground("最近左侧地面", 3, -1, one_x1 - GROUND_M, one_x1, one_y0, one_y0 + GROUND_M),
        _ground("最近右侧地面", 3, 3, five_x0, five_x0 + GROUND_M, five_y0, five_y0 + GROUND_M),
        _ground("右侧地面远一格", 1, 3, six_x0, six_x0 + GROUND_M, six_y0, six_y0 + GROUND_M),
    )


def tower_cells() -> tuple[Cell, ...]:
    by_name = {cell.name: cell for cell in all_cells() if cell.tower}
    for cell in ground_towers():
        by_name[cell.name] = cell
    return tuple(by_name[name] for name in DISPLAY)


def search_contains(cell: Cell, xy: np.ndarray, margin: float = MARGIN_M) -> bool:
    x0, x1, y0, y1 = cell.search_bounds(margin)
    x, y = float(xy[0]), float(xy[1])
    return x0 <= x <= x1 and y0 <= y <= y1

