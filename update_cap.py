from pathlib import Path
import ast
import shutil
import datetime
import sys

root = (
    Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) > 1
    else Path.home() / "robocon/robocon3"
)
files = {}

p = root / "classify.py"
s = p.read_text(encoding="utf-8")
if "def _cap_from_depth(" in s:
    sys.exit("已经安装带顶块判定补丁")

helper = '''
def _cap_from_depth(cell, world, xs, ys, distance, color, mask, intr):
    """整塔高点必须同时有白盒支撑和相接的红蓝顶块证据。"""
    support = np.abs(world[:, 2] - cell.two_z) <= 0.08
    if int(support.sum()) < 10:
        return None, "疑似带顶块：缺少两层白盒顶附近的有效白色深度"

    support_xy = world[support, :2]
    span = (
        np.percentile(support_xy, 95, axis=0)
        - np.percentile(support_xy, 5, axis=0)
    )
    if float(np.linalg.norm(span)) < 0.12:
        return None, "疑似带顶块：两层白盒支撑证据过窄"

    # 只用白盒高度范围构造顶边，排除整塔顶部的高点。
    body = world[:, 2] <= cell.two_z + 0.08
    white_body = np.zeros(mask.shape, np.uint8)
    white_body[ys[body], xs[body]] = 255

    z_cam = float(np.median(distance[support]))
    cap_px = max(
        4, int(round(0.20 * intr.fy / max(z_cam, 0.3)))
    )
    band, tops = _band_above(white_body, mask, cap_px)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = (
        (cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1])) > 0)
        | (cv2.inRange(
            hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1])
        ) > 0)
    )
    blue = cv2.inRange(
        hsv, np.array(BLUE[0]), np.array(BLUE[1])
    ) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ((red | blue) & band).astype(np.uint8),
        connectivity=8,
    )
    rows = np.arange(mask.shape[0])[:, None]
    contact = band & (
        rows >= tops - max(3, int(0.12 * cap_px))
    )

    accepted = np.zeros(mask.shape, np.uint8)
    candidates = 0

    for index in range(1, count):
        x, y, width, height, pixels = stats[index]
        part = labels == index
        width_m = _metric(float(width), z_cam, intr.fx)
        height_m = _metric(float(height), z_cam, intr.fy)

        if (
            pixels >= max(12, int(0.03 * cap_px**2))
            and 0.08 <= width_m <= 0.34
            and 0.04 <= height_m <= 0.28
            and np.any(part & contact)
        ):
            candidates += 1
            accepted[part] = 255

    if candidates != 1:
        return None, (
            "疑似带顶块：没有唯一且尺寸合理、与白盒相接的红蓝块"
        )

    faction, unclear = _faction(
        color, white_body, accepted, z_cam, intr
    )
    if unclear or faction is None:
        return None, "已见两层白盒和疑似顶块：红蓝阵营仍不明确"

    return faction, None


'''

anchor = "def _depth_on_cell("
if anchor not in s:
    raise RuntimeError("缺少深度判定函数，未修改文件")
s = s.replace(anchor, helper + anchor, 1)

old = '''    if not one_ok and not two_ok:
        return fail('''

start = s.index("def _depth_on_cell(")
if old not in s[start:]:
    raise RuntimeError("判层代码版本不匹配，未修改文件")
index = s.index(old, start)

new = '''    # 整塔高点与白盒层高分开判定。
    if top_z > cell.two_z + 0.08:
        if abs(top_z - cell.cap_z) > 0.06:
            return fail(
                f"整塔高度不符：{top_z*100:.1f}厘米，"
                f"带顶块允许{(cell.cap_z-.06)*100:.1f}"
                f"至{(cell.cap_z+.06)*100:.1f}厘米",
                piece, outside, True,
            )

        faction, cap_reason = _cap_from_depth(
            cell, world, xs, ys, distance, color, mask, intr
        )
        if faction is None:
            return fail(cap_reason, piece, outside, True)

        if diagnostic is not None:
            diagnostic["height_source"] = (
                "整塔高度＋两层白盒支撑＋红蓝顶块"
            )
            diagnostic["cap_verified"] = True

        return faction, piece, outside, False

    if not one_ok and not two_ok:
        return fail('''

s = s[:index] + s[index:].replace(old, new, 1)
ast.parse(s)
files[p] = s

p = root / "任务三说明.md"
files[p] = p.read_text(encoding="utf-8") + """
- 2026-11-08：深度高点超过两层白盒顶8厘米时，单独检查带顶块结构。整塔顶允许相对标称高度偏6厘米，还必须有两层白盒顶附近的白色支撑深度，以及尺寸合理、贴着白盒的唯一红蓝块，才输出阵营。高度接近90厘米本身不直接判两层或阵营，证据不足写具体原因。相机安装高度沿用现值，本次只更正整塔与白盒层高的判定分工。
"""

backup = root.parent / (
    root.name + "_cap_backup_"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)

for p in files:
    target = backup / p.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, target)

try:
    for p, text in files.items():
        p.write_text(text, encoding="utf-8")
except Exception:
    for p in files:
        shutil.copy2(backup / p.relative_to(root), p)
    raise

print("更新完成，原文件备份：", backup)