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


def load(name):
    return files.get(name, (root / name).read_text(encoding="utf-8"))


def change(name, old, new):
    text = load(name)
    if old not in text:
        raise RuntimeError("版本不匹配，未写入：" + name + " " + old[:45])
    files[name] = text.replace(old, new, 1)


n = "classify.py"
if "measurements: dict | None = None" in load(n):
    sys.exit("已经更新，无需重复运行")

change(
    n,
    "    reason: str | None = None",
    "    reason: str | None = None\n    measurements: dict | None = None",
)
change(
    n,
    "    reason = None\n\n    def pack",
    "    reason = None\n    diagnostic = {}\n\n    def pack",
)
change(
    n,
    "            reason if conclusion is None else None,",
    "            reason if conclusion is None else None,\n            diagnostic,",
)
change(n, "    diagnostic = {}\n    depth_state", "    depth_state")

change(
    n,
    "    top_z = float(np.median(world[top, 2]))",
    '''    top_z = float(np.median(world[top, 2]))
    if diagnostic is not None:
        diagnostic["top_pixels"] = np.column_stack(
            (xs[top], ys[top])
        )[::max(1, int(top.sum())//80)].tolist()
        diagnostic["depth_height_cm"] = top_z * 100
        diagnostic["height_source"] = "深度最高切片"''',
)
change(
    n,
    "            top_z = seen\n    one_ok",
    '''            top_z = seen
            if diagnostic is not None:
                diagnostic["height_source"] = "彩色顶边补高"
    one_ok''',
)

# 兼容上一轮更新脚本的两种等价换行格式。
variants = [
    'f"层高不符：最高点离地{top_z*100:.0f}厘米，预期{cell.one_z*100:.0f}或{cell.two_z*100:.0f}厘米"',
    'f"层高不符：最高点离地{top_z*100:.0f}厘米，"\n'
    '            f"预期{cell.one_z*100:.0f}或{cell.two_z*100:.0f}厘米"',
]
for old in variants:
    if old in load(n):
        change(
            n,
            old,
            '''f"层高不符：采用高度{top_z*100:.1f}厘米；一层允许{cell.one_z*80:.1f}至{cell.one_z*120:.1f}，两层允许{cell.two_z*80:.1f}至{cell.two_z*120:.1f}厘米；来源{diagnostic.get('height_source', '深度') if diagnostic is not None else '深度'}"''',
        )
        break
else:
    raise RuntimeError("层高说明版本不匹配，未写入")

for function, next_function in [
    ("_color_reading", "_far_reading"),
    ("_far_reading", "_classify_cell"),
]:
    text = load(n)
    start = text.index("def " + function + "(")
    end = text.index("def " + next_function + "(", start)
    block = text[start:end]

    block = block.replace(
        "    two_h: float,\n)",
        "    two_h: float,\n    diagnostic: dict | None = None,\n)",
    )
    block = block.replace(
        "    # ",
        '''    def fail(text):
        if diagnostic is not None:
            diagnostic["reason"] = text
        return None

    # ''',
        1,
    )

    if function == "_far_reading":
        replacements = [
            (
                "    if int(local.max()) == 0:\n        return None",
                '''    if int(local.max()) == 0:
        return fail("远处：搜索框内没有通过颜色阈值的白色")''',
            ),
            (
                "    if not found:\n        return None",
                '''    if not found:
        return fail("远处：没有宽15至65厘米且至少2×2像素的白块")''',
            ),
            (
                "    if contour is None:\n        return None",
                '''    if contour is None:
        return fail("远处：补缝后没有有效轮廓")''',
            ),
            (
                "        return None\n    _x, _y, w, h =",
                '''        return fail("远处轮廓不符：周长无效或轮廓过圆")
    _x, _y, w, h =''',
            ),
            (
                "    if w < 2 or h < 2 or area < FILL_MIN * float(w * h):\n        return None",
                '''    if w < 2 or h < 2 or area < FILL_MIN * float(w * h):
        return fail(
            f"远处填充不足：{100*area/max(1,w*h):.1f}%，"
            f"要求至少75%；框{w}×{h}像素"
        )''',
            ),
            (
                "    if width_m < WIDTH_MIN_M or width_m > WIDTH_MAX_M:\n        return None",
                '''    if width_m < WIDTH_MIN_M or width_m > WIDTH_MAX_M:
        return fail(
            f"远处宽度不符：{width_m*100:.1f}厘米，要求15至65厘米"
        )''',
            ),
        ]
        for old, new in replacements:
            if old not in block:
                raise RuntimeError("远处诊断版本不匹配，未写入")
            block = block.replace(old, new, 1)

    block = block.replace(
        "    if not one_ok and not two_ok:\n        return None",
        '''    if not one_ok and not two_ok:
        return fail(
            f"彩色层高不符：可见高{height_m*100:.1f}厘米；"
            f"一层允许{one_h*50:.1f}至{one_h*150:.1f}，"
            f"两层允许{two_h*50:.1f}至{two_h*150:.1f}厘米"
        )''',
        1,
    )
    block = block.replace(
        "    if unclear:\n        return None",
        '''    if unclear:
        return fail("阵营不清：颜色有效面积不足或上下关系不明确")''',
        1,
    )
    files[n] = text[:start] + block + text[end:]

change(
    n,
    "reading = _color_reading(cell, accepted, color, mask, intr, blocked, one_h, two_h)",
    "reading = _color_reading(cell, accepted, color, mask, intr, blocked, one_h, two_h, diagnostic)",
)
change(
    n,
    'reason = "彩色判定未通过：层高不符或塔顶阵营不清"',
    'reason = diagnostic.get("reason", "彩色判定未通过")',
)
change(
    n,
    "far = _far_reading(color, white, mask, z_cam, intr, blocked, one_h, two_h)",
    "far = _far_reading(color, white, mask, z_cam, intr, blocked, one_h, two_h, diagnostic)",
)
change(
    n,
    'reason = "远处彩色未通过：宽高、填充或轮廓不符"',
    'reason = diagnostic.get("reason", "远处彩色未通过")',
)

change(
    "report.py",
    '            rows.append(f"{TOWER_NO[name]}：{plain_state(self.views[name], locked[name], mode)}")',
    '''            view, vote = self.views[name], self.votes[name]
            text = plain_state(view, locked[name], mode)
            if (
                locked[name] is None and view.conclusion is not None
                and self.stable_frames >= 5
            ):
                text += f"；连续{vote.count}/15帧"
            if vote.recheck:
                text += f"；待重看，旧结果暂留；新结果{vote.count}/15帧"
            rows.append(f"{TOWER_NO[name]}：{text}")''',
)

change(
    "view_field.py",
    "            anchor = view.piece if view.piece is not None else view.region",
    '''            measure = getattr(view, "measurements", None) or {}
            points = measure.get("top_pixels", [])
            for x, y in points:
                cv2.circle(
                    board, (int(x), int(y)), 2, (255, 0, 255), -1
                )
            if points:
                x, y = points[len(points)//2]
                height = measure.get("depth_height_cm", 0)
                tags.append((
                    int(x)+6, max(0,int(y)-18),
                    f"{TOWER_NO[name]} 深度取样 {height:.1f}cm",
                    (255,0,255),
                ))
            anchor = view.piece if view.piece is not None else view.region''',
)

files["任务三说明.md"] = load("任务三说明.md") + """
- 2026-11-08：远处失败原因按颜色、尺寸、填充、轮廓、层高分别显示，层高保留一位小数并写允许区间和来源。紫色点标出最高有效深度切片，窗口补上连续计票和重看进度。识别阈值保持不变。
"""

# 全部检查完成后才备份和写入。
for name, text in files.items():
    if name.endswith(".py"):
        ast.parse(text)

backup = root.parent / (
    root.name + "_detail_backup_"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
for name in files:
    target = backup / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, target)

try:
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
except Exception:
    for name in files:
        shutil.copy2(backup / name, root / name)
    raise

print("更新完成，原文件备份：", backup)