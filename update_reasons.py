from pathlib import Path
import sys
import shutil
import datetime
import ast

root = (
    Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) > 1
    else Path.home() / "robocon/robocon3"
)
changes = {}

p = root / "classify.py"
s = p.read_text(encoding="utf-8")
if "reason: str | None = None" in s:
    sys.exit("已经安装具体原因显示，无需重复运行。")


def replace(old, new):
    global s
    if old not in s:
        raise RuntimeError("文件版本不匹配，未写入文件：" + old[:70])
    s = s.replace(old, new, 1)


replace(
    "    skip_vote: bool = False",
    "    skip_vote: bool = False\n    reason: str | None = None",
)

start = s.index("def _depth_on_cell(")
end = s.index("\ndef _outline(", start)
a = s[start:end]

a = a.replace(
    "    roll: float = 0.0,\n)",
    "    roll: float = 0.0,\n    diagnostic: dict | None = None,\n)",
)

a = a.replace(
    "    if cloud is None:",
    '''    def fail(reason, piece=None, outside=False, veto=False):
        if diagnostic is not None:
            diagnostic["reason"] = reason
        return None, piece, outside, veto

    if cloud is None:''',
)

a = a.replace(
    "return None, None, False, False",
    'return fail("深度不足：没有有效白点云")',
    1,
)
a = a.replace(
    "return None, None, False, False",
    'return fail("深度不足：格内有效白点少于10个")',
    1,
)
a = a.replace(
    "return None, None, False, False",
    'return fail("深度不足：盒身有效点少于30个")',
    1,
)
a = a.replace(
    "return None, None, False, True",
    '''return fail(
            f"尺寸不符：深度水平跨度{float(spread.max())*100:.0f}厘米，要求15至65厘米",
            veto=True,
        )''',
    1,
)
a = a.replace(
    "return None, None, outside, False",
    'return fail("深度不足：没有连续的高度切片", outside=outside)',
    1,
)
a = a.replace(
    "return None, None, outside, False",
    'return fail("顶边出画：深度最高一段贴近画面边缘", outside=outside)',
    1,
)
a = a.replace(
    "return None, piece, outside, True",
    '''return fail(
            f"层高不符：最高点离地{top_z*100:.0f}厘米，"
            f"预期{cell.one_z*100:.0f}或{cell.two_z*100:.0f}厘米",
            piece, outside, True,
        )''',
    1,
)
a = a.replace(
    "return None, piece, outside, True",
    'return fail("阵营不清：塔顶颜色不足或上下关系不明确", piece, outside, True)',
    1,
)
s = s[:start] + a + s[end:]

replace(
    "    region = _outline(mask)\n\n    def pack",
    "    region = _outline(mask)\n    reason = None\n\n    def pack",
)
replace(
    "            skip,\n        )",
    "            skip,\n            reason if conclusion is None else None,\n        )",
)
replace(
    '''    if color is None:
        return pack(None, 0.0, 255.0)''',
    '''    if color is None:
        reason = "等待有效画面或曝光稳定"
        return pack(None, 0.0, 255.0)''',
)
replace(
    '''        return pack(None, frac, mean_v, skip=True)
    accepted:''',
    '''        reason = "前景遮挡：近处杂物盖住建塔区域"
        return pack(None, frac, mean_v, skip=True)
    accepted:''',
)
replace(
    "    depth_state, depth_piece, saw_outside, cap_unclear = _depth_on_cell(",
    "    diagnostic = {}\n"
    "    depth_state, depth_piece, saw_outside, cap_unclear = _depth_on_cell(",
)
replace(
    '''        cell, cloud, mask, intr, color, blocked, labels, pose, offset, pitch, roll
    )
    if cap_unclear:''',
    '''        cell, cloud, mask, intr, color, blocked, labels, pose, offset, pitch, roll,
        diagnostic=diagnostic,
    )
    reason = diagnostic.get("reason")
    if cap_unclear:''',
)
replace(
    '''        if reading is None:
            return pack(None, frac, mean_v)''',
    '''        if reading is None:
            reason = "彩色判定未通过：层高不符或塔顶阵营不清"
            return pack(None, frac, mean_v)''',
)
replace(
    '''        if far is None:
            return pack(None, frac, mean_v)''',
    '''        if far is None:
            reason = "远处彩色未通过：宽高、填充或轮廓不符"
            return pack(None, frac, mean_v)''',
)
replace(
    '''    if depth is not None and (clipped or overwide or saw_outside):
        return pack(None, frac, mean_v)
    if not color_usable or blocked or missed_shape or missed_width:
        return pack(None, frac, mean_v)''',
    '''    if depth is not None and (clipped or overwide or saw_outside):
        reason = (
            "尺寸不符：白块宽度超过65厘米" if overwide else
            "白块出画：顶边或左右边被切掉" if clipped else
            "投影不符：白点落在搜索框外"
        )
        return pack(None, frac, mean_v)
    if not color_usable or blocked or missed_shape or missed_width:
        reason = (
            "不在画面里，要先转向" if needs_yaw else
            "高度不在画面里，要改俯仰" if needs_pitch else
            "被高台遮挡：一层可见高度不足8厘米" if blocked else
            "轮廓不符：白块破碎、被遮挡或直边检查未通过" if missed_shape else
            "尺寸不符：白块宽度不足15厘米" if missed_width else
            "距离过近：彩色估距不足20厘米"
        )
        return pack(None, frac, mean_v)''',
)
replace(
    '''    if covered or holds:
        #''',
    '''    if covered or holds:
        reason = "前面有东西挡住了或格内有物体：不能确认为空点"
        #''',
)
replace(
    '''    if not _empty_evidence(cell, pose, offset, pitch, intr, depth, roll):
        return pack(None, frac, mean_v)''',
    '''    if not _empty_evidence(cell, pose, offset, pitch, intr, depth, roll):
        reason = ((reason + "；") if reason else "") + (
            "空点未证实：支撑面未完整入画、深度不足或高度不符"
        )
        return pack(None, frac, mean_v)''',
)
changes[p] = s

p = root / "report.py"
s = p.read_text(encoding="utf-8")

old = "    shown = locked or view.conclusion"
new = '''    # 只调整窗口说明，裁判输出和计票规则保持原样。
    if not locked:
        detail = getattr(view, "reason", None)
        if (
            view.conclusion is None and detail
            and not (view.needs_yaw or view.needs_pitch)
        ):
            text = detail
        elif detail and detail.startswith("等待相机稳定"):
            text = detail
        elif mode == "basic" and view.conclusion is not None:
            text = f"这一眼像是{STATE_WORDS[view.conclusion]}，等待连续帧锁定"
    shown = locked or view.conclusion'''

if old not in s:
    raise RuntimeError("report.py版本不匹配，未写入文件")
s = s.replace(old, new, 1)

old = "        frac = max((view.white_frac for view in self.views.values()), default=0.0)"
new = '''        if color is not None and self.stable_frames < 5:
            for view in self.views.values():
                if view.conclusion is None and view.reason:
                    continue
                view.reason = f"等待相机稳定：{self.stable_frames}/5帧"
        frac = max((view.white_frac for view in self.views.values()), default=0.0)'''

if old not in s:
    raise RuntimeError("report.py版本不匹配，未写入文件")
s = s.replace(old, new, 1)
changes[p] = s

p = root / "任务三说明.md"
changes[p] = p.read_text(encoding="utf-8") + """
- 2026-11-08：窗口增加未锁定的具体原因，包括深度点数不足、实测尺寸或层高不符、出画、轮廓破碎、位置不符、阵营不清和稳定等待。空点缺少支撑面证据时明确写出，深度不足仍可走原有彩色识别。尺寸、高度的明确冲突继续保持未知，不通过放宽门槛猜结果。裁判输出和连续15帧锁定不变。
"""

p = root / "tests/test_review_regressions.py"
changes[p] = p.read_text(encoding="utf-8") + '''

def test_depth_rejection_has_specific_reason():
    from report import plain_state
    cell, pose, c, d = scene(width=.015)
    view = inspect_frame(pose, OFFSET, -8., INTR, c, d)[cell.name]
    assert view.conclusion is None
    assert '尺寸不符' in view.reason
    assert '尺寸不符' in plain_state(view, None, 'extend')
    assert '尺寸不符' in plain_state(view, None, 'basic')


def test_height_rejection_reports_measured_height():
    cell, pose, c, d = scene(top=.50)
    view = inspect_frame(pose, OFFSET, -8., INTR, c, d)[cell.name]
    assert view.conclusion is None
    assert '层高不符' in view.reason and '厘米' in view.reason


def test_sparse_depth_allows_valid_color():
    setup, pose, offset = m['_basic']()
    cell = m['_cell']('右侧地面远一格')
    color = m['_blank']()
    m['_paint'](
        color, pose, 0., cell, .35, .35, 0.,
        (255, 255, 255), offset=offset,
    )
    depth = np.full(color.shape[:2], np.nan, np.float32)
    view = inspect_frame(pose, offset, 0., INTR, color, depth)[cell.name]
    assert view.conclusion == '1'
'''

# 全部读取和语法检查通过后再备份、写入。
for path, text in changes.items():
    if path.suffix == ".py":
        ast.parse(text)

backup = root.parent / (
    root.name + "_diagnostics_backup_"
    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
for path in changes:
    dst = backup / path.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)

try:
    for path, text in changes.items():
        path.write_text(text, encoding="utf-8")
except Exception:
    for path in changes:
        shutil.copy2(backup / path.relative_to(root), path)
    raise

print("更新完成。原文件备份：", backup)