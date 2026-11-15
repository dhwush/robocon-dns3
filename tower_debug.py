"""实地颜色/深度诊断；所有深度输入单位为米。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classify import RED, RED_WRAP, BLUE, _white_mask
from field_map import DISPLAY, TOWER_NO


def _label(board, lines):
    # 中文字体与主窗口一致。
    from view_field import _FONT
    panel = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    step = min(22, max(12, (board.shape[0]-10)//max(1, len(lines))))
    font = ImageFont.truetype(_FONT, min(16, step-2))
    for row, line in enumerate(lines):
        draw.text((8, 5 + row * step), line, font=font, fill='white',
                  stroke_width=1, stroke_fill='black')
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def paint_diagnostics(color, depth, views, selected=0, locked=None):
    """四联图：原图、HSV分割、实际取色证据、对齐深度。"""
    name = DISPLAY[selected % len(DISPLAY)]
    view = (views or {}).get(name)
    measure = (getattr(view, 'measurements', None) or {})
    faction = measure.get('faction') or {}
    height, width = color.shape[:2]
    raw = color.copy()
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1])) |
           cv2.inRange(hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1]))) > 0
    blue = cv2.inRange(hsv, np.array(BLUE[0]), np.array(BLUE[1])) > 0
    masks = np.zeros_like(color)
    masks[_white_mask(color) > 0] = (210, 210, 210)
    masks[red] = (0, 0, 255)
    masks[blue] = (255, 80, 0)
    evidence = (color.astype(np.float32) * .3).astype(np.uint8)
    box = faction.get('bbox')
    if box is not None:
        x0, y0, x1, y1 = box
        patch = evidence[y0:y1, x0:x1]
        patch[faction['band']] = (0, 110, 110)
        for key, bgr in [('red', (0, 0, 150)), ('blue', (150, 40, 0))]:
            if key in faction:
                patch[faction[key]] = bgr
        kept = faction.get('kept')
        if kept is not None:
            patch[kept & faction['red']] = (0, 0, 255)
            patch[kept & faction['blue']] = (255, 100, 0)
        if 'contact' in faction:
            contours, _ = cv2.findContours(faction['contact'].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(patch, contours, -1, (0, 255, 0), 1)
        tops = faction['tops']
        xs = np.arange(x0, x1)
        valid = tops < height
        evidence[tops[valid], xs[valid]] = (255, 255, 255)
        row = faction.get('decision_row')
        if row is not None:
            cv2.line(evidence, (x0, row), (x1-1, row), (255, 0, 255), 3)
    deep = np.zeros_like(color)
    valid = np.zeros((height, width), dtype=bool)
    if depth is not None and depth.shape == valid.shape:
        valid = np.isfinite(depth) & (depth > 0)
        # 固定0.3–3米色标，禁止每帧自动拉伸导致颜色含义变化。
        scaled = np.zeros(valid.shape, dtype=np.uint8)
        scaled[valid] = (np.clip((depth[valid] - .3) / 2.7, 0, 1) * 255).astype(np.uint8)
        deep = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        deep[~valid] = 0
    roi = np.zeros(valid.shape, dtype=np.uint8)
    if view is not None and view.region is not None:
        cv2.drawContours(roi, [view.region], -1, 255, -1)
        for board in (raw, masks, evidence, deep):
            cv2.drawContours(board, [view.region], -1, (0, 255, 255), 2)
    far_support = measure.get('far_support') or {}
    for x, y in far_support.get('support_pixels', []):
        cv2.circle(deep, (int(x), int(y)), 1, (0, 255, 0), -1)
    far_white = measure.get('far_white') or {}
    for x, y in far_white.get('support_pixels', []):
        cv2.circle(deep, (int(x), int(y)), 1, (0, 255, 0), -1)
    points = measure.get('top_pixels', [])
    for x, y in points:
        cv2.circle(deep, (int(x), int(y)), 2, (255, 0, 255), -1)
    inside = roi > 0
    count = int(inside.sum())
    values = depth[inside & valid] if depth is not None else np.array([])
    stats = '无有效深度'
    if len(values):
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        stats = f'框内有效 {len(values)/max(count,1):.0%}，P10/中位/P90={p10:.2f}/{p50:.2f}/{p90:.2f}m'
    state = getattr(view, 'conclusion', None) or '未知'
    locked_state = (locked or {}).get(name) or '未锁定'
    why = getattr(view, 'reason', None) or measure.get('height_source', '')
    descriptions = [
        [f'{TOWER_NO[name]}号 原图：本帧={state}，锁定={locked_state}', str(why)],
        ['HSV阈值：白/红/蓝（整幅，含未被采用的像素）'],
        ['实际取色：黄=带，绿=接触区，白=顶边，紫=判定行',
         '暗红蓝=阈值像素，亮红蓝=连接且面积达标的像素',
         f"距离={faction.get('distance_m', 0):.2f}m 带高={faction.get('cap_px', 0)}px 门槛={faction.get('need', 0)}px",
         f"有效红/蓝={faction.get('red_pixels', 0)}/{faction.get('blue_pixels', 0)}；{faction.get('status', '本帧未执行阵营判定')}"],
        ['对齐深度：固定0.3–3m；黑=无效；超范围饱和显示', stats,
         f"紫点=最高深度切片，离地高={measure.get('depth_height_cm', 0):.1f}cm" if points else '本帧无最高深度切片'],
    ]
    rgb_cap = measure.get('rgb_cap') or {}
    if rgb_cap:
        descriptions[2].append(f"RGB顶块：合格{rgb_cap['candidates']}个；{rgb_cap.get('reason', '')}")
        if 'distance_m' in rgb_cap:
            descriptions[2].append(
                f"估距{rgb_cap['distance_m']:.2f}m 俯仰{rgb_cap['pitch_deg']:+.1f}° "
                f"横滚{rgb_cap['roll_deg']:+.1f}° 闭合半径{rgb_cap['close_radius_px']}px")
        checks = sorted(rgb_cap.get('checks', []),
                        key=lambda c: c['inside_pixels'], reverse=True)
        descriptions[1].append('编号=RGB连通块；绿=通过，橙=拒绝；下方列格内像素最多的两块')
        for item in checks:
            x, y, w, h = item['bbox']
            ink = (0, 165, 255) if item['failures'] else (0, 255, 0)
            cv2.rectangle(masks, (x, y), (x+w-1, y+h-1), ink, 1)
            cv2.putText(masks, f"C{item['id']}", (x, min(height-3, y+h+14)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, ink, 1)
        for item in checks[:2]:
            descriptions[2].extend([
                f"C{item['id']} 宽/高{item['width_cm']:.1f}/{item['height_cm']:.1f}cm "
                f"格内{item['coverage']:.0%} 填充{item['fill']:.0%}",
                f"红/蓝{item['red_pixels']}/{item['blue_pixels']}px 各需≥{item['need']}px；"
                + ('通过' if not item['failures'] else '拒绝'),
            ])
            if 'bottom_cm' in item:
                descriptions[2].append(
                    f"底/顶{item['bottom_cm']:.1f}/{item['top_cm']:.1f}cm "
                    f"误差{item['bottom_error_cm']:+.1f}/{item['top_error_cm']:+.1f}(限±12/15)")
                descriptions[2].append(
                    f"横向重叠{item['horizontal_overlap']:.0%}(≥50%) "
                    f"上下间距{item['vertical_gap_px']:.1f}px(>0)")
            # 长原因分行，避免超出640像素面板。
            reasons = item['failures']
            for reason in reasons[:2]:
                descriptions[2].append('× ' + reason)
            if len(reasons) > 2:
                descriptions[2].append(f"另{len(reasons)-2}项拒绝原因见s保存数据")
        if not checks:
            descriptions[2].append('当前格内没有红蓝连通块；先查颜色阈值及搜索框位置')
    if far_support:
        descriptions[3].append('远处融合：' + far_support.get('reason', ''))
        if 'width_cm' in far_support:
            descriptions[3].append(
                f"绿点=格内白盒支撑{far_support['support_points']}点，宽{far_support['width_cm']:.1f}cm")
    if far_white:
        descriptions[3].append('远处白盒：' + far_white.get('reason', ''))
        if 'width_cm' in far_white:
            descriptions[3].append(
                f"支撑{far_white['support_points']}点，宽{far_white['width_cm']:.1f}cm，顶高{far_white['top_height_cm']:.1f}cm")
    recovery = measure.get('depth_boundary_recovery')
    if recovery:
        descriptions[3].append(
            f"3米边界补全：宽{recovery['original_width_cm']:.1f}→{recovery['recovered_width_cm']:.1f}cm，补{recovery['extra_points']}点")
    elif 'depth_width_cm' in measure:
        descriptions[3].append(f"深度水平跨度={measure['depth_width_cm']:.1f}cm")
    if measure.get('depth_fallback_reason'):
        descriptions[3].append('边界深度不完整，本帧改用合格彩色白盒轮廓')
    tiles = []
    for board, lines in zip((raw, masks, evidence, deep), descriptions):
        tile = cv2.resize(board, (640, 360), interpolation=cv2.INTER_NEAREST)
        tiles.append(_label(tile, lines))
    result = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    # 独立底栏，不遮挡取色或深度区域。
    footer = np.zeros((28, 1280, 3), dtype=np.uint8)
    footer = _label(footer, ['d 开关诊断 | n 下一座塔 | s 保存原始BGR/米制深度/位姿/诊断图；近端深蓝→青绿→黄→远端红'])
    return np.vstack((result, footer))


def save_capture(directory, color, depth, views, selected, locked, metadata):
    """保存同帧原始数据和诊断，供离线重放；不保存叠字后的图作原始输入。"""
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    metadata = dict(metadata, depth_unit='m', color_format='BGR', selected=DISPLAY[selected], locked=locked,
                    views={name: {'conclusion': v.conclusion, 'reason': v.reason,
                                  'method': v.method, 'measurements': v.measurements}
                           for name, v in (views or {}).items()})
    np.savez_compressed(str(stem) + '.npz', color=color, depth=depth,
                        metadata=np.array(json.dumps(metadata, ensure_ascii=False, default=serial)))
    if not cv2.imwrite(str(stem) + '.png', paint_diagnostics(color, depth, views, selected, locked)):
        raise OSError('诊断图保存失败')
    return stem
