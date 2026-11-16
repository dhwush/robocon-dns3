"""一帧里每一格的结论。对不上三档就保持未知。3 m 以内，白块不是实心直边也保持未知。

搜索框只决定白块属于哪一格：质心落在框里，宽度和高度量整块，不再先和框相与。
近处白块顶边或左右贴到画面边，或整块宽过 65 cm，彩色这条不收。
底边被切掉、顶边还在里面时，用车位把顶边换成离地高度，仍走彩色。
0.3 m 到 3 m 里，只要有足够的白点落在这一格，层数用最高一截的离地高度。
整块在画面里、彩色轮廓收不成直边时，这段对上了层高的深度仍然写。
近处两层的顶面常常没有深度。同一块白的顶边按车位换成离地高度，对上两层就写两层。
深度最高一截落在层高之间时保持未知；仅在顶边有证据时补判。
远处超过 3 m、没有深度时，框里的白先补上几像素宽的黑缝，再量整块的宽和高，不再要求直边。
3 m 以内仍要直边；轮廓不整齐时只靠深度写层数。
偏黄的地板从白掩膜里拿掉，白盒正面留着。层数仍看剩下的白块。
框里的白被切碎，或只剩不够 15 cm 的一截，同样保持未知，不写成空。
三层塔优先独立识别对应塔位和高度的红蓝顶块，不要求白盒或深度支撑。
独立顶块不成立时，仍可看贴在白盒顶上的像素带；分不出先后保持未知。
0.3 m 到 3 m 里，挡在格子近面前面、又落在这块塔的画面里的箱子和杂物不写成塔，也不写成空。
像方块、但深度不在台心 50 cm 里的白块同样丢掉。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from car_pose import CAMERA_HEIGHT_M, CarPose, Intrinsics, bearing_deg, camera_frame, camera_position, project, to_camera
from field_map import MARGIN_M, Cell, tower_cells
from sight import sees, visible_height


# 白盒阴影保留到 V=105，减少为补亮白盒而把顶块曝白的需要。
# 红保持窄色相，允许高亮时饱和度降到120；尺寸/高度/两色关系仍需通过。
# 蓝的色相仍是105–130，饱和度下限80保留偏淡的塔顶蓝面。
WHITE_S = 110
WHITE_V = 105
# 黄地面色相大约 16，亮处大约 12–19，饱和度大约 49，明度大约 135。
# 这块地上的阴影色相大约 19，大约 13–24，饱和度大约 27，明度大约 113。色相还是黄，靠明度分开。
# 饱和度达到 48 的黄才从白里拿掉，盒边大约 40 的黄留着。盒子底边连着格子外面的地板，所以整幅都拿。
FLOOR_H = (10, 26)
FLOOR_S = 48
RED = ((0, 120, 110), (8, 255, 255))
RED_WRAP = ((172, 120, 110), (180, 255, 255))
# 保留偏淡的塔顶蓝面，仅放宽饱和度；排除偏青/偏紫背景。
BLUE = ((105, 80, 50), (130, 255, 255))

WIDTH_M = 0.35
# 搜索框宽 60 cm。白块铺满框时，按格子中心的距离会量到 60 cm 出头。收到 65 cm，两块并排的 70 cm 仍然不要。
WIDTH_MIN_M = 0.15
WIDTH_MAX_M = 0.65
# 实心：白像素至少占外轮廓内部的七成五。线、空心框留不住。
FILL_MIN = 0.75
# 边比较直。先按短边的 8% 收成折线，折线离原轮廓也不超过这个距离。
# 白掩膜边上大约 2.5 像素的台阶和毛刺，这个数留给它。
# 四到七边直接留下：正面、看见顶面的六边、远处多出来的一个角、错开的台阶。
# 八边才比圆度，高于方块（π/4）和圆（1）的中点就丢掉。齿轮的角多于 8 个，也丢掉。
EDGE_SIMPLIFY = 0.08
CORNER_MIN = 4
CORNER_MAX = 8
_CIRCLE_LIMIT = (np.pi / 4.0 + 1.0) / 2.0
VIEW_FRACTION = 0.70
# 可见高度小于 8 cm 的那一档，当成没看见。
MIN_VISIBLE_M = 0.08
# 白盒 0.35 m，两层 0.70 m，塔顶再高 0.20 m。搜索框比塔顶再高出 0.10 m。
_TOP_MARGIN_M = 0.10
# 深度和视线一样，每 2 cm 看一截。相机深度只在大约 0.3 m 到 3 m 之间有数。
_DEPTH_SLICE_M = 0.02
_DEPTH_NEAR_M = 0.3
_DEPTH_FAR_M = 3.0
# 仅用于修复跨过3米边界、被硬截断的同一座塔；不扩大空点/遮挡判定范围。
_DEPTH_BOUNDARY_M = 0.30
_DEPTH_EDGE_M = 0.05
_DEPTH_MIN_POINTS = 10
# 远处黑缝大约几像素。补上这一道，两块并排的 70 cm 仍然合不到一起。
_FAR_SEAM_PX = 6
# 比这一格 50 cm 的近面再近 25 cm，才算挡在塔前面。塔自己的正面在近面里侧，不会被算进去。
_FRONT_M = 0.25
# 这些点占到搜索框的一成，或盖住一层顶到两层顶，这一眼就不认塔。
_OCCLUDE_FRAC = 0.10


@dataclass
class CellView:
    conclusion: str | None
    in_view: bool
    needs_yaw: bool
    needs_pitch: bool
    blocked: bool
    bearing_deg: float
    pitch_target_deg: float
    white_frac: float
    mean_v: float
    region: np.ndarray | None = None
    piece: np.ndarray | None = None
    method: str | None = None
    skip_vote: bool = False
    reason: str | None = None
    measurements: dict | None = None
    needs_parallax: bool = False


def heights_match(measured: float, predicted: float) -> bool:
    # 五成。没有深度时用像素高度，近处抬头会把顶面算进去，两层会比 70 cm 高一截。
    if predicted < MIN_VISIBLE_M:
        return False
    return abs(measured - predicted) <= 0.50 * predicted + 1e-6


def heights_match_level(measured: float, predicted: float) -> bool:
    # 两成。离地高度对一层、两层的绝对档，间隔是 35 cm，两成以内两档不重叠。
    if predicted < MIN_VISIBLE_M:
        return False
    return abs(measured - predicted) <= 0.20 * predicted + 1e-6


def _samples(cell: Cell) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    xs = np.linspace(x0, x1, 5)
    ys = np.linspace(y0, y1, 5)
    return [(float(x), float(y)) for x in xs for y in ys]


def _inside(pixel: np.ndarray | None, intr: Intrinsics) -> bool:
    if pixel is None:
        return False
    return 0.0 <= float(pixel[0]) < intr.width and 0.0 <= float(pixel[1]) < intr.height


def _bearing_deg(point: np.ndarray, pose: CarPose, offset: np.ndarray, pitch: float) -> float:
    # 俯仰不参与。身后的格子按真实转角算，不再被夹成 90 度。
    del pitch
    return bearing_deg(point, pose, offset)


def _pitch_target(cell: Cell, pose: CarPose, offset: np.ndarray) -> float:
    # 角度中点而非高度中点；近处上下仰角不对称，需按真实视线对中。
    origin = camera_position(pose, offset, 0.)
    horizontal = max(float(np.linalg.norm(cell.center(cell.one_z)[:2]-origin[:2])), 1e-6)
    low = np.arctan2(cell.one_z-origin[2], horizontal)
    high = np.arctan2(cell.cap_z-origin[2], horizontal)
    return float(np.degrees((low+high)/2))


def _mask(cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0) -> np.ndarray:
    shape = (intr.height, intr.width)
    pixels = []
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    low_z = cell.base_z
    high_z = cell.cap_z + _TOP_MARGIN_M
    for x, y, z in (
        (x0, y0, low_z),
        (x1, y0, low_z),
        (x1, y1, low_z),
        (x0, y1, low_z),
        (x0, y0, high_z),
        (x1, y0, high_z),
        (x1, y1, high_z),
        (x0, y1, high_z),
    ):
        pixel = project(np.array([x, y, z]), pose, offset, pitch, intr, roll)
        if pixel is not None:
            pixels.append(pixel)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pixels) < 3:
        return mask
    hull = cv2.convexHull(np.round(np.stack(pixels)).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _view_flags(
    cell: Cell, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0
) -> tuple[bool, bool, bool]:
    # 左右仍要这块 50 cm 的大部分进画面。
    # 竖直只看一层顶和塔顶。近处从台面到搜索框顶已经塞满彩色的垂直视场，
    # 台面掉在画面下面不因此算没对准。
    horizontal = 0
    vertical = 0
    total = 0
    for x, y in _samples(cell):
        total += 1
        low = project(np.array([x, y, cell.one_z]), pose, offset, pitch, intr, roll)
        high = project(np.array([x, y, cell.cap_z]), pose, offset, pitch, intr, roll)
        low_h = low is not None and 0.0 <= float(low[0]) < intr.width
        high_h = high is not None and 0.0 <= float(high[0]) < intr.width
        if low_h and high_h:
            horizontal += 1
        if _inside(low, intr) and _inside(high, intr):
            vertical += 1
    if total == 0:
        return False, True, False
    horiz_ok = horizontal / total >= VIEW_FRACTION
    vert_ok = vertical / total >= VIEW_FRACTION
    return horiz_ok and vert_ok, not horiz_ok, horiz_ok and not vert_ok


def _split_white(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 第二项是从白里拿掉的地板。这一格的框里有这种黄，彩色结论才标成黄地。
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    floor = (
        (hsv[:, :, 0] >= FLOOR_H[0])
        & (hsv[:, :, 0] <= FLOOR_H[1])
        & (hsv[:, :, 1] >= FLOOR_S)
        & (hsv[:, :, 1] <= WHITE_S)
        & (hsv[:, :, 2] >= WHITE_V)
    )
    white = (hsv[:, :, 1] <= WHITE_S) & (hsv[:, :, 2] >= WHITE_V) & ~floor
    return white.astype(np.uint8) * 255, floor


def _white_mask(color: np.ndarray) -> np.ndarray:
    white, _floor = _split_white(color)
    return white


def _color_method(mask: np.ndarray, floor: np.ndarray | None) -> str:
    if floor is not None and mask.size and bool(np.any(floor & (mask > 0))):
        return "黄地"
    return "彩色"


def _band_above(white: np.ndarray, cell_mask: np.ndarray, cap_px: int) -> tuple[np.ndarray, np.ndarray]:
    # 每一列从这一列的白盒顶边往上 cap_px。再收进这一格的投影里。
    present = white > 0
    height = present.shape[0]
    rows = np.arange(height, dtype=np.int32)[:, None]
    tops = np.where(present, rows, height).min(axis=0)
    y0 = np.clip(tops - int(cap_px), 0, height)
    band = (rows >= y0) & (rows < tops) & (tops < height)
    if cell_mask.shape == band.shape:
        band &= cell_mask > 0
    return band, tops


def _cap_colors(color, diagnostic=None, possible=False):
    """顶块路径共用阈值与归属掩膜，避免旧分支再次使用其他塔的颜色。"""
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    red=((cv2.inRange(hsv,np.array(RED[0]),np.array(RED[1]))
          |cv2.inRange(hsv,np.array(RED_WRAP[0]),np.array(RED_WRAP[1])))>0)
    blue=cv2.inRange(hsv,np.array(BLUE[0]),np.array(BLUE[1]))>0
    allowed=(diagnostic or {}).get('_cap_suspected_allowed' if possible else '_cap_allowed')
    if allowed is not None:
        red &= allowed
        blue &= allowed
    return red,blue


def _cap_world_support(part, colored, depth, x, y, pose, offset, pitch, intr, roll):
    """只使用颜色块内部成片一致的实测深度；缺测和零星深度不冒充归属。"""
    if depth is None or depth.shape != colored.shape:
        return None
    inner=cv2.erode(part.astype(np.uint8),np.ones((3,3),np.uint8),
                    borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    yy,xx=np.nonzero(inner & colored[y:y+part.shape[0],x:x+part.shape[1]])
    if len(xx)<12:
        return None
    values=depth[y+yy,x+xx].astype(float)
    valid=np.isfinite(values)&(values>=_DEPTH_NEAR_M)&(values<=6.)
    if valid.sum()<12 or valid.mean()<.25:
        return None
    yy,xx,values=yy[valid]+y,xx[valid]+x,values[valid]
    middle=float(np.median(values))
    same=np.abs(values-middle)<=max(.05,.025*middle)
    if same.sum()<12 or same.mean()<.7:
        return None
    yy,xx,values=yy[same],xx[same],values[same]
    if len(xx)>4000:
        take=np.linspace(0,len(xx)-1,4000).astype(int)
        yy,xx,values=yy[take],xx[take],values[take]
    origin,right,down,ahead=camera_frame(pose,offset,pitch,roll)
    world=(origin+((xx-intr.cx)*values/intr.fx)[:,None]*right
           +((yy-intr.cy)*values/intr.fy)[:,None]*down+values[:,None]*ahead)
    return world,middle


def _assign_caps(cells, color, pose, offset, pitch, intr, roll=0., depth=None):
    """整帧顶块唯一归属：成片实测深度优先，缺测时用RGB塔顶投影。"""
    red, blue = _cap_colors(color)
    colored=red | blue
    # 统一闭合半径，确保各塔使用同一颜色块身份。
    closed = cv2.morphologyEx(colored.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    owners = np.full(count, -1, np.int32)
    possible = np.zeros((len(cells), count), bool)
    reports = {cell.name: [] for cell in cells}
    masks = [_mask(cell, pose, offset, pitch, intr, roll) for cell in cells]
    for index in range(1, count):
        x, y, w, h, pixels = map(int, stats[index])
        if pixels < 12:
            continue
        part = labels[y:y+h, x:x+w] == index
        support=_cap_world_support(part,colored,depth,x,y,pose,offset,pitch,intr,roll)
        depth_status='missing_or_incoherent'
        if support is not None:
            world,_=support
            matched=False
            for candidate_cell in cells:
                bx0,bx1,by0,by1=candidate_cell.search_bounds(MARGIN_M)
                near=((world[:,0]>=bx0)&(world[:,0]<=bx1)&(world[:,1]>=by0)&(world[:,1]<=by1)
                      &(world[:,2]>=candidate_cell.two_z-.08)&(world[:,2]<=candidate_cell.cap_z+.10))
                if np.mean(near)>=.6:
                    matched=True
                    break
            depth_status='matched_tower' if matched else 'outside_all_tower_volumes'
            # 穿透到背景或坐标不一致的深度不作归属依据；不凭它否决独立RGB。
            if not matched:
                support=None
        paired=(np.count_nonzero(red[y:y+h,x:x+w]&part)>=8
                and np.count_nonzero(blue[y:y+h,x:x+w]&part)>=8)
        scores = []
        details = []
        for number, (cell, mask) in enumerate(zip(cells, masks)):
            coverage = float(np.count_nonzero(part & (mask[y:y+h, x:x+w] > 0)) / pixels)
            if coverage < .35:
                continue
            distance = float(to_camera(cell.center(cell.two_z), pose, offset, pitch, roll)[2])
            if distance <= .2:
                continue
            bottom = _z_at_cell(cell, x+w/2, y+h, pose, offset, pitch, intr, roll)
            top = _z_at_cell(cell, x+w/2, y, pose, offset, pitch, intr, roll)
            errors = []
            if bottom is not None and y+h < intr.height-1:
                errors.append(abs(bottom-cell.two_z))
            if top is not None and y > 1:
                errors.append(abs(top-cell.cap_z))
            detail = dict(component=index, bbox=[x,y,w,h], coverage=coverage,
                          bottom_cm=None if bottom is None else bottom*100,
                          top_cm=None if top is None else top*100,
                          owner=None, source='depth' if support is not None else 'rgb',
                          depth_status=depth_status, reason='顶块位置与塔位不匹配')
            reports[cell.name].append(detail)
            details.append(detail)
            if support is not None:
                world,measured_distance=support
                x0,x1,y0,y1=cell.search_bounds(MARGIN_M)
                own=((world[:,0]>=x0)&(world[:,0]<=x1)
                     &(world[:,1]>=y0)&(world[:,1]<=y1)
                     &(world[:,2]>=cell.two_z-.08)&(world[:,2]<=cell.cap_z+.10))
                fraction=float(np.mean(own))
                detail.update(depth_points=len(world),depth_distance_m=measured_distance,
                              depth_coverage=fraction,world_center=np.median(world,axis=0).tolist())
                if fraction<.6:
                    detail['reason']='实测顶块不在本塔水平范围或顶块高度带'
                    continue
                score=float(np.linalg.norm(np.median(world[own,:2],axis=0)-[cell.x,cell.y]))
            else:
                # 不恢复完整尺寸要求；出画边不作高度证据，局部可见保留容差。
                if not errors or max(errors) > .18:
                    continue
                # 单色碎片只有贴近白盒顶才保留占用证据；不要求露出完整20cm。
                if (not paired and bottom is not None and y+h<intr.height-1
                        and abs(bottom-cell.two_z)>.115):
                    detail['reason']='单色区域没有贴近本塔白盒顶'
                    continue
                score = float(np.mean(errors))
            possible[number, index] = True
            detail['height_error_cm'] = None if not errors else float(np.mean(errors))*100
            detail['ownership_score_m']=score
            scores.append((score, number))
        scores.sort()
        owner = None
        if scores and (len(scores) == 1 or scores[1][0]-scores[0][0] >= .04):
            owners[index] = scores[0][1]
            owner = cells[scores[0][1]].name
        for detail in details:
            detail['owner'] = owner
            if 'ownership_score_m' in detail:
                detail['reason'] = '唯一归属' if owner else '塔位证据接近，归属未定'
    return {cell.name: (owners[labels] == number,
                       (owners[labels] == number) | ((owners[labels] < 0) & possible[number, labels]),
                       reports[cell.name])
            for number, cell in enumerate(cells)}


def _faction(color: np.ndarray, white: np.ndarray, cell_mask: np.ndarray, z_cam: float, intr: Intrinsics, diagnostic: dict | None = None, linked_cap: bool = False) -> tuple[str | None, bool]:
    # 只认贴着白盒顶边的那条像素带，从上往下数。
    # 和顶边隔开的红蓝不算塔顶。红蓝从上面一起数够，第二项为真，调用处保持未知。
    cap_px = max(4, int(round(0.20 * intr.fy / max(float(z_cam), 0.3))))
    band, tops = _band_above(white, cell_mask, cap_px)
    if linked_cap:
        # 独立核验过完整顶块后，保留整块高度，防止透视或小黑缝截掉上色。
        cap_y, cap_x = np.nonzero(cell_mask)
        usable = tops[cap_x] < white.shape[0]
        if np.any(usable):
            cap_px = max(cap_px, int(np.max(tops[cap_x[usable]] - cap_y[usable])))
            band, tops = _band_above(white, cell_mask, cap_px)
    debug = None
    if diagnostic is not None:
        debug = {"distance_m": float(z_cam), "cap_px": cap_px,
                 "red_pixels": 0, "blue_pixels": 0, "result": None,
                 "status": "无取色带", "need": 0}
        diagnostic["faction"] = debug
        by, bx = np.nonzero(band)
        if len(bx):
            x0, x1 = int(bx.min()), int(bx.max()) + 1
            y0, y1 = int(by.min()), int(by.max()) + 1
            debug.update(bbox=(x0, y0, x1, y1),
                         band=band[y0:y1, x0:x1].copy(),
                         tops=tops[x0:x1].copy(),
                         status="带内没有红蓝像素")
    if not np.any(band):
        return None, False
    red,blue=_cap_colors(color,diagnostic,possible=True)
    red &= band
    blue &= band
    colored = red | blue
    if debug is not None:
        debug["raw_red_pixels"] = int(red.sum())
        debug["raw_blue_pixels"] = int(blue.sum())
        debug["red"] = red[y0:y1, x0:x1].copy()
        debug["blue"] = blue[y0:y1, x0:x1].copy()
    if not np.any(colored):
        return None, False
    # 顶边上大约 8% 的一条算贴着。膨胀只用来跨过这一条里的小缝，不伸出带子。
    touch = max(2, int(round(0.08 * cap_px)))
    rows = np.arange(band.shape[0], dtype=np.int32)[:, None]
    contact = band & (rows >= (tops - touch))
    radius = max(1, touch // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    grown = (cv2.dilate(colored.astype(np.uint8), kernel) > 0) & band
    _count, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    hit = np.unique(labels[contact])
    hit = hit[hit > 0]
    if debug is not None:
        debug["contact"] = contact[y0:y1, x0:x1].copy()
        debug["status"] = "红蓝未连接白盒顶边"
    if hit.size == 0:
        return None, False
    keep = (cell_mask > 0) & band if linked_cap else np.isin(labels, hit)
    # 颜色至少覆盖候选带的 3%，不能由一排噪点决定阵营。
    need = max(8, int(0.03 * np.count_nonzero(band)))
    for colored_part in (red, blue):
        total = int(np.count_nonzero(colored_part & keep))
        if total < need:
            colored_part[:] = False
    if debug is not None:
        debug.update(need=need, red_pixels=int((red & keep).sum()),
                     blue_pixels=int((blue & keep).sum()),
                     kept=(keep & (red | blue))[y0:y1, x0:x1].copy(),
                     status="有效面积不足或先后不明确")
    ry, rx = np.nonzero(red & keep)
    by, bx = np.nonzero(blue & keep)
    if len(ry) < need or len(by) < need:
        if debug is not None:
            debug['status'] = '红蓝两色未同时达到阈值，不影响白盒层数'
        return None, False
    allowed=(diagnostic or {}).get('_cap_allowed')
    if allowed is not None and np.any((red|blue)&keep&~allowed):
        if debug is not None:
            debug['status']='顶块归属未定，禁止判阵营或回退为两层'
        return None,True
    roll = float((diagnostic or {}).get('roll_deg', 0.0))
    angle = np.deg2rad(roll)
    # 重力投影的向下坐标，横滚时不能直接按图像行判断上下。
    red_down = -np.sin(angle)*(rx-intr.cx)+np.cos(angle)*(ry-intr.cy)
    blue_down = -np.sin(angle)*(bx-intr.cx)+np.cos(angle)*(by-intr.cy)
    red_above = np.percentile(red_down,75) < np.percentile(blue_down,25)
    blue_above = np.percentile(blue_down,75) < np.percentile(red_down,25)
    overlap = min(rx.max(),bx.max())-max(rx.min(),bx.min())+1
    if overlap < .5*min(np.ptp(rx)+1,np.ptp(bx)+1) or not (red_above or blue_above):
        return None, True
    name = '红' if red_above else '蓝'
    if debug is not None:
        debug.update(result=name, status='红蓝完整且重力上下关系通过',
                     decision_row=int(min(ry.min(),by.min())))
    return name, False


def _metric(pixels: float, z_cam: float, focal: float) -> float:
    return float(pixels) * z_cam / focal


def _z_at_cell(
    cell: Cell, u: float, v: float, pose: CarPose, offset: np.ndarray, pitch: float, intr: Intrinsics, roll: float = 0.0
) -> float | None:
    # 与朝向相机的格心竖直平面相交；不能固定用场地Y平面，侧视时会病态。
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    direction = ((u-intr.cx)/intr.fx)*right + ((v-intr.cy)/intr.fy)*down + ahead
    normal = np.array([cell.x, cell.y])-origin[:2]
    denom = float(direction[:2] @ normal)
    if abs(denom) < 1e-6:
        return None
    distance = float(normal @ normal)/denom
    if distance <= .2:
        return None
    return float(origin[2]+distance*direction[2])


def _depth_in_range(z_cam: float) -> bool:
    return _DEPTH_NEAR_M <= z_cam <= _DEPTH_FAR_M


def _enter_depth(
    origin: np.ndarray,
    direction: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> np.ndarray:
    # 每条射线进入台心柱体时的相机深度。进不去的是 nan。
    count = len(direction)
    enter = np.full(count, -np.inf)
    leave = np.full(count, np.inf)
    valid = np.ones(count, dtype=bool)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        comp = direction[:, axis]
        parallel = np.abs(comp) < 1e-9
        valid &= ~(parallel & ((float(origin[axis]) < lo) | (float(origin[axis]) > hi)))
        inv = np.zeros(count, dtype=np.float64)
        moving = ~parallel
        inv[moving] = 1.0 / comp[moving]
        near = (lo - float(origin[axis])) * inv
        far = (hi - float(origin[axis])) * inv
        swap = near > far
        near_s = np.where(swap, far, near)
        far_s = np.where(swap, near, far)
        near_s = np.where(parallel, -np.inf, near_s)
        far_s = np.where(parallel, np.inf, far_s)
        enter = np.maximum(enter, near_s)
        leave = np.minimum(leave, far_s)
        valid &= enter <= leave + 1e-6
    valid &= leave >= 0.0
    return np.where(valid, np.maximum(enter, 0.0), np.nan)


def _tower_clutter(
    cell: Cell,
    depth: np.ndarray | None,
    mask: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float = 0.0,
    diagnostic: dict | None = None,
    color: np.ndarray | None = None,
) -> tuple[bool, bool]:
    # 第一项：搜索框里有东西比近面更近，挡住了这一格的塔。
    # 第二项：台心柱体里、台面 8 cm 以上有东西，不能把这一格写成空。
    if depth is None or depth.shape[:2] != mask.shape:
        return False, False
    ys, xs = np.nonzero(mask)
    if len(xs) < _DEPTH_MIN_POINTS:
        return False, False
    values = depth[ys, xs].astype(np.float64)
    good = np.isfinite(values) & (values >= _DEPTH_NEAR_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return False, False
    examined = float(len(xs))
    xs = xs[good]
    ys = ys[good]
    values = values[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    px = (xs.astype(np.float64) - intr.cx) / intr.fx
    py = (ys.astype(np.float64) - intr.cy) / intr.fy
    direction = px[:, None] * right + py[:, None] * down + ahead
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    entered = _enter_depth(origin, direction, x0, x1, y0, y1)
    front = np.isfinite(entered) & (values <= entered - _FRONT_M)
    world = origin + values[:, None] * direction
    risen = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
        & (world[:, 2] >= cell.base_z + MIN_VISIBLE_M)
        & (world[:, 2] <= cell.cap_z + _TOP_MARGIN_M)
    )
    holds = int(np.count_nonzero(risen)) >= _DEPTH_MIN_POINTS
    if diagnostic is not None:
        # 只有接在本塔两层白盒顶部附近的上层点才构成否决证据。
        # 仅落进带定位余量的搜索柱体，不足以证明柜子/纸箱属于塔。
        white = (_white_mask(color)[ys,xs]>0) if color is not None else np.ones(len(xs),bool)
        visible_body=risen & white & (world[:,2]<=cell.two_z+.035)
        # 锚点必须位于最近的白盒表面深度带，不能用背景自身的上下连续性自证归属。
        body_depth=values[visible_body]
        front_depth=float(np.percentile(body_depth,10)) if len(body_depth)>=30 else -np.inf
        body = world[visible_body & (values<=front_depth+.08)
                     & (world[:,2]>=cell.two_z-.12)]
        upper = world[risen & (world[:,2]>=cell.two_z+.06)
                      & (world[:,2]<=cell.two_z+.15)]
        if len(body)>=30 and len(upper)>=30:
            anchors=body[::max(1,len(body)//400)]
            linked=[]
            for part in np.array_split(upper,max(1,int(np.ceil(len(upper)/1000)))):
                delta=part[:,None,:2]-anchors[None,:,:2]
                linked.append(np.min(np.sum(delta*delta,axis=2),axis=1)<=.13**2)
            attached=upper[np.concatenate(linked)]
            if len(attached)>=30:
                span=np.percentile(attached[:,:2],95,axis=0)-np.percentile(attached[:,:2],5,axis=0)
                if np.linalg.norm(span)>=.08:
                    diagnostic['upper_object']=dict(points=len(attached),
                        width_cm=float(np.linalg.norm(span))*100,attached_to_body=True)
    front_n = int(np.count_nonzero(front))
    covers = front_n / examined >= _OCCLUDE_FRAC
    if not covers and front_n >= _DEPTH_MIN_POINTS:
        low = project(cell.center(cell.one_z), pose, offset, pitch, intr, roll)
        high = project(cell.center(cell.two_z), pose, offset, pitch, intr, roll)
        if low is not None and high is not None:
            row0, row1 = sorted((float(low[1]), float(high[1])))
            span = row1 - row0
            band = front & (ys.astype(np.float64) >= row0) & (ys.astype(np.float64) <= row1)
            if span >= 4.0 and int(np.count_nonzero(band)) >= _DEPTH_MIN_POINTS:
                covers = float(ys[band].max() - ys[band].min()) >= 0.6 * span
    return covers, holds


def _depth_place(
    cell: Cell,
    depth: np.ndarray | None,
    ys: np.ndarray,
    xs: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    z_cam: float,
    roll: float = 0.0,
) -> tuple[str, float]:
    # 深度只把像素换成米。反投影落在这一格里才采用，不再拿和格子中心相差的成数否决。
    # 读不到就用格子的距离。点够了却不在这一格里，是挡在画面上的箱子，调用处丢掉。
    # 远处超过大约 3 m 常常没有深度，不因此把这块丢掉。
    if depth is None or len(xs) == 0:
        return "none", z_cam
    values = depth[ys, xs].astype(np.float64)
    good = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return "none", z_cam
    xs = xs[good]
    ys = ys[good]
    values = values[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    cam_x = (xs.astype(np.float64) - intr.cx) * values / intr.fx
    cam_y = (ys.astype(np.float64) - intr.cy) * values / intr.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + values[:, None] * ahead
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    inside = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
    )
    if int(inside.sum()) < _DEPTH_MIN_POINTS:
        return "out", float(np.median(values))
    return "in", float(np.median(values[inside]))


def _white_cloud(
    color: np.ndarray,
    depth: np.ndarray | None,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float = 0.0,
) -> dict | None:
    # 多保留3米后30厘米的白点，常规判定仍在3米内；仅在边界截断时尝试补全。
    if depth is None or depth.shape[:2] != color.shape[:2]:
        return None
    ys, xs = np.nonzero(_white_mask(color))
    if len(xs) < _DEPTH_MIN_POINTS:
        return None
    measured = depth[ys, xs].astype(np.float64)
    good = np.isfinite(measured) & (measured >= _DEPTH_NEAR_M) & (measured <= _DEPTH_FAR_M + _DEPTH_BOUNDARY_M)
    if int(good.sum()) < _DEPTH_MIN_POINTS:
        return None
    xs = xs[good]
    ys = ys[good]
    measured = measured[good]
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    cam_x = (xs.astype(np.float64) - intr.cx) * measured / intr.fx
    cam_y = (ys.astype(np.float64) - intr.cy) * measured / intr.fy
    world = origin + cam_x[:, None] * right + cam_y[:, None] * down + measured[:, None] * ahead
    return {"xs": xs, "ys": ys, "z": measured, "world": world}


def _silhouette_top(
    cell: Cell,
    labels: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    roll: float,
) -> float | None:
    # 顶面经常没有深度，深度云的最高点就停在正面中间。
    # 和这些深度点连在同一块白上的顶边还在，用车位把顶边换成离地高度。
    if labels is None or len(xs) == 0:
        return None
    ids, counts = np.unique(labels[ys, xs], return_counts=True)
    best = None
    for index, count in zip(ids, counts):
        if int(index) <= 0 or int(count) < _DEPTH_MIN_POINTS:
            continue
        component = labels == int(index)
        comp_y, comp_x = np.nonzero(component)
        # 补高必须来自独立完整盒形，不能取与墙/地面粘连的全局最高白点。
        if (comp_x.min()<=1 or comp_x.max()>=intr.width-2
                or not _block_shaped(component)):
            continue
        distance = float(to_camera(cell.center(cell.one_z),pose,offset,pitch,roll)[2])
        width = _metric(float(np.ptp(comp_x)+1),distance,intr.fx)
        if not WIDTH_MIN_M<=width<=WIDTH_MAX_M:
            continue
        top_row = int(comp_y.min())
        if top_row <= 1 or top_row >= intr.height - 2:
            continue
        on_top = comp_y == top_row
        height = _z_at_cell(
            cell,
            float(np.median(comp_x[on_top])),
            float(top_row),
            pose,
            offset,
            pitch,
            intr,
            roll,
        )
        if height is None:
            continue
        if best is None or height > best:
            best = height
    return best



def _cap_from_depth(cell, world, xs, ys, distance, color, mask, intr, diagnostic=None):
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

    red,blue=_cap_colors(color,diagnostic,possible=True)

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
        color, white_body, accepted, z_cam, intr, diagnostic
    )
    if unclear or faction is None:
        return None, "已见两层白盒和疑似顶块：红蓝阵营仍不明确"

    return faction, None


def _near_top_observed(world, xs, ys, color, depth, intr, diagnostic):
    """横向多数实测列到达颜色边界或深度后退处；中段深度截断不当作盒顶。"""
    columns, first = np.unique(xs, return_index=True)
    if len(columns)<8:
        return False
    gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY).astype(float)
    heights = world[first,2]
    middle = float(np.median(heights))
    stable = np.abs(heights-middle)<=.04
    seen = 0
    for index in first[stable]:
        u,v = int(xs[index]),int(ys[index])
        if not (1<=u<intr.width-1 and 4<=v<intr.height-2):
            continue
        # 顶边混合深度允许3像素误差；比较盒面与顶边上方。
        below = gray[v:v+3,u-1:u+2]
        above = gray[v-4:v-1,u-1:u+2]
        edge = abs(float(np.median(below))-float(np.median(above)))>=12
        behind = False
        if depth is not None:
            patch = depth[v-4:v-1,u-1:u+2]
            behind = np.count_nonzero(np.isfinite(patch)&(patch>depth[v,u]+.25))>=7
        if edge or behind:
            seen += 1
    if diagnostic is not None:
        diagnostic['near_top_check'] = dict(columns=len(columns),valid_columns=seen,
                                            height_cm=middle*100)
    return seen>=max(6,.45*len(columns))


def _near_depth_top(world,xs,ys,depth,intr,diagnostic=None):
    """实测深度的盒顶支撑；未知列不当反证，顶部缺测也不当空域。"""
    info={'state':'unknown','valid_columns':0,'unknown_columns':0,'occluded_columns':0,
          'continued_columns':0}
    if diagnostic is not None:
        diagnostic['depth_top_check']=info
    if depth is None or len(xs)<30:
        return False
    columns,first=np.unique(xs,return_index=True)
    if len(columns)<12:
        return False
    heights=world[first,2]
    top_height=float(np.percentile(heights,80))
    good_columns=[]
    for index in first:
        u,v=int(xs[index]),int(ys[index])
        if not (1<=u<intr.width-1 and 4<=v<intr.height-2):
            info['unknown_columns']+=1
            continue
        # 只采用到达同一顶高的列；低侧面、前景或缺测列不拉低顶高。
        if abs(world[index,2]-top_height)>.04:
            info['unknown_columns']+=1
            continue
        distance=float(depth[v,u])
        patch=depth[v-4:v-1,u-1:u+2]
        valid=np.isfinite(patch)&(patch>=_DEPTH_NEAR_M)
        n=int(valid.sum())
        if n<5:
            info['unknown_columns']+=1
        elif np.count_nonzero(valid&(patch>distance+.08))>=.7*n:
            good_columns.append(index)
        elif np.count_nonzero(valid&(patch<distance-.12))>=.7*n:
            info['occluded_columns']+=1
        else:
            info['continued_columns']+=1
    info['valid_columns']=len(good_columns)
    info['height_cm']=top_height*100
    if len(good_columns)<12:
        return False
    support=world[good_columns]
    xy=support[:,:2]-np.median(support[:,:2],axis=0)
    _,_,axes=np.linalg.svd(xy,full_matrices=False)
    spans=np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    width=float(spans.max())
    info['support_width_cm']=width*100
    # 最少12列、12cm实测横向支撑；不能靠一个小角认定整座塔。
    # 仍连续的同深度表面超过有效观测20%时，顶部证据不足。
    observed=len(good_columns)+info['continued_columns']
    good=(width>=.12 and len(good_columns)>=.8*observed)
    info['state']='supported' if good else 'unknown'
    return good


def _near_body_support(cell,world,distance,xs,ys,shape,diagnostic=None):
    """先在三维中选择前侧盒体，再量高度；搜索范围不是物体归属证明。"""
    info={'groups':[]}
    if diagnostic is not None:
        diagnostic['near_body']=info
    raised=world[:,2]>=cell.base_z+.06
    labels,ids=_far_spatial_labels(world,raised,ys,xs,shape)
    groups=[]
    for label in ids:
        own=raised & (labels[ys,xs]==label)
        points=world[own]
        xy=points[:,:2]-np.median(points[:,:2],axis=0)
        _,_,axes=np.linalg.svd(xy,full_matrices=False)
        width=float(np.max(np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)))
        span=float(np.percentile(points[:,2],98)-np.percentile(points[:,2],2))
        front=float(np.percentile(distance[own],20))
        detail=dict(id=label,points=int(own.sum()),width_cm=width*100,
                    visible_cm=span*100,front_m=front)
        info['groups'].append(detail)
        if WIDTH_MIN_M<=width<=WIDTH_MAX_M and span>=.035:
            groups.append((front,label,own))
    if not groups:
        info['reason']='没有独立且宽高支撑充分的盒体'
        return None
    groups.sort(key=lambda item:item[0])
    _,label,selected=groups[0]
    # 叠放错位或接缝造成的断开，只在实测表面空间相邻时合并。
    # 深度跳变、远处高标签不能因为投影重叠就合进来。
    for _ in range(2):
        changed=False
        for _,other_label,own in groups[1:]:
            if np.any(selected & own):
                continue
            a=world[selected];b=world[own]
            low_a,high_a=np.percentile(a[:,2],[2,98])
            low_b,high_b=np.percentile(b[:,2],[2,98])
            # 两块上下接续；并排的墙面/纸板不凭深度相近合并。
            gap=min(abs(low_b-high_a),abs(low_a-high_b))
            overlap=min(high_a,high_b)-max(low_a,low_b)
            if gap>.08 or overlap>.08:
                continue
            aa=a[::max(1,len(a)//300)];bb=b[::max(1,len(b)//300)]
            delta=aa[:,None,:]-bb[None,:,:]
            if np.min(np.sum(delta*delta,axis=2))>.10**2:
                continue
            points=world[selected|own]
            spans=np.percentile(points[:,:2],95,axis=0)-np.percentile(points[:,:2],5,axis=0)
            if np.linalg.norm(spans)>WIDTH_MAX_M:
                continue
            selected|=own
            changed=True
        if not changed:
            break
    info.update(selected_group=label,selected_points=int(selected.sum()),
                excluded_points=int(raised.sum()-selected.sum()))
    # 不把同位置的两个相近主体随意选一个；无法分清时保持未知。
    selected_front=float(np.percentile(distance[selected],20))
    for front,other_label,own in groups[1:]:
        if not np.any(selected&own) and abs(front-selected_front)<.08:
            info['reason']='多个独立盒体距离接近，归属不唯一'
            return None
    return selected


def _depth_on_cell(
    cell: Cell,
    cloud: dict | None,
    mask: np.ndarray,
    intr: Intrinsics,
    color: np.ndarray,
    blocked: bool,
    labels: np.ndarray | None = None,
    pose: CarPose | None = None,
    offset: np.ndarray | None = None,
    pitch: float = 0.0,
    roll: float = 0.0,
    diagnostic: dict | None = None,
    depth: np.ndarray | None = None,
) -> tuple[str | None, np.ndarray | None, bool, bool]:
    # 只留下落在这一格 50 cm 里的白点。最高那一截的离地高度对上某一档，才写层数。
    # 最高一截若只剩画面边上的点，顶被切掉了，不写。
    # 最后一项为真：看见塔顶红蓝，但分不出哪一色在上面。
    def fail(reason, piece=None, outside=False, veto=False):
        if diagnostic is not None:
            diagnostic["reason"] = reason
        return None, piece, outside, veto

    if cloud is None:
        return fail("深度不足：没有有效白点云")
    x0, x1, y0, y1 = cell.search_bounds(MARGIN_M)
    world = cloud["world"]
    chosen = (
        (world[:, 0] >= x0)
        & (world[:, 0] <= x1)
        & (world[:, 1] >= y0)
        & (world[:, 1] <= y1)
        & (world[:, 2] >= cell.base_z)
        & (world[:, 2] <= cell.cap_z + _TOP_MARGIN_M)
    )
    spatial = chosen.copy()
    chosen &= cloud["z"] <= _DEPTH_FAR_M
    if int(chosen.sum()) < _DEPTH_MIN_POINTS:
        return fail("深度不足：格内有效白点少于10个")
    xs = cloud["xs"][chosen]
    ys = cloud["ys"][chosen]
    distance = cloud["z"][chosen]
    world = world[chosen]
    risen = world[:, 2] >= cell.base_z + MIN_VISIBLE_M
    body = world[risen]
    # 深度不能绕过基本尺寸检查；用水平主轴跨度拒绝细杆、窄板。
    if len(body) < 30:
        return fail("深度不足：盒身有效点少于30个")
    def width_of(points):
        xy = points[:, :2] - np.median(points[:, :2], axis=0)
        _u, _s, axes = np.linalg.svd(xy, full_matrices=False)
        spans = np.percentile(xy @ axes.T, 95, axis=0) - np.percentile(xy @ axes.T, 5, axis=0)
        return float(spans.max())

    width = width_of(body)
    at_boundary = bool(np.any(distance >= _DEPTH_FAR_M - _DEPTH_EDGE_M))
    if width < WIDTH_MIN_M and at_boundary:
        # 只有原来已有格内近3米的支撑点，才考虑同一柱体里3–3.3米的点。
        # 远处整片墙、没有近侧支撑的点云不能靠这个入口变成塔。
        extended = spatial & (cloud["z"] <= _DEPTH_FAR_M + _DEPTH_BOUNDARY_M)
        extra = extended & (cloud["z"] > _DEPTH_FAR_M)
        if int(extra.sum()) >= _DEPTH_MIN_POINTS:
            extended_world = cloud["world"][extended]
            extended_risen = extended_world[:, 2] >= cell.base_z + MIN_VISIBLE_M
            extended_body = extended_world[extended_risen]
            recovered_width = width_of(extended_body)
            if WIDTH_MIN_M <= recovered_width <= WIDTH_MAX_M:
                world = extended_world
                xs, ys = cloud["xs"][extended], cloud["ys"][extended]
                distance = cloud["z"][extended]
                risen = extended_risen
                if diagnostic is not None:
                    diagnostic["depth_boundary_recovery"] = {
                        "original_width_cm": width * 100,
                        "recovered_width_cm": recovered_width * 100,
                        "extra_points": int(extra.sum()),
                    }
                width = recovered_width
    selected_body=_near_body_support(cell,world,distance,xs,ys,mask.shape,diagnostic)
    if selected_body is None:
        # 仍保留窄板/3米截断的原有诊断，但不从多个杂点拼出完整盒宽。
        if WIDTH_MIN_M<=width<=WIDTH_MAX_M:
            return fail("深度归属未证实：没有唯一可靠盒体",veto=True)
    else:
        world,xs,ys,distance=world[selected_body],xs[selected_body],ys[selected_body],distance[selected_body]
        risen=world[:,2]>=cell.base_z+MIN_VISIBLE_M
        if np.count_nonzero(risen)<30:
            return fail("深度归属不足：盒身支撑少于30点",veto=True)
        width=width_of(world[risen])
    if diagnostic is not None:
        diagnostic["depth_width_cm"] = width * 100
    if not (WIDTH_MIN_M <= width <= WIDTH_MAX_M):
        partial = width < WIDTH_MIN_M and at_boundary
        if diagnostic is not None and partial:
            diagnostic["depth_partial_at_boundary"] = True
        return fail(
            f"尺寸不符：深度水平跨度{width*100:.0f}厘米，要求15至65厘米"
            + ("；接近3米边界，需完整彩色轮廓核验" if partial else ""),
            veto=not partial,
        )
    outside = int(np.count_nonzero((mask[ys, xs] == 0) & risen)) >= _DEPTH_MIN_POINTS
    step = _DEPTH_SLICE_M
    bins = np.floor((world[:, 2] - cell.base_z) / step).astype(int)
    counts = np.bincount(np.clip(bins, 0, None))
    filled = np.flatnonzero(counts >= _DEPTH_MIN_POINTS)
    if len(filled) == 0:
        return fail("深度不足：没有连续的高度切片", outside=outside)
    top = bins == int(filled[-1])
    # 最高一截贴着画面上沿或下沿，说明顶已经出画，量到的只是正面中间。
    cut = (ys <= 1) | (ys >= intr.height - 2)
    if bool(np.any(top & cut)):
        return fail("顶边出画：深度最高一段贴近画面边缘", outside=outside)
    top_z = float(np.median(world[top, 2]))
    if diagnostic is not None:
        diagnostic["top_pixels"] = np.column_stack(
            (xs[top], ys[top])
        )[::max(1, int(top.sum())//80)].tolist()
        diagnostic["depth_height_cm"] = top_z * 100
        diagnostic["height_source"] = "深度最高切片"
    if (pose is not None and offset is not None
            and (diagnostic or {}).get('near_body',{}).get('excluded_points',0)==0):
        seen = _silhouette_top(cell, labels, xs, ys, pose, offset, pitch, intr, roll)
        # 彩色顶边对得上两层、深度却停在下面，用顶边。不要把一面墙的高点当成两层。
        if (
            seen is not None
            and seen > top_z
            and heights_match_level(seen, cell.two_z)
            and not heights_match_level(top_z, cell.two_z)
        ):
            top_z = seen
            if diagnostic is not None:
                diagnostic["height_source"] = "彩色顶边补高"
    one_ok = (not blocked) and heights_match_level(top_z, cell.one_z)
    two_ok = heights_match_level(top_z, cell.two_z)
    # 中间高度没有足够证据，保持未知。
    piece_mask = np.zeros(mask.shape, dtype=np.uint8)
    piece_mask[ys, xs] = 255
    piece = _outline(piece_mask)
    # 整塔高点与白盒层高分开判定。
    if top_z > cell.two_z + 0.08:
        if abs(top_z - cell.cap_z) > 0.06:
            return fail(
                f"整塔高度不符：{top_z*100:.1f}厘米，"
                f"带顶块允许{(cell.cap_z-.06)*100:.1f}"
                f"至{(cell.cap_z+.06)*100:.1f}厘米",
                piece, outside, True,
            )

        faction, cap_reason = _cap_from_depth(
            cell, world, xs, ys, distance, color, mask, intr, diagnostic
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
        return fail(
            f"层高不符：采用高度{top_z*100:.1f}厘米；一层允许{cell.one_z*80:.1f}至{cell.one_z*120:.1f}，两层允许{cell.two_z*80:.1f}至{cell.two_z*120:.1f}厘米；来源{diagnostic.get('height_source', '深度') if diagnostic is not None else '深度'}",
            piece, outside, True,
        )
    if pose is not None and offset is not None:
        depth_top=_near_depth_top(world,xs,ys,depth,intr,diagnostic)
        if depth_top:
            if diagnostic is not None:
                diagnostic['height_source']='深度最高切片＋实测顶边支撑'
                diagnostic['depth_primary']=True
        elif ((diagnostic or {}).get('height_source')!='彩色顶边补高'
              and not _near_top_observed(world,xs,ys,color,depth,intr,diagnostic)):
            return fail("盒顶未证实：实测深度只到盒面中段或顶边被遮挡",piece,outside,True)
    if one_ok and two_ok:
        use_two = abs(top_z - cell.two_z) <= abs(top_z - cell.one_z)
    else:
        use_two = two_ok
    if not use_two:
        return "1", piece, outside, False
    if (diagnostic or {}).get('depth_primary'):
        # 完整顶块由独立RGB/整塔高度路径仲裁；零碎颜色不推翻可靠白盒深度。
        return "2",piece,outside,False
    faction, unclear = _faction(color, piece_mask, mask, float(np.median(distance)), intr, diagnostic)
    if unclear:
        return fail("阵营不清：塔顶颜色不足或上下关系不明确", piece, outside, True)
    return (faction if faction is not None else "2"), piece, outside, False


def _outline(mask: np.ndarray) -> np.ndarray | None:
    if mask.size == 0 or int(mask.max()) == 0:
        return None
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None
    return max(found, key=cv2.contourArea)


def _max_edge_deviation(contour: np.ndarray, approx: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) > 400:
        points = points[:: max(1, len(points) // 400)]
    verts = approx.reshape(-1, 2).astype(np.float64)
    nearest = np.full(len(points), np.inf)
    count = len(verts)
    for index in range(count):
        start = verts[index]
        edge = verts[(index + 1) % count] - start
        length2 = float(edge @ edge)
        if length2 < 1e-6:
            dist = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
        else:
            along = np.clip((points - start) @ edge / length2, 0.0, 1.0)
            proj = start + along[:, None] * edge
            dist = np.hypot(points[:, 0] - proj[:, 0], points[:, 1] - proj[:, 1])
        nearest = np.minimum(nearest, dist)
    return float(nearest.max()) if len(nearest) else 0.0


def _close_small(component: np.ndarray) -> np.ndarray:
    # 补上短边 8% 以内的孔。线缆和标签留下的小洞不算空心，大的空心框仍然留不住。
    mask = component.astype(np.uint8) * 255
    _x, _y, width, height = cv2.boundingRect(mask)
    short = float(min(width, height))
    radius = max(1, int(round(EDGE_SIMPLIFY * short / 2.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0


def _block_shaped(component: np.ndarray) -> bool:
    closed = _close_small(component)
    contour = _outline(closed.astype(np.uint8) * 255)
    if contour is None or len(contour) < CORNER_MIN:
        return False
    area = float(cv2.contourArea(contour))
    if area < 4.0:
        return False
    if float(np.count_nonzero(closed)) < FILL_MIN * area:
        return False
    _x, _y, width, height = cv2.boundingRect(contour)
    short = float(min(width, height))
    if short < 2.0:
        return False
    approx = cv2.approxPolyDP(contour, max(2.0, EDGE_SIMPLIFY * short), True)
    if len(approx) < CORNER_MIN or len(approx) > CORNER_MAX:
        return False
    # 圆收成折线后仍接近圆。像素台阶会把原轮廓的圆度拉低，所以用折线来比。
    # 六边和七边是看见顶面的白盒。远处多出一个角仍留下，八边才比圆度。
    poly_peri = float(cv2.arcLength(approx, True))
    poly_area = float(cv2.contourArea(approx))
    circular = poly_peri >= 1e-6 and 4.0 * np.pi * poly_area / (poly_peri * poly_peri) > _CIRCLE_LIMIT
    if poly_peri < 1e-6 or (len(approx) > 7 and circular):
        return False
    if poly_area < 0.80 * area or area < 0.80 * poly_area:
        return False
    return _max_edge_deviation(contour, approx) <= max(2.5, EDGE_SIMPLIFY * short)


def _white_pieces(white: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, float, float]]]:
    # 整幅白掩膜上找块。框只在后面用来看质心落进哪一格。
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
    pieces = []
    for index in range(1, count):
        x, y, w, h, _pixels = [int(stats[index, k]) for k in range(5)]
        if w < 2 or h < 2:
            continue
        pieces.append((index, x, y, w, h, float(centroids[index, 0]), float(centroids[index, 1])))
    return labels, pieces


def _centroid_inside(mask: np.ndarray, cx: float, cy: float) -> bool:
    ix, iy = int(round(cx)), int(round(cy))
    height, width = mask.shape
    if not (0 <= ix < width and 0 <= iy < height):
        return False
    return bool(mask[iy, ix])


def _color_reading(
    cell: Cell,
    accepted: list[tuple],
    color: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    blocked: bool,
    one_h: float,
    two_h: float,
    diagnostic: dict | None = None,
) -> tuple[str, np.ndarray | None, float] | None:
    def fail(text):
        if diagnostic is not None:
            diagnostic["reason"] = text
        return None

    # 一格里多块错开的白合在一起，取从最低底到最高顶的整段。单块仍量它自己。
    union = np.zeros(mask.shape, dtype=np.uint8)
    distances = []
    pixel_tops = []
    pixel_bots = []
    world_tops = []
    top_y = None
    bot_y = None
    for _x, y, _w, h, distance, component, top_z in accepted:
        union[component] = 255
        distances.append(distance)
        if top_z is None:
            pixel_tops.append(y)
            pixel_bots.append(y + h)
        else:
            world_tops.append(top_z)
        top_y = y if top_y is None else min(top_y, y)
        bot_y = y + h if bot_y is None else max(bot_y, y + h)
    piece = _outline(union)
    distance = float(np.median(distances))
    if len(accepted) == 1 and accepted[0][6] is not None:
        height_m = float(accepted[0][6]) - cell.base_z
    elif pixel_tops and not world_tops:
        height_m = _metric(max(pixel_bots) - min(pixel_tops), distance, intr.fy)
    elif world_tops and not pixel_tops:
        height_m = max(world_tops) - cell.base_z
    else:
        span = _metric(max(pixel_bots) - min(pixel_tops), distance, intr.fy) if pixel_tops else 0.0
        lifted = (max(world_tops) - cell.base_z) if world_tops else 0.0
        height_m = max(span, lifted)
    one_ok = (not blocked) and heights_match(height_m, one_h)
    two_ok = heights_match(height_m, two_h)
    if not one_ok and not two_ok:
        return fail(
            f"彩色层高不符：可见高{height_m*100:.1f}厘米；"
            f"一层允许{one_h*50:.1f}至{one_h*150:.1f}，"
            f"两层允许{two_h*50:.1f}至{two_h*150:.1f}厘米"
        )
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return ("1", piece, distance)
    faction, unclear = _faction(color, union, mask, distance, intr, diagnostic)
    if unclear:
        return fail("阵营不清：颜色有效面积不足或上下关系不明确")
    return (faction if faction is not None else "2", piece, distance)


def _side_visibility(cell,x,width,pose,offset,pitch,intr,z,side,roll=0.):
    """只衡量左右出画；图像边界不是物体边界。位置/高度另行核验。"""
    if x>1 and x+width<intr.width-1:
        return 1.0
    uv=project(cell.center(z),pose,offset,pitch,intr,roll)
    if uv is None or not 0<=uv[0]<intr.width:
        return 0.0
    # 中心到未截断侧边估计全宽，同时限制不能小于真实物体的最小投影宽。
    center=float(uv[0])
    full=2*(x+width-center) if x<=1 else 2*(center-x)
    distance=float(to_camera(cell.center(z),pose,offset,pitch,roll)[2])
    expected=max(full,side*intr.fx/max(.2,distance))
    return min(1.,(width+2)/max(1.,expected))


def _upper_white_reading(cell,color,white,depth,mask,pose,offset,pitch,intr,roll=0.,diagnostic=None):
    """底部前景遮挡时，用本塔上部实测盒面、真实顶边和上方空域确认层数。"""
    info={}
    if diagnostic is not None:
        diagnostic['upper_white']=info
    if depth is None:
        return None
    ys,xs=np.nonzero((mask>0)&(white>0)&np.isfinite(depth)&(depth>=_DEPTH_NEAR_M))
    if len(xs)<30:
        return None
    values=depth[ys,xs].astype(float)
    origin,right,down,ahead=camera_frame(pose,offset,pitch,roll)
    world=(origin+((xs-intr.cx)*values/intr.fx)[:,None]*right
           +((ys-intr.cy)*values/intr.fy)[:,None]*down+values[:,None]*ahead)
    x0,x1,y0,y1=cell.search_bounds(0.)
    own=((world[:,0]>=x0)&(world[:,0]<=x1)&(world[:,1]>=y0)&(world[:,1]<=y1)
         &(world[:,2]>=cell.base_z+.025)&(world[:,2]<=cell.cap_z+.1))
    world,xs,ys=world[own],xs[own],ys[own]
    if len(xs)<30:
        return None
    # 先看全部格内支撑，再匹配一/两层，不能先裁成一层。
    highest=float(np.percentile(world[:,2],98))
    state,level=min([('1',cell.one_z),('2',cell.two_z)],key=lambda item:abs(highest-item[1]))
    tolerance=.065
    if abs(highest-level)>tolerance:
        info['reason']='实测盒面顶高不符合一层或两层'
        return None
    columns,first=np.unique(xs,return_index=True)
    top=float(np.median(world[first,2]))
    span=float(np.percentile(world[:,2],98)-np.percentile(world[:,2],2))
    xy=world[:,:2]-np.median(world[:,:2],axis=0)
    _,_,axes=np.linalg.svd(xy,full_matrices=False)
    widths=np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    info.update(top_cm=top*100,visible_cm=span*100,width_cm=float(max(widths))*100)
    if abs(top-level)>tolerance or span<.035 or not .12<=max(widths)<=WIDTH_MAX_M:
        return None
    # 至少三成盒宽可见；层数仍需实测顶边支撑。
    if _side_visibility(cell,int(xs.min()),int(xs.max()-xs.min()+1),pose,offset,pitch,
                        intr,(cell.base_z+level)/2,.35,roll)<.3:
        return None
    if not _near_top_observed(world,xs,ys,color,depth,intr,info):
        info['reason']='未证实真实盒顶'
        return None
    if not _upper_space_clear(cell,level,pose,offset,pitch,intr,depth,roll,info.setdefault('upper_space',{})):
        info['reason']='盒顶上方空域未证实'
        return None
    piece=np.zeros(mask.shape,np.uint8)
    piece[ys,xs]=255
    info.update(reason='本塔上部盒面＋真实顶边＋上方空域',layers=state)
    return state,_outline(piece)


def _cap_width_limit(cell, pose, offset, pitch, intr, z_cam, roll=0.0):
    """顶块随盒体旋转，两个面同时可见；投影包络还包含安装/定位容差。"""
    widths = []
    # 24cm包络包含20cm顶块的尺寸与轮廓误差；偏移只改变透视比例，
    # 不能把不同位置的包络并起来虚增物体自身宽度。
    for dx in (-.10, 0., .10):
        for dy in (-.10, 0., .10):
            for angle in np.linspace(0., np.pi/2, 9):
                c, sn = np.cos(angle), np.sin(angle)
                pixels = []
                for a in (-.12, .12):
                    for b in (-.12, .12):
                        for z in (cell.two_z, cell.cap_z):
                            uv = project(np.array([cell.x+dx+c*a-sn*b,
                                                   cell.y+dy+sn*a+c*b,z]),
                                         pose,offset,pitch,intr,roll)
                            if uv is not None:
                                pixels.append(float(uv[0]))
                if len(pixels) == 8:
                    widths.append(max(pixels)-min(pixels))
    predicted = _metric(max(widths)+2, z_cam, intr.fx) if widths else .34
    return max(.34, min(.40, predicted))


def _rgb_cap_reading(cell, color, mask, pose, offset, pitch, intr, z_cam,
                     roll=0.0, diagnostic=None):
    """顶块独立提供阵营证据；白盒遮挡、粘连和缺失深度不否定它。"""
    red,blue=_cap_colors(color)
    allowed=(diagnostic or {}).get('_cap_allowed')
    suspected_allowed=(diagnostic or {}).get('_cap_suspected_allowed')
    width_limit = _cap_width_limit(cell,pose,offset,pitch,intr,z_cam,roll)
    radius = max(1, min(3, int(round(.015 * intr.fy / z_cam))))
    colored = (red | blue).astype(np.uint8)
    closed = cv2.morphologyEx(colored, cv2.MORPH_CLOSE,
                             np.ones((2*radius+1, 2*radius+1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    candidates = []
    checks = []
    suspected = []
    for index in range(1, count):
        x, y, w, h, pixels = [int(v) for v in stats[index]]
        # 小噪点不构造整帧布尔图；仍计入components原始块数。
        if pixels < 12:
            continue
        inside = int(np.count_nonzero((labels[y:y+h, x:x+w] == index)
                                      & (mask[y:y+h, x:x+w] > 0)))
        part = labels == index
        # 仅诊断与当前格相交的颜色块；识别仍检查全部连通块。
        item = dict(id=index, bbox=[x, y, w, h], pixels=pixels,
                    inside_pixels=inside, coverage=inside/max(1, pixels),
                    width_cm=_metric(w, z_cam, intr.fx)*100,
                    width_limit_cm=width_limit*100,
                    height_cm=_metric(h, z_cam, intr.fy)*100,
                    fill=pixels/max(1, w*h), failures=[])
        if diagnostic is not None and inside:
            checks.append(item)
        def reject(reason):
            item['failures'].append(reason)
        if pixels < 12:
            reject('颜色块不足12像素')
        side_cut=x<=1 or x+w>=intr.width-1
        side_fraction=_side_visibility(cell,x,w,pose,offset,pitch,intr,
                                       (cell.two_z+cell.cap_z)/2,.20,roll)
        item['side_visible_fraction']=side_fraction
        # 已知场地的红蓝配对本身就是顶块证据。裁切、遮挡、位姿误差
        # 会改变尺寸及推算顶高，不能再用完整顶块的尺寸否决可见配对。
        if pixels < .30*w*h:
            reject('填充率不足30%')
        if inside < .35*pixels:
            reject('格内覆盖不足35%')
        # 两色面积提前记录，单色碎片也能查明是否断开。
        ry, rx = np.nonzero(red & part)
        by, bx = np.nonzero(blue & part)
        need = max(8, int(.05*pixels))
        item.update(red_pixels=len(ry), blue_pixels=len(by), need=need)
        # 单色背景不构成顶块占用证据；红蓝均成块时才保留待确认状态。
        item['suspected'] = False
        paired_blocks = (len(ry)>=need and len(by)>=need and all(
            xx.max()-xx.min()+1>=3 and yy.max()-yy.min()+1>=3
            and len(yy)>=.35*(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)
            for yy,xx in ((ry,rx),(by,bx))))
        if (paired_blocks
                and (suspected_allowed is None or np.count_nonzero(part & suspected_allowed)>=.8*pixels)
                and .08 <= item['width_cm']/100 <= .40
                and .04 <= item['height_cm']/100 <= .32
                and item['coverage'] >= .80 and item['fill'] >= .30
                and y>1 and y+h<intr.height-1 and (not side_cut or side_fraction>=.8)):
            samples = [x+w*.25, x+w*.5, x+w*.75]
            bottoms = [_z_at_cell(cell, u, y+h, pose, offset, pitch, intr, roll)
                       for u in samples]
            tops = [_z_at_cell(cell, u, y, pose, offset, pitch, intr, roll)
                    for u in samples]
            if all(v is not None for v in bottoms+tops):
                bottom, top = float(np.median(bottoms)), float(np.median(tops))
                item.update(suspected_bottom_cm=bottom*100, suspected_top_cm=top*100)
                if (bottom >= cell.two_z-.05 and bottom <= cell.cap_z-.025
                        and top >= cell.two_z+.04 and top <= cell.cap_z+.10):
                    item['suspected'] = True
                    suspected.append(index)
        if len(ry) < need or len(by) < need:
            reject('红或蓝面积不足（可能单色/断开）')
        if item['failures']:
            continue
        # 各色也须呈块状，避免两根细线凑成红蓝配对。
        for yy,xx in ((ry,rx),(by,bx)):
            if (xx.max()-xx.min()+1<3 or yy.max()-yy.min()+1<3
                    or len(yy)<.35*(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)):
                reject('单色不是有面积的块状区域')
        bottom=_z_at_cell(cell,x+w/2,y+h,pose,offset,pitch,intr,roll)
        top=_z_at_cell(cell,x+w/2,y,pose,offset,pitch,intr,roll)
        if bottom is not None and top is not None:
            item.update(bottom_cm=bottom*100,top_cm=top*100,
                        bottom_error_cm=(bottom-cell.two_z)*100,
                        top_error_cm=(top-cell.cap_z)*100)
        overlap = min(rx.max(), bx.max()) - max(rx.min(), bx.min()) + 1
        overlap_ratio = float(overlap / min(rx.max()-rx.min()+1, bx.max()-bx.min()+1))
        item['horizontal_overlap'] = overlap_ratio
        if overlap_ratio < .5:
            reject('红蓝横向重叠不足50%')
        angle = np.deg2rad(roll)
        rd = -np.sin(angle)*(rx-intr.cx)+np.cos(angle)*(ry-intr.cy)
        bd = -np.sin(angle)*(bx-intr.cx)+np.cos(angle)*(by-intr.cy)
        red_above = np.percentile(rd,75) < np.percentile(bd,25)
        blue_above = np.percentile(bd,75) < np.percentile(rd,25)
        item['vertical_gap_px'] = float(max(np.percentile(bd,25)-np.percentile(rd,75),
                                            np.percentile(rd,25)-np.percentile(bd,75)))
        if not (red_above or blue_above):
            reject('红蓝上下未分开（分位间距须大于0）')
        if not item['failures'] and allowed is not None and np.count_nonzero(part & allowed)<.8*pixels:
            if suspected_allowed is not None and np.count_nonzero(part & suspected_allowed)>=.8*pixels:
                # 合格红蓝配对仅归属未定，即使出画也不能降级为普通白盒。
                item['suspected']=True
                suspected.append(index)
            reject('顶块未唯一归属本塔')
        if item['failures']:
            continue
        item['result'] = '红' if red_above else '蓝'
        candidates.append((item['result'], part))
    if diagnostic is not None:
        diagnostic['rgb_cap'] = dict(
            candidates=len(candidates), components=count-1, checks=checks,
            state=('confirmed' if len(candidates) == 1 else
                   'suspected' if suspected or candidates else 'absent'),
            suspected_components=suspected,
            distance_m=z_cam, pitch_deg=float(pitch), roll_deg=float(roll),
            yaw_deg=float(np.degrees(pose.yaw)),
            expected_bottom_cm=cell.two_z*100, expected_top_cm=cell.cap_z*100,
            close_radius_px=radius,
            reason=('唯一候选通过' if len(candidates) == 1 else
                    '多个合格候选，无法唯一归属' if candidates else '没有合格候选'))
    if len(candidates) != 1:
        return None
    faction, part = candidates[0]
    return faction, _outline(part.astype(np.uint8)*255)


def _far_cap_reading(cell, color, white, depth, mask, pose, offset, pitch,
                     intr, z_cam, roll=0.0, diagnostic=None):
    """远处白盒粘墙时，用塔顶定位，再用格内深度分离白盒；不量整片白墙。"""
    info = {}
    if diagnostic is not None:
        diagnostic["far_support"] = info

    def fail(reason):
        info["reason"] = reason
        return None

    if depth is None or depth.shape != white.shape:
        return fail("远处粘连：缺少可核验白盒归属的深度")
    red,blue=_cap_colors(color,diagnostic)
    colored=red|blue
    # 只跨过红蓝之间很小的黑缝，不能把相隔的彩色杂物拼成顶块。
    radius = max(1, min(3, int(round(.015 * intr.fy / z_cam))))
    closed = cv2.morphologyEx(colored.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((2*radius+1, 2*radius+1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    candidates = []
    width_limit = _cap_width_limit(cell,pose,offset,pitch,intr,z_cam,roll)
    for index in range(1, count):
        x, y, w, h, pixels = stats[index]
        if pixels < 12 or x <= 1 or y <= 1 or x+w >= intr.width-1 or y+h >= intr.height-1:
            continue
        if not (.08 <= _metric(w, z_cam, intr.fx) <= width_limit
                and .14 <= _metric(h, z_cam, intr.fy) <= .28
                and pixels >= .45*w*h):
            continue
        part = labels == index
        if np.count_nonzero(part & (mask > 0)) < .9*pixels:
            continue
        # 顶块底边必须位于两层白盒顶，顶部也要符合整塔高度。
        py, px = np.nonzero(part)
        low = np.array([py[px == col].max()+1 for col in range(x,x+w)])
        high = np.array([py[px == col].min() for col in range(x,x+w)])
        bottom_z = [_z_at_cell(cell, col, row, pose, offset, pitch, intr, roll)
                    for col, row in zip(range(x,x+w),low)]
        top_z = [_z_at_cell(cell, col, row, pose, offset, pitch, intr, roll)
                 for col, row in zip(range(x,x+w),high)]
        if any(v is None for v in bottom_z+top_z):
            continue
        if abs(float(np.median(bottom_z))-cell.two_z) > .12:
            continue
        if abs(float(np.median(top_z))-cell.cap_z) > .15:
            continue
        # 白盒必须直接接在顶块下面；这个顶边由彩色块定位，不由整片白色的最高点定位。
        anchor = np.zeros(white.shape, np.uint8)
        contacts = 0
        for col, row in zip(range(x,x+w),low):
            end = min(intr.height, row+max(3, radius*2+1))
            hit = np.flatnonzero(white[row:end,col] > 0)
            if hit.size:
                anchor[row+int(hit[0]),col] = 255
                contacts += 1
        if contacts < .5*w:
            continue
        candidates.append((part, anchor, x, y, int(w), int(h)))
    info["cap_candidates"] = len(candidates)
    if len(candidates) != 1:
        return fail("远处粘连：没有唯一且位置/尺寸正确、下接白盒的塔顶")
    cap, anchor, x, y, w, h = candidates[0]

    # 深度可以超过3米，但须反投影落在这座塔的柱体内，白墙和前景不能当支撑。
    good = ((white > 0) & (mask > 0) & np.isfinite(depth)
            & (depth >= _DEPTH_NEAR_M))
    ys, xs = np.nonzero(good)
    values = depth[ys,xs].astype(np.float64)
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    x0,x1,y0,y1 = cell.search_bounds(MARGIN_M)
    own = ((world[:,0] >= x0) & (world[:,0] <= x1)
           & (world[:,1] >= y0) & (world[:,1] <= y1)
           & (world[:,2] >= cell.base_z+MIN_VISIBLE_M)
           & (world[:,2] <= cell.two_z+.08)
           & (xs >= x+w/2-.325*intr.fx/z_cam)
           & (xs <= x+w/2+.325*intr.fx/z_cam)
           & (ys >= y+h-3))
    support = world[own]
    info["support_points"] = len(support)
    if len(support) < 30:
        return fail("远处粘连：塔顶下方格内白盒深度不足30点")
    xy = support[:,:2]-np.median(support[:,:2],axis=0)
    _u,_s,axes = np.linalg.svd(xy,full_matrices=False)
    spans = np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    width_m = float(spans.max())
    low_z, high_z = np.percentile(support[:,2],[10,90])
    info.update(width_cm=width_m*100, height_span_cm=float(high_z-low_z)*100,
                top_height_cm=float(high_z)*100)
    if not (WIDTH_MIN_M <= width_m <= WIDTH_MAX_M):
        return fail("远处粘连：分离后的白盒宽度不符")
    if high_z-low_z < .20 or high_z < cell.two_z-.15:
        return fail("远处粘连：缺少足够的白盒高度支撑")
    faction, unclear = _faction(color, anchor, cap.astype(np.uint8)*255,
                                z_cam, intr, diagnostic, linked_cap=True)
    if unclear or faction is None:
        return fail("远处粘连：塔顶阵营不明确")
    piece = np.zeros(mask.shape,np.uint8)
    piece[ys[own],xs[own]] = 255
    info["reason"] = "塔顶定位＋格内深度分离白盒"
    info["support_pixels"] = np.column_stack((xs[own],ys[own]))[::max(1,int(own.sum())//300)].tolist()
    return faction, _outline(piece)


def _upper_space_clear(cell, top_z, pose, offset, pitch, intr, depth, roll=0.0, diagnostic=None):
    """验证采样射线已穿过上方塔位；近背景也可证明空域，缺测不充当占用。"""
    if diagnostic is not None:
        diagnostic.update(state='unknown',levels=[])
    if depth is None:
        return False
    half=min(.20,cell.foot/2-.02)
    unknown=False
    for level in np.arange(top_z+.08,cell.cap_z+.081,.06):
        clear=observed=occupied=0
        for dx in np.linspace(-half,half,5):
            for dy in np.linspace(-half,half,5):
                point=np.array([cell.x+dx,cell.y+dy,level])
                pixel=project(point,pose,offset,pitch,intr,roll)
                if pixel is None:
                    continue
                u,v=np.round(pixel).astype(int)
                if not (1<=u<intr.width-1 and 1<=v<intr.height-1):
                    continue
                patch=depth[v-1:v+2,u-1:u+2]
                valid=np.isfinite(patch)&(patch>=_DEPTH_NEAR_M)
                if np.count_nonzero(valid)<5:
                    continue
                observed+=1
                expected=float(to_camera(point,pose,offset,pitch,roll)[2])
                # 深度确实越过本采样位置即可，不再强求背景还远40cm。
                # 5cm余量抑制深度噪声；至少80%的有效邻域支持同一结论。
                if np.count_nonzero(valid&(patch>expected+.05))>=.8*np.count_nonzero(valid):
                    clear+=1
                elif np.count_nonzero(valid&(np.abs(patch-expected)<=.08))>=.8*np.count_nonzero(valid):
                    occupied+=1
        # 一些缺测允许存在，但每个高度仍需至少半数位置有观测，不能凭空补空域。
        if diagnostic is not None:
            diagnostic['levels'].append(dict(height_cm=float(level*100),observed=observed,
                                               clear=clear,occupied=occupied))
        if occupied>=2:
            if diagnostic is not None:
                diagnostic['state']='occupied'
            return False
        unknown |= observed<13 or clear/max(1,observed)<.8
    if diagnostic is not None:
        diagnostic['state']='unknown' if unknown else 'clear'
    return not unknown


def _rgb_upper_consistent(cell, level, pose, offset, pitch, intr, depth, roll, info):
    """RGB已有真实顶边时，上方缺测只记未知；实测占用仍作反证。"""
    _upper_space_clear(cell,level,pose,offset,pitch,intr,depth,roll,info)
    if info.get('state')=='clear':
        return True
    return info.get('state')!='occupied'


def _far_white_group(world, own, labels, ys, xs, component):
    """只合并同一竖直表面上相邻的实测碎片，不补造深度。"""
    selected = own & (labels[ys, xs] == component)
    seed = world[selected]
    if len(seed) < 30:
        return selected
    center = np.median(seed[:, :2], axis=0)
    _, _, axes = np.linalg.svd(seed[:, :2]-center, full_matrices=False)
    coords = (world[:, :2]-center) @ axes.T
    # 种子自身深度已经散乱时，不拿它去吸收其他白色物体。
    if np.percentile(np.abs(coords[selected, 1]), 90) > .035:
        return selected
    for candidate in np.unique(labels[ys[own], xs[own]]):
        if candidate in (0, component):
            continue
        part = own & (labels[ys, xs] == candidate)
        if part.sum() < 10 or np.percentile(np.abs(coords[part, 1]), 90) > .035:
            continue
        a = np.percentile(coords[selected, 0], [5, 95])
        b = np.percentile(coords[part, 0], [5, 95])
        za = np.percentile(world[selected, 2], [5, 95])
        zb = np.percentile(world[part, 2], [5, 95])
        if max(a[0]-b[1], b[0]-a[1]) > .10 or max(za[0]-zb[1], zb[0]-za[1]) > .12:
            continue
        if max(a[1], b[1])-min(a[0], b[0]) <= WIDTH_MAX_M:
            selected |= part
    return selected


def _vertical_faces(world, selected):
    """确定性稳健拟合竖直面：把同一连通组的正面/侧面与横向杂点分开。"""
    indices = np.flatnonzero(selected)
    points = world[indices]
    if len(points) < 30:
        return []
    xy = points[:, :2]-np.median(points[:, :2],axis=0)
    _,_,axes = np.linalg.svd(xy,full_matrices=False)
    if np.percentile(np.abs(xy@axes[-1]),90) <= .035:
        return [selected]
    rng = np.random.default_rng(0)
    remaining = np.ones(len(indices),bool)
    faces = []
    for _ in range(4):
        ids = np.flatnonzero(remaining)
        if len(ids)<30:
            break
        sample = ids[np.linspace(0,len(ids)-1,min(256,len(ids))).astype(int)]
        pairs = rng.choice(sample,size=(64,2))
        best = None
        for first,second in pairs:
            direction = points[second,:2]-points[first,:2]
            length = np.linalg.norm(direction)
            if length < .12:
                continue
            normal = np.array([-direction[1],direction[0]])/length
            hits = np.abs((points[sample,:2]-points[first,:2])@normal) <= .02
            score = int(hits.sum())
            if score < 12:
                continue
            if np.ptp(points[sample[hits],2]) < .03:
                continue
            if best is None or score>best[0]:
                best = score,normal,points[first,:2]
        if best is None:
            break
        _,normal,anchor = best
        inlier = remaining & (np.abs((points[:,:2]-anchor)@normal)<=.02)
        if inlier.sum()<30:
            break
        center = np.median(points[inlier,:2],axis=0)
        _,_,axes = np.linalg.svd(points[inlier,:2]-center,full_matrices=False)
        inlier = remaining & (np.abs((points[:,:2]-center)@axes[-1])<=.025)
        face = np.zeros(len(world),bool)
        face[indices[inlier]] = True
        faces.append(face)
        remaining[inlier] = False
    return faces


def _far_spatial_labels(world, own, ys, xs, shape):
    """在4厘米三维网格上分组，禁止二维闭运算跨过前后表面的深度跳变。"""
    indices = np.flatnonzero(own)
    voxels, inverse = np.unique(np.floor(world[own]/.04).astype(np.int32),
                                axis=0, return_inverse=True)
    lookup = {tuple(v): i for i, v in enumerate(voxels)}
    groups = np.zeros(len(voxels), np.int32)
    neighbors = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1)
                 for c in (-1,0,1) if (a,b,c) != (0,0,0)]
    number = 0
    for start in range(len(voxels)):
        if groups[start]:
            continue
        number += 1
        groups[start] = number
        queue = [start]
        while queue:
            x,y,z = map(int, voxels[queue.pop()])
            for a,b,c in neighbors:
                index = lookup.get((x+a,y+b,z+c))
                if index is not None and not groups[index]:
                    groups[index] = number
                    queue.append(index)
    labels = np.zeros(shape, np.int32)
    labels[ys[indices], xs[indices]] = groups[inverse]
    counts = np.bincount(groups[inverse], minlength=number+1)
    return labels, [int(i) for i in np.argsort(counts[1:])[::-1]+1 if counts[i]>=30]


def _far_white_top_edge(support, xs, ys, depth, intr):
    """盒底缺失时，要求横向分布的平顶及紧邻顶边后的背景深度。"""
    columns, first = np.unique(xs, return_index=True)
    # 调用者提供的点按图像行排序，同一列的第一个点就是实测顶边。
    edge = support[first]
    if len(columns) < 8:
        return False
    top = float(np.median(edge[:, 2]))
    flat = np.abs(edge[:, 2]-top) <= .03
    if flat.mean() < .80:
        return False
    columns, rows, edge = columns[flat], ys[first][flat], edge[flat]
    if np.linalg.norm(np.percentile(edge[:, :2], 95, axis=0)
                      - np.percentile(edge[:, :2], 5, axis=0)) < WIDTH_MIN_M:
        return False
    clear = 0
    for col, row in zip(columns, rows):
        value = float(depth[row, col])
        # 上移约3厘米后采样，排除顶边混合深度；无效深度不能证明背景。
        above = int(row-max(3, round(.03*intr.fy/value)))
        if not (1 <= col < intr.width-1 and 1 <= above < intr.height-1):
            continue
        patch = depth[above-1:above+2, col-1:col+2]
        if np.count_nonzero(np.isfinite(patch) & (patch > value+.40)) >= 7:
            clear += 1
    return clear >= .80*len(columns)


def _far_rgb_top(support, xs, ys, color, white, depth, pose, offset,
                 pitch, intr, roll, diagnostic=None):
    """实测竖直盒面限定搜索；原图顶边须横向一致且上方有后退的背景深度。"""
    detail = {} if diagnostic is None else diagnostic
    def fail(reason):
        detail['reason'] = reason
        return None
    center = np.median(support[:, :2], axis=0)
    _, _, axes = np.linalg.svd(support[:, :2]-center, full_matrices=False)
    normal = axes[-1]
    detail['plane_error_cm'] = float(np.percentile(np.abs((support[:, :2]-center) @ normal), 90))*100
    if detail['plane_error_cm'] > 3.5:
        return fail('盒面不共面')
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    columns, first = np.unique(xs, return_index=True)
    detail['support_columns'] = len(columns)
    if len(columns) < 8:
        return fail('横向支撑不足8列')
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(float)
    hits = []
    edge_seen = background_seen = 0
    for col, index in zip(columns, first):
        row = int(ys[index])
        # 最多补回12厘米；不能向上无限搜索柜门、墙线或预期层高。
        reach = max(3, int(np.ceil(.12*intr.fy/depth[row, col])))
        found = None
        for edge in range(row, max(2, row-reach)-1, -1):
            if edge+2 >= intr.height or not 1 <= col < intr.width-1:
                continue
            below = gray[edge:edge+3, col-1:col+2]
            above = gray[edge-3:edge, col-1:col+2]
            if abs(float(np.median(below))-float(np.median(above))) < 12:
                continue
            if np.count_nonzero(white[edge:edge+3, col-1:col+2]) < 7:
                continue
            ray = (col-intr.cx)/intr.fx*right + (edge-intr.cy)/intr.fy*down + ahead
            denom = float(ray[:2] @ normal)
            if abs(denom) < 1e-6:
                continue
            distance = float((center-origin[:2]) @ normal)/denom
            point = origin+distance*ray
            if distance <= 0 or not -.02 <= point[2]-support[index, 2] <= .12:
                continue
            edge_seen += 1
            # 顶边附近常有深度空洞；向上3至12厘米寻找实测背景，不填补空洞。
            clear = False
            for rise in (.03, .06, .09, .12):
                back_row = edge-max(3, int(round(rise*intr.fy/distance)))
                if back_row < 1:
                    continue
                patch = depth[back_row-1:back_row+2, col-1:col+2]
                if np.count_nonzero(np.isfinite(patch) & (patch > distance+.40)) >= 7:
                    clear = True
                    break
            if not clear:
                continue
            background_seen += 1
            found = point
            break
        if found is not None:
            hits.append(found)
    detail.update(edge_samples=edge_seen, background_samples=background_seen,
                  accepted_columns=len(hits))
    if len(hits) < max(8, .80*len(columns)):
        return fail('原图边界或后方背景有效列不足80%')
    hits = np.asarray(hits)
    height = float(np.median(hits[:, 2]))
    flat = np.abs(hits[:, 2]-height) <= .03
    if flat.mean() < .80:
        return fail('顶边高度不一致')
    edge = hits[flat]
    width = float(np.ptp((edge[:, :2]-center) @ axes[0]))
    if not WIDTH_MIN_M <= width <= WIDTH_MAX_M:
        return fail('顶边宽度不符')
    detail['reason'] = '原图顶边及后方背景通过'
    return height, len(edge)


def _far_white_reading(cell, color, white, depth, mask, pose, offset, pitch,
                       intr, roll=0., diagnostic=None):
    """没有顶块锚点时，独立用格内白色深度及上方空域判断一/两层。"""
    info = {}
    if diagnostic is not None:
        diagnostic["far_white"] = info
    def fail(reason):
        info["reason"] = reason
        return None
    if depth is None or depth.shape != white.shape:
        return fail("白盒与背景未分离：缺少深度，需要侧向视角")
    ys,xs = np.nonzero((white > 0) & (mask > 0) & np.isfinite(depth)
                       & (depth >= _DEPTH_NEAR_M))
    if len(xs) < 30:
        return fail("白盒深度不足，需要侧向视角")
    values = depth[ys,xs].astype(float)
    origin,right,down,ahead = camera_frame(pose,offset,pitch,roll)
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    x0,x1,y0,y1 = cell.search_bounds(MARGIN_M)
    own = ((world[:,0]>=x0)&(world[:,0]<=x1)&(world[:,1]>=y0)&(world[:,1]<=y1)
           &(world[:,2]>=cell.base_z+MIN_VISIBLE_M)&(world[:,2]<=cell.cap_z+.10))
    if own.sum() < 30:
        return fail("格内白盒支撑不足，需要侧向视角")
    labels, components = _far_spatial_labels(world, own, ys, xs, white.shape)
    if not components:
        return fail("没有完整白盒支撑，需要侧向视角")
    if len(components) > 12:
        return fail("格内独立表面过多，需要侧向视角")
    candidates, checked = [], set()
    for component in components:
        selected = _far_white_group(world, own, labels, ys, xs, component)
        for face in _vertical_faces(world, selected):
            key = np.packbits(face).tobytes()
            if key in checked:
                continue
            checked.add(key)
            detail = {}
            reading = _far_white_candidate(cell,color,white,depth,mask,pose,offset,
                pitch,intr,roll,world,face,xs,ys,detail)
            candidates.append((reading,detail,face))
    info['spatial_groups'] = len(components)
    info['faces'] = len(candidates)
    info['groups'] = [dict(result=None if r is None else r[0], **d)
                      for r,d,_ in candidates]
    if not candidates:
        return fail("格内没有可靠竖直盒面，需要侧向视角")
    accepted = [(r,d,f) for r,d,f in candidates if r is not None]
    if not accepted:
        info.update(candidates[0][1])
        return None
    states = {r[0] for r,_,_ in accepted}
    if len(states)>1:
        return fail("多个盒面层高冲突，需要侧向视角")
    # 不挑最容易通过的低面；另一个可信盒面还有更高支撑时必须复核。
    level = cell.one_z if accepted[0][0][0] == '1' else cell.two_z
    if any(d.get('width_cm',0)>=WIDTH_MIN_M*100 and
           d.get('top_height_cm',0)>100*(level+.10) for _,d,_ in candidates):
        return fail("其他盒面存在更高支撑，不能用低面确定层数，需要侧向视角")
    combined = np.logical_or.reduce([f for _,_,f in accepted])
    support = world[combined]
    xy = support[:,:2]-np.median(support[:,:2],axis=0)
    _,_,axes = np.linalg.svd(xy,full_matrices=False)
    width = np.max(np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0))
    if width>WIDTH_MAX_M:
        return fail("多个盒面合并宽度超限，需要侧向视角")
    info.update(accepted[0][1])
    info['accepted_faces'] = len(accepted)
    info['support_points'] = len(support)
    info['support_pixels'] = np.column_stack((xs[combined],ys[combined]))[::max(1,int(combined.sum())//300)].tolist()
    piece = np.zeros(white.shape,np.uint8)
    piece[ys[combined],xs[combined]] = 255
    return accepted[0][0][0], _outline(piece)


def _far_white_candidate(cell, color, white, depth, mask, pose, offset,
                         pitch, intr, roll, world, own, xs, ys, info):
    def fail(reason):
        info['reason'] = reason
        return None
    support = world[own]
    info["support_points"] = len(support)
    info["support_pixels"] = np.column_stack((xs[own],ys[own]))[::max(1,int(own.sum())//300)].tolist()
    if len(support)<30:
        return fail("白盒有效支撑不足，需要侧向视角")
    # 搜索余量用于找到候选；独立盒面整体位于塔位外沿时属于背景，
    # 不能再作为“另一个更高盒面”否决已经核验的目标。
    x0,x1,y0,y1=cell.search_bounds(0.)
    centered=((support[:,0]>=x0-.015)&(support[:,0]<=x1+.015)
              &(support[:,1]>=y0-.015)&(support[:,1]<=y1+.015))
    if np.mean(centered)<.1:
        return fail("独立表面位于塔位外侧，仅落入定位搜索余量")
    xy = support[:,:2]-np.median(support[:,:2],axis=0)
    _u,_s,axes = np.linalg.svd(xy,full_matrices=False)
    span = np.percentile(xy@axes.T,95,axis=0)-np.percentile(xy@axes.T,5,axis=0)
    width = float(span.max())
    low,top = np.percentile(support[:,2],[5,98])
    span_z = float(top-low)
    info.update(width_cm=width*100,top_height_cm=float(top)*100,
                height_span_cm=span_z*100)
    if not WIDTH_MIN_M<=width<=WIDTH_MAX_M:
        return fail("白盒宽度支撑不足，需要侧向视角")
    # 保留深度原始顶高供诊断；明暗边界只在可靠盒面和背景证据齐备时补顶。
    info['depth_top_height_cm'] = float(top)*100
    rgb_top = None
    if span_z < .18 or min(abs(top-cell.one_z), abs(top-cell.two_z)) > .08:
        rgb_top = _far_rgb_top(support, xs[own], ys[own], color, white, depth,
                               pose, offset, pitch, intr, roll, info.setdefault('rgb_top_check', {}))
    if rgb_top is not None and rgb_top[0] < top-.02:
        rgb_top = None
    if rgb_top is not None:
        top, edge_columns = rgb_top
        info.update(rgb_top_height_cm=top*100, rgb_top_columns=edge_columns,
                    top_height_cm=top*100)
    partial = span_z < .18
    info['partial_body'] = partial
    if partial:
        # 只有顶边确实与后方背景分离，才允许盒底遮挡；小白片不能只靠顶高过关。
        xy = support[:,:2]-np.median(support[:,:2],axis=0)
        plane_error = float(np.percentile(np.abs(xy @ axes[-1]),90))
        info['top_edge_clear'] = (plane_error <= .035 and
            (rgb_top is not None or (span_z >= .06 and
             _far_white_top_edge(support,xs[own],ys[own],depth,intr))))
        if not info['top_edge_clear']:
            return fail("白盒高度支撑不足且顶边未分离，需要侧向视角")
    levels = [(abs(top-cell.one_z), "1", cell.one_z),
              (abs(top-cell.two_z), "2", cell.two_z)]
    error,state,level = min(levels)
    if error>.08:
        return fail("白盒顶高不确定，需要侧向视角")
    info["upper_clear"] = _upper_space_clear(cell,level,pose,offset,pitch,intr,depth,roll,info.setdefault('upper_space',{}))
    if not info["upper_clear"]:
        return fail("未证实白盒上方为空：可能还有上层或顶块，需要侧向视角")
    piece = np.zeros(white.shape,np.uint8)
    piece[ys[own],xs[own]] = 255
    info["support_pixels"] = np.column_stack((xs[own],ys[own]))[::max(1,int(own.sum())//300)].tolist()
    info["reason"] = ("格内白盒深度＋分离顶边＋顶高＋上方空域（盒底不完整）"
                      if partial else "格内白盒深度＋顶高＋上方空域")
    return state,_outline(piece)


def _resolve_layer_seams(candidates, color, white, mask, cell, pose, offset,
                         pitch, intr, z_cam, roll=0., depth=None, diagnostic=None):
    """仅在同帧一/两层证据竞争时，检查低横线是否为同塔盒体的接缝。"""
    if {item[0] for item in candidates} != {'1', '2'}:
        return candidates
    info={'triggered':True,'checks':[],'removed_indices':[],'state':'unresolved'}
    if diagnostic is not None:
        diagnostic.update(info)
        info=diagnostic
    gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY).astype(float)
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    valid_white=(white>0)&(hsv[:,:,1]<=100)&(mask>0)
    origin,right,down,ahead=camera_frame(pose,offset,pitch,roll)

    def top_edge(item):
        # 轮廓可能从左下角开始，不能假定前两个点就是顶边。
        xy=np.asarray(item[1]).reshape(-1,2)
        if len(xy)==4:
            edges=[]
            for index,a in enumerate(xy):
                b=xy[(index+1)%len(xy)]
                dx=float(b[0]-a[0])
                if abs(dx)>=8 and abs((b[1]-a[1])/dx-np.tan(np.deg2rad(roll)))<=.4:
                    gravity=(-np.sin(np.deg2rad(roll))*(a[0]+b[0])
                             +np.cos(np.deg2rad(roll))*(a[1]+b[1]))
                    edges.append((gravity,a,b))
            if edges:
                _,a,b=min(edges,key=lambda q:q[0])
                a,b=sorted((a,b),key=lambda q:q[0])
                return np.asarray(a,float),np.asarray(b,float)
        y=float(xy[:,1].min())
        top=xy[xy[:,1]<=y+max(3,.02*intr.fy/z_cam)]
        a,b=top[np.argmin(top[:,0])],top[np.argmax(top[:,0])]
        return np.asarray(a,float),np.asarray(b,float)

    def world_samples(rows,cols):
        values=depth[rows,cols].astype(float)
        good=np.isfinite(values)&(values>=_DEPTH_NEAR_M)
        if good.sum()<12:
            return None
        u,v,d=cols[good],rows[good],values[good]
        return (origin+((u-intr.cx)*d/intr.fx)[:,None]*right
                +((v-intr.cy)*d/intr.fy)[:,None]*down+d[:,None]*ahead)

    removed=set()
    for low_index,low in enumerate(candidates):
        if low[0]!='1':
            continue
        la,lb=top_edge(low)
        for high_index,high in enumerate(candidates):
            if high[0]!='2':
                continue
            ha,hb=top_edge(high)
            gap=high[2]-low[2]
            overlap=min(lb[0],hb[0])-max(la[0],ha[0])
            width=min(lb[0]-la[0],hb[0]-ha[0])
            ratio=(lb[0]-la[0])/max(1,hb[0]-ha[0])
            if not .70<=ratio<=1.40:
                continue
            if not .20<=gap<=.50 or width<8 or overlap<.5*width:
                continue
            item={'low_index':low_index,'high_index':high_index,
                  'height_gap_cm':float(gap*100),'overlap':float(overlap/width)}
            info['checks'].append(item)
            # 允许上下盒错位12cm，但不把另一个横向分离的背景盒面拼进来。
            aligned=min(abs(la[0]-ha[0]),abs(lb[0]-hb[0]))*z_cam/intr.fx<=.12
            if not aligned:
                item['reason']='侧边横向不对应'
                continue
            cols=np.arange(int(max(la[0],ha[0]))+3,int(min(lb[0],hb[0]))-2,dtype=int)
            if len(cols)<8:
                continue
            seam=la[1]+(cols-la[0])*(lb[1]-la[1])/max(1,lb[0]-la[0])
            upper=ha[1]+(cols-ha[0])*(hb[1]-ha[1])/max(1,hb[0]-ha[0])
            # 跨过4cm暗缝，检查其上方4~16cm处是否恢复为连续盒面。
            start=max(4,int(.04*intr.fy/z_cam))
            stop=max(start+4,int(.16*intr.fy/z_cam))
            above=np.round(seam[:,None]-np.arange(start,stop)[None,:]).astype(int)
            below=np.round(seam[:,None]+np.arange(start,stop)[None,:]).astype(int)
            xx=np.broadcast_to(cols[:,None],above.shape)
            if (above.min()<3 or below.max()>=intr.height-3
                    or cols.min()<4 or cols.max()>=intr.width-4
                    or np.any(above[:,0]<=upper+3)):
                item['reason']='上下盒面可见支撑不足'
                continue
            column_fill=np.mean(valid_white[above,xx],axis=1)
            fill=float(np.mean(column_fill>=.65))
            item['upper_face_columns']=fill
            if fill<.65:
                item['reason']='跨过缝隙后没有连续上层白面'
                continue
            # 同一上层盒面至少一侧从更高候选延续到接缝附近。
            sides=0
            for u,top_y in (ha,hb):
                u=int(round(u))
                bottom_y=float(la[1]+(u-la[0])*(lb[1]-la[1])/max(1,lb[0]-la[0]))
                rows=np.arange(int(top_y)+start,int(bottom_y)-start,dtype=int)
                if not 4<=u<intr.width-4 or len(rows)<6 or rows.min()<0 or rows.max()>=intr.height:
                    continue
                contrast=np.abs(gray[rows,u-3]-gray[rows,u+3])
                if np.mean(contrast>=8)>=.5:
                    sides+=1
            item['continued_sides']=sides
            # 接缝下方也必须仍有本塔盒面；单盒底边上方当然有白面，
            # 不能因此把真实盒底当作两层接缝。
            lower_fill=float(np.mean(np.mean(valid_white[below,xx],axis=1)>=.65))
            lower_sides=0
            for u,top_y in (la,lb):
                u=int(round(u))
                rows=np.arange(int(top_y)+start,int(top_y)+stop,dtype=int)
                if 4<=u<intr.width-4 and rows.min()>=0 and rows.max()<intr.height:
                    if np.mean(np.abs(gray[rows,u-3]-gray[rows,u+3])>=8)>=.5:
                        lower_sides+=1
            item.update(lower_face_columns=lower_fill,lower_continued_sides=lower_sides)
            if lower_fill<.65:
                item['reason']='低横线下方没有连续盒面，可能是真实盒底'
                continue
            depth_connected=False
            if depth is not None:
                up=world_samples(above.ravel(),xx.ravel())
                dn=world_samples(below.ravel(),xx.ravel())
                if up is not None and dn is not None:
                    x0,x1,y0,y1=cell.search_bounds(MARGIN_M)
                    def own(points):
                        return ((points[:,0]>=x0)&(points[:,0]<=x1)
                                &(points[:,1]>=y0)&(points[:,1]<=y1))
                    up_own,dn_own=own(up),own(dn)
                    item['depth_ownership']=[float(up_own.mean()),float(dn_own.mean())]
                    if up_own.mean()<.6 or dn_own.mean()<.6:
                        item['reason']='上下深度不属于同塔，保留原判断'
                        continue
                    up_mid=np.median(up[up_own],axis=0)
                    dn_mid=np.median(dn[dn_own],axis=0)
                    distance=float(np.linalg.norm(up_mid[:2]-dn_mid[:2]))
                    item['depth_xy_gap_cm']=distance*100
                    depth_connected=(distance<=.16 and up_mid[2]>dn_mid[2]+.04)
                    if not depth_connected:
                        item['reason']='上下实测表面不连续，保留原判断'
                        continue
                item['depth_state']='connected' if depth_connected else 'unknown'
            # 若较高盒面已有完整单盒相对高度，RGB须看到接缝下两侧
            # 均延续，才推翻它；防止盒底后的台面边缘冒充下层。
            if high[3] and lower_sides<2 and not depth_connected:
                item['reason']='完整单盒高度仍成立，下层侧边不足以推翻'
                continue
            if (sides==0 or lower_sides==0) and not depth_connected:
                item['reason']='缺少上下侧边延续或深度连接证据'
                continue
            item['reason']='一层横线上方仍是同塔盒面，判为接缝'
            item['seam']=True
            removed.add(low_index)
    info['removed_indices']=sorted(removed)
    if removed:
        info['state']='seam_rejected'
    return [item for index,item in enumerate(candidates) if index not in removed]


def _white_face_continues_above(piece,color,white,mask,z_cam,intr,roll=0.):
    """无上方顶边候选时，也能否决盒面内的接缝；不据此补造两层结果。"""
    x,y,w,h=cv2.boundingRect(piece)
    if _metric(w,z_cam,intr.fx)<.20:
        return False
    gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY).astype(float)
    cols=np.arange(x+4,x+w-4)
    rows=y+(cols-(x+w/2))*np.tan(np.deg2rad(roll))
    start=max(5,int(.05*intr.fy/z_cam))
    stop=max(start+6,int(.22*intr.fy/z_cam))
    above=np.round(rows[:,None]-np.arange(start,stop)[None,:]).astype(int)
    below=np.round(rows[:,None]+np.arange(start,max(start+4,int(.12*intr.fy/z_cam)))[None,:]).astype(int)
    if not len(cols) or above.min()<4 or below.max()>=intr.height-4:
        return False
    valid=(white>0)&(mask>0)
    if (np.mean(np.mean(valid[above,cols[:,None]],axis=1)>=.75)<.75
            or np.mean(np.mean(valid[below,cols[:,None]],axis=1)>=.75)<.75):
        return False
    # 白墙本身不构成上层盒面：左右两条真实侧边必须向上延续，
    # 允许上下盒错位8cm；单边、窗框和纯亮度横纹不触发。
    radius=max(3,int(.08*intr.fx/z_cam))
    for side,u0 in [('left',x),('right',x+w-1)]:
        found=False
        for u in range(max(4,u0-radius),min(intr.width-4,u0+radius+1)):
            top=y+(u-(x+w/2))*np.tan(np.deg2rad(roll))
            rr=np.round(top-np.arange(start,stop)).astype(int)
            inner=u+3 if side=='left' else u-3
            if (np.mean(np.abs(gray[rr,u-3]-gray[rr,u+3])>=8)>=.7
                    and np.mean(valid[rr,inner])>=.75):
                found=True
                break
        if not found:
            return False
    return True


def _merge_white_top_lines(lines,z_cam,intr,roll=0.):
    """合并短缺口两侧共线的顶边；原始线段保留，合并结果仍须核验盒面。"""
    original=np.asarray(lines,dtype=np.int32).reshape(-1,4)
    segments=[]
    for x0,y0,x1,y1 in original:
        if x1<x0:
            x0,y0,x1,y1=x1,y1,x0,y0
        if x1-x0<8:
            continue
        slope=(y1-y0)/(x1-x0)
        if abs(slope-np.tan(np.deg2rad(roll)))<=.25:
            segments.append((int(x0),int(y0),int(x1),int(y1),float(slope)))
    segments.sort(key=lambda q:(q[0],q[1]))
    gap=max(4,int(.05*intr.fx/z_cam))
    merged=[]
    for x0,y0,x1,y1,slope in segments:
        for a,b,c,d,other_slope in segments:
            if a<=x0 or a> x1+gap or c<=x1 or abs(other_slope-slope)>.06:
                continue
            if abs(b-(y0+(a-x0)*slope))>3 or _metric(c-x0,z_cam,intr.fx)>.65:
                continue
            merged.append((x0,y0,c,d))
    return np.unique(np.asarray([*original,*merged],np.int32).reshape(-1,4),axis=0)


def _rgb_white_top(cell,color,white,mask,pose,offset,pitch,intr,z_cam,roll=0.,diagnostic=None,depth=None):
    """从真实顶边和局部盒面生成候选，不要求可见盒底或独立白色连通块。"""
    info={'checks':[]}
    if diagnostic is not None:
        diagnostic['rgb_white_top']=info
    gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    edges=cv2.Canny(gray,18,50)
    # 饱和颜色不是白盒边；同时避免录屏中彩色框线成为几何证据。
    ink=cv2.dilate((hsv[:,:,1]>100).astype(np.uint8),np.ones((5,5),np.uint8))
    edges[(mask==0)|(ink>0)]=0
    min_length=max(10,int(.12*intr.fx/z_cam))
    full_length=max(12,int(.16*intr.fx/z_cam))
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(10,full_length//2),
                          minLineLength=full_length,maxLineGap=4)
    short=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(10,min_length//2),
                         minLineLength=min_length,maxLineGap=4)
    if short is not None:
        lines=short if lines is None else np.concatenate((lines.reshape(-1,4),short.reshape(-1,4)))
    bridged=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(10,full_length//4),
                           minLineLength=full_length,maxLineGap=max(4,int(.05*intr.fx/z_cam)))
    if bridged is not None:
        lines=bridged if lines is None else np.concatenate((lines.reshape(-1,4),bridged.reshape(-1,4)))
    hough_lines=set() if lines is None else {tuple(map(int,q)) for q in lines.reshape(-1,4)}
    # Hough会随搜索掩膜内其它边缘的增减而漏掉同一条顶边。小roll时
    # 再逐行收集已有边缘的连续横段；不创造边缘，后续仍核验盒面和侧边。
    if abs(roll)<3:
        joined=cv2.dilate(edges,np.ones((3,1),np.uint8))
        joined=cv2.morphologyEx(joined,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))
        transitions=np.diff(np.pad(joined>0,((0,0),(1,1))).astype(np.int8),axis=1)
        yy,starts=np.nonzero(transitions==1)
        _,ends=np.nonzero(transitions==-1)
        spans=[]
        for row,left,right in zip(yy,starts,ends):
            if row%2==0 and right-left>=full_length:
                spans.append([left,row,right-1,row])
        if spans:
            extra=np.asarray(spans,np.int32)
            lines=extra if lines is None else np.concatenate((lines.reshape(-1,4),extra))
    if lines is None:
        return None
    lines=_merge_white_top_lines(lines,z_cam,intr,roll)
    candidates=[]
    pending=[]
    band=max(8,int(.12*intr.fy/z_cam))
    # 绝对顶高由格心距离反投影，前侧盒面及小角度姿态误差会随距离
    # 放大。两层留出1.2度角误差及4cm近面误差，上限14cm小于半层高；
    # 一层仍用7.5cm，避免低背景白块趁放宽后成为一层候选。
    # 顶边、侧边、连续盒面及深度归属的核验继续独立执行。
    height_tolerance=max(.075,min(.14,.04+z_cam*np.tan(np.deg2rad(1.2))))
    info['height_tolerance_cm']=height_tolerance*100
    for x0,y0,x1,y1 in lines.reshape(-1,4):
        from_hough=tuple(map(int,(x0,y0,x1,y1))) in hough_lines
        if x1<x0:
            x0,y0,x1,y1=x1,y1,x0,y0
        w=int(x1-x0)
        if w<min_length or abs((y1-y0)/max(1,w)-np.tan(np.deg2rad(roll)))>.25:
            continue
        if not .12<=_metric(w,z_cam,intr.fx)<=.65:
            continue
        z=_z_at_cell(cell,(x0+x1)/2,(y0+y1)/2,pose,offset,pitch,intr,roll)
        if z is None:
            continue
        state,level=min([('1',cell.one_z),('2',cell.two_z)],key=lambda item:abs(item[1]-z))
        if abs(z-level)>(height_tolerance if state=='2' else .075):
            continue
        # 盒顶在其50cm占地内的投影，而非搜索框内任意同高度线段。
        projected=[project(np.array([xx,yy,level]),pose,offset,pitch,intr,roll)
                   for xx in (cell.x-.25,cell.x+.25) for yy in (cell.y-.25,cell.y+.25)]
        projected=[q for q in projected if q is not None]
        if not projected or not min(q[0] for q in projected)-3<=(x0+x1)/2<=max(q[0] for q in projected)+3:
            continue
        cols=np.arange(x0+3,x1-2,dtype=int)
        rows=np.round(y0+(cols-x0)*(y1-y0)/w).astype(int)
        if len(cols)<8 or rows.min()<5 or rows.max()+band+3>=intr.height:
            continue
        below=rows[:,None]+np.arange(3,band+3)[None,:]
        valid_face=ink[below,cols[:,None]]==0
        if np.mean(valid_face)<.5:
            continue
        inside_white=white[below,cols[:,None]]>0
        fill=float(np.mean(inside_white[valid_face]))
        contrast=np.abs(gray[rows+3,cols].astype(float)-gray[rows-3,cols].astype(float))
        edge_support=float(np.mean(contrast>=8))
        if fill<.75 or edge_support<.5:
            continue
        # 白盒接缝不是顶边：上方连续白面、且亮度相近时不接受。
        above=rows[:,None]-np.arange(4,band+4)[None,:]
        above_white=float(np.mean(white[above,cols[:,None]]>0))
        above_gray=float(np.median(gray[above,cols[:,None]]))
        below_gray=float(np.median(gray[below,cols[:,None]]))
        if above_white>.5 and abs(above_gray-below_gray)<12:
            continue
        # 顶边可能被纹理/叠字分成短段，沿两端寻找真实向下侧边，
        # 不能把Hough短段的端点直接当成盒子左右边。
        extent=max(8,int(.22*intr.fx/z_cam))
        side_cols=[]
        for side,start,end,step in [('left',int(x0)+3,max(4,int(x0)-extent),-1),
                                     ('right',int(x1)-3,min(intr.width-5,int(x1)+extent),1)]:
            found=None
            for u in range(start,end,step):
                v=int(round(y0+(u-x0)*(y1-y0)/w))
                if not 5<=u<intr.width-5 or v+band+3>=intr.height or v<0:
                    continue
                a=gray[v+3:v+band+3,u-3].astype(float)
                b=gray[v+3:v+band+3,u+3].astype(float)
                inner=u+3 if side=='left' else u-3
                body=white[v+3:v+band+3,inner]>0
                strength=float(np.mean(np.abs(a-b)>=8))
                if strength>=.65 and np.mean(body)>=.65:
                    found=u
                    break
            side_cols.append(found)
        check=dict(state=state,top_cm=z*100,line=[int(x0),int(y0),int(x1),int(y1)],
                   fill=fill,edge_support=edge_support,sides=sum(q is not None for q in side_cols))
        info['checks'].append(check)
        if all(q is None for q in side_cols):
            continue
        if not from_hough and check['sides']<2:
            # 逐行补线只补有双侧边的盒面，避免杂物横边与单侧边拼接。
            continue
        if check['sides']==1 and edge_support<(.8 if _metric(w,z_cam,intr.fx)<.23 else .6):
            continue
        # 一条真实侧边加连续顶边也可定位局部盒面；未见侧不能伪称完整边。
        left=side_cols[0] if side_cols[0] is not None else int(x0)
        right=side_cols[1] if side_cols[1] is not None else int(x1)
        if right-left<w*.8 or not .12<=_metric(right-left,z_cam,intr.fx)<=.65:
            continue
        if not from_hough and w<.75*(right-left):
            # 圆盘/齿轮顶部的短平段不能向两侧扩成矩形盒面。
            continue
        if all(q is not None for q in side_cols) and _metric(right-left,z_cam,intr.fx)<.12:
            continue
        face_cols=np.arange(left+3,right-2,dtype=int)
        face_rows=np.round(y0+(face_cols-x0)*(y1-y0)/w).astype(int)
        face_below=face_rows[:,None]+np.arange(3,band+3)[None,:]
        valid=ink[face_below,face_cols[:,None]]==0
        face=(white[face_below,face_cols[:,None]]>0)&(mask[face_below,face_cols[:,None]]>0)
        if np.mean(valid)<.5 or np.mean(face[valid])<.75:
            continue
        column_valid=valid.sum(axis=1)
        continuous=(face & valid).sum(axis=1)/np.maximum(1,column_valid)
        usable_columns=column_valid>=.5*valid.shape[1]
        if not usable_columns.any() or np.mean(continuous[usable_columns]>=.75)<.75:
            continue
        # 短白条、窗框和印字也可能恰好落在预期顶高处。顶边下方应有
        # 至少半个盒高的连续盒面，而不只是12cm的浅色条带。
        support_band=max(band,int(.16*intr.fy/z_cam))
        if face_rows.max()+support_band+3>=intr.height:
            continue
        support_rows=face_rows[:,None]+np.arange(3,support_band+3)[None,:]
        support_valid=ink[support_rows,face_cols[:,None]]==0
        reference=np.median(gray[face_below,face_cols[:,None]][valid])
        face_consistent=((white[support_rows,face_cols[:,None]]>0)
                         &(np.abs(gray[support_rows,face_cols[:,None]].astype(float)-reference)<28))
        if np.mean(support_valid)<.5 or np.mean(face_consistent[support_valid])<.8:
            continue
        check['body_support']=float(np.mean(face_consistent[support_valid]))
        check['side_columns']=[left,right]
        check['complete_sides']=all(q is not None for q in side_cols)
        # 完整盒面的底边提供相对高度，抵消相机姿态/实际支撑面造成的
        # 绝对顶高偏移。底部被遮住时不从可见短段反推层数。
        if check['complete_sides'] and abs(roll)<15:
            bottom_evidence=[]
            top_gray=gray[face_below,face_cols[:,None]].astype(float)
            reference=np.median(top_gray,axis=1)
            max_drop=min(intr.height-5-int(face_rows.max()),int(.82*intr.fy/z_cam))
            for drop in range(band+5,max_drop):
                rr=face_rows+drop
                a=gray[rr-3,face_cols].astype(float)
                b=gray[rr+3,face_cols].astype(float)
                # 底边上侧应仍是这一盒面，下侧必须离开盒面；连续白色
                # 且同亮度的接缝不构成底边证据。
                same_above=np.abs(a-reference)<18
                leaves_face=np.abs(b-reference)>=18
                usable=ink[rr-3,face_cols]==0
                # 深色前景挡住下半塔时，这条横线是遮挡边而不是盒底。
                below_visible=b>=90
                support=np.mean(same_above & leaves_face & (np.abs(a-b)>=12) & usable & below_visible)
                if support<.7:
                    continue
                bottom_z=_z_at_cell(cell,float(np.median(face_cols)),float(np.median(rr)),
                                       pose,offset,pitch,intr,roll)
                if bottom_z is None or bottom_z<cell.base_z-.12:
                    continue
                height=z-bottom_z
                # 横线下方仍有同亮度白面时，它更可能是上下盒接缝。
                below_span=max(6,int(.10*intr.fy/z_cam))
                below_rows=rr[:,None]+np.arange(4,below_span+4)[None,:]
                if below_rows.max()<intr.height:
                    lower_gray=gray[below_rows,face_cols[:,None]].astype(float)
                    continues_white=((white[below_rows,face_cols[:,None]]>0)
                                     &(np.abs(lower_gray-reference[:,None])<28))
                    if np.mean(continues_white)>.5:
                        continue
                # 两侧都必须在底边结束；只有一侧结束时，仍可能是接缝或遮挡边。
                continued=[]
                for u in (left,right):
                    v=int(round(y0+(u-x0)*(y1-y0)/w))+drop+5
                    if v+band>=intr.height:
                        continued.append(True)
                    else:
                        continued.append(np.mean(np.abs(gray[v:v+band,u-3].astype(float)
                            -gray[v:v+band,u+3].astype(float))>=8)>.65)
                if any(continued):
                    continue
                relative=min((1,2),key=lambda n:abs(height-.35*n))
                if abs(height-.35*relative)<=.065:
                    bottom_evidence.append((support,relative,height,bottom_z))
            if bottom_evidence:
                _,relative,height,bottom_z=max(bottom_evidence,key=lambda q:q[0])
                check.update(absolute_state=state,body_height_cm=height*100,
                             bottom_cm=bottom_z*100,relative_state=str(relative))
                state=str(relative)
                level=cell.one_z if state=='1' else cell.two_z
        # 窄而两侧闭合的小白面易来自椅腿/窗框。它若没有盒底证据，
        # 不能冒充完整盒体；单侧出画及有真实底边的半盒仍允许。
        if (check['complete_sides'] and _metric(right-left,z_cam,intr.fx)<.23
                and 'relative_state' not in check):
            continue
        ly=int(round(y0+(left-x0)*(y1-y0)/w))
        ry=int(round(y0+(right-x0)*(y1-y0)/w))
        piece=np.array([[[left,ly]],[[right,ry]],[[right,ry+band]],[[left,ly+band]]],np.int32)
        pending.append((state,piece,z,'relative_state' in check,check,face_cols,face_rows,face_below))
    absolute=[(q[4]['state'],q[1],q[2],q[3] and q[0]!=q[4]['state']) for q in pending]
    arbitration={}
    if {q[0] for q in absolute}=={'1','2'}:
        _resolve_layer_seams(absolute,color,white,mask,cell,pose,offset,pitch,intr,
                             z_cam,roll,depth,arbitration)
        info['seam_conflict']=arbitration
    removed=set(arbitration.get('removed_indices',[]))
    confirmed_high={q['high_index'] for q in arbitration.get('checks',[]) if q.get('seam')}
    for index,(state,piece,z,relative,check,cols,face_rows,below) in enumerate(pending):
        if index in removed:
            check['seam_rejected']=True
            continue
        if index in confirmed_high:
            state='2'
            # 盒面中的横线已证实是接缝，不能再用它充当底边把两层缩为一层。
            relative=False
            check['seam_restored_state']='2'
        # 先保留低横线供上下盒面仲裁；没有配对顶边时再独立排除接缝。
        if (state=='1' and check['state']=='1' and depth is None
                and _white_face_continues_above(piece,color,white,mask,z_cam,intr,roll)):
            check['seam_rejected']=True
            continue
        level=cell.one_z if state=='1' else cell.two_z
        if depth is not None:
            yy,xx=below.ravel(),np.broadcast_to(cols[:,None],below.shape).ravel()
            values=depth[yy,xx].astype(float)
            good=np.isfinite(values)&(values>=_DEPTH_NEAR_M)
            if good.sum()>=12:
                values,xx,yy=values[good],xx[good],yy[good]
                origin,cam_right,down,ahead=camera_frame(pose,offset,pitch,roll)
                world=(origin+((xx-intr.cx)*values/intr.fx)[:,None]*cam_right
                       +((yy-intr.cy)*values/intr.fy)[:,None]*down+values[:,None]*ahead)
                bx0,bx1,by0,by1=cell.search_bounds(MARGIN_M)
                own=((world[:,0]>=bx0)&(world[:,0]<=bx1)&(world[:,1]>=by0)&(world[:,1]<=by1)
                     &(world[:,2]<=level+.08)&(world[:,2]>=level-.22))
                check['depth_support']=float(np.mean(own))
                if np.mean(own)<.7:
                    continue
            elif z_cam<=_DEPTH_FAR_M:
                continue
            # 提供了深度时，缺失的上层不能被RGB色差伪装为空域。
            if not _rgb_upper_consistent(cell,level,pose,offset,pitch,intr,depth,roll,
                                      check.setdefault('upper_space',{})):
                continue
            if check['upper_space'].get('state')=='unknown':
                top_values=depth[face_rows[:,None]+np.arange(1,5)[None,:],cols[:,None]]
                measured_top=np.isfinite(top_values)&(top_values>z_cam-.5)&(top_values<z_cam+.5)
                body_measured=((mask>0)&np.isfinite(depth)&(depth>z_cam-.5)&(depth<z_cam+.5))
                if np.count_nonzero(measured_top)<5 and np.any(body_measured):
                    check['depth_state']='body_only_without_top_or_upper'
                    continue
        # 同层候选比较真实盒面支撑，不再只挑反投影最高的一小段。
        # 层高偏差作软评分；层数仍先由几何、相对盒高及接缝核验决定。
        width=_metric(cv2.boundingRect(piece)[2],z_cam,intr.fx)
        score=(check['sides']+check['edge_support']+check['body_support']
               +min(1.,width/.35)-.25*abs(z-level)/height_tolerance)
        check['score']=float(score)
        candidates.append((state,piece,z,relative,float(score)))
    # 两层的真实顶边优先于下方接缝；几何验证不通过的线段不进入比较。
    info['candidates']=len(candidates)
    if not candidates:
        return None
    chosen=max(candidates,key=lambda item:(item[3],item[0]=='2',item[4]))
    info['state']=chosen[0]
    info['selected_score']=chosen[4]
    return chosen[:2]


def _rgb_white_edges(cell,color,white,mask,pose,offset,pitch,intr,z_cam,roll=0.,diagnostic=None,depth=None):
    """用实际灰度边界拆开同为白色的盒面与背景；不以搜索框裁切边当盒边。"""
    info = {'candidates': 0}
    if diagnostic is not None:
        diagnostic['rgb_white_edges'] = info
    gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,20,60)
    barrier = cv2.dilate(edges,np.ones((3,3),np.uint8))
    separated = ((white>0)&(mask>0)&(barrier==0)).astype(np.uint8)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(separated,8)
    candidates = []
    for i in range(1,count):
        x,y,w,h,pixels = map(int,stats[i])
        if pixels<30 or w<8 or h<8 or pixels<.60*w*h:
            continue
        side_cut=x<=1 or x+w>=intr.width-1
        if y<5 or y+h>=intr.height-5:
            continue
        if side_cut and _side_visibility(cell,x,w,pose,offset,pitch,intr,
                                         (cell.base_z+cell.one_z)/2,.35,roll)<.3:
            continue
        if not side_cut and (x<5 or x+w>=intr.width-5):
            continue
        if not (.12<=_metric(w,z_cam,intr.fx)<=.65):
            continue
        # 分割会向内缩1至3像素，用边缘带恢复盒体的完整边界。
        x0,x1,y0,y1 = max(0,x-2),min(intr.width,x+w+2),y-2,y+h+2
        if np.mean(mask[y0:y1,x0:x1]>0)<.5:
            continue
        strengths = []
        for side in ('top','bottom','left','right'):
            if (side=='left' and x<=1) or (side=='right' and x+w>=intr.width-1):
                continue
            if side in ('top','bottom'):
                row = y0 if side=='top' else y1
                a=gray[row-3,x:x+w].astype(float)
                b=gray[row+3,x:x+w].astype(float)
            else:
                col = x0 if side=='left' else x1
                a=gray[y:y+h,col-3].astype(float)
                b=gray[y:y+h,col+3].astype(float)
            strengths.append(float(np.mean(np.abs(a-b)>=10)))
        if strengths[0]<.7 or sum(q>=.65 for q in strengths)<2:
            continue
        # 连续白面上的亮度接缝不能作为盒顶，缺测也不能把接缝变成顶边。
        above=gray[max(0,y0-6):y0-2,x:x+w]
        below=gray[y0+2:min(intr.height,y0+6),x:x+w]
        if (above.size and below.size
                and np.mean(white[max(0,y0-6):y0-2,x:x+w]>0)>.8
                and abs(float(np.median(above))-float(np.median(below)))<12):
            continue
        lookback=max(8,int(.10*intr.fy/z_cam))
        upper_body=gray[max(0,y0-lookback):max(0,y0-6),x:x+w]
        if (upper_body.size and below.size
                and np.mean(np.abs(upper_body.astype(float)-float(np.median(below)))<12)>.55):
            continue
        top = _z_at_cell(cell,(x0+x1)/2,y0,pose,offset,pitch,intr,roll)
        bottom = _z_at_cell(cell,(x0+x1)/2,y1,pose,offset,pitch,intr,roll)
        if top is None or bottom is None or abs(bottom-cell.base_z)>.15:
            continue
        levels=[(state,z) for state,z in [('1',cell.one_z),('2',cell.two_z)]
                if abs(top-z)<=.09]
        if len(levels)!=1:
            continue
        part=np.zeros(mask.shape,np.uint8)
        part[y0:y1,x0:x1]=255
        candidates.append((levels[0][0],_outline(part),strengths,top,bottom))
    info['candidates']=len(candidates)
    if len(candidates)!=1:
        info['reason']='没有唯一且四边、顶底高度完整的RGB盒面'
        return None
    state,piece,strengths,top,bottom=candidates[0]
    info.update(edge_coverage=strengths,top_cm=top*100,bottom_cm=None if bottom is None else bottom*100)
    if depth is not None:
        region=np.zeros(mask.shape,np.uint8)
        cv2.drawContours(region,[piece],-1,255,-1)
        ys,xs=np.nonzero((region>0)&(white>0)&np.isfinite(depth)&(depth>=_DEPTH_NEAR_M))
        if len(xs)<12:
            info['depth_state']='unknown'
            return (state,piece) if _rgb_upper_consistent(cell,top,pose,offset,pitch,intr,depth,roll,info.setdefault('upper_space',{})) else None
        values=depth[ys,xs].astype(float)
        origin,right,down,ahead=camera_frame(pose,offset,pitch,roll)
        world=(origin+((xs-intr.cx)*values/intr.fx)[:,None]*right
               +((ys-intr.cy)*values/intr.fy)[:,None]*down+values[:,None]*ahead)
        x0,x1,y0,y1=cell.search_bounds(MARGIN_M)
        own=((world[:,0]>=x0)&(world[:,0]<=x1)&(world[:,1]>=y0)&(world[:,1]<=y1)
             &(world[:,2]>=cell.base_z-.06)&(world[:,2]<=top+.06))
        info['depth_coverage']=float(np.mean(own))
        level=cell.one_z if state=='1' else cell.two_z
        upper=info.setdefault('upper_space',{})
        _rgb_upper_consistent(cell,level,pose,offset,pitch,intr,depth,roll,upper)
        # 盒身有深度，但顶边和上方同时缺测，不能将中间接缝补成顶边。
        px,py,pw,ph=cv2.boundingRect(piece)
        top_patch=depth[py:min(py+6,depth.shape[0]),px:px+pw]
        top_valid=np.isfinite(top_patch)&(top_patch>z_cam-.5)&(top_patch<z_cam+.5)
        if upper.get('state')=='unknown' and np.count_nonzero(top_valid)<5:
            info['reason']='盒身有深度，但顶边与上方同时缺测'
            return None
        if np.mean(own)<.6 or not _rgb_upper_consistent(cell,level,pose,offset,pitch,intr,depth,roll,info.setdefault('upper_space',{})):
            info['reason']='RGB边界未通过深度归属或上方空间核验'
            return None
    return state,piece


def _far_reading(
    color: np.ndarray,
    white: np.ndarray,
    mask: np.ndarray,
    z_cam: float,
    intr: Intrinsics,
    blocked: bool,
    one_h: float,
    two_h: float,
    diagnostic: dict | None = None,
) -> tuple[str, np.ndarray | None] | None:
    def fail(text):
        if diagnostic is not None:
            diagnostic["reason"] = text
        return None

    # 只在超过深度范围、直边又收不上时用。先留下够宽的白，再补黑缝，量整块宽高。
    # 窄于 15 cm 的碎片不补到一起。圆和齿轮仍然不写。
    local = np.zeros(mask.shape, dtype=np.uint8)
    local[(white > 0) & (mask > 0)] = 255
    if int(local.max()) == 0:
        return fail("远处：搜索框内没有通过颜色阈值的白色")
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(local, connectivity=8)
    wide = np.zeros(mask.shape, dtype=np.uint8)
    found = False
    for index in range(1, count):
        _x, _y, w, h, pixels = [int(stats[index, k]) for k in range(5)]
        if pixels < 4 or w < 2 or h < 2:
            continue
        if not (WIDTH_MIN_M <= _metric(w, z_cam, intr.fx) <= WIDTH_MAX_M):
            continue
        wide[labels == index] = 255
        found = True
    if not found:
        return fail("远处：没有宽15至65厘米且至少2×2像素的白块")
    radius = _FAR_SEAM_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    closed = cv2.morphologyEx(wide, cv2.MORPH_CLOSE, kernel)
    contour = _outline(closed)
    if contour is None:
        return fail("远处：补缝后没有有效轮廓")
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri < 1e-6 or 4.0 * np.pi * area / (peri * peri) > _CIRCLE_LIMIT:
        return fail("远处轮廓不符：周长无效或轮廓过圆")
    _x, _y, w, h = cv2.boundingRect(contour)
    if w < 2 or h < 2 or area < FILL_MIN * float(w * h):
        return fail(
            f"远处填充不足：{100*area/max(1,w*h):.1f}%，"
            f"要求至少75%；框{w}×{h}像素"
        )
    width_m = _metric(float(w), z_cam, intr.fx)
    if width_m < WIDTH_MIN_M or width_m > WIDTH_MAX_M:
        return fail(
            f"远处宽度不符：{width_m*100:.1f}厘米，要求15至65厘米"
        )
    height_m = _metric(float(h), z_cam, intr.fy)
    one_ok = (not blocked) and heights_match(height_m, one_h)
    two_ok = heights_match(height_m, two_h)
    piece = contour
    if not one_ok and not two_ok:
        return fail(
            f"彩色层高不符：可见高{height_m*100:.1f}厘米；"
            f"一层允许{one_h*50:.1f}至{one_h*150:.1f}，"
            f"两层允许{two_h*50:.1f}至{two_h*150:.1f}厘米"
        )
    if one_ok and two_ok:
        use_two = abs(height_m - two_h) <= abs(height_m - one_h)
    else:
        use_two = two_ok
    if not use_two:
        return "1", piece
    faction, unclear = _faction(color, closed, mask, z_cam, intr, diagnostic)
    if unclear:
        return fail("阵营不清：颜色有效面积不足或上下关系不明确")
    return (faction if faction is not None else "2"), piece


def _classify_cell(
    cell: Cell,
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None,
    depth: np.ndarray | None,
    cloud: dict | None,
    labels: np.ndarray | None = None,
    pieces: list | None = None,
    roll: float = 0.0,
    floor: np.ndarray | None = None,
    cap_assignment=None,
) -> CellView:
    in_view, needs_yaw, needs_pitch = _view_flags(cell, pose, offset, pitch, intr, roll)
    # 一层只露出不到 8 cm 时，这一档不能写成空，也不能写成一层。
    one_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.one_z)
    blocked = one_h < MIN_VISIBLE_M
    bearing = _bearing_deg(cell.center(cell.one_z), pose, offset, pitch)
    pitch_target = _pitch_target(cell, pose, offset)
    mask = _mask(cell, pose, offset, pitch, intr, roll)
    region = _outline(mask)
    reason = None
    diagnostic = {"roll_deg": float(roll), "pitch_deg": float(pitch)}
    if cap_assignment is not None:
        diagnostic['_cap_allowed'],diagnostic['_cap_suspected_allowed'],diagnostic['cap_ownership']=cap_assignment
    rgb_cap = None
    rgb_structure = None
    rgb_top_structure = None
    view_distance = float(np.linalg.norm(camera_position(pose,offset,pitch)[:2]-[cell.x,cell.y]))

    def pack(conclusion, frac: float, mean_v: float, piece=None, method=None, hold=False, skip=False) -> CellView:
        # 深度已经量到、彩色又没把整块看全的格，先留在这里记满 15 帧。
        shown_reason = reason
        if (depth is None and conclusion in ('1','2') and piece is not None
                and method in ('彩色','黄地','远处')):
            # 只按像素轮廓高度估层数，会把窗边小白块碰巧判成一层。
            # 还要核对这块轮廓的实际顶边确实落在对应塔高附近。
            xy=np.asarray(piece).reshape(-1,2)
            top_row=float(np.min(xy[:,1]))
            top_x=float(np.median(xy[xy[:,1]<=top_row+3,0]))
            measured=_z_at_cell(cell,top_x,top_row,pose,offset,pitch,intr,roll)
            expected=cell.one_z if conclusion=='1' else cell.two_z
            diagnostic['rgb_contour_top_cm']=None if measured is None else measured*100
            tolerance=max(.18,.5*(expected-cell.base_z))
            if measured is None or abs(measured-expected)>tolerance:
                diagnostic['rgb_contour_rejected']=conclusion
                conclusion,piece,method=None,None,None
                shown_reason='RGB轮廓顶高不符：可能识别到背景白块'
        if rgb_cap is None and method!='深度':
            rgb_options=[]
            if conclusion in ('1','2') and piece is not None:
                rgb_options.append((conclusion,piece))
            rgb_options += [q for q in (rgb_top_structure,rgb_structure) if q is not None]
            if {q[0] for q in rgb_options}=={'1','2'}:
                contenders=[]
                for state,outline in rgb_options:
                    xy=np.asarray(outline).reshape(-1,2)
                    row=float(xy[:,1].min())
                    u=float(np.median(xy[xy[:,1]<=row+3,0]))
                    height=_z_at_cell(cell,u,row,pose,offset,pitch,intr,roll)
                    if height is not None:
                        contenders.append((state,outline,height,False))
                arbitration={}
                resolved=_resolve_layer_seams(contenders,color,white,mask,cell,pose,
                    offset,pitch,intr,z_cam,roll,depth,arbitration)
                if arbitration.get('triggered'):
                    diagnostic['rgb_layer_conflict']=arbitration
                if arbitration.get('removed_indices'):
                    best=max(resolved,key=lambda q:q[2])
                    conclusion,piece=best[:2]
                    method,hold,skip='RGB接缝仲裁',True,False
                    shown_reason=None
        if (rgb_top_structure is not None
                and (rgb_structure is None or (rgb_top_structure[0]=='2' and rgb_structure[0]=='1'))
                and (conclusion is None or (depth is None and conclusion in ('1','2')
                                           and rgb_top_structure[0]!=conclusion))):
            conclusion,piece=rgb_top_structure
            method,hold,skip='RGB局部盒面',True,False
            shown_reason=None
        if conclusion is None and rgb_structure is not None:
            conclusion,piece = rgb_structure
            method,hold,skip = 'RGB盒体边界',True,False
            shown_reason = None
        if (depth is None and conclusion=='1' and piece is not None
                and _white_face_continues_above(piece,color,white,mask,z_cam,intr,roll)):
            diagnostic['rgb_single_seam_rejected']=True
            conclusion,piece,method,hold,skip=None,None,None,False,True
            shown_reason='横线上方仍有连续盒面和两侧边：接缝不能作为一层盒顶'
        if conclusion is None and view_distance<1.1 and needs_pitch:
            diagnostic['needs_standoff'] = True
            shown_reason = '近处竖直视野不足：先调整俯仰，仍无法看全时后退换位；' + (reason or '')
        diagnostic['white_reading'] = dict(
            result=conclusion if conclusion in ('1', '2') else None,
            method=method, reason=reason)
        if conclusion in ('1', '2') and piece is not None:
            diagnostic['white_piece'] = piece
        if rgb_cap is not None:
            # 顶块与白盒都运行检测；白盒失败不能否定独立通过校验的顶块。
            conclusion, piece = rgb_cap
            method, hold, skip = 'RGB顶块', True, False
        elif ((diagnostic.get('rgb_cap', {}).get('state') == 'suspected'
               or diagnostic.get('upper_object'))
              and conclusion not in ('红', '蓝')):
            # 各白盒/深度/空点路径统一仲裁；可信顶块待确认时禁止降级。
            diagnostic['cap_blocks_white'] = True
            conclusion, piece, method, hold, skip = None, None, None, False, True
            shown_reason = ('上层物体待确认：深度证实两层以上仍有物体，禁止回退为白盒层数'
                            if diagnostic.get('upper_object') else
                            '疑似顶块待确认：禁止回退为白盒层数或空点')
        if (conclusion is None and diagnostic.get("needs_parallax")
                and not diagnostic.get('cap_blocks_white')):
            detail = diagnostic.get("far_white", {}).get("reason", "证据不足，需要侧向视角")
            shown_reason = detail + (("；" + reason) if reason else "")
        diagnostic.pop('_cap_allowed',None)
        diagnostic.pop('_cap_suspected_allowed',None)
        return CellView(
            conclusion,
            True if hold else in_view,
            False if hold else needs_yaw,
            False if hold else needs_pitch,
            blocked,
            bearing,
            pitch_target,
            frac,
            mean_v,
            region,
            piece,
            method,
            skip,
            shown_reason if conclusion is None else None,
            diagnostic,
            needs_parallax=conclusion is None and diagnostic.get("needs_parallax", False),
        )

    if color is None:
        reason = "等待有效画面或曝光稳定"
        return pack(None, 0.0, 255.0)
    white = _white_mask(color)
    inside = (white > 0) & (mask > 0)
    area = int(np.count_nonzero(mask))
    frac = 0.0 if area == 0 else float(np.count_nonzero(inside)) / float(area)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mean_v = 255.0 if area == 0 else float(np.mean(hsv[:, :, 2][mask > 0]))
    cam = to_camera(cell.center(cell.one_z), pose, offset, pitch, roll)
    z_cam = float(cam[2])
    color_usable = in_view and z_cam > 0.2
    if z_cam>.2 and np.count_nonzero(mask)>=30:
        rgb_cap = _rgb_cap_reading(cell, color, mask, pose, offset, pitch, intr,
                                   z_cam, roll, diagnostic)
    if z_cam>.2 and not in_view:
        rgb_structure=_rgb_white_edges(cell,color,white,mask,pose,offset,pitch,
                                       intr,z_cam,roll,diagnostic,depth)
    if z_cam>.2:
        rgb_top_structure=_rgb_white_top(cell,color,white,mask,pose,offset,pitch,
                                         intr,z_cam,roll,diagnostic,depth)
    clipped = False
    overwide = False
    missed_shape = False
    missed_width = False
    covered = False
    occludes, holds = (False, False)
    if depth is not None:
        occludes, holds = _tower_clutter(cell, depth, mask, pose, offset, pitch, intr, roll, diagnostic, color)
    near_depth_result=None
    if depth is not None and .2<z_cam<=_DEPTH_FAR_M:
        near_depth_result=_depth_on_cell(cell,cloud,mask,intr,color,blocked,labels,
            pose,offset,pitch,roll,diagnostic=diagnostic,depth=depth)
        state,piece,_,unclear=near_depth_result
        if state is not None and not unclear:
            # 分离后的本塔深度是主判。搜索区底部杂物及RGB轮廓破损不否决它。
            return pack(state,frac,mean_v,piece,"深度",hold=True)
    side_white = bool(np.any((mask[:,0]>0)&(white[:,0]>0))
                      or np.any((mask[:,-1]>0)&(white[:,-1]>0)))
    if side_white and depth is None and rgb_structure is None and z_cam>.2:
        rgb_structure=_rgb_white_edges(cell,color,white,mask,pose,offset,pitch,
                                       intr,z_cam,roll,diagnostic)
    if occludes or blocked or side_white:
        partial=_upper_white_reading(cell,color,white,depth,mask,pose,offset,pitch,intr,roll,diagnostic)
        if partial is not None:
            state,piece=partial
            return pack(state,frac,mean_v,piece,"上部盒面",hold=True)
    if occludes:
        # 箱子或杂物挡在这块塔的画面里。不写成层数，也不写成空，这一帧不计票。
        reason = "前景遮挡：近处杂物盖住建塔区域"
        diagnostic['needs_parallax'] = True
        return pack(None, frac, mean_v, skip=True)
    if z_cam > _DEPTH_FAR_M and color_usable:
        far_cap = None if rgb_cap is not None else _far_cap_reading(
            cell, color, white, depth, mask, pose, offset, pitch, intr, z_cam, roll, diagnostic)
        if far_cap is not None:
            state, piece = far_cap
            return pack(state, frac, mean_v, piece, "远处融合")
        far_white = _far_white_reading(cell,color,white,depth,mask,pose,offset,pitch,
                                       intr,roll,diagnostic)
        if far_white is not None:
            state,piece = far_white
            return pack(state,frac,mean_v,piece,"远处白盒")
        # RGB拆分粘连盒面；有深度时仍必须通过归属及上方空间核验。
        rgb_structure = _rgb_white_edges(cell,color,white,mask,pose,offset,pitch,
                                         intr,z_cam,roll,diagnostic,depth)
        if frac >= .05:
            diagnostic["needs_parallax"] = True
    accepted: list[tuple] = []
    if z_cam > 0.2 and labels is not None:
        for index, x, y, w, h, cx, cy in pieces or []:
            if not _centroid_inside(mask, cx, cy):
                continue
            component = labels == index
            ys, xs = np.nonzero(component)
            sits, own = _depth_place(cell, depth, ys, xs, pose, offset, pitch, intr, z_cam, roll)
            if _depth_in_range(z_cam) and sits == "out":
                # 质心落在塔的框里，深度却不在台心。大小又像一块白盒时，就是挡在前面的箱子。
                width_own = _metric(w, own, intr.fx)
                if WIDTH_MIN_M <= width_own <= WIDTH_MAX_M and _block_shaped(component):
                    covered = True
                continue
            touches_side = x <= 1 or x + w >= intr.width - 1
            touches_top = y <= 1
            touches_bottom = y + h >= intr.height - 1
            component_overwide = _metric(w, z_cam, intr.fx) > WIDTH_MAX_M
            overwide = overwide or component_overwide
            # 背景候选过宽，不连带排除后面的正常白盒。
            if touches_side or touches_top or component_overwide:
                clipped = True
                continue
            if not color_usable:
                continue
            distance = own if sits == "in" else z_cam
            width_m = _metric(w, distance, intr.fx)
            if width_m < WIDTH_MIN_M:
                # 阴影、线缆、标签切出来的段。看见了白，不能写成空。
                missed_width = True
                continue
            if width_m > WIDTH_MAX_M:
                overwide = True
                continue
            if not _block_shaped(component):
                missed_shape = True
                continue
            top_z = None
            if touches_bottom:
                top_rows = ys == int(ys.min())
                top_z = _z_at_cell(
                    cell,
                    float(np.median(xs[top_rows])),
                    float(ys.min()),
                    pose,
                    offset,
                    pitch,
                    intr,
                    roll,
                )
                if top_z is None:
                    clipped = True
                    continue
            accepted.append((x, y, w, h, distance, component, top_z))
    diagnostic['white_candidates'] = [_outline(item[5].astype(np.uint8)*255)
                                       for item in accepted] if rgb_cap is not None else []
    depth_state, depth_piece, saw_outside, cap_unclear = (near_depth_result if near_depth_result is not None else _depth_on_cell(
        cell, cloud, mask, intr, color, blocked, labels, pose, offset, pitch, roll,
        diagnostic=diagnostic, depth=depth,
    ))
    reason = diagnostic.get("reason")
    if cap_unclear:
        return pack(None, frac, mean_v, depth_piece)
    if diagnostic.get("depth_partial_at_boundary"):
        # 不把不完整深度当成反证，但必须有独立通过尺寸/直边检查的彩色白盒。
        if not accepted:
            return pack(None, frac, mean_v, depth_piece)
        diagnostic["depth_fallback_reason"] = reason
    # 白点落在这一格 50 cm 里，最高一截对上一层或两层，就写层数。
    # 整块在画面里、和木板粘在一起收不成直边时，也用这段深度。
    if depth is not None and depth_state is not None:
        return pack(depth_state, frac, mean_v, depth_piece, "深度", hold=not accepted)
    if accepted:
        two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
        reading = _color_reading(cell, accepted, color, mask, intr, blocked, one_h, two_h, diagnostic)
        if reading is None:
            reason = diagnostic.get("reason", "彩色判定未通过")
            return pack(None, frac, mean_v)
        state, piece, _distance = reading
        return pack(state, frac, mean_v, piece, _color_method(mask, floor))
    if (
        z_cam > _DEPTH_FAR_M
        and color_usable
        and (missed_shape or missed_width)
        and not overwide
        and not clipped
    ):
        two_h = visible_height(pose, offset, pitch, cell, cell.base_z, cell.two_z)
        far = _far_reading(color, white, mask, z_cam, intr, blocked, one_h, two_h, diagnostic)
        if far is None:
            reason = diagnostic.get("reason", "远处彩色未通过")
            return pack(None, frac, mean_v)
        state, piece = far
        return pack(state, frac, mean_v, piece, "远处")
    if depth is not None and (clipped or overwide or saw_outside):
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
        return pack(None, frac, mean_v)
    if covered or holds:
        reason = "前面有东西挡住了或格内有物体：不能确认为空点"
        # 框里有一块像塔的箱子，或柱体里已经有东西。没有合格白盒也不能写成空。
        return pack(None, frac, mean_v, skip=True)
    if not _empty_evidence(cell, pose, offset, pitch, intr, depth, roll):
        reason = ((reason + "；") if reason else "") + (
            "空点未证实：支撑面未完整入画、深度不足或高度不符"
        )
        return pack(None, frac, mean_v)
    return pack("0", frac, mean_v, method=_color_method(mask, floor))


def _empty_evidence(cell, pose, offset, pitch, intr, depth, roll=0.0) -> bool:
    """在台面投影内部看到覆盖充分的空支撑面，才确认空点。"""
    if depth is None or depth.shape != (intr.height, intr.width):
        return False
    x0, x1, y0, y1 = cell.search_bounds(0.0)
    verts = [project(np.array([x, y, cell.base_z]), pose, offset, pitch, intr, roll)
             for x, y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
    if any(v is None or not _inside(v, intr) for v in verts):
        return False
    patch = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(patch, np.round(verts).astype(np.int32), 255)
    ys, xs = np.nonzero(patch)
    if len(xs) < 30:
        return False
    values = depth[ys, xs]
    valid = np.isfinite(values) & (values >= _DEPTH_NEAR_M) & (values <= _DEPTH_FAR_M)
    if valid.mean() < 0.80:
        return False
    origin, right, down, ahead = camera_frame(pose, offset, pitch, roll)
    xs, ys, values = xs[valid], ys[valid], values[valid]
    world = (origin + ((xs-intr.cx)*values/intr.fx)[:,None]*right
             + ((ys-intr.cy)*values/intr.fy)[:,None]*down + values[:,None]*ahead)
    return bool(np.mean(np.abs(world[:,2]-cell.base_z) <= 0.03) >= 0.95)



def _filter_edge_depth(color: np.ndarray, depth: np.ndarray | None) -> np.ndarray | None:
    """只剔除颜色边界内一像素带中、被同类邻域多数否定的深度；不填补深度。"""
    if depth is None or depth.shape != color.shape[:2]:
        return depth
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    kinds = np.zeros(depth.shape, np.uint8)
    kinds[_white_mask(color) > 0] = 1
    red = (cv2.inRange(hsv, np.array(RED[0]), np.array(RED[1]))
           | cv2.inRange(hsv, np.array(RED_WRAP[0]), np.array(RED_WRAP[1]))) > 0
    kinds[red] = 2
    kinds[cv2.inRange(hsv, np.array(BLUE[0]), np.array(BLUE[1])) > 0] = 3
    edge = np.zeros(depth.shape, bool)
    for kind in (1, 2, 3):
        part = (kinds == kind).astype(np.uint8)
        inner = cv2.erode(part, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        edge |= (part > 0) & (inner == 0)
    valid = np.isfinite(depth) & (depth >= _DEPTH_NEAR_M)
    ys, xs = np.nonzero(edge & valid)
    if not len(xs):
        return depth
    padded = np.pad(depth.astype(np.float32), 2, constant_values=np.nan)
    labels = np.pad(kinds, 2)
    neighbors = []
    for dy in range(5):
        for dx in range(5):
            if dy == 2 and dx == 2:
                continue
            values = padded[ys + dy, xs + dx]
            same = labels[ys + dy, xs + dx] == kinds[ys, xs]
            neighbors.append(np.where(same & np.isfinite(values)
                                      & (values >= _DEPTH_NEAR_M), values, np.nan))
    samples = np.stack(neighbors, axis=1)
    enough = np.sum(np.isfinite(samples), axis=1) >= 8
    if not enough.any():
        return depth
    samples = samples[enough]
    ys, xs = ys[enough], xs[enough]
    middle = np.nanmedian(samples, axis=1)
    # 至少相差8厘米，远处按距离的3%提高门槛；只拒绝明显跳变。
    tolerance = np.maximum(0.08, middle * 0.03)
    coherent = np.sum(np.abs(samples - middle[:, None]) <= tolerance[:, None], axis=1)
    majority = coherent >= 0.65 * np.sum(np.isfinite(samples), axis=1)
    reject = majority & (np.abs(depth[ys, xs] - middle) > tolerance)
    if not reject.any():
        return depth
    cleaned = depth.astype(np.float32, copy=True)
    cleaned[ys[reject], xs[reject]] = np.nan
    return cleaned


def inspect_frame(
    pose: CarPose,
    offset: np.ndarray,
    pitch: float,
    intr: Intrinsics,
    color: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    roll: float = 0.0,
) -> dict[str, CellView]:
    labels = None
    pieces = None
    floor = None
    if color is not None:
        depth = _filter_edge_depth(color, depth)
        white, floor = _split_white(color)
        labels, pieces = _white_pieces(white)
    cloud = None if color is None else _white_cloud(color, depth, pose, offset, pitch, intr, roll)
    cells=list(tower_cells())
    assignments={} if color is None else _assign_caps(cells,color,pose,offset,pitch,intr,roll,depth)
    return {
        cell.name: _classify_cell(
            cell, pose, offset, pitch, intr, color, depth, cloud, labels, pieces, roll, floor, assignments.get(cell.name)
        )
        for cell in cells
    }
