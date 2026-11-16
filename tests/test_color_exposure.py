"""同一帧中的阴影白盒和亮红蓝顶块不能靠切换曝光二选一。"""
import runpy

import cv2
import numpy as np
import pytest

from classify import BLUE, _white_mask, inspect_frame

SCENE = runpy.run_path('tests/test_far_merged_wall.py')['merged_scene']


def bgr(h, s, v):
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


@pytest.mark.parametrize('red,blue,white', [
    ((0, 183, 143), (107, 177, 91), 110),
    ((177, 144, 254), (108, 130, 196), 220),
    ((0, 183, 143), (114, 95, 139), 220),
])
def test_shaded_body_and_bright_cap_are_detected_in_same_frame(red, blue, white):
    cell, pose, offset, intr, color, depth = SCENE(bgr(*blue), bgr(*red))
    neutral = np.max(color, axis=2) == np.min(color, axis=2)
    body = neutral & (depth < 6.)
    color[neutral] = 35
    color[body] = white
    view = inspect_frame(pose, offset, 0., intr, color, depth)[cell.name]
    assert view.conclusion == '蓝'
    assert view.method == 'RGB顶块'
    assert np.mean(_white_mask(color)[body] > 0) > .95
    assert (view.measurements.get('white_piece') is not None
            or len(view.measurements.get('white_candidates', [])) > 0)


def test_saturated_floor_is_not_added_to_white_by_shadow_threshold():
    pixels = np.uint8([[bgr(18, 80, 110), bgr(0, 0, 110), bgr(0, 0, 80)]])
    assert _white_mask(pixels).tolist() == [[0, 255, 0]]


def test_expanded_blue_still_rejects_gray_and_dark_pixels():
    pixels=np.uint8([[bgr(114,40,139), bgr(114,100,30), bgr(114,95,139)]])
    hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    assert cv2.inRange(hsv,np.array(BLUE[0]),np.array(BLUE[1])).tolist()==[[0,0,255]]


@pytest.mark.parametrize('hue',[100,101,104,131,133,135])
def test_cyan_and_purple_backgrounds_are_not_in_blue_mask(hue):
    pixel=np.uint8([[bgr(hue,160,160)]])
    hsv=cv2.cvtColor(pixel,cv2.COLOR_BGR2HSV)
    assert not cv2.inRange(hsv,np.array(BLUE[0]),np.array(BLUE[1])).any()
