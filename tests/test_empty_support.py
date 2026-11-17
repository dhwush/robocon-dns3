"""空点必须有均匀底面证据，且不能漏掉连续小物体。"""
import runpy
import cv2
import numpy as np
from car_pose import camera_frame, project
from classify import _empty_evidence

M = runpy.run_path('tests/test_review_regressions.py')


def support_scene():
    cell, pose, _, depth = M['scene']()
    intr, offset, pitch = M['INTR'], M['OFFSET'], -18.
    origin, right, down, ahead = camera_frame(pose, offset, pitch)
    ys, xs = np.indices(depth.shape)
    rays = (((xs-intr.cx)/intr.fx)[...,None]*right
            + ((ys-intr.cy)/intr.fy)[...,None]*down + ahead)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = ((cell.base_z-origin[2])/rays[:,:,2]).astype(np.float32)
    x0,x1,y0,y1 = cell.search_bounds(0.)
    vertices = [project(np.array([x,y,cell.base_z]),pose,offset,pitch,intr)
                for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
    patch = np.zeros(depth.shape,np.uint8)
    cv2.fillConvexPoly(patch,np.round(vertices).astype(np.int32),255)
    return cell,pose,offset,pitch,intr,depth,patch,rays,origin


def test_empty_ignores_invalid_one_pixel_boundary():
    cell,pose,offset,pitch,intr,depth,patch,_,_ = support_scene()
    inner = cv2.erode(patch,np.ones((3,3),np.uint8))
    depth[(patch>0)&(inner==0)] = np.nan
    assert _empty_evidence(cell,pose,offset,pitch,intr,depth)


def test_empty_rejects_concentrated_missing_depth():
    cell,pose,offset,pitch,intr,depth,patch,rays,origin = support_scene()
    floor = origin + rays*depth[...,None]
    x0,x1,y0,y1 = cell.search_bounds(0.)
    region = (patch>0)&(floor[:,:,0]>(x0+x1)/2)&(floor[:,:,1]>(y0+y1)/2)
    yy,xx = np.nonzero(region)
    # 丢失一个分区约一半的深度，总体仍超过80%。
    depth[yy[::2],xx[::2]] = np.nan
    assert not _empty_evidence(cell,pose,offset,pitch,intr,depth)


def test_empty_rejects_small_connected_raised_object():
    cell,pose,offset,pitch,intr,depth,patch,rays,origin = support_scene()
    yy,xx = np.nonzero(patch)
    cy,cx = int(np.median(yy)),int(np.median(xx))
    # 小块物体所占面积低于总体5%，仍应阻止判空。
    region = np.zeros(depth.shape,bool)
    region[cy-2:cy+3,cx-20:cx+21] = True
    depth[region] = ((cell.base_z+.08-origin[2])/rays[:,:,2][region])
    assert not _empty_evidence(cell,pose,offset,pitch,intr,depth)


def test_empty_tolerates_isolated_depth_spike():
    cell,pose,offset,pitch,intr,depth,patch,rays,origin = support_scene()
    yy,xx = np.nonzero(patch)
    cy,cx = int(np.median(yy)),int(np.median(xx))
    depth[cy,cx] = (cell.base_z+.08-origin[2])/rays[cy,cx,2]
    assert _empty_evidence(cell,pose,offset,pitch,intr,depth)
