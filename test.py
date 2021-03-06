#!/usr/bin/python3
#
# GpuWrapper example code
#
# OpenCV: Understanding warpPerspective / perspective transform
# https://stackoverflow.com/questions/45717277

import time

import cv2
import numpy as np

from cv2cuda import gpuwrapper


def test_warpperspective():
    img = cv2.imread('test_transform.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # source points
    top_left = [521, 103]
    top_right = [549, 131]
    bottom_right = [222, 458]
    bottom_left = [194, 430]
    pts = np.array([bottom_left, bottom_right, top_right, top_left])

    # target points
    y_off = 50;  # y offset
    top_left_dst = [top_left[0], top_left[1] - y_off]
    top_right_dst = [top_left_dst[0] + 39.6, top_left_dst[1]]
    bottom_right_dst = [top_right_dst[0], top_right_dst[1] + 462.4]
    bottom_left_dst = [top_left_dst[0], bottom_right_dst[1]]
    dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

    # generate a preview to show where the warped bar would end up
    preview = np.copy(img)
    cv2.polylines(preview, np.int32([dst_pts]), True, (0,0,255), 5)
    cv2.polylines(preview, np.int32([pts]), True, (255,0,255), 1)
    cv2.imwrite('preview.jpg', preview)

    # calculate transformation matrix
    pts = np.float32(pts.tolist())
    dst_pts = np.float32(dst_pts.tolist())
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # wrap image and draw the resulting image
    image_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, dsize = image_size, flags = cv2.INTER_LINEAR)
    cv2.imwrite('warped.jpg', warped)

    h, w = img.shape
    warped = gpuwrapper.cudaWarpPerspectiveWrapper(
                 img, M.astype(np.float32), (w, h), cv2.INTER_LINEAR)
    cv2.imwrite('warped_gpu.jpg', warped)


def test_resize():
    img = cv2.imread('test_transform.jpg')
    img = gpuwrapper.cudaResizeWrapper(img, (30, 30))
    cv2.imwrite('resize.jpg', img)


def test_resize_performance():
    imgs = [np.random.randint(255, size=(s, s, 3), dtype=np.uint8)
            for s in (1000, 2000, 3000, 4000)]

    for i, img in enumerate(imgs):
        t_start = time.time()
        img_r = cv2.resize(img.astype(np.float32), (500, 500))
        t_end = time.time()
        print('CPU resize time #{0}, {1}: {2} ms'.format(
            i, img.shape, (t_end - t_start) * 1000))

    print('GPU round 1')
    for i, img in enumerate(imgs):
        t_start = time.time()
        img_r = gpuwrapper.cudaResizeWrapper(img, (500, 500))
        t_end = time.time()
        print('GPU resize time #{0}, {1}: {2} ms'.format(
            i, img.shape, (t_end - t_start) * 1000))

    print('GPU round 2')
    for i, img in enumerate(imgs):
        t_start = time.time()
        img_r = gpuwrapper.cudaResizeWrapper(img, (500, 500))
        t_end = time.time()
        print('GPU resize time #{0}, {1}: {2} ms'.format(
            i, img.shape, (t_end - t_start) * 1000))


test_warpperspective()
test_resize()
test_resize_performance()
