import numpy as np
import cv2

def get_bbox_from_mask_3D(mask):
    # Axes in the order x,y,z
    xmin, xmax = get_xmin_xmax(mask)
    ymin, ymax = get_ymin_ymax(mask)
    zmin, zmax = get_zmin_zmax(mask)
    return xmin, xmax, ymin, ymax, zmin, zmax


def get_ymin_ymax(mask):
    c = np.any(mask, axis=(0, 2))
    ymin, ymax = np.where(c)[0][[0, -1]]
    return ymin,ymax


def get_xmin_xmax(mask):
    r = np.any(mask, axis=(1, 2))
    xmin, xmax = np.where(r)[0][[0, -1]]
    return xmin,xmax


def get_zmin_zmax(mask):
    z = np.any(mask, axis=(0, 1))
    zmin, zmax = np.where(z)[0][[0, -1]]
    return zmin, zmax


def getCentroid(cnt):
    M = cv2.moments(cnt)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    return (cX, cY)


def getEuclidanDistance(x,y):
    if isinstance(x, tuple):
        x = np.array([xi for xi in x])
    if isinstance(y, tuple):
        y = np.array([yi for yi in y])
    return np.linalg.norm(x-y)