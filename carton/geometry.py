import cv2
import numpy as np


def poly_mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
    return m


def mask_outside(img, poly, fill=114):
    """Grey out everything outside the carton polygon, so the item detector never sees it."""
    m = poly_mask(poly, img.shape)
    out = img.copy()
    out[m == 0] = fill
    return out


def box_center_in(poly, box):
    c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), c, False) >= 0
