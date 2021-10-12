
import os
import numpy as np
import cv2


def get_bounding_box(mat):
    assert len(mat.shape) == 2
    h, w = mat.shape
    row = (np.sum(mat, axis=0) > 0).astype(np.uint8)
    x0 = np.argmax(row)
    x1 = w - np.argmax(row[::-1])
    col = (np.sum(mat, axis=1) > 0).astype(np.uint8)
    y0 = np.argmax(col)
    y1 = h - np.argmax(col[::-1])
    return x0, y0, x1, y1

def crop_by_bounding_box(img, mat, x0, y0, x1, y1, ratio:float=0.1):
    h, w = mat.shape
    hh = y1 - y0 + 1
    ww = x1 - x0 + 1
    # while True:
    yy0 = int(y0 - hh * ratio)
    yy1 = int(y1 + hh * ratio)
    xx0 = int(x0 - ww * ratio)
    xx1 = int(x1 + ww * ratio)
    yy0 = max(0, yy0)
    yy1 = min(h, yy1)
    xx0 = max(0, xx0)
    xx1 = min(w, xx1)
    new_img = img[yy0:yy1, xx0:xx1, ...]
    new_mat = mat[yy0:yy1, xx0:xx1, ...]
    return new_img, new_mat

def augment_interest_object(img, mat):
    x0, y0, x1, y1 = get_bounding_box(mat)
    new_img, new_mat = crop_by_bounding_box(img, mat, x0, y0, x1, y1)
    print('aug:', new_mat.shape)
    return new_img, new_mat