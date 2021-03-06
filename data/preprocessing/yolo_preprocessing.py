#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : yolo_preprocessing.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/23 下午1:26
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import tensorflow as tf
import numpy as np


def image_resize_padding(image, target_size, gt_boxes=None, is_rgb=False):

    if is_rgb is False:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # convert pixel scale to (0, 1)
    image_paded = image_paded / 255.
    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def image_resize_padding_tensor(image, target_size, gt_boxes=None):

    image = tf.cast(image, dtype=tf.float32)
    ih, iw = target_size
    h, w, _ = image.shape
    h, w = int(h), int(w)

    scale = min(iw / int(w), ih / h)
    nw, nh = int(scale * w), int(scale * h)

    image_resized = tf.image.resize(image, (nw, nh))

    dw = iw - nw
    dh = ih - nh
    dw_left = int(np.floor(dw / 2))
    dw_right = dw - dw_left
    dh_up = int(np.floor(dh / 2))
    dh_down = dh - dh_up
    image_padded = tf.pad(image_resized, paddings=[[dw_left, dw_right], [dh_up, dh_down], [0, 0]], constant_values=128.0)

    image_paded = tf.cast(image_padded / 255., dtype=tf.float32)

    if gt_boxes is None:
        return image_paded
    else:
        x_min, y_min, x_max, y_max, labels = tf.unstack(gt_boxes, axis=1)
        x_min, x_max = x_min * scale + dw_left, x_max * scale + dw_left
        y_min, y_max = y_min * scale + dw_left, y_max * scale + dw_left
        gt_boxes = tf.transpose(tf.stack([x_min, y_min, x_max, y_max, labels], axis=0))
        return image_paded, gt_boxes


if __name__ == "__main__":
    image_tensor = tf.constant(shape=(576, 448, 3), value=128.0, dtype=tf.float32)
    gt_boxes = tf.constant([[112.0, 158.0, 276.0, 360.0, 3], [180.0, 330, 398, 412, 6]], dtype=tf.float32)
    target_image, target_gt_boxes = image_resize_padding_tensor(image_tensor, target_size=(416, 416), gt_boxes=gt_boxes)
    print(target_image)
    print(target_gt_boxes)


