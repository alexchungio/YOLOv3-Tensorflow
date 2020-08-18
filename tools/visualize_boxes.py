#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : visualize_boxes.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/18 上午9:31
# @ Software   : PyCharm
#-------------------------------------------------------

import cv2 as cv
import numpy as np
from PIL import Image


if __name__ == "__main__":
    ID = 0
    label_txt = "../data/annotation/voc_test.txt"
    image_info = open(label_txt).readlines()[ID].split()

    image_path = image_info[1]
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    for bbox in image_info[4:]:
        bbox = bbox.split(",")
        image = cv.rectangle(image,(int(float(bbox[0])),
                                     int(float(bbox[1]))),
                                    (int(float(bbox[2])),
                                     int(float(bbox[3]))), (255,0,0), 2)

    image = Image.fromarray(np.uint8(image))
    image.show()
