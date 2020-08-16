#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : voc_annotation.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/16 下午5:19
# @ Software   : PyCharm
#-------------------------------------------------------
import os
from utils.tools import makedir
import argparse
import xml.etree.ElementTree as ET
import libs.configs.cfgs as cfgs

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=cfgs.DATASET_DIR)
    parser.add_argument("--train_annotation", default=os.path.join(cfgs.ANNOTATION_DIR, "voc_train.txt"))
    parser.add_argument("--test_annotation",  default=os.path.join(cfgs.ANNOTATION_DIR, "voc_test.txt"))
    flags = parser.parse_args()

    makedir(os.path.dirname(flags.train_annotation))
    makedir(os.path.dirname(flags.train_annotation))

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    train_path_1 = os.path.join(flags.data_path, 'train', 'VOC2007')
    train_path_2 = os.path.join(flags.data_path, 'train', 'VOC2012')
    test_path_1 = os.path.join(flags.data_path, 'test', 'VOC2007')



    num1 = convert_voc_annotation(train_path_1, 'trainval', flags.train_annotation, False)
    num2 = convert_voc_annotation(train_path_2, 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(test_path_1,  'test', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 + num2, num3))