#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/16 下午5:24
# @ Software   : PyCharm
#-------------------------------------------------------
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
from enum import Enum

# ------------------------------------------------
# VERSION = 'FPN_Res101_20181201'
VERSION = 'YOLOv3_2020816'
MODEL_NAME = 'yolo_v3'

ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"

SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

#----------------------data config------------------------------------
DATASET_DIR = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_voc'

ORG_WEIGHTS = ROOT_PATH + '/data/pretrained_weights/yolov3_coco.ckpt'
PRETRAINED_WEIGHTS = ROOT_PATH + '/data/pretrained_weights/yolov3_coco_demo.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH,  'outputs', 'trained_weights', VERSION)
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'
SUMMARY_PATH = os.path.join(ROOT_PATH, 'outputs', 'summary', VERSION)

INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'


#---------------------- network config-----------------------------
# IMAGE_SHPAE = [416, 416]

CLASS_NAME             = ROOT_PATH + "/data/classes/coco.names"
ANCHORS                = ROOT_PATH + "/data/anchors/baseline_anchors.txt"
MOVING_AVE_DECAY       = 0.9995
STRIDES                = [8, 16, 32]
ANCHOR_PER_SCALE       = 3
IOU_LOSS_THRESH        = 0.5
UPSAMPLE_METHOD        = "resize"


# -------------------------Train options-------------------------------
TRAIN_ANNOT_PATH       = ROOT_PATH + "/data/annotation/voc_train.txt"
TRAIN_BATCH_SIZE       = 2
TRAIN_INPUT_SIZE       = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TRAIN_DATA_AUG         = True
LEARNING_RATE_INIT     = 1e-4
LEARNING_RATE_END      = 1e-6
WARMUP_EPOCHS           = 2
FIRST_STAGE_EPOCHS      = 20
SECOND_STAGE_EPOCHS     = 30

# --------------------------Test options---------------------------------

TEST_ANNOT_PATH             = ROOT_PATH + "/data/annotation/voc_test.txt"
TEST_BATCH_SIZE             = 2
TEST_INPUT_SIZE             = 544
TEST_DATA_AUG               = False
TEST_WRITE_IMAGE            = True
TEST_WRITE_IMAGE_SHOW_LABEL = True
TEST_SHOW_LABEL             = True
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45



#-----------------------misc config------------------------------
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
