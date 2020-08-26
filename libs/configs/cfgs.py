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

DATASET_DIR = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_voc'

ORG_WEIGHTS = ROOT_PATH + '/data/pretrained_weights/yolov3_coco.ckpt'
PRETRAINED_WEIGHTS = ROOT_PATH + '/data/pretrained_weights/yolov3_coco_demo.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH,  'outputs', 'trained_weights')
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'
SUMMARY_PATH = os.path.join(ROOT_PATH, 'outputs', 'summary')

INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_IMAGE_PATH = ROOT_PATH + '/outputs/test_results/detect_images'
TEST_SAVE_MAP_PATH = ROOT_PATH + '/outputs/test_results/detect_images/mAP'
#----------------------data config------------------------------------

NUM_READER = 4  # The number of parallel readers that read data from the dataset.
NUM_THREADS = 4

#---------------------- network config-----------------------------
# IMAGE_SHPAE = [416, 416]
NUM_CLASSES         = 20
CLASSES             = ROOT_PATH + "/data/classes/voc.names"
ANCHORS                = ROOT_PATH + "/data/anchors/baseline_anchors.txt"
MOVING_AVE_DECAY       = 0.9995
STRIDES                = [8, 16, 32]
ANCHOR_PER_SCALE       = 3
IOU_LOSS_THRESH        = 0.5
UPSAMPLE_METHOD        = "resize"


# -------------------------Train options-------------------------------
TRAIN_RECORD_DIR       = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd/train'
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
TEST_RECORD_DIR             = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd/test'
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
    'aeroplane': (0, 'Vehicle'),
    'bicycle': (1, 'Vehicle'),
    'bird': (2, 'Animal'),
    'boat': (3, 'Vehicle'),
    'bottle': (4, 'Indoor'),
    'bus': (5, 'Vehicle'),
    'car': (6, 'Vehicle'),
    'cat': (7, 'Animal'),
    'chair': (8, 'Indoor'),
    'cow': (9, 'Animal'),
    'diningtable': (10, 'Indoor'),
    'dog': (11, 'Animal'),
    'horse': (12, 'Animal'),
    'motorbike': (13, 'Vehicle'),
    'person': (14, 'Person'),
    'pottedplant': (15, 'Indoor'),
    'sheep': (16, 'Animal'),
    'sofa': (17, 'Indoor'),
    'train': (18, 'Vehicle'),
    'tvmonitor': (18, 'Indoor'),
}
