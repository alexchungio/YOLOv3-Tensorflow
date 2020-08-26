#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : read_tfrecord.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/17 下午4:40
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import random
import numpy as np
import glob
import cv2 as cv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.python_io import tf_record_iterator

from utils import tools
from data.preprocessing.yolo_preprocessing import image_resize_padding

from libs.configs import cfgs

# origin_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_split/val'
# tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd/train'


class Dataset(object):

    def __init__(self, is_training=False):

        self.is_training = is_training
        self.record_dir = cfgs.TRAIN_RECORD_DIR if is_training else cfgs.TEST_RECORD_DIR

        self.input_sizes = cfgs.TRAIN_INPUT_SIZE if self.is_training else cfgs.TRAIN_INPUT_SIZE
        self.batch_size = cfgs.TRAIN_BATCH_SIZE if self.is_training else cfgs.TEST_BATCH_SIZE

        self.train_input_size = 416
        self.strides = np.array(cfgs.STRIDES)
        self.num_classes = cfgs.NUM_CLASSES
        self.anchors = np.array(tools.get_anchors(cfgs.ANCHORS))
        self.anchor_per_scale = cfgs.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150
        self.num_samples = self.get_num_samples()
        # self.num_steps_per_epoches = int(np.ceil( self.get_num_samples()/ self.batch_size))
        # self.batch_count = 0

    def read_parse_single_example(self, serialized_sample, is_training=False):
        """
        parse tensor
        :param image_sample:
        :return:
        """
        # construct feature description
        keys_to_features = {
            'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/num_object': tf.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)
        }
        features = tf.io.parse_single_example(serialized=serialized_sample, features=keys_to_features)

        # parse feature
        image_name = tf.cast(features['image/filename'], dtype=tf.string)
        num_objects = tf.cast(features['image/object/num_object'], dtype=tf.int32)

        height = tf.cast(features['image/height'], dtype=tf.int32)
        width = tf.cast(features['image/width'], dtype=tf.int32)
        depth = tf.cast(features['image/channels'], dtype=tf.int32)

        # shape = tf.cast(feature['shape'], tf.int32)

        # actual data shape
        image_shape = [height, width, depth]
        bbox_shape = [num_objects, 1]

        image = tf.decode_raw(features['image/encoded'], out_type=tf.uint8)
        image = tf.reshape(image, image_shape)

        # parse gtbox
        x_min = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'], default_value=0)
        y_min = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'], default_value=0)
        x_max = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'], default_value=0)
        y_max = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'], default_value=0)
        label = tf.sparse_tensor_to_dense(features['image/object/bbox/label'],default_value=0)

        x_min = tf.reshape(x_min, bbox_shape)
        y_min = tf.reshape(y_min, bbox_shape)
        x_max = tf.reshape(x_max, bbox_shape)
        y_max = tf.reshape(y_max, bbox_shape)
        label = tf.reshape(label, bbox_shape)

        # bboxes = tf.concat([x_min[:, tf.newaxis], y_min[:, tf.newaxis], x_max[:, tf.newaxis], y_max[:, tf.newaxis], tf.cast(label[:, tf.newaxis], dtype=tf.float32)], axis=-1)
        bboxes = tf.concat([x_min, y_min, x_max, y_max, tf.cast(label, dtype=tf.float32)], axis=-1)
        bboxes = tf.reshape(bboxes, shape=[-1, 5])

        self.train_output_sizes = self.train_input_size // self.strides

        image, bboxes = tf.numpy_function(self.image_processing, inp=[image, bboxes, is_training], Tout=[tf.float32, tf.float32])
        image = tf.reshape(image, shape=(self.train_input_size, self.train_input_size, 3))
        bboxes = tf.reshape(bboxes, shape=(-1, 5))
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = tf.numpy_function(self.preprocess_true_boxes,
                                                                                             inp=[bboxes],
                                                                                             Tout=[tf.float32,
                                                                                                   tf.float32,
                                                                                                   tf.float32,
                                                                                                   tf.float32,
                                                                                                   tf.float32,
                                                                                                   tf.float32])

        label_sbbox = tf.reshape(label_sbbox, shape=(self.train_output_sizes[0], self.train_output_sizes[0],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        label_mbbox = tf.reshape(label_mbbox, shape=(self.train_output_sizes[1], self.train_output_sizes[1],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        label_lbbox = tf.reshape(label_lbbox, shape=(self.train_output_sizes[2], self.train_output_sizes[2],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        sbboxes = tf.reshape(sbboxes, shape=(self.max_bbox_per_scale, 4))
        mbboxes = tf.reshape(mbboxes, shape=(self.max_bbox_per_scale, 4))
        lbboxes = tf.reshape(lbboxes, shape=(self.max_bbox_per_scale, 4))


        return image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def dataset_tfrecord(self, batch_size=2, epoch=None, shuffle=False):
        """
        construct iterator to read image
        :param record_file:
        :return:
        """
        # comprise record file
        record_list = [os.path.join(self.record_dir, filename) for filename in  os.listdir(self.record_dir)]
        record_dataset = tf.data.TFRecordDataset(record_list)
        # check record file format
        # execute parse function to get dataset
        # This transformation applies map_func to each element of this dataset,
        # and returns a new dataset containing the transformed elements, in the
        # same order as they appeared in the input.
        # when parse_example has only one parameter (office recommend)
        # parse_img_dataset = raw_img_dataset.map(parse_example)
        # when parse_example has more than one parameter which used to process data
        parse_img_dataset = record_dataset.map(lambda series_record:
                                               self.read_parse_single_example(serialized_sample=series_record,
                                                                              is_training=self.is_training))
        # get dataset batch
        if shuffle:
            shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size * 4).repeat(epoch).batch(
                batch_size=batch_size)
        else:
            shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
        # make dataset iterator
        iterator = shuffle_batch_dataset.make_one_shot_iterator()
        # get element
        image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = iterator.get_next()


        return image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes


    def image_processing(self, image, bboxes, is_training=False):
        if is_training:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = image_resize_padding(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes), is_rgb=True)
        return np.ndarray.astype(image, np.float32), np.ndarray.astype(bboxes, np.float32)

    def bbox_iou(self, boxes1, boxes2):
        """
        compute bboxes iou
        :param boxes1:
        :param boxes2:
        :return:
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        # small_label, middle_label, larger_label
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # small_anchor, middle_scale, large_scale
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])


            # one_hot_label
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0

            # label_smooth
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            delta = 0.01
            smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

            # (x_min, y_min, x_max, y_max) -> (x, y, w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # get bbox of three feature(small, middle, large)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False

            # iterate small middle and large
            for i in range(3):
                # get anchor bbox of three scale
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                # computer one bbox iou at three scale
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    # save bbox box, conf and classify witch iou value over threshold
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    # save bbox box
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return np.ndarray.astype(label_sbbox, np.float32), np.ndarray.astype(label_mbbox, np.float32), \
               np.ndarray.astype(label_lbbox, np.float32), np.ndarray.astype(sbboxes, np.float32),\
               np.ndarray.astype(mbboxes, np.float32), np.ndarray.astype(lbboxes, np.float32)

    def get_num_samples(self):
        """
        get tfrecord numbers
        :param record_file:
        :return:
        """
        # check record file format
        # record_list = glob.glob(os.path.join(self.record_dir, '*.record'))
        file_pattern = os.path.join(self.record_dir, '*.record')
        input_files = tf.io.gfile.glob(file_pattern)
        num_samples = 0
        print("counting number of sample, please waiting...")
        # convert to dynamic mode
        tf.enable_eager_execution()
        for _ in tf.data.TFRecordDataset(input_files):
            num_samples += 1
        # recover to static mode
        tf.disable_eager_execution()
        return num_samples


if __name__ == "__main__":

    dataset = Dataset(is_training=True)

    # gtboxes_and_label_tensor = tf.reshape(gtboxes_and_label_batch, [-1, 5])
    image_batch, label_sbbox_batch, label_mbbox_batch, label_lbbox_batch, sbboxes_batch, mbboxes_batch, lbboxes_batch = \
        dataset.dataset_tfrecord(batch_size=2, shuffle=True)


    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()

        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        try:
            if not coord.should_stop():
                for _ in range(10):
                    image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = sess.run([image_batch, label_sbbox_batch, label_mbbox_batch, label_lbbox_batch,
                                                                             sbboxes_batch, mbboxes_batch, lbboxes_batch])
                    #
                    print(image.shape, label_sbbox.shape, label_mbbox.shape, label_lbbox.shape)

                    plt.imshow(image[0])
                    plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()
        # waiting all threads safely exit
        sess.close()


# [207.      208.41602 118.97601 385.216