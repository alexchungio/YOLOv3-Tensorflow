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
tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd/train'


class Dataset(object):

    def __init__(self, record_dir, is_training=False):

        self.record_dir = record_dir
        self.is_training = is_training

        self.input_sizes = cfgs.TRAIN_INPUT_SIZE if self.is_training else cfgs.TRAIN_INPUT_SIZE
        self.batch_size = cfgs.TRAIN_BATCH_SIZE if self.is_training else cfgs.TEST_BATCH_SIZE

        self.train_input_sizes = cfgs.TRAIN_INPUT_SIZE
        self.strides = np.array(cfgs.STRIDES)
        self.num_classes = cfgs.NUM_CLASSES
        self.anchors = np.array(tools.get_anchors(cfgs.ANCHORS))
        self.anchor_per_scale = cfgs.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150
        self.num_samples = self.get_num_samples()
        self.num_steps = int(np.ceil( self.get_num_samples()/ self.batch_size))
        self.batch_count = 0


    def dataset_tfrecord(self, batch_size=2, num_epochs=None, shuffle=True, num_threads=4, is_training=False):
        """
        parse tensor
        :param image_sample:
        :return:
        """
        # construct feature description
        # Features in Pascal VOC TFRecords.
        keys_to_features = {
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/num_object': tf.FixedLenFeature([1], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'filename': slim.tfexample_decoder.Tensor('image/filename'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/num_object': slim.tfexample_decoder.Tensor('image/object/num_object'),
            'object/bboxes': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax',], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
            'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        labels_to_names = {}
        for name, pair in cfgs.VOC_LABELS.items():
            labels_to_names[pair[0]] = name

        dataset = slim.dataset.Dataset(
            data_sources=os.path.join(self.record_dir, '*'),
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=self.num_samples,
            items_to_descriptions=None,
            num_classes=self.num_classes,
            labels_to_names=labels_to_names)

        with tf.name_scope('dataset_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=cfgs.NUM_READER,
                common_queue_capacity=32 * batch_size,
                common_queue_min=8 * batch_size,
                shuffle=shuffle,
                num_epochs=num_epochs)

        self.train_input_size = 416
        self.train_output_sizes = self.train_input_size // self.strides

        [image, filename, shape, bboxes, labels] = provider.get(['image', 'filename', 'shape', 'object/bboxes', 'object/label'])

        bboxes = tf.concat([bboxes, tf.cast(labels[:, tf.newaxis], tf.float32)], axis=-1)
        bboxes = tf.reshape(bboxes, (-1, 5))
        image, bboxes = tf.py_func(self.image_processing, inp=[image, bboxes], Tout=[tf.float32, tf.float32])

        image = tf.reshape(image, shape=(self.train_input_size, self.train_input_size, 3))
        bboxes = tf.reshape(bboxes, shape=(-1, 5))
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = tf.py_func(self.preprocess_true_boxes,
                                                                                      inp=[bboxes],
                                                                                      Tout=[tf.float32, tf.float32,
                                                                                            tf.float32, tf.float32,
                                                                                            tf.float32, tf.float32])

        label_sbbox = tf.reshape(label_sbbox, shape=(self.train_output_sizes[0], self.train_output_sizes[0],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        label_mbbox = tf.reshape(label_mbbox, shape=(self.train_output_sizes[1], self.train_output_sizes[1],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        label_lbbox = tf.reshape(label_lbbox, shape=(self.train_output_sizes[2], self.train_output_sizes[2],
                                                     self.anchor_per_scale, 5 + self.num_classes))
        sbboxes = tf.reshape(sbboxes, shape=(self.max_bbox_per_scale, 4))
        mbboxes = tf.reshape(mbboxes, shape=(self.max_bbox_per_scale, 4))
        lbboxes = tf.reshape(lbboxes, shape=(self.max_bbox_per_scale, 4))


        return  tf.train.batch([image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes],
                                 dynamic_pad=False,
                                 batch_size=batch_size,
                                 allow_smaller_final_batch=(not is_training),
                                 num_threads=num_threads,
                                 capacity=5 * batch_size)


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


    def image_processing(self, image, bboxes):
        if self.is_training:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = image_resize_padding(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
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
        record_list = glob.glob(os.path.join(self.record_dir, '*.record'))

        num_samples = 0
        for record_file in record_list:
            for record in tf_record_iterator(record_file):
                num_samples += 1
        return num_samples


if __name__ == "__main__":


    dataset = Dataset(tfrecord_dir, is_training=False)


    # gtboxes_and_label_tensor = tf.reshape(gtboxes_and_label_batch, [-1, 5])
    image_batch, label_sbbox_batch, label_mbbox_batch, label_lbbox_batch, sbboxes_batch, mbboxes_batch, lbboxes_batch = \
        dataset.dataset_tfrecord(shuffle=False)
    # gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=image_batch,
    #                                                                boxes=gtboxes_and_label_tensor[:, :-1],
    #                                                                labels=gtboxes_and_label_tensor[:, -1])
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
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            if not coord.should_stop():
                image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = sess.run([image_batch, label_sbbox_batch, label_mbbox_batch, label_lbbox_batch,
                                                                         sbboxes_batch, mbboxes_batch, lbboxes_batch])

                plt.imshow(image[0])
                # print(filename[0])
                print(sbboxes[0])
                print(mbboxes[0])
                print(lbboxes[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()
        # waiting all threads safely exit
        coord.join(threads)
        sess.close()