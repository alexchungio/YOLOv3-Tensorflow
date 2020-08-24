#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/23 下午5:15
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import cv2 as cv
import numpy as np
import time
import tensorflow as tf
from PIL import Image

from libs.configs import cfgs
from libs.nets.yolo_v3 import YOLOV3
from libs.boxes import box_utils
from libs.boxes import draw_box_in_image
from utils.tools import read_class_names
from data.preprocessing import yolo_preprocessing
from utils.tools import makedir, view_bar


class ObjectInference():
    def __init__(self, input_size=(416, 416), ckpt_path=None, score_threshold=0.3, num_threshold=0.45):
        self.input_size    = input_size
        self.ckpt_path    = ckpt_path
        self.score_threshold = score_threshold
        self.num_threshold = num_threshold
        self.class_name   = read_class_names(cfgs.CLASS_NAME)
        self.num_classes  = len(self.class_name)
        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
        self.trainable = tf.placeholder(dtype=tf.bool, name="training")
        self.detector   = YOLOV3(self.input_data, self.trainable)

    def exucute_detect(self, image_path, save_path):
        """
        execute object detect
        :param detect_net:
        :param image_path:
        :return:
        """
        # load detect network
        pred_sbbox_batch, pred_mbbox_batch, pred_lbbox_batch = self.detector.pred_sbbox, self.detector.pred_mbbox, self.detector.pred_lbbox
        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            # restore pretrain weight
            if self.ckpt_path is not None:
                restorer = tf.train.Saver()
                restorer.restore(sess, self.ckpt_path)
            else:
                restorer, ckpt_path = self.detector.get_restorer(is_training=False)
                restorer.restore(sess, ckpt_path)
            print('*'*80 +'\nSuccessful restore model from {0}\n'.format(self.ckpt_path) + '*'*80)

            # construct image path list
            format_list = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
            if os.path.isfile(image_path):
                image_name_list = [image_path]
            else:
                image_name_list = [img_name for img_name in os.listdir(image_path)
                              if img_name.endswith(format_list) and os.path.isfile(os.path.join(image_path, img_name))]

            assert len(image_name_list) != 0
            print("test_dir has no imgs there. Note that, we only support img format of {0}".format(format_list))
            #+++++++++++++++++++++++++++++++++++++start detect+++++++++++++++++++++++++++++++++++++++++++++++++++++=++
            makedir(save_path)
            fw = open(os.path.join(save_path, 'detect_bbox.txt'), 'w')

            for index, img_name in enumerate(image_name_list):

                detect_dict = {}

                original_image, image_batch, original_size = self.image_process(img_path=os.path.join(image_path, img_name))

                start_time = time.perf_counter()
                # image resize and white process
                # construct feed_dict
                # Run SSD network.]
                feed_dict = {self.input_data: image_batch,
                             self.trainable: False}

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run([pred_sbbox_batch, pred_mbbox_batch, pred_lbbox_batch],
                                                              feed_dict=feed_dict)

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

                bboxes = box_utils.postprocess_boxes(pred_bbox, original_size, self.input_size[0], self.score_threshold)
                bboxes = box_utils.nms(bboxes, self.num_threshold, method='nms')
                end_time = time.perf_counter()

                image = draw_box_in_image.draw_bbox(original_image, bboxes, classes=self.class_name)
                image = Image.fromarray(image)
                image.save(os.path.join(save_path, img_name))

                # resize boxes and image according to raw input image
                # final_detections= cv.resize(final_detections[:, :, ::-1], (raw_w, raw_h))

                # recover to raw size
                bboxes = np.array(bboxes)
                rbboxes = bboxes[:, :4]
                rscores = bboxes[:, 4]
                rclasses = bboxes[:, 5]
                # convert from RGB to BG
                fw.write(f'\n{img_name}')
                for score, boxes, categories in zip(rscores, rbboxes, rclasses):
                    fw.write('\n\tscore:' + str(score))
                    fw.write('\tbboxes:' + str(boxes))
                    fw.write('\tcategories:' + str(int(categories)))

                view_bar('{} image cost {} second'.format(img_name, (end_time - start_time)), index + 1,
                               len(image_name_list))
            fw.close()

    def image_process(self, img_path, input_size=(416, 416), img_format='NHWC'):
        """

        :param img_path:
        :param input_size:
        :param img_format:
        :return:
        """
        original_image = cv.imread(img_path)
        if img_format == 'NCHW':
            original_image = np.transpose(original_image, perm=(2, 0, 1))
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        original_size = original_image.shape[:2]
        image_data = yolo_preprocessing.image_resize_padding(np.copy(original_image), input_size)
        # expend dimension
        image_batch = image_data[np.newaxis, ...]  # (1, None, None, 3)
        # image_batch = tf.expand_dims(input=image_data, axis=0)
        return original_image, image_batch, original_size


if __name__ == "__main__":

    yolo_inference = ObjectInference(input_size=(416, 416), ckpt_path=None)
    yolo_inference.exucute_detect(image_path='./demo', save_path=cfgs.INFERENCE_SAVE_PATH)