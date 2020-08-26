#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : yolo_v3.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/18 上午10:13
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from utils import tools
import libs.nets.custom_layers as common
import libs.nets.backbone as backbone
import  libs.configs.cfgs as cfgs


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable):

        self.trainable        = trainable
        self.classes          = tools.read_class_names(cfgs.CLASSES)
        self.num_class        = len(self.classes)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfgs.STRIDES)
        self.anchors          = tools.get_anchors(cfgs.ANCHORS)
        self.anchor_per_scale = cfgs.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfgs.IOU_LOSS_THRESH
        self.upsample_method  = cfgs.UPSAMPLE_METHOD

        try:
            with tf.variable_scope(cfgs.MODEL_NAME, default_name="yolo_v3"):
                self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data):

        route_1, route_2, route_3 = backbone.darknet53(input_data, self.trainable)

        input_data = common.convolutional(route_3, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        # [batch_size, target_size/32, target_size/32, anchor_per_scale * (4+1+NUM_CLASS)]
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        # [batch_size, target_size/16, target_size/16, anchor_per_scale * (4+1+NUM_CLASS)]
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        # [batch_size, target_size/8, target_size/8, anchor_per_scale * (4+1+NUM_CLASS)]
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        """
        reference paper Bounding Box Prediction part
        # couv_output [batch_size, output_size, output_size, (anchor_per_scale * 5 + num_classes)]
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        # [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))


        conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # [batch_size, output_size, output_size, anchor_per_scale, 2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # [batch_size, output_size, output_size, anchor_per_scale, 2]
        conv_raw_conf = conv_output[:, :, :, :, 4:5] # [batch_size, output_size, output_size, anchor_per_scale, 1]
        conv_raw_prob = conv_output[:, :, :, :, 5: ] # [batch_size, output_size, output_size, anchor_per_scale, num_classes]

        # reference Bounding Box section of paper
        # get grids coordinate of feature map, the top-left grid = (1, 1), down-right gird = (output_size, output_size)
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])  # (output_size, output_size)
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1]) # (output_size, output_size)

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)  # (output_size, output_size, 2)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1]) # (batch_size, output_size, output_size, 3, 2)
        xy_grid = tf.cast(xy_grid, tf.float32)

        #+++++++++++++++++++++++reference YOLOv3 paper Bounding Box Prediction part++++++++++++++++++++++++++++++++
        # conv_raw_dxdy => t_x, t_y
        # conv_raw_dwdh => t_w, t_h
        # xy_grid => c_x, x_y
        # anchor =>  p_w, p_h
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)  # reference class prediction section

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):
        """

        :param boxes1: # [batch_size, output_size, output_size, anchor_per_scale, 4]
        :param boxes2: # [batch_size, output_size, output_size, anchor_per_scale, 4]
        :return:
        """
        # (x, y, w, h) => (x_min, y_min, x_max, y_max)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # ensure x_min < x_max, y_min < y_max
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        # get boxes area
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # get inter area of two boxes (x_min, y_min), (x_max, y_max)
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        # (x_min, y_min), (x_max, y_max) => (w, h)
        inter_section = tf.maximum(right_down - left_up, 0.0)
        # get inter area value
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # get union area value
        union_area = boxes1_area + boxes2_area - inter_area
        # get iou
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):
        """

        :param boxes1: [batch_size, target_seize, target_size, 3, 1,   4]
        :param boxes2: [batch_size, 1,            1,           1, 150, 4]
        :return:
        """

        # get boxes1 area
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # [batch_size, target_seize, target_size, 3, 1]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # [batch_size, 1,            1,           1, 150]

        # (x, y, w, h) => (x_min, y_min, x_max, y_max)
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5],
                           axis=-1)  # [batch_size, target_seize, target_size, 3, 1,   4]
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5],
                           axis=-1)  # [batch_size, 1,            1,           1, 150, 4]

        # get inter bbox
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])  # [batch_size, target_seize, target_size, 3, 150,  2]
        right_down = tf.minimum(boxes1[..., 2:],
                                boxes2[..., 2:])  # # [batch_size, target_seize, target_size, 3, 150,  2]
        inter_section = tf.maximum(right_down - left_up, 0.0)

        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area  # [batch_size, target_seize, target_size, 3, 150]

        return iou


    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        """
        :param conv: [batch_size, output_size, output_size, anchor_per_scale*(4+1+NUM_CLASSES)]
        :param pred: [batch_size, output_size, output_size, anchor_per_scale，(4+1+NUM_CLASSES)]
        :param label: [batch_size, output_size, output_size, anchor_per_scale，(4+1+NUM_CLASSES)]
        :param bboxes: [batch_size, max_box_per_scale, 4] = [batch_size, 150, 4]
        :param anchors: [anchor_per_scale, 2] = [3,2]
        :param stride: [, 3] = [8, 16, 32]
        :return:
        """

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        # [batch_size, output_size, output_size, anchor_per_scale, 4+1+NUM_CLASSES]
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5] # [batch_size, output_size, output_size, anchor_per_scale, 1]
        conv_raw_prob = conv[:, :, :, :, 5:]  # [batch_size, output_size, output_size, anchor_per_scale, NUM_CLASSES]

        pred_xywh     = pred[:, :, :, :, 0:4] # [batch_size, output_size, output_size, anchor_per_scale, 4]
        pred_conf     = pred[:, :, :, :, 4:5] # [batch_size, output_size, output_size, anchor_per_scale, 1]

        label_xywh    = label[:, :, :, :, 0:4] # [batch_size, output_size, output_size, anchor_per_scale, 4]
        respond_bbox  = label[:, :, :, :, 4:5] # [batch_size, output_size, output_size, anchor_per_scale, 1]
        label_prob    = label[:, :, :, :, 5:] # [batch_size, output_size, output_size, anchor_per_scale, NUM_CLASS]

        # -------------------compute bbox loss-------------------------------------
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        # --------------------compute conf loss-----------------------------------
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # [batch_size, target_seize, target_size, 3, 1]

        # get back ground mask
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )
        #-----------------------cmpute class loss-------------------------------------------------
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4])) # (1,)
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4])) # (1,)
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4])) # (1,)

        return giou_loss, conf_loss, prob_loss


    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss  # (1,), (1,), (1,)


    def get_restorer(self, is_training=False):
        """
        restore pretrain weight
        :return:
        """

        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:

            # model_variables = tf.model_variables()
            restorer = tf.train.Saver()
            print("model restore from {0}".format(checkpoint_path))
        else:
            checkpoint_path = cfgs.PRETRAINED_WEIGHTS
            if is_training:
                custom_scope = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
            else:
                custom_scope = []
            model_variables = tf.global_variables()
            ckpt_var_dict = {}
            for var in model_variables:
                if var.name.split('/')[1] in custom_scope:
                    continue
                else:
                    var_name_ckpt = var.op.name
                    ckpt_var_dict[var_name_ckpt] = var
            restore_variables = ckpt_var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)

            restorer = tf.compat.v1.train.Saver(restore_variables)

            print("restore from pretrained_weighs by COCO")

        return restorer, checkpoint_path

    def cosine_decay_with_warmup(self, global_step,
                                 learning_rate_base,
                                 total_decay_steps,
                                 learning_rate_end=0.0,
                                 warmup_learning_rate=0.0,
                                 warmup_steps=0,
                                 hold_base_rate_steps=0):
        """Cosine decay schedule with warm up period.

        """
        if total_decay_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')

        def eager_decay_rate():
            """Callable to compute the learning rate."""
            # reference cosine_decay
            # cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
            # decayed = (1 - alpha) * cosine_decay + alpha
            # decayed_learning_rate = learning_rate * decayed
            # where alpha = 0
            # global_step = global_step - (warmup_steps + hold_base_rate_steps)
            # decay_step = total_steps - (warmup_steps + hold_base_rate_steps)
            # learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
            #     np.pi *(tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
            #     ) / float(total_decay_steps - warmup_steps - hold_base_rate_steps)))
            learning_rate = tf.train.cosine_decay(learning_rate=learning_rate_base,
                                                  decay_steps=total_decay_steps - warmup_steps - hold_base_rate_steps,
                                                  global_step=global_step - warmup_steps - hold_base_rate_steps,
                                                  alpha=learning_rate_end)
            if hold_base_rate_steps > 0:
                learning_rate = tf.where(
                    global_step > warmup_steps + hold_base_rate_steps,
                    learning_rate, learning_rate_base)
            if warmup_steps > 0:
                if learning_rate_base < warmup_learning_rate:
                    raise ValueError('learning_rate_base must be larger or equal to '
                                     'warmup_learning_rate.')
                slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
                warmup_rate = slope * tf.cast(global_step,
                                              tf.float32) + warmup_learning_rate
                learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                         learning_rate)
            return tf.where(global_step > total_decay_steps, learning_rate_end, learning_rate,
                            name='learning_rate')

        if tf.executing_eagerly():
            return eager_decay_rate
        else:
            return eager_decay_rate()

