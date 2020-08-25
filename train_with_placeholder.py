#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train_with_placeholder.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/25 下午3:05
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
import shutil
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from utils import tools
from data.pascal.read_tfrecord import Dataset
from libs.nets.yolo_v3 import YOLOV3
from libs.configs import cfgs


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfgs.ANCHOR_PER_SCALE
        self.classes             = tools.read_class_names(cfgs.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfgs.LEARNING_RATE_INIT
        self.learn_rate_end      = cfgs.LEARNING_RATE_END
        self.first_stage_epochs  = cfgs.FIRST_STAGE_EPOCHS
        self.second_stage_epochs = cfgs.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfgs.WARMUP_EPOCHS
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfgs.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.log_dir             = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        self.train_dataset       = Dataset(is_training=True)
        self.test_dataset        = Dataset(is_training=False)
        self.steps_per_period    = self.train_dataset.num_steps_per_epoches
        self.test_steps_per_period = self.train_dataset.num_steps_per_epoches
        self.train_data_batch = self.train_dataset.dataset_tfrecord(batch_size=cfgs.TRAIN_BATCH_SIZE, is_training=True)
        self.test_data_batch = self.test_dataset.dataset_tfrecord(batch_size=cfgs.TEST_BATCH_SIZE, is_training=False)

        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8,
                                                                                              allow_growth=True)))

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            # self.net_var = tf.global_variables()
            # get loader and saver
            self.loader, self.checkpoint_path = self.model.get_restorer(is_training=True)
            # self.global_step = tf.train.get_or_create_global_step()

            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss


        with tf.name_scope('learning_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            # # self.global_step = self.model.global_step
            # # self.global_step = tf.train.get_or_create_global_step()
            # warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
            #                             dtype=tf.float64, name='warmup_steps')
            # train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
            #                             dtype=tf.float64, name='train_steps')
            # self.learn_rate = tf.cond(
            #     pred=self.global_step < warmup_steps,
            #     true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
            #     false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
            #                         (1 + tf.cos(
            #                             (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            # )
            warmup_steps = int(self.warmup_periods * self.steps_per_period)
            total_train_step = int((self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period)
            self.learning_rate = self.model.cosine_decay_with_warmup(learning_rate_base=self.learn_rate_init,
                                                                     learning_rate_end=self.learn_rate_end,
                                                                     total_decay_steps=total_train_step,
                                                                     warmup_steps=warmup_steps,
                                                                     global_step=self.global_step)
            global_step_update = tf.assign_add(self.global_step, 1.0)
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[1] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            # self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", self.learning_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)


            if os.path.exists(self.log_dir): shutil.rmtree(self.log_dir)
            tools.makedir(self.log_dir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)


    def train(self):

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        try:
            print('=> Restoring weights from: {0} ... '.format(self.checkpoint_path))
            self.loader.restore(self.sess, self.checkpoint_path)
        except:
            print('=> {0}does not exist !!!'.format(self.checkpoint_path))
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        # ++++++++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            if not coord.should_stop():
                for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
                    if epoch <= self.first_stage_epochs:
                        train_op = self.train_op_with_frozen_variables
                    else:
                        train_op = self.train_op_with_all_variables

                    train_epoch_loss, test_epoch_loss = [], []
                    train_bar = tqdm(range(self.steps_per_period))
                    for _ in train_bar:
                        train_data = self.sess.run(self.train_data_batch)
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                        self.input_data:   train_data[0],
                                                        self.label_sbbox:  train_data[1],
                                                        self.label_mbbox:  train_data[2],
                                                        self.label_lbbox:  train_data[3],
                                                        self.true_sbboxes: train_data[4],
                                                        self.true_mbboxes: train_data[5],
                                                        self.true_lbboxes: train_data[6],
                                                        self.trainable:    True,
                        })

                        train_epoch_loss.append(train_step_loss)
                        self.summary_writer.add_summary(summary, global_step_val)

                        train_bar.set_description("train loss: {:.2f}".format(train_step_loss))

                    for _ in range(self.test_steps_per_period):
                        test_data = self.sess.run(self.test_data_batch)

                        test_step_loss = self.sess.run( self.loss, feed_dict={
                                                        self.input_data:   test_data[0],
                                                        self.label_sbbox:  test_data[1],
                                                        self.label_mbbox:  test_data[2],
                                                        self.label_lbbox:  test_data[3],
                                                        self.true_sbboxes: test_data[4],
                                                        self.true_mbboxes: test_data[5],
                                                        self.true_lbboxes: test_data[6],
                                                        self.trainable:    False,
                        })

                        test_epoch_loss.append(test_step_loss)

                    train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
                    save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                    tools.makedir(save_dir)
                    ckpt_file = os.path.join(save_dir, "yolov3_loss={:.4f}.ckpt".format(test_epoch_loss))
                    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print("=> Epoch: {0} Time: {1} Train loss: {2:.4f} Test loss: {3:.4f} Saving {4} ...".
                          format(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                    self.saver.save(self.sess, ckpt_file, global_step=epoch)

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
            print('all threads are asked to stop!')

if __name__ == '__main__':
    YoloTrain().train()