#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : convert_tfrecord.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/4 下午3:06
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import io
import glob
import random
import tensorflow.compat.v1 as tf
from lxml import etree
import PIL.Image

from libs.configs import cfgs
from utils.tools import view_bar, read_class_names


'''How to organize your dataset folder:
  VOCROOT/
       |->VOC2007/
       |    |->Annotations/
       |    |->ImageSets/
       |    |->...
       |->VOC2012/
       |    |->Annotations/
       |    |->ImageSets/
       |    |->...
'''

original_dataset_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_voc/train'
tfrecord_dir = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd/train'

tf.app.flags.DEFINE_string('dataset_dir', original_dataset_dir, 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('year', '2007,2012', 'Desired challenge year.')
tf.app.flags.DEFINE_string('output_dir', tfrecord_dir, 'save name')
tf.app.flags.DEFINE_string('img_format', 'jpg', 'format of image')
tf.app.flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                            'difficult instances')
FLAGS = tf.app.flags.FLAGS


# check route
try:
    if os.path.exists(FLAGS.dataset_dir) is False:
        raise IOError('dataset is not exist please check the path')
except FileNotFoundError as e:
    print(e)
finally:
    # makedir(FLAGS.output_dir)
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)


def convert_pascal_to_tfrecord(dataset_path, save_path, record_capacity=2000, shuffling=False):
    """
    convert pascal dataset to rfrecord
    :param img_path:
    :param xml_path:
    :param save_path:
    :param record_capacity:
    :return:
    """
    index_name = read_class_names(cfgs.CLASSES)
    name_index = {}
    for index, name in index_name.items():
        name_index[name] = int(index)
    years = [s.strip() for s in FLAGS.year.split(',')]
    # record_file = os.path.join(FLAGS.save_dir, FLAGS.save_name+'.tfrecord')

    # get image and xml list
    img_name_list = []
    img_xml_list = []

    for year in years:
        img_path = os.path.join(dataset_path, 'VOC'+year, FLAGS.image_dir)
        xml_path = os.path.join(dataset_path, 'VOC'+year, FLAGS.xml_dir)
        xml_list = [xml_file for xml_file in glob.glob(os.path.join(xml_path, '*.xml'))]
        img_list = [os.path.join(img_path, os.path.basename(xml).replace('xml', FLAGS.img_format)) for xml in xml_list]
        img_name_list.extend(img_list)
        img_xml_list.extend(xml_list)


    if shuffling:
        shuffled_index = list(range(len(img_name_list)))
        random.seed(0)
        random.shuffle(shuffled_index)
        img_name_shuffle = [img_name_list[index] for index in shuffled_index]
        img_xml_shuffle = [img_xml_list[index] for index in shuffled_index]
        img_name_list = img_name_shuffle
        img_xml_list = img_xml_shuffle

    remainder_num = len(img_name_list) % record_capacity
    if remainder_num == 0:
        num_record = int(len(img_name_list) / record_capacity)
    else:
        num_record = int(len(img_name_list) / record_capacity) + 1

    num_samples = 0
    for index in range(num_record):
        record_filename = os.path.join(save_path, f'{index}.record')
        write = tf.io.TFRecordWriter(record_filename)
        if index < num_record - 1:
            sub_img_list = img_name_list[index * record_capacity: (index + 1) * record_capacity]
            sub_xml_list = img_xml_list[index * record_capacity: (index + 1) * record_capacity]
        else:
            sub_img_list = img_name_list[(index * record_capacity): (index * record_capacity + remainder_num)]
            sub_xml_list = img_xml_list[(index * record_capacity): (index * record_capacity + remainder_num)]

        try:
            for img_file, xml_file in zip(sub_img_list, sub_xml_list):

                encoded_img, shape, bboxes, labels, labels_text, difficult, truncated = process_image(img_file, xml_file, class_name=name_index)

                image_record = serialize_example(img_file, encoded_img, labels, labels_text, bboxes, shape, difficult, truncated)
                write.write(record=image_record)

                num_samples += 1
                view_bar(message='\nConversion progress', num=num_samples, total=len(img_name_list))
        except Exception as e:
            print(e)
            continue
        write.close()
    print('\nThere are {0} samples convert to {1}'.format(num_samples, save_path))


def process_image(img_path, xml_path, class_name=None):

    # process image
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = PIL.Image.open(encoded_img_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    # process xml
    # with tf.gfile.GFile(xml_path, 'r') as fid:
    #     xml_str = fid.read()
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(cfgs.VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult') is not None:
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated') is not None:
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('xmin').text),
                       float(bbox.find('ymin').text),
                       float(bbox.find('xmax').text),
                       float(bbox.find('ymax').text)
                       ))
    return encoded_img, shape, bboxes, labels, labels_text, difficult, truncated


def serialize_example(filename, image_data, labels, labels_text, bboxes, shape, difficult, truncated):
    """
    create a tf.Example message to be written to a file
    :param label: label info
    :param image: image content
    :param filename: image name
    :return:
    """
    # create a dict mapping the feature name to the tf.Example compatible
    # image_shape = tf.image.decode_jpeg(image_string).eval().shape
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    feature = {
            'image/filename':_bytes_feature(filename.encode()),
            'image/height': _int64_feature(shape[0]),
            'image/width': _int64_feature(shape[1]),
            'image/channels': _int64_feature(shape[2]),
            'image/shape': _int64_feature(shape),
            'image/object/num_object': _int64_feature(len(bboxes)),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/bbox/label': _int64_feature(labels),
            'image/object/bbox/label_text': _bytes_feature(labels_text),
            'image/object/bbox/difficult': _int64_feature(difficult),
            'image/object/bbox/truncated': _int64_feature(truncated),
            'image/encoded': _bytes_feature(image_data),
            'image/format': _bytes_feature(b'JPEG')}
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main(argc):
    convert_pascal_to_tfrecord(FLAGS.dataset_dir, FLAGS.output_dir, record_capacity=2000, shuffling=False)


if __name__ == "__main__":
    tf.app.run()











