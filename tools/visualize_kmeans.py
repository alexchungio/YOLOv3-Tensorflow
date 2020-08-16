#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : visualize_kmeans.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/16 下午8:00
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import numpy as np
import libs.configs.cfgs as cfgs
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
         'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
         'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Parse annotations
def parse_annotation(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    boxes = []
    labels = {}
    num_samples = 0
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        box_label_list = s[4:]


        for box_label in box_label_list:
            x_min, y_min, x_max, y_max, label = [float(coord) for coord in box_label.split(',')]
            label = int(label)
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0

            if LABELS[label] not in labels.keys():
                labels[LABELS[label]] = 1
            else:
                labels[LABELS[label]] += 1

            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                boxes.append([width, height])
            # get k-means anchors on the original image size
            else:
                boxes.append([width, height])
        num_samples += 1

    boxes = np.asarray(boxes)
    return boxes, labels, num_samples


# visual class distribution
def visualize_class(boxes, labels, num_samples, label_map):
    y_pos = np.arange(len(label_map))
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y_pos, [labels[label] for label in label_map])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_map)
    ax.set_title("The total number of objects = {} in {} images".format(
        len(boxes), num_samples))
    plt.show()


def visualize_cluster(boxes):

    plt.figure(figsize=(10, 10))
    plt.scatter(boxes[:, 0], boxes[:, 1], alpha=0.3)
    plt.title("Clusters", fontsize=20)
    plt.xlabel("normalized width", fontsize=20)
    plt.ylabel("normalized height", fontsize=20)
    plt.show()


def visualize_kmeans(clusters, nearest_clusters, boxes, k):
    current_palette = list(sns.xkcd_rgb.values())
    for cluster_index in np.unique(nearest_clusters):
        index = np.equal(nearest_clusters, cluster_index)

        plt.rc('font', size=8)
        plt.plot(boxes[index][:, 0], boxes[index][:, 1], "p",
                 color= current_palette[cluster_index],
                 alpha=0.5, label="cluster = {}, N = {:6.0f}".format(cluster_index, np.sum(cluster_index)))
        plt.text(clusters[cluster_index][0],
                 clusters[cluster_index][1],
                 "c{}".format(cluster_index),
                 fontsize=20, color="red")
        plt.title("Clusters={0}".format(k))
        plt.xlabel("width")
        plt.ylabel("height")
    plt.show()
    # plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters


def get_kmeans(annotation, cluster_num=9):

    anchors, nearest_clusters = kmeans(annotation, cluster_num)
    ave_iou = avg_iou(annotation, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou, nearest_clusters

if __name__ == "__main__":

    train_boxes, train_labels, num_samples = parse_annotation(cfgs.TRAIN_ANNOTATION)
    print("N train = {}".format(len(train_boxes)))

    visualize_class(train_boxes, train_labels, num_samples, LABELS)
    visualize_cluster(train_boxes)

    anchors, ave_iou, nearest_clusters = get_kmeans(train_boxes, 9)

    visualize_kmeans(anchors, nearest_clusters, train_boxes, 9)