#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  :   mrx
@Contact :   mengruxing@gmail.com
@Date    :   2019/7/22
@Project :   work_shop
@File    :   voc_verification.py
@Desc    :   
"""

import glob
import json
import re
import os
import numpy as np

from numba import jit
from scipy.optimize import linear_sum_assignment

from voc_io import PascalVocReader


PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))


class Detector(object):

    def __init__(self, labels=(), input_size=(300, 300)):
        """
        目标检测器
        :param input_size: 网络图片输入大小 (height, width)
        """
        self.input_size = input_size
        self.labels = labels
        self.label_index = {l: i for i, l in enumerate(labels)}

    def __call__(self, image_path, image_shape):
        """
        检测目标
        :param image_path: 图片路径
        :param image_shape: 图片维度 (height, width, depth)
        :return: list bonding_boxes of all classes
        """
        raise NotImplementedError


class JsonNoDetector(Detector):

    def __call__(self, image_path, image_shape):
        with open(re.sub(re.compile(r"\.(jpg|png)$", re.S), ".json", image_path), 'r') as f:
            j = json.load(f)
        return [np.asarray(j.get(label, []), dtype=np.float32) for label in self.labels]


class VocXmlNoDetector(Detector):

    def __init__(self, labels=(), input_size=(300, 300), root_path=None):
        """
        读取voc格式的检测框
        :param labels:
        :param input_size:
        :param root_path:
        """
        super(VocXmlNoDetector).__init__(labels=labels, input_size=input_size)
        self.root_path = root_path

    def __call__(self, image_path, image_shape):
        if self.root_path is None:
            xml_path = re.sub(re.compile(r"\.(jpg|png)$", re.S), ".xml", image_path)
        else:
            xml_path = os.path.join(self.root_path, re.sub(re.compile(r"\.(jpg|png)$", re.S), ".xml", os.path.basename(image_path)))
        reader = PascalVocReader().parse_xml(xml_path=xml_path)
        return [np.asarray(reader.bboxes.get(label, [])) for label in self.labels]


@jit
def iou(bbox_1, bbox_2):
    """
    计算两个 bounding box 的 IoU 值
    必备条件:
        左上角是 [x1, y1]
        右下角是 [x2, y2]
    """
    i_x1 = np.maximum(bbox_1[0], bbox_2[0])
    i_y1 = np.maximum(bbox_1[1], bbox_2[1])
    i_x2 = np.minimum(bbox_1[2], bbox_2[2])
    i_y2 = np.minimum(bbox_1[3], bbox_2[3])
    inter_area = np.maximum(0., i_x2 - i_x1) * np.maximum(0., i_y2 - i_y1)
    sum_area = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1]) + (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])
    return inter_area / (sum_area - inter_area)


def verify_voc(detector, xml_path, root_path=PROJECT_HOME, areas_limit=64, iou_limit=0.45):
    """
    验证 voc.xml 文件
    :param detector:    目标检测器
    :param xml_path:    voc.xml 文件路径
    :param root_path:   数据根目录
    :param areas_limit: 目标框在网络中输入的有效面积阈值
    :param iou_limit:   iou阈值
    :return: 正确个数, 总个数
    """
    voc_reader = PascalVocReader().parse_xml(xml_path)

    img_path = voc_reader.image_path
    if img_path is None or not os.path.exists(img_path):
        img_path = os.path.join(root_path, voc_reader.folder_name, voc_reader.image_name)

    input_height, input_width = detector.input_size
    img_height, img_width, img_depth = voc_reader.image_shape
    real_limit = areas_limit * (img_height * img_width) / (input_height * input_width)

    scores = {}

    det_bboxes = detector(image_path=img_path, image_shape=voc_reader.image_shape)
    for label, bboxes in voc_reader.bboxes.items():
        np_bboxes = np.asarray(bboxes)
        areas = (np_bboxes[:, 3] - np_bboxes[:, 1]) * (np_bboxes[:, 2] - np_bboxes[:, 0])
        valid_bboxes = np_bboxes[areas > real_limit]
        if valid_bboxes.shape[0] == 0:
            scores[label] = [0, 0]
            continue
        try:
            idx = detector.label_index[label]
        except KeyError:
            scores[label] = [0, len(valid_bboxes)]
            continue
        iou_matrix = np.asarray([[iou(voc_bbox, det_bbox) for det_bbox in det_bboxes[idx]] for voc_bbox in valid_bboxes], dtype=np.float32)
        iou_matrix[iou_matrix < iou_limit] = 0
        voc_idx, det_idx = linear_sum_assignment(-iou_matrix)
        match_iou = iou_matrix[voc_idx, det_idx]
        scores[label] = [len(voc_idx) - len(match_iou[match_iou < iou_limit]), len(valid_bboxes)]
    print(scores)
    return scores


class VocValidator(object):

    def __init__(self, detector, xml_dir_path, areas_limit=100, iou_limit=0.45):
        """

        :param detector:
        :param xml_dir_path:
        :param areas_limit:
        :param iou_limit:
        """
        self.detector = detector
        self.xml_dir_path = xml_dir_path
        self.areas_limit = areas_limit
        self.iou_limit = iou_limit

    def run(self):
        """
        运行
        :return:
        """
        xml_files = glob.glob(os.path.join(self.xml_dir_path, '*.xml'))
        map_scores = {}
        for xml_file in xml_files:
            ap_scores = verify_voc(
                detector=self.detector,
                xml_path=xml_file,
                areas_limit=self.areas_limit,
                iou_limit=self.iou_limit
            )
            for label, (matches, total) in ap_scores.items():
                try:
                    label_ap_scores = map_scores[label]
                except KeyError:
                    label_ap_scores = [0, 0]
                    map_scores[label] = label_ap_scores
                label_ap_scores[0] += matches
                label_ap_scores[1] += total

        all_matches, all_bboxes = np.sum([score for score in map_scores.values()], axis=0)
        print('acc: {:.3f}'.format(all_matches / all_bboxes))
        for label, (matches, total) in map_scores.items():
            map_scores[label] = matches / total
        print(map_scores)
