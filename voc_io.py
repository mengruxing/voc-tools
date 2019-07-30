#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  :   mrx
@Contact :   mengruxing@gmail.com
@Date    :   2019/7/21
@Project :   work_shop
@File    :   voc_io.py
@Desc    :   生成和读取 voc 格式的 xml 文件
"""

import os
import re
import codecs
import logging

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    return etree.tostring(
        etree.fromstring(ElementTree.tostring(elem, encoding='utf8')),
        pretty_print=True,
        encoding='utf-8'
    ).replace("  ".encode(), "\t".encode())


class PascalVoc(object):

    def __init__(self):
        self.verified = False
        self.folder_name = None
        self.image_name = None
        self.image_path = None
        self.image_shape = [0, 0, 0]

        self.database = 'Unknown'
        self.segmented = False

        self.bboxes = {}
        self.shapes = []

    def get_image_path(self):
        return self.image_path if os.path.exists(self.image_path) else os.path.join(self.folder_name, self.image_name)


class PascalVocWriter(object):

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        """
        构建PascalVoc文件写入工具
        :param folder_name:     folder,             图片所在文件夹名称
        :param filename:        filename,           图片名称
        :param img_size:        size,               图片大小 (height, width, depth)
        :param database_src:    source.database,    数据集
        :param local_img_path:  path,               图片绝对路径
        """
        self.folder_name = folder_name
        self.filename = filename
        self.local_img_path = local_img_path
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.verified = False

    def add_bbox(self, xmin, ymin, xmax, ymax, name, difficult=False):
        """
        添加一个框
        :param xmin: xmin
        :param ymin: ymin
        :param xmax: xmax
        :param ymax: ymax
        :param name: label
        :param difficult: 0
        """
        self.box_list.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'name': name, 'difficult': difficult})

    def save_xml(self, target_path=None):
        """
        保存 xml 文件
        :param target_path:
        :return:
        """
        if self.filename is None or self.folder_name is None or self.img_size is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.folder_name

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.local_img_path is not None:
            local_img_path = SubElement(top, 'path')
            local_img_path.text = self.local_img_path

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.database_src

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        width.text = str(self.img_size[1])
        height = SubElement(size_part, 'height')
        height.text = str(self.img_size[0])
        depth = SubElement(size_part, 'depth')
        depth.text = str(self.img_size[2]) if len(self.img_size) == 3 else '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'

        for each_object in self.box_list:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.img_size[0])) or (int(float(each_object['ymin'])) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(float(each_object['xmax'])) == int(float(self.img_size[1]))) or (int(float(each_object['xmin'])) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object['difficult']) & 1)
            bbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bbox, 'ymax')
            ymax.text = str(each_object['ymax'])

        if target_path is None:
            xml_file_name = re.sub(re.compile(r"\.(jpg|png)$", re.S), ".xml", self.filename)
            target_path = os.path.join(self.folder_name, xml_file_name)
        if not target_path.endswith('.xml'):
            target_path += '.xml'
        out_file = codecs.open(target_path, 'w', encoding='utf-8')
        out_file.write(prettify(top).decode('utf8'))
        out_file.close()


class PascalVocReader(object):

    def __init__(self):
        """
        构建PascalVoc文件写入工具
        """
        self.shapes = []
        self.bboxes = {}
        self.verified = False
        self.folder_name = None
        self.image_name = None
        self.image_path = None
        self.image_shape = (0, 0, 0)

    def get_image_path(self):
        return self.image_path if os.path.exists(self.image_path) else os.path.join(self.folder_name, self.image_name)

    def _add_bbox(self, label, bbox, difficult=False):
        """
        添加一个框 (内部调用)
        :param label:
        :param bbox:
        :param difficult:
        :return:
        """
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        try:
            self.bboxes[label].append([xmin, ymin, xmax, ymax])
        except KeyError:
            self.bboxes[label] = [[xmin, ymin, xmax, ymax]]
        self.shapes.append((label, [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], None, None, difficult))

    def parse_xml(self, xml_path):
        """
        解析 voc.xml 文件
        :param xml_path: voc.xml 文件路径
        :return:
        """
        assert xml_path.endswith('.xml'), "Unsupported file format"
        xml_tree = ElementTree.parse(xml_path, parser=etree.XMLParser(encoding='utf-8')).getroot()
        try:
            self.verified = xml_tree.attrib['verified'] == 'yes'
        except KeyError:
            pass

        try:
            self.folder_name = xml_tree.find('folder').text
        except AttributeError:
            logging.warning('AttributeError: catch exception while parsing folder.')
        try:
            self.image_name = xml_tree.find('filename').text
        except AttributeError:
            logging.warning('AttributeError: catch exception while parsing filename.')
        try:
            self.image_path = xml_tree.find('path').text
        except AttributeError:
            logging.warning('AttributeError: catch exception while parsing path.')
        try:
            size = xml_tree.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
        except AttributeError:
            logging.warning('AttributeError: catch exception while parsing size.')
        else:
            self.image_shape = (height, width, depth)

        for object_iter in xml_tree.findall('object'):
            label = object_iter.find('name').text
            bbox = object_iter.find('bndbox')
            difficult = object_iter.find('difficult')
            self._add_bbox(label, bbox, False if difficult is None else bool(int(difficult.text)))

        return self
