#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  :   mrx
@Contact :   mengruxing@gmail.com
@Date    :   2019/7/29
@Project :   work_shop
@File    :   lisa_to_voc.py
@Desc    :   
"""

import os
import re
import glob
import csv

from voc_io import PascalVocWriter


class LisaData(object):

    def __init__(self, strings, filename=None, tag=None, xmin=None, ymin=None, xmax=None, ymax=None):
        self.data = strings.strip().split(';')
        self.filename = self.data[0] if filename is None else filename
        self.tag = self.data[1] if tag is None else tag
        self.xmin = int(float(self.data[2] if xmin is None else xmin))
        self.ymin = int(float(self.data[3] if ymin is None else ymin))
        self.xmax = int(float(self.data[4] if xmax is None else xmax))
        self.ymax = int(float(self.data[5] if ymax is None else ymax))
        self.bbox = (self.xmin, self.ymin, self.xmax, self.ymax)

    def __cmp__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, LisaData):
            return False
        if self.filename != other.filename:
            return False
        if self.tag != other.tag:
            return False
        if self.xmin != other.xmin:
            return False
        if self.ymin != other.ymin:
            return False
        if self.xmax != other.xmax:
            return False
        if self.ymax != other.ymax:
            return False
        return True


class LisaManager(object):

    def __init__(self):
        self.lisa_dict = {}

    def add_lisa_data(self, lisa_data):
        assert isinstance(lisa_data, LisaData)
        try:
            self.lisa_dict[lisa_data.filename].append(lisa_data)
        except KeyError:
            self.lisa_dict[lisa_data.filename] = [lisa_data]

    def gen_voc_data(self, root_path):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        for filename, lisa_data_list in self.lisa_dict.items():
            folder_name, file_name = filename.split('/')
            writer = PascalVocWriter(folder_name=folder_name, filename=file_name, img_size=(960, 1280, 3))
            for lisa_data in lisa_data_list:
                writer.add_bbox(
                    xmin=lisa_data.xmin,
                    ymin=lisa_data.ymin,
                    xmax=lisa_data.xmax,
                    ymax=lisa_data.ymax,
                    name=lisa_data.tag
                )
            save_dir = os.path.join(root_path, folder_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            writer.save_xml(os.path.join(save_dir, re.sub(re.compile(r"\.(jpg|png)$", re.S), ".xml", file_name)))


def convert(lisa_csv_path, voc_root_path):
    manager = LisaManager()
    with open(lisa_csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            manager.add_lisa_data(LisaData(row[0]))
    manager.gen_voc_data(voc_root_path)
    print('{} -> {}'.format(lisa_csv_path, voc_root_path))


def main(csv_glob, voc_root_path):
    for f in glob.glob(csv_glob):
        convert(lisa_csv_path=f, voc_root_path=voc_root_path)


if __name__ == '__main__':
    main('/home/mrx/dataset/lisa/*/*/frameAnnotationsBOX.csv', '/home/mrx/tmp/lisa_traffic_light_voc_format')
    print('all done.')
