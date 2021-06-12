#!/usr/bin/env python3.7

import os
import mrcnn
import cv2
from mrcnn.utils import Dataset, extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import numpy
from numpy.core.numeric import ones
from pascal_voc_writer import Writer
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from skimage.util import dtype
from matplotlib import pyplot
from matplotlib.patches import Rectangle


'''
Class for Phone Dataset

@param Dataset: Dataset Object from R-CNN class
'''

class PhoneDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train = True):
        self.add_class("dataset", 1, "phone")
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            if is_train and int(image_id) >= 109:
                continue
            if not is_train and int(image_id) < 109:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id=image_id, path= img_path, annotation=ann_path)
    
    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h,w,len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('phone'))
        return masks, asarray(class_ids, dtype='int32')
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
