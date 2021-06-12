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
from phone_dataset import PhoneDataset

'''
Get Dataset Info from Input Folder

@param foldername: Name of the input folder in cwd
'''
def getDatasetInfo(foldername):
    file = open(foldername + 'labels.txt', 'r')
    save_dir = 'dataset/images/'
    lines = file.read().splitlines()
    obj_coordinates = {}
    for line in lines:
        label = line.split(' ')
        obj_coordinates[label[0]] = {'coordinates' : (float(label[1]), float(label[2]))}
    for file in os.listdir(foldername):
        if file.endswith('.jpg'):
            img = cv2.imread(foldername + file)
            cv2.imwrite(save_dir + file, img)
            h, w, c = img.shape
            obj_coordinates[file]['size'] = (h,w)
    return obj_coordinates

'''
Get Pixel Pose from X-Coordinate of Image

@param img_w: Image Width
@param x: x-coordinate
'''
def getPixelfromCoordsinX(img_w, x):
    pix_x = img_w*x
    return round(pix_x)


'''
Get Pixel Pose from Y-Coordinate of Image

@param img_h: Image Height
@param y: y-coordinate
'''
def getPixelfromCoordsinY(img_h, y):
    pix_y = img_h*y
    return round(pix_y)


'''
Generate Annotation File for all images in PASCAL VOC format

@param foldername: Name of the input folder in cwd
@param dataset_info: Dictionary Containing information about image and labels
'''
def generateAnnotFiles(foldername, dataset_info):
    save_dir = 'dataset/annots/'
    for file in os.listdir(foldername):
        if file.endswith('.jpg'):
            h, w = dataset_info[file]['size']
            writer = Writer(foldername + file, w, h)
            c1, c2 = dataset_info[file]['coordinates']
            pix_x_min = getPixelfromCoordsinX(w, c1-0.05)
            pix_y_min = getPixelfromCoordsinY(h, c2-0.05)
            pix_x_max = getPixelfromCoordsinX(w, c1+0.05)
            pix_y_max = getPixelfromCoordsinY(h, c2+0.05)
            writer.addObject('phone', pix_x_min, pix_y_min, pix_x_max, pix_y_max)
            writer.save(save_dir + file.rstrip('.jpg') + '.xml')


'''
Config Class for Training Phone Dataset

@param Config: Config Instance from R-CNN class
'''
class PhoneConfig(Config):
    NAME = "phone_cfg"
    NUM_CLASSES = 1+1
    STEPS_PER_EPOCH = 131


##############################################################################################
#                       Loading Dataset Folder
##############################################################################################
foldername = ''
if len(sys.argv)>1:
    foldername = sys.argv[1]    #'find_phone/'
else:
    print('No Dataset Folder provided')
    raise ValueError('Please Provide a Dataset Folder')
dataset_info = getDatasetInfo(foldername)
generateAnnotFiles(foldername, dataset_info)

##############################################################################################
#                       Loading Train Dataset
##############################################################################################
train_set = PhoneDataset()
train_set.load_dataset('dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

##############################################################################################
#                       Loading Test Dataset
##############################################################################################
# test/val set
test_set = PhoneDataset()
test_set.load_dataset('dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

##############################################################################################
#                       Training and Saving the Model
##############################################################################################
# prepare config
config = PhoneConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

