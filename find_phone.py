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
import skimage.io
import argparse
import sys

'''
Config Class for Prediction of Phone Dataset

@param Config: Config Instance from R-CNN class
'''
class PredictionConfig(Config):
    NAME = "phone_cfg"
    NUM_CLASSES = 1+1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


'''
Plot Actual and Prediction of Images Side by Side

@param dataset: Image Dataset Class Object
@param model: R-CNN model
@param cfg: Config Object of PredictionConfig Class
@param n_images: No. of images to compare
'''
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# get coordinates
		y1, x1, y2, x2 = yhat['rois'][0]
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the figure
	pyplot.show()


'''
Evaluate Trained Model

@param dataset: Image Dataset Class Object
@param model: R-CNN model
@param cfg: Config Object of PredictionConfig Class
'''
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask = False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose = 1)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"])
        APs.append(AP)
    mAP = mean(APs)
    return mAP

'''
Get X-Coordinate from Pixel Width of Image

@param img_w: Image Width
@param pix_x: x-coordinate in pixel
'''
def getCoordsfromPixelinX(img_w, pix_x):
    x = pix_x/img_w
    return x


'''
Get Y-Coordinate from Pixel Height of Image

@param img_h: Image Height
@param pix_y: y-coordinate in pixel
'''
def getCoordsfromPixelinY(img_h, pix_y):
    y = pix_y/img_h
    return y



'''
Predict Center of Phone from Input Image

@param image_filepath: Path to image file
@param model: Pretrained Model
@param cfg: Prediction Config Object
'''

def predict_phone_center(image_filepath, model, cfg):
    image = skimage.io.imread(image_filepath)
    yhat = model.detect([image], verbose=0)[0]
    pyplot.imshow(image)
    ax = pyplot.gca()
    y1,x1,y2,x2 = yhat['rois'][0]
    pyplot.plot((x1+x2)/2, (y1+y2)/2, 'ro')
    width, height = x2 - x1, y2 - y1
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    ax.add_patch(rect)
    pyplot.savefig('prediction_result.png')
    center_x = getCoordsfromPixelinX(image.shape[1],(x1+x2)/2)
    center_y = getCoordsfromPixelinY(image.shape[0],(y1+y2)/2)
    print(center_x, center_y)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Image File Path')
    parser.add_argument('inputFile', help='Path to Image File')
    return parser


##############################################################################################
#                      Checking if Query Image File is Provided
##############################################################################################
image_file_path = ''
if len(sys.argv)>1:
    image_filepath = sys.argv[1]
else:
    print('No query image provided')
    raise ValueError('Please Provide an Input Image File')


##############################################################################################
#                       Loading the Training Dataset
##############################################################################################
train_set = PhoneDataset()
train_set.load_dataset('dataset', is_train=True)
train_set.prepare()

##############################################################################################
#                       Loading the Test Dataset
##############################################################################################
test_set = PhoneDataset()
test_set.load_dataset('dataset', is_train=False)
test_set.prepare()

##############################################################################################
#                       Setting and Loading Model Parameters
##############################################################################################
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
cfg_folder = ''
for folder in os.listdir():
    if 'phone_cfg' in folder:
        cfg_folder = folder
model_path = cfg_folder + '/mask_rcnn_phone_cfg_0005.h5'
model.load_weights(model_path, by_name=True)

##############################################################################################
#                       Evaluating Model Performance
##############################################################################################
# evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP) 

##############################################################################################
#                      Performing Prediction of I/P image
##############################################################################################
predict_phone_center(image_filepath, model, cfg)

