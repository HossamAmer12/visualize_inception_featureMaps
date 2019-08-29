# -*- coding: utf-8 -*-
# Hossam Amer
# Run using this way: python3 visualize_featureMaps.py

# Inception image recognition attempt v1
import tensorflow as tf
import numpy as np
import re
import os
import time
from tkinter import *
import tkinter.filedialog
import matplotlib.pyplot as plt
import logging

# Video capture and convert rgb 
from video_capture import VideoCaptureYUV
import cv2

# Node look up
from node_lookup import NodeLookup

import time 

# for fetching files
import glob

import math

from random import randrange

# needs more work
# MODEL_PATH = '/Users/hossam.amer/7aS7aS_Works/work/jpeg_ml_research/inceptionv3/inception_model'
MODEL_PATH = './inception_model'

# YUV Path
PATH_TO_RECONS = '/Volumes/MULTICOMHD2/set_yuv/Seq-RECONS/'

# JPEG Path
path_to_valid_images    = '/Volumes/MULTICOMHD2/validation_original/';
path_to_valid_QF_images = '/Volumes/MULTICOMHD2/validation_generated_QF/';

#读取训练好的Inception-v3模型来创建graph
def create_graph():
  # the class that's been created from the textual definition in graph.proto
  #with tf.gfile.FastGFile('./inception_model/inception_v3_2016_08_28_frozen.pb', 'rb') as f:  
    with tf.gfile.FastGFile(MODEL_PATH + '/classify_image_graph_def.pb', 'rb') as f:  
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def print_ops(sess):
  constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
  for constant_op in constant_ops:
    print(constant_op.name)  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide the warning information from Tensorflow - annoying...



def get_image_data(imgID, QF, isCast = True):
	# Parse the YUV and convert it into RGB
	original_img_ID = imgID
	imgID = str(imgID).zfill(8)
	shard_num  = round(original_img_ID/10000);
	folder_num = math.ceil(original_img_ID/1000);
	if isCast:
		path_to_recons = PATH_TO_RECONS
		# Get files list to fetch the correct name
		filesList = glob.glob(path_to_recons + str(folder_num) + '/ILSVRC2012_val_' + imgID + '*.yuv')
		name = filesList[0].split('/')[-1]
		rgbStr = name.split('_')[5]
		width  = int(name.split('_')[-5])
		height = int(name.split('_')[-4])
		is_gray_str = name.split('_')[-3]
		
		image = path_to_recons + str(folder_num) + '/ILSVRC2012_val_' + imgID + '_' + str(width) + '_' + str(height) + '_' + rgbStr + '_' + str(QF) + '_1.yuv'
		figure_title = 'ILSVRC2012_val_' + imgID + '_' + str(width) + '_' + str(height) + '_' + rgbStr + '_' + str(QF) + '_1.yuv'
		print(image)
		size = (height, width) # height and then width
		videoObj = VideoCaptureYUV(image, size, isGrayScale=is_gray_str.__contains__('Y'))
		ret, yuv, rgb = videoObj.getYUVAndRGB()
		image_data = rgb
	else:
		if QF == 110:
			image = path_to_valid_images + str(folder_num) + '/ILSVRC2012_val_' + imgID + '.JPEG'
			figure_title = 'ILSVRC2012_val_' + imgID + '.JPEG'
		else:
			image = path_to_valid_QF_images + str(folder_num) + '/ILSVRC2012_val_' + imgID + '-QF-' + str(QF) + '.JPEG'
			figure_title = 'ILSVRC2012_val_' + imgID + '-QF-' + str(QF) + '.JPEG'
		print(image)
		image_data = tf.gfile.FastGFile(image, 'rb').read()
	return image_data, figure_title



def plot(feature_maps, featureMapIdx, figure_title):
	K     = feature_maps.shape[2]
	nRows = K//8
	nCols = K//nRows
	if featureMapIdx < 0:
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(10, 5))
		plt.figure(figureID)
		for irow, row in enumerate(ax):
			for icol, col in enumerate(row):
				idx = irow + icol * nRows
				if idx >= K:
					continue

				m = feature_maps[:, :, idx]
				if isGrayScaleNorm:
					A   = np.double(m)
					out = np.zeros(A.shape, np.double)
					m   = cv2.normalize(A, out, 255.0, 0.0, cv2.NORM_MINMAX)
					col.imshow(m, cmap='gray', vmin=0.0, vmax=255.0)
				else:
					col.imshow(m)
				col.axis('off')
	else:
		plt.figure(figureID)
		m = feature_maps[:, :, featureMapIdx]
		if isGrayScaleNorm:
			A   = np.double(m)
			out = np.zeros(A.shape, np.double)
			m   = cv2.normalize(A, out, 255.0, 0.0, cv2.NORM_MINMAX)
			plt.imshow(m, cmap='gray', vmin=0.0, vmax=255.0)
		else:
			plt.imshow(m)
			plt.axis('off')
		figure_title = str(featureMapIdx) + ')' + figure_title
	plt.suptitle(figure_title, fontsize=16)
	plt.subplots_adjust(wspace=0.0, hspace=0.0)


# Visualizes feature map of a specific image in the validation set
def visualize_image(imgID, QF, layerID = 1, figureID = 1, isCast = True, isGrayScaleNorm = False, featureMapIdx = -1):


    # Create graph:
    create_graph()
    sess = tf.Session()
	
    # Inception-v3: last layer is output as softmax
    conv1_tensor = sess.graph.get_tensor_by_name('conv_' + str(layerID) + ':0')
  
    # Title of the figure
    figure_title = ''

    # Get image data
    image_data, figure_title = get_image_data(imgID, QF, isCast)
   
    if isCast:
      feature_maps = sess.run(conv1_tensor, {'Cast:0': image_data}) # n, m, 3
    else:
      feature_maps = sess.run(conv1_tensor, {'DecodeJpeg/contents:0': image_data}) # n, m, 3

    # print(type(feature_maps))
    # print(feature_maps.shape)

    feature_maps = np.reshape(feature_maps, [feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]])
    plot(feature_maps, featureMapIdx, figure_title)

    print('GrayScale: ', isGrayScaleNorm)
    print ('Layer ID: %d' % layerID)
    if featureMapIdx > 0:
    	print('Feature Map Index: %d' % featureMapIdx)
    	
    print('\n')


    		
# Main

# Print ops:
# print_ops(sess)

imgID = 37
layerID = 1
featureMapIdx = 18
isGrayScaleNorm = True

QF = 22
figureID = 1
visualize_image(imgID, QF, layerID, figureID, True, isGrayScaleNorm)


QF = 110
figureID = figureID + 1
visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm)


figureID = figureID + 1
visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm, featureMapIdx)


QF = 10
figureID = figureID + 1
visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm)

figureID = figureID + 1
visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm, featureMapIdx)


# Show plt at the end
plt.show()

