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

MODEL_PATH = '../alexnet_model'

# YUV Path
PATH_TO_RECONS = '/Volumes/MULTICOMHD2/set_yuv/Seq-RECONS/'

# JPEG Path
path_to_valid_images    = '/Volumes/MULTICOMHD2/validation_original/';
path_to_valid_QF_images = '/Volumes/MULTICOMHD2/validation_generated_QF/';

#mean of imagenet dataset in BGR
#imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
# mean in rgb 
imagenet_mean = np.array([ 124. , 117., 104.], dtype=np.float32)

#读取训练好的Inception-v3模型来创建graph
def create_graph():
  # the class that's been created from the textual definition in graph.proto
  #with tf.gfile.FastGFile('./inception_model/inception_v3_2016_08_28_frozen.pb', 'rb') as f:  
    with tf.gfile.FastGFile(MODEL_PATH + '/alexNet_graph.pb', 'rb') as f:  
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')



def print_all_ops_in_graph():
  with tf.Session() as sess:
    for op in tf.get_default_graph().get_operations():
        print(str(op.name))


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
		image_data = cv2.imread(image) # use open cv to read the image


	# AlexNet requirements
	# Convert image to float32 and resize to (227x227)
	image_data = cv2.resize(image_data.astype(np.float32), (227, 227))

	
	# Subtract the ImageNet mean
	image_data -= imagenet_mean

	# Reshape as needed to feed into model
	image_data = image_data.reshape((1, 227, 227, 3))

	# cv2.imshow('image',image_data[0,:,:,: ])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return image_data, figure_title


def YUV2RGB( yuv ):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return rgb

def RGB2YUV( rgb ):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv


def plot(feature_maps, featureMapIdx, figure_title, is_pool):

	if not is_pool:
		K     = feature_maps.shape[3]
	else:
		K     = feature_maps.shape[2]

	# K     = 32
	nRows = K//8
	nCols = K//nRows
	if featureMapIdx < 0:
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols, figsize=(10, 5), sharex=True, sharey=True) # sharex: makes you zoom in the entire plot
		plt.figure(figureID)
		for irow, row in enumerate(ax):
			for icol, col in enumerate(row):
				idx = irow + icol * nRows
				if idx >= K:
					continue

				if is_pool:
					m = feature_maps[:, :, idx]
				else:
					m = feature_maps[:, :, :, idx]

				# the values of feature maps are between 0 and 1; Scale it between 0 to 255
				m = (m * 255).astype(np.uint8)


				if isGrayScaleNorm:
					# Convert RGB to YUV
					if not is_pool:
						m = RGB2YUV(m)
						Y = m[:, :, 0]
					else:
						Y = m
					col.imshow(Y, cmap='gray', vmin=0.0, vmax=255.0)
				else:
					col.imshow(m)
				col.axis('off')
	else:
		plt.figure(figureID)

		if is_pool:
			m = feature_maps[:, :, featureMapIdx]
		else:
			m = feature_maps[:, :, :, featureMapIdx]

		# the values of feature maps are between 0 and 1; Scale it between 0 to 255
		m = (m * 255).astype(np.uint8)
		if isGrayScaleNorm:
			# Convert RGB to YUV
			if not is_pool:
				m = RGB2YUV(m)
				Y = m[:, :, 0]
			else:
				Y = m
			plt.imshow(Y, cmap='gray', vmin=0.0, vmax=255.0)
		else:
			plt.imshow(m)
			plt.axis('off')
		figure_title = str(featureMapIdx) + ')' + figure_title
	plt.suptitle(figure_title, fontsize=16)
	if K <= 32:
		plt.subplots_adjust(wspace=0.1, hspace=0.05)
	else:
		plt.subplots_adjust(wspace=0, hspace=0)


# Visualizes feature map of a specific image in the validation set
def visualize_image(imgID, QF, layerID = 1, figureID = 1, isCast = True, isGrayScaleNorm = False, is_pool = False, featureMapIdx = -1):


    # Create graph:
    create_graph()
    sess = tf.Session()
	
    # Inception-v3: last layer is output as softmax
    conv1_tensor = sess.graph.get_tensor_by_name('conv' + str(layerID) + '/weights:0')
    if is_pool:
    	conv1_tensor = sess.graph.get_tensor_by_name('pool' + str(layerID) + ':0')

    # Print all ops
    # print_all_ops_in_graph()

    # Title of the figure
    figure_title = ''

    # Get image data
    image_data, figure_title = get_image_data(imgID, QF, isCast) # (1, 227, 227, 3)


    # RGB input:
    feature_maps = sess.run(conv1_tensor, {'Placeholder:0': image_data, 'Placeholder_1:0': 1}) # (11, 11, 3, 96) n, m, 3


    # print(type(feature_maps))
    print(feature_maps.shape) # (11, 11, 3, 96)

    if is_pool:
    	feature_maps = np.reshape(feature_maps, [feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]])
    	# feature_maps = np.moveaxis(feature_maps, 0, -2)

    	# print(np.moveaxis(feature_maps, 0, -2).shape)
    	# print(np.swapaxes(feature_maps, 0, -1).shape)
    	# exit(0)
    plot(feature_maps, featureMapIdx, figure_title, is_pool)

    print('GrayScale: ', isGrayScaleNorm)
    print ('Layer ID: %d' % layerID)
    if featureMapIdx > 0:
    	print('Feature Map Index: %d' % featureMapIdx)
    	
    print('\n')


    		
# Main


imgID = 37
layerID = 1
featureMapIdx = 44
isGrayScaleNorm = True
is_pool  = True

QF = 0
figureID = 1
visualize_image(imgID, QF, layerID, figureID, True, isGrayScaleNorm, is_pool, featureMapIdx)


QF = 110
figureID = figureID + 1
visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm, is_pool, featureMapIdx)


# figureID = figureID + 1
# visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm, featureMapIdx)


# QF = 10
# figureID = figureID + 1
# visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm)

# figureID = figureID + 1
# visualize_image(imgID, QF, layerID, figureID, False, isGrayScaleNorm, featureMapIdx)


# Show plt at the end
plt.show()

