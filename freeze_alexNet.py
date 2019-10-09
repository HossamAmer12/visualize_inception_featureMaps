import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# Video capture and convert rgb 
import cv2
import time

from video_capture import VideoCaptureYUV
import cv2
import math
import glob
import sys
# Excel sheet stuff:
import xlrd
from xlwt import *
from xlutils.copy import copy


from alexnet import AlexNet
from caffe_classes import class_names

# Path stuff
from pathlib import Path

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


final_tensor_name = 'softmax'
 # path_to_excel = PATH_TO_DISC + '/Accuracy Record ILSVRC2012 Top 1 - HEVC.xls'

QP = []
QP.append(51)
qp = QP[0]
for i in range(50, 0, -2):
    QP.append(i)
QP.append(0)

# path to class file names 

MAIN_PATH    = '/Volumes/MULTICOM105/103_HA/MULTICOM103/set_yuv'
image_dir    = os.path.join(MAIN_PATH, 'pics')
# output_path  = os.path.join(MAIN_PATH, 'Seq-RECONS-ffmpeg')
output_path  = os.path.join(MAIN_PATH, 'Seq-RECONS')
path_to_textfile = os.path.join(MAIN_PATH, 'Gen/Seq-Stats')


#mean of imagenet dataset in BGR
#imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
# mean in rgb 
imagenet_mean = np.array([ 124. , 117., 104.], dtype=np.float32)


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def freeze_graph(epoch, accuracy_so_far, sess, graph, glob_step):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """

    # We precise the file fullname of our freezed graph
    accuracy_so_far     = accuracy_so_far * 100
    acc_str             = str(int(100 * accuracy_so_far))
    actual_index        = glob_step
    idx_str             = str(actual_index)
    model_dir_name      = 'model-' + idx_str + '-' + acc_str
    absolute_model_dir = os.path.join('./frozen/', model_dir_name)
    ensure_dir_exists(absolute_model_dir)
    absolute_output_graph_dir = os.path.join(absolute_model_dir, "alexNet_graph.pb")
    my_file_exist        = Path(absolute_output_graph_dir)

     # if directory already exists, just return from the method
    if my_file_exist.is_file():
      print('----------------')
      print("****[Avoid Double Bass] FROZEN Model ALREADY exists in path: %s" % absolute_output_graph_dir)
      print('----------------')
      return;

    # Write it this way because you need the accuracy results when you are predicting:
    output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [final_tensor_name, 'fc8/fc8', 'Placeholder'])

    with gfile.FastGFile(absolute_output_graph_dir, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    print('----------------')
    print("****[Freeze Model] Wrote %d ops in the final graph at %s" % (len(output_graph_def.node), absolute_model_dir))
    print('----------------')

    return output_graph_def

def construct_all_list(classNo):
    print('Constructing all list for Class-%d...' % classNo)
    class_filename       = 'class' + str(classNo) + '.txt'
    path_to_class_file   = os.path.join(PATH_TO_DISC + '\\classes', class_filename)
    all_list  = [x.split('\n')[0] for x in open(path_to_class_file).readlines()]
    return all_list



def mak_list():
    print('list of images is generated')
    all_list =  [x.split('\n')[0] for x in open(PATH_TO_FILE_NAMES).readlines() ]
    return all_list 


def print_all_ops_in_graph(sess):
    for op in sess.graph.get_operations():
        print(str(op.name))
    # for op in tf.get_default_graph().get_operations():

     
# def print_all_tensors_in_graph():
#   with tf.Session() as sess:
#     tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
#     print(tensor_names)

def print_all_tensors_in_graph(sess):
    tensor_names = [n.name for n in sess.graph.as_graph_def().node]
    print(tensor_names)


def rank_estimate(probs , label_list , idx , sheet , col_idx  ):
    predictions = np.squeeze(probs)
    N = - 1000
    rank = -1 
    predictions = np.squeeze(predictions)
    top_5 = predictions.argsort()[N:][::-1]
    for rank, node_id in enumerate(top_5):
        human_string = class_names[node_id]
        score = predictions[node_id]
        if(label_list[idx] == human_string):
            row = idx            
            current_rank =  1 + rank 
            style = XFStyle()
            style.num_format_str = 'general'
            sheet.write(row , col_idx , current_rank , style)            
            sheet.write(row , 1 + col_idx , score.item() , style)

            print(current_rank, human_string)
            break 
    return current_rank
    


def readAndpredictloop():
        
    ## reset default graph 
    tf.reset_default_graph()
    
    #placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    
    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])
    
    #define activation of last layer as score
    score = model.fc8    
    
    #create op to calculate softmax 
    softmax = tf.nn.softmax(score, name=final_tensor_name)
    
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
      
            
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Load the pretrained weights into the model
        model.load_initial_weights(sess)

        epoch = validation_accuracy = i = 1

        #print_all_ops_in_graph(sess)
        print('---------------')
        print_all_tensors_in_graph(sess)
        print('---------------')

        # conv1_tensor = sess.graph.get_tensor_by_name('conv' + str(1) + '/biases:0')
        # print(sess.run(conv1_tensor))
        # exit(0)
        freeze_graph(epoch, validation_accuracy, sess, sess.graph, i)
        
        

readAndpredictloop()
print('Freeze graph is done!')