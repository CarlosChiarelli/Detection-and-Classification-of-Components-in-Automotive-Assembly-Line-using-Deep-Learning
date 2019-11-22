######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test3.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','object-detection.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

font = cv2.FONT_HERSHEY_SIMPLEX
no_detection_flag = 0
frame_count = 0
frame_count_final = 0
class_count = np.array([0]*NUM_CLASSES)
class_count_final = np.array([0]*NUM_CLASSES)
caliper_type = -1
disc_type = -1
# Apply detection to each frame and then save
ret, frame = video.read()

w_adjust_size = int((frame.shape[1]-frame.shape[0])/2)
h_adjust_size = 0

#recortar video
x1, y1 = w_adjust_size, h_adjust_size
x2, y2 = frame.shape[1]-w_adjust_size, frame.shape[0]-h_adjust_size
frame = frame[y1:y2, x1:x2]

height,width,layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
while ret:
    frame_count = frame_count + 1
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    #ret, frame = video.read()
    
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=16,
        min_score_thresh=0.9)
    
    score_list = np.squeeze(scores)
    class_list = np.squeeze(classes).astype(np.int32)
    detection_flag = 0
    for i in range(len(score_list)):
        if score_list[i] < 0.9:
            break
        else:
            detection_flag = 1
            class_count[class_list[i]-1] = class_count[class_list[i]-1] + 1
    if not detection_flag: #none class found
        no_detection_flag = no_detection_flag + 1

    k = cv2.waitKey(1)
    # Press 'esc' to quit
    if k == 27:
        break
    # Press 'r' to reset estatistics
    #if k == ord('r'):
    # 100 sec without any detection
    if no_detection_flag > 0 and no_detection_flag < 100 and detection_flag:    
        no_detection_flag = 0
    elif no_detection_flag == 100:
        frame_count_final = frame_count - 100 #discard last 100 count
        frame_count = 0
        mx = 0
        caliper_type = -1
        disc_type = -1
        for i in range(3):
            if class_count[i] > mx:
                mx = class_count[i]
                disc_type = i+1
        mx = 0
        for i in range(3,6):
            if class_count[i] > mx:
                mx = class_count[i]
                caliper_type = i-2
        class_count_final = class_count
        class_count = np.array([0]*NUM_CLASSES)
    elif no_detection_flag > 100 and detection_flag:    
        no_detection_flag = 0
        frame_count = 1
    
 # All the results have been drawn on the frame, so it's time to display it.
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.putText(frame,"Frames: {0:.2f}".format(frame_count),(30,50),font,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,"Count: {}".format(class_count),(30,100),font,1,(0,0,255),2,cv2.LINE_AA)
    #cv2.putText(frame,"Flag: {}".format(no_detection_flag),(30,150),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Caliper type: {}".format(caliper_type),(30,height-200),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Disc type: {}".format(disc_type),(30,height-150),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Frames: {}".format(frame_count_final),(30,height-100),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"Final Count: {}".format(class_count_final),(30,height-50),font,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Video', frame)
    cv2.resizeWindow('Video', 1000,800)
    out_video.write(frame)
    ret, frame = video.read()
    #recortar video
    x1, y1 = w_adjust_size, h_adjust_size
    x2, y2 = frame.shape[1]-w_adjust_size, frame.shape[0]-h_adjust_size
    frame = frame[y1:y2, x1:x2]

# Clean up
out_video.release()
video.release()
cv2.destroyAllWindows()
