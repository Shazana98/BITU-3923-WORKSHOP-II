######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

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
from time import sleep
#from object_detection.utils import label_map_util

#import socket to pass value from tensorflow to ev3
import socket
import pickle
import threading

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `cat`.
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



# Initialize webcam feed
print("[INFO] starting video stream...")
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("169.254.70.137", 3000))



while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    #print(frame)
    #print(ret)

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
        line_thickness=7,
        min_score_thresh=0.60)

    threshold = 0.5

    #print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])

    objects = []
    for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > threshold:
            object_dict[(category_index.get(value)).get('name').encode('utf8')] = scores[0,index]
            objects.append(object_dict)
    print (objects)

    if objects:
        exist = 1
        if exist != '0':
            s.send(pickle.dumps(exist))
            print(exist)

    
        

    #name, score = object_dict.items()[0]
    #print ("The value: " + str(name))

    

    

    
    #print(len(np.where(scores[0] > threshold)[0]/num_detections[0])
        
        
   #import subprocess
   #if [category_index.get(i) for i in classes[0]] ==category_index.get('Cat'):
    # subprocess.call(['python', 'main.py'])
  # else:
    # subprocess.call(['python', 'Object_detection_webcam_updated.py'])

        
   
            
    #do nothing
    
    #if scores > 1:
    #if detection_scores > 1:
    #if detection_classes > 1:
    #final_score = np.squeeze(scores)
    #count = 0
    #for i in range(100):
    #if scores is None or final_score[i] > 0.5:
            

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    sleep(1.0)

    
    #final_scores = np.squeeze(scores)
    #count = 0
    #for i in range(100):
    #def GetClassName(data):
        #for cl in data:
            #return cl['name']
        
    #if detection_classes != '0':
        #data = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0,index]
        #verified = str(detection_classes[0])
        #s.send(pickle.dumps(verified))
        
 

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
print("[INFO] cleaning up...")
video.release()
cv2.destroyAllWindows()

