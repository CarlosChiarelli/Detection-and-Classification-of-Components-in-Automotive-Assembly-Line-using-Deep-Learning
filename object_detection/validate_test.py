import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
import xml.etree.ElementTree as ET

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1, 6) ]
TEST_IMAGE_PATHS = glob.glob("test_images/*.jpg")
print(TEST_IMAGE_PATHS)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def intersection_over_union(ymin_gt, xmin_gt, ymax_gt, xmax_gt, ymin, xmin, ymax, xmax):
  xA = max(xmin_gt, xmin)
  yA = max(ymin_gt, ymin)
  xB = min(xmax_gt, xmax)
  yB = min(ymax_gt, ymax)
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)#plus one because the pixel count

  # compute the area of both the prediction and ground-truth
  # rectangles
  boxArea_gt = (xmax_gt - xmin_gt + 1) * (ymax_gt - ymin_gt + 1)
  boxArea_predict = (xmax - xmin + 1) * (ymax - ymin + 1)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxArea_gt + boxArea_predict - interArea)

  # return the intersection over union value
  return iou

def getid(class_name, category_index):
  for idx, name in category_index.items():
    if class_name == name['name']:
      return idx
  return -1

#EVALUATE IOU, PRECISION AND RECALL FROM MODEL
TP = [0] * len(category_index) # TP: are the Bounding Boxes (BB) that the intersection over union (IoU) with the ground truth (GT) is above 0.5
FP = [0] * len(category_index) # FP: BB that the IoU with GT is below 0.5 also the BB that have IoU with a GT that has already been detected. Also detections that does not go on image.
FN = [0] * len(category_index) # FN: those images were the method failed to produce a BB. Images without BB
 
FP_names = []
FN_names = []

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  min_score_thresh=.9
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'], #ymin, xmin, ymax, xmax
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      min_score_thresh=min_score_thresh,
      line_thickness=8)
  #plt.figure(figsize=IMAGE_SIZE)
  #plt.imshow(image_np)

  pil_image = Image.fromarray(image_np.astype(dtype=np.uint8))
  im_width, im_height = pil_image.size #get size informations of image
  path, filename = os.path.split(image.filename) #split the name of the image to search by xml file
  #print("largura e altura: " + str(im_width) + " " + str(im_height) + "\n")
  tree = ET.parse(image.filename[:-4]+".xml")
  root = tree.getroot()
  GT_objects = root.findall('object')
  for i in range(len(output_dict['detection_classes'])):
    if output_dict['detection_scores'][i] > min_score_thresh:
      #print("\nclasse verificada: " + str(category_index[output_dict['detection_classes'][i]]['name']) + "\n")
      obj_class = category_index[output_dict['detection_classes'][i]]['name']
      ymin, xmin, ymax, xmax = int(output_dict['detection_boxes'][i][0] * im_height), int(output_dict['detection_boxes'][i][1] * im_width), int(output_dict['detection_boxes'][i][2] * im_height), int(output_dict['detection_boxes'][i][3] * im_width)
      #print(ymin, xmin, ymax, xmax) 
      #get box and compativel class from xml
      xmin_gt, ymin_gt, xmax_gt, ymax_gt = -1, -1, -1, -1
      for member in GT_objects:
        #print(member[0].text)
        if member[0].text == obj_class:
          member_class = member
          xmin_gt, ymin_gt, xmax_gt, ymax_gt = int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text)
      #print(ymin_gt, xmin_gt, ymax_gt, xmax_gt)
      if xmin_gt != -1: #Class exists on image
        iou = intersection_over_union(ymin_gt, xmin_gt, ymax_gt, xmax_gt, ymin, xmin, ymax, xmax)
        #print(iou)
        if iou > 0.5:
          TP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] = TP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] + 1
          GT_objects.remove(member_class) #if object was detected, then it can not be found again
        else:
          FP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] = FP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] + 1
          FP_names.append(filename)
      else: #Class predicted does not exist on image
        FP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] = FP[int(category_index[output_dict['detection_classes'][i]]['id'])-1] + 1
        FP_names.append(filename)
  for member in GT_objects: # objects that was not detected
    idx = getid(member[0].text, category_index)
    FN[idx-1] = FN[idx-1] + 1
    FN_names.append(filename)
  #pil_image = Image.fromarray(image_np.astype(dtype=np.uint8))
  with tf.gfile.Open(image_path[:-4]+"_predict.png", mode='w') as f:
    pil_image.save(f, 'PNG')  

print("True Positives: " + str(TP))
print("\nFalse Positives: " + str(FP))
print(FP_names)
print("\nFalse Negatives: " + str(FN))
print(FN_names)

























