import tensorflow as tf
import cv2
import numpy as np
# Load the saved model
model = tf.saved_model.load('saved_model')

image = cv2.imread("data/dataset/test/auto63_jpg.rf.eb25b79864e340f69ac05f83188adf7d.jpg")

image_h,image_w ,_= image.shape
input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)

# Perform inferenc
detections = model.signatures['serving_default'](input_tensor)
num_detections = int(detections.pop('num_detections'))

detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

detections = {key: value for key, value in detections.items() if key in key_of_interest}

box = ((detections['detection_boxes'])[0])
if(detections['detection_scores'][0]> 0.7):
    y1abs, x1abs = int(box[0] * image_h), int(box[1] * image_w)
    y2abs, x2abs = int(box[2] * image_h), int(box[3] * image_w)
    cv2.rectangle(image, (x1abs, y1abs), (x2abs,y2abs), (10, 255, 0), 2)
    cv2.imwrite("ma.jpg" ,image)

