
# Import packages
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


### Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.8, num_test_images=10):
  tutte = 0
  giuste = 0
  sbagliate = 0
  non_detect = 0
  object_name = ""
  # Grab filenames of all images in test folder
  images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')


  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)


  # Randomly select test images
  images_to_test = random.sample(images, num_test_images)
  print(len(images_to_test))

  # Loop over every image and perform detection
  for image_path in images_to_test:
      tutte += 1
      xml_path = image_path.replace(".jpg",".xml")
      with open(xml_path, 'r') as f:
        data = f.read()
      bs_data = BeautifulSoup(data, 'xml')
      for tag in bs_data.find_all('name'):
        vero = (str(tag).split(">")[1].split("<")[0])

      # Load image and resize to expected shape [1xHxWx3]
      image = cv2.imread(image_path)
      imH, imW, _ = image.shape 
      image_resized = cv2.resize(image, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      input_mean = np.mean(input_data)
      input_std = np.std(input_data)
      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()
      
      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
      detections = []


      for i in range(len(scores)):
        if (scores[i] *100) > min_conf*100:
           # Get bounding box coordinates and draw box
          # Interpreter can rn coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
          ymin = int(max(1,(boxes[i][0] * imH)))
          xmin = int(max(1,(boxes[i][1] * imW)))
          ymax = int(min(imH,(boxes[i][2] * imH)))
          xmax = int(min(imW,(boxes[i][3] * imW)))
          
          cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

          # Draw label
          object_name = labels[int(classes[i])] 
          label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
          label_ymin = max(ymin, labelSize[1] + 10) 
          cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
          cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
          detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
          object_name = labels[int(classes[i])] 
          save = image_path.replace(imgpath,"")
          cv2.imwrite(imgpath + "/out/" + save ,image)

      if (object_name == vero):
        giuste += 1
      else:
        print(image_path)
        print("identificato " , object_name)
        print("reale ", vero)
        sbagliate += 1


  print(giuste)
  print(sbagliate)
  print((giuste+sbagliate) == tutte)
  print(tutte)

  return ((giuste *100)/ tutte )

# Set up variables for running user's model
PATH_TO_IMAGES='data/dataset/test/'   # Path to test images folder
PATH_TO_MODEL='saved_model/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS='/home/michele/Desktop/stella_mb2/stella_dataset/labels.txt'   # Path to labelmap.txt file
min_conf_threshold=0.4  # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 115   # Number of images to run detection on


# Run inferencing function!
accuracy = tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

print(accuracy)