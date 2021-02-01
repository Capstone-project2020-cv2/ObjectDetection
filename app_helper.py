
import tensorflow as tf
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import mimetypes
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import imageio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import h5py

def get_image(image_path,img_name):
    MODEL_PATH='./weights'
    LB_PATH='./weights'
    output_path = './static/detections/'
    MODEL_PATH = os.path.sep.join([MODEL_PATH, "detector.h5"])
    LB_PATH = os.path.sep.join([LB_PATH, "lb.pickle"])

    print("[INFO] loading object detector...")
    model = load_model(MODEL_PATH)
    lb = pickle.loads(open(LB_PATH, 'rb').read())
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # determine the class label with the largest predicted
    # probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # load the input image (in OpenCV format), resize it such that it
    # fits on our screen, and grab its dimensions
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_TRIPLEX,
                0.65, (0, 0, 255), 2)
    img=cv2.rectangle(image, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)

    # show the output image
    # probability
    cv2.imwrite(output_path + '{}'.format(img_name), img)
    #cv2.imshow(image)
    #cv2.waitKey(0)
    print('output saved to: {}'.format(output_path + '{}'.format(img_name)))

