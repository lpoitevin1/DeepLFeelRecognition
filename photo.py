#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# import the necessary packages
from imutils import face_utils
from calc_points import *
import numpy as np
import argparse
import imutils
import dlib
import cv2

def photo (image,detector,predictor):
    
    # load the input image, resize it, and convert it to grayscale
    #image = cv2.imread("im")

    image = imutils.resize(image, width=400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        '''pointr=[1,17,49,55,28,34,43,46,34,36,23,27]
        i = 0
        while i<len(pointr):
            point.append(calc_points(shape,pointr[i],pointr[i+1],w))
            i += 2 '''

        return calc_points(shape,w)