#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import datetime

def capture_cam (i,detector,predictor,vs) :
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    #print("Lecture du fichier : facial predictor...")
    ''' detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") '''

    # initialize the video stream and allow the cammera sensor to warmup
    ''' print("Demarrage de la webcam...")
    vs = VideoStream(-1).start()
    time.sleep(2.0) ''' 

	# loop over the frames from the video stream
    while True:
        key = cv2.waitKey(1) & 0xFF
		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        a = 125
        b = 75
        c = 150
        d = 150
        cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)

        # show the frame
        cv2.imshow("Webcam", frame)

        if key == ord("z"):
            print ("Capture de la webcam effectué")
            camera_capture = gray
            cv2.imwrite("images/image"+str(i)+".png",camera_capture)
            break

		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print ("Fin de la fonction capture_cam...")
            break

	# do a bit of cleanup
    cv2.destroyAllWindows()

