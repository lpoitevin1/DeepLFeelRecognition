#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# packages
from imutils.video import VideoStream
from imutils import face_utils
from video_facial_landmarks import *
from photo import *
from comparaison2 import *
from capture_cam import *
import ipdb
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os

ok = True
# initialiser dlib face detector et lis le fichier shape predictor
print("Lecture du fichier : facial predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# demarre la webcam et integre le flux a une variable
print("Demarrage de la webcam...")
vs = VideoStream(-1).start()
time.sleep(2.0)
while True :
    
    # boucle pour prendre une photo si l'utilisateur n'en a pas 
    rep = input ("Avez vous deja une photo dans la base de données ? (o/n) ")
    nb_image = len(os.listdir("images/test/"))
    if (rep == "n" or rep == "N") :
        ccam = True
        while (ccam) :
            print (nb_image)
            capture_cam(nb_image+1,detector,predictor,vs)
            img = cv2.imread("images/test/image" + str(nb_image+1) + ".png")
            img = imutils.resize(img, width=400)
            print ("Creation de l'image : image" + str(nb_image+1) + " terminé !")
            ccam = False
            '''cv2.imshow("capture_cam",img)
            rep2 = input ("Satisfait ? (o/n) ")
            if (rep2 == "o" or rep2 == "O") :
            ccam = False '''
                    

        
    nb_image = len(os.listdir("images/test/"))
    i = 1
    liste = []
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print ("Debut de l'analyse des images 132145")
    print ("Lecture du fichier : shape predictor...")
    detector2 = dlib.get_frontal_face_detector()
    predictor2 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while (i<(nb_image+1)) :
        image = cv2.imread("images/test/image" + str(i) +".png")
        print ("Analyse de l'image n° ",i)
        pointcomp = photo (image,detector2,predictor2)
        liste.append(pointcomp)
        i += 1

    ''' image = cv2.imread("images/image0.jpg")
    photo (image,point1,detector,predictor)
    image = cv2.imread("images/image1.jpg")
    photo (image,point2,detector,predictor)
    liste.append(point1)
    liste.append(point2) '''

    print ("Debut de la fonction webcam")
    point = webcam(detector,predictor,vs)

    i = comparaison2(point,liste)
    print ("Tu es sur l'image numero ",i)
    cv2.destroyAllWindows()

    rep3 = input("Voulez vous recommencez ? (o/n) : ")
    if (rep3 == "n" or rep3 == "N") :
        break