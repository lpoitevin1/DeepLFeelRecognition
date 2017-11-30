from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from imutils.video import VideoStream
from imutils import face_utils
from image_learning import *

import os
import urllib

import math
import numpy as np
import cv2
import imutils
import dlib
import random
import time
import argparse
from threading import Thread


import keras
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=-1,
	help="Choisir une video exploitable par le programme. Par defaut, utilise la webcam")
args = vars(ap.parse_args())

def lecture_tab (res) :
    n = 0
    test = 0
    for i in range (0, len(res[0])) :
        if np.any(res [0][i] > test) :
            test = res[0][i]
            n = i
    #tab = ['Surprise','Sadness','Happiness','Fear','Digust','Anger']
    tab = ['Happiness','Anger']
    return tab[n]


def main() :
    # charge et compile le modele
    model = load_model("model.h5")
    print("Le modele est chargé")
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  
    print("Le modele est compilé") 
    
    # initialiser dlib face detector et lis le fichier shape predictor
    print("Lecture du fichier : facial predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



    # demarre la webcam et integre le flux a une variable
    print("Demarrage de la webcam...")
    vs = VideoStream(-1).start()
    time.sleep(2.0)


    
	# boucle sur chaque image du flux de la webcam
    while True:
        key = cv2.waitKey(1) & 0xFF

		# lis chaque image de la webcam, limite la taille de l'image
		# a 400 pixels max et convertie ça en gray, pour la couleur
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)


		# boucle sur chaque image qui detecte le visage
        for (i, rect) in enumerate (rects):
			# detecte le visage et convertie les coordonnées des 
			# points caracteristique pour les inserer dans un 
			# tableau
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

			
			#affiche un cadre autour des visages detecté 
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
			# affiche le numero du visage detecté
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
			# boucle sur les coordonnées de chaque point du visage et les 
			# affiche sur l'image
            '''
            tab = []
            shape_a = []
            shape_b = []
            '''
            for (x1, y1) in shape:
                cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)
            '''
                shape_a.append(x1)
                shape_b.append(y1)
            tab = np.concatenate((shape_a, shape_b))
            print (tab)
            print ("Nombre de points : ", len(tab))
            '''

            a = []
            a = selection_point (shape,a,w)

            '''
            pt_a, pt_b = point_central_bouche(shape)
            a1, b1 = shape[49]
            a2, b2 = shape[55]
            a.append(sqrt(((pt_a-a1)*(pt_a-a1))+((pt_b-b1)*(pt_b-b1)))/w)
            a.append(sqrt(((pt_a-a2)*(pt_a-a2))+((pt_b-b2)*(pt_b-b2)))/w)
            '''

            # renvoie le resultat des points apres analyse par le modele du reseau (error)
            res = model.predict (np.array([a]))
            print (a)
            print(res)

            emotion = lecture_tab (res)
            print (emotion)

            cv2.putText(frame, emotion, (x + int(w/4), y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)



		# affiche l'image 
        cv2.imshow("Webcam", frame)		

		# si l'utilisateur presse "q", le programme prend fin
        if key == ord("q") :
            print ("Fin de la fonction webcam...")
            break
	
	# ferme la fenetre de la webcam
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()