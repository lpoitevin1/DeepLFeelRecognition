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
from threading import Thread


import keras
from keras.models import load_model


def main() :
    # charge et compile le modele
    model = load_model("model.h5")
    print("Le modele est chargé")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
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
            tab = []
            shape_a = []
            shape_b = []
            for (x1, y1) in shape:
                cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)
                shape_a.append(x1)
                shape_b.append(y1)
            tab = np.concatenate((shape_a, shape_b))
            print (tab)
            print ("Nombre de points : ", len(tab))

            # renvoie le resultat des points apres analyse par le modele du reseau (error)
            res = model.predict (np.array([tab]))
            print(res)
            

		# affiche l'image 
        cv2.imshow("Webcam", frame)		

		# si l'utilisateur presse "q", le programme prend fin
        if key == ord("q") :
            print ("Fin de la fonction webcam...")
            break
	
	# ferme la fenetre de la webcam
    cv2.destroyAllWindows()

    ''' 
    ramener le code de la webcam ici, predire les resultats a chaque image (voir image_learning)
    en lui donnant les points de shape (mis dasn un tableau de 136 points)
    faire une fonction enterpretant le resultat et afficher ce resultat (happiness, surprise etc..)
    a cote de la face sur la webcam
    '''
if __name__ == "__main__":
    main()