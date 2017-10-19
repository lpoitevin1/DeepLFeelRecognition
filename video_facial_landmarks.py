#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# packages
from imutils.video import VideoStream
from imutils import face_utils
from calc_points import *
import datetime
import argparse
import imutils
import time
import dlib
import cv2

def webcam (detector,predictor,vs):
	# initialiser dlib face detector et lis le fichier shape predictor
	''' print("Lecture du fichier : facial predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") '''

	# demarre la webcam et integre le flux a une variable
	''' print("Demarrage de la webcam...")
	vs = VideoStream(-1).start()
	time.sleep(2.0) '''

	# boucle sur chaque image du flux de la webcam
	while True:
		key = cv2.waitKey(1) & 0xFF

		# lis chaque image de la webcam, limite la taille de l'image
		# a 400 pixels max et convertie ça en gray, pour la couleur
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
		# affiche le rectangle au centre de l'image
		rects = detector(gray, 0)
		a = 125
		b = 75
		c = 150
		d = 150
		cv2.rectangle(frame,(a,b),(a+c,b+d),(0,255,0),2)

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
			for (x1, y1) in shape:
				cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)


		# affiche l'image 
		cv2.imshow("Webcam", frame)
 			
		# si l'utilisateur presse "a", le programme enregistre ses points
		# caracteristique et se termine
		if key == ord("a"):
			print ("Enregistrement des points...")
			'''pointr=[1,17,49,55,28,34,43,46,34,36,23,27]
			while i<len(pointr):
				point.append(calc_points(shape,pointr[i],pointr[i+1],w))
				i += 2'''
			print ("Fin de la fonction webcam..")
			return calc_points(shape,w)				

		# si l'utilisateur presse "q", le programme prend fin
		if key == ord("q"):
			print ("Fin de la fonction webcam...")
			break
	
	# ferme la fenetre de la webcam
	cv2.destroyAllWindows()
