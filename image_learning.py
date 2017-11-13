from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from imutils.video import VideoStream
from imutils import face_utils

import os
import urllib

import math
from math import sqrt
import numpy as np
import cv2
import imutils
import dlib
import random
import time
from threading import Thread

import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Reseau de neurone sur les points des images
print ("Debut de l'analyse des images")
print ("Lecture du fichier : shape predictor...")
detector2 = dlib.get_frontal_face_detector()
predictor2 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
debut = time.time()

# Choisie aleatoirement une image entre 6 emotions et recupere les
# points caracteristiques + associe a une emotion 
def point_image (detector2,predictor2,n) :
    j = random.randrange(1,7,1)
    #print ("Le dossier sera le numero ",j)
    x = []
    shape_a = []
    shape_b = []
    if (j == 1) :
        max = len(os.listdir("images/anger_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/anger_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([0,0,0,0,0,1])
    elif (j == 2) :
        max = len(os.listdir("images/disgust_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/disgust_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([0,0,0,0,1,0])
    elif (j == 3) :
        max = len(os.listdir("images/fear_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/fear_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([0,0,0,1,0,0])
    elif (j == 4) :
        max = len(os.listdir("images/happiness_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/happiness_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([0,0,1,0,0,0])
    elif (j == 5) :
        max = len(os.listdir("images/sadness_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/sadness_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([0,1,0,0,0,0]) 
    else :
        max = len(os.listdir("images/surprise_photo/"))
        k = random.randrange(1,max,1)
        img = cv2.imread("images/surprise_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
        y = np.array([1,0,0,0,0,0])          
    i = 0
    pt_a, pt_b = point_central(shape)
    while (i<len(shape)):
        (a,b) = shape[i]
        # teste le reseau en renvoyant des vecteurs distance entre le point i et le point central
        # normalisé par rapport a la distance w du carre autour du visage
        #shape_a.append(((a+pt_a)/2)/w) 
        #shape_b.append(((b+pt_b)/2)/h)
        x.append(sqrt(((pt_a-a)*(pt_a-a))+((pt_b-b)*(pt_b-b)))/w)
        i += 1
    #x = np.concatenate((shape_a, shape_b))
    #print("Anayse de l'image n°", k," dans le y ", y , " . Nombre de point : ", len(x))
    #affiche une image toute les 10 images pour verifier les points
    """
    if (n == 10) :
        pt_a = int(pt_a)
        pt_b = int(pt_b)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
            (c, d, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img, (c, d), (c + w, d + h), (255, 0, 0), 2)

            cv2.circle(img, (pt_a, pt_b), 1, (0, 255, 0), 5)

            for (xx, yy) in shape:
                cv2.circle(img, (xx, yy), 1, (0, 0, 255), -1)

        cv2.imshow("Test image", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return x, y

def point_central(shape) :
    x = 0
    y = 0
    i = 0
    while i < len(shape) :
        (a,b) = shape[i]
        x += a
        y += b
        i += 1
    x = x / len(shape)
    y = y / len(shape)
    return x, y


def nb_random():
    x = []
    j = random.randrange(1,3,1)
    if (j == 1):
        for i in range (0,136) :
            x.append(random.randrange (1,121,1))
        y = [0,1]
    else :
        for i in range (0,136) :
            x.append(random.randrange (121,141,1))
        y = [1,0]
        
    print ("Tableau de ", len(x)," nombres random crée : ", x," qui est classé dans le y = ",y)
    return x, y       



# Boucle n fois pour recuperer les données de n images
def next_batch(n):
    x = np.zeros( shape=(n,68), dtype=np.float32)
    y = np.zeros( shape=(n,6), dtype=np.float32)
    for i in range(0, n):
        x[i],y[i] = point_image(detector2,predictor2,i)
        """x[i],y[i] = nb_random();"""
    print ("Fin de la creation du batch de ", n," elements")
    return x,y
        
# Creation, parametrage et utilisation du reseau
def main():

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 2-dimensional vectors.
	#The last output has to be the number of class
    model.add(Dense(68, activation='relu', input_dim=68))
    model.add(Dropout(0,5))
    model.add(Dense(136, activation='relu'))
    model.add(Dropout(0,5))
    model.add(Dense(136, activation='relu'))
    model.add(Dropout(0,5))
    model.add(Dense(68, activation='relu'))
    model.add(Dropout(0,5))
    model.add(Dense(6, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
		  
    training_epochs = 20
    i = 1
    for epoch in range(training_epochs):
        x_train, y_train = next_batch(20)
        model.fit(x_train, y_train, epochs=15, batch_size=20, shuffle=True)
        print ("nombre de training_epochs = ", i)
        i += 1
	
    x_test, y_test = next_batch(20)
    score = model.evaluate(x_test, y_test, batch_size=20)
    print("score=", score)
    print(model.metrics_names[1], score[1]*100)
    
    single_x_test, single_y_result = point_image(detector2,predictor2,10)
    """single_x_test, single_y_result = nb_random()"""
    q = model.predict( np.array([single_x_test]))
    print("La prediction est classé dans ", q, " et le resultat réel est ", single_y_result)

    """single_x_load, single_y_load = point_image(detector2,predictor2)
    q1 = model.predict (np.array ([single_x_load]))
    print("Prediction du modele chargé : Il est classé dans ", q1," et le resultat réel est ", single_y_load)
    """

    # enregistre les resultats du modele .h5
    model.save("model.h5")
    del model
    print("Le modele est sauvegardé")

    # chrono pour avoir le temps d'execution du reseau de neurones
    fin = time.time()
    print("Temps total : ", (fin-debut)/60 ,"minutes")

if __name__ == "__main__":
    main()