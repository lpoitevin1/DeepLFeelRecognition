from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from imutils.video import VideoStream
from imutils import face_utils

import os
import urllib

import math
import numpy as np
import cv2
import imutils
import dlib
import random
import time

import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Reseau de neurone sur les points des images
print ("Debut de l'analyse des images")
print ("Lecture du fichier : shape predictor...")
detector2 = dlib.get_frontal_face_detector()
predictor2 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
debut = time.time()

# Choisie aleatoirement une image entre 2 emotions (happiness/surprise) et recupere les
# points caracteristiques + associe a une emotion 
def point_image (detector2,predictor2) :
    j = random.randrange(1,3,1)
    k = random.randrange(1,61,1)
    shape_a = []
    shape_b = []
    if (j == 1) :
        img = cv2.imread("images/happiness_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
        y = np.array([1,0])
    else :
        img = cv2.imread("images/surprise_photo/"+ str(k) +".png")
        img = imutils.resize(img, width=400)
        rects = detector2(img,1)
        for (i, rect) in enumerate(rects):
            shape = predictor2(img, rect)
            shape = face_utils.shape_to_np(shape)
        y = np.array([0,1])          
    i = 0
    while (i<len(shape)):
        (a,b) = shape[i]
        shape_a.append(a)
        shape_b.append(b)
        i += 1
    x = np.concatenate((shape_a, shape_b))
    print("Anayse de l'image n°", k," dans le y ", y , " . Nombre de point : ", len(x))         
    return x, y            

# Boucle n fois pour recuperer les données de n images
def next_batch(n):
    x = np.zeros( shape=(n,136), dtype=np.float32)
    y = np.zeros( shape=(n,2), dtype=np.float32)
    for i in range(0, n):
        x[i],y[i] = point_image(detector2,predictor2)
    print ("Tableau de valeur : ", x)
    return x,y
        
# Creation, parametrage et utilisation du reseau
def main():

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 2-dimensional vectors.
	#The last output has to be the number of class
    model.add(Dense(136, activation='relu', input_dim=136))
    model.add(Dropout(0.5))
    model.add(Dense(68, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
		  
    training_epochs = 50
    i = 1
    for epoch in range(training_epochs):
        x_train, y_train = next_batch(100)
        model.fit(x_train, y_train, epochs=50, batch_size=100)
        print ("nombre d'iteration = ", i)
        i += 1 
	
    x_test, y_test = next_batch(20)
    score = model.evaluate(x_test, y_test, batch_size=20)
    print("score=", score)
    print(model.metrics_names[1], score[1]*100)
    
    single_x_test, single_y_result = point_image(detector2,predictor2)
    q = model.predict( np.array( [single_x_test] )  )
    print(single_x_test, "is classified as ", q, " and real result is ", single_y_result)

    fin = time.time()
    print("Temps total : ", fin-debut,"s")

if __name__ == "__main__":
    main()