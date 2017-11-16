from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from imutils.video import VideoStream
from imutils import face_utils
from image_learning import *

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
import threading

import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# Code pour un reseau de neurones, voir image_learning.py pour plus de commentaires
def network(a,n) :
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 2-dimensional vectors.
	#The last output has to be the number of class

    # Le paramtetre n module le nombre de couche du reseau
    model.add(Dense(68, activation='relu', input_dim=68))
    model.add(Dropout(0,5))
    for i in range (0,n) :
        model.add(Dense(68, activation='relu'))
        model.add(Dropout(0,5))
    model.add(Dense(6, activation='softmax'))
    
    # Le parametre a module le lr
    sgd = SGD(lr=a, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
		  
    training_epochs = 10
    i = 1
    for epoch in range(training_epochs):
        x_train, y_train = next_batch(10)
        model.fit(x_train, y_train, epochs=10, batch_size=10, shuffle=True)
        print ("nombre de training_epochs = ", i)
        i += 1
	
    x_test, y_test = next_batch(10)
    score = model.evaluate(x_test, y_test, batch_size=20)
    print("score=", score)
    print(model.metrics_names[1], score[1]*100)

    return score, model      

# Le main teste plusieurs reseaux en changeant les parametres a et n
# Et renvoie celui qui a le plus haut taux de reussite
# Ici on teste 100 configs differents 
# n allant de 0 a 10 -> nb de couches supplementaires
# a allant de 0,1 a 10^⁻10
def main() :

    debut = time.time()
    res = 0
    for i in range (0,10) :
        n = i
        for j in range (0,10) :
            a = (1/10)/(10**j)
            print ("Test du reseau n°",i,".",j," avec ",n," couches supplementaires, lr = ", a)
            score, model = network(a,n)
            inter = time.time()
            print ("Temps intermediaire : ", (inter-debut)/60, " min")

            if score[1] > res :
                final = ""
                res = score[1]
                final += str(i)
                final += "."
                final += str(j)
                # enregistre les resultats du modele .h5
                model.save("model.h5")
                del model
                print("Le modele est sauvegardé") 

            print ("Le meilleur modele est ", final, " avec un taux de ", res)

    fin = time.time()
    print("Le reseau utilisé est le n°", final, " avec un score de ", res)
    print("Temps du test : ", (fin-debut)/60," min")


if __name__ == "__main__":
    main()

        
    
    
