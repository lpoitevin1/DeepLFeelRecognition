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
import glob
import os

def nombre_image (video) :
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    i = 0
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            i += 1 
    # Break the loop
        else: 
            break
    return i

def capture_image(video,n_image,dossier) :
    cap = cv2.VideoCapture(video)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    i = 0
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if i == n_image - 1  :
                nb = len(os.listdir("images/"+dossier+"_photo/"))
                cv2.imwrite("images/"+dossier+"_photo/"+ str(nb+1)+".png",frame)
                print("images/"+dossier+"_photo/"+ str(nb+1)+".png")
            i += 1
    # Break the loop
        else: 
            break     

def modif_image(image,dossier) :
    i = len(os.listdir("images/"+dossier+"_photo/"))
    img = cv2.imread(image)
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+1)+".png",dst)
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
    dst2 = cv2.warpAffine(img,M2,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+2)+".png",dst2)


def creation_image (dossier) :
    liste = glob.glob("images/"+dossier+"/*.avi") + glob.glob("images/"+dossier+"/*.mpeg")
    len_liste = len(liste)
    print ("Nombre de videos dans le dossier : ", len_liste)
    i = 0
    while (i<len_liste) : 
        nb = nombre_image(liste[i])
        capture_image(liste[i],i,dossier)
        i += 1
    i = 0
    max = len(os.listdir("images/"+dossier+"_photo/"))
    print ("Nombre d'image sur le nombre de video du dossier : ", max,"/",len_liste, " taux : ", (max/len_liste)*100)
    while (i<max) :
        modif_image("images/"+dossier+"_photo/"+str(i+1)+".png",dossier)
        print ("Modif effectue sur l'image ", i+1 )
        i += 1
    
def main () :
    tab = ['anger','happiness','disgust','fear','sadness','surprise']
    print (tab)
    i = 0
    while i < len(tab) :
        dossier = tab[i]
        print ("Debut de traitement du dossier ", dossier)
        creation_image(dossier)
        print ("Le dossier ",dossier," est complete")
        i += 1



if __name__ == "__main__":
    main()