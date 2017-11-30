#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# import the necessary packages
from imutils import face_utils
from calc_points import *
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import glob
import os


# Fonction qui retourne le nombre d'image de la video lu
def nombre_image (video) :
    # Ouvre le flux video
    cap = cv2.VideoCapture(video)
    
    # Check si le flux est ouvert ou pas 
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    # Compteur d'image initialisé a 0
    i = 0
    # Lit la video
    while(cap.isOpened()):
    # A chaque frame (image) incremente i
        ret, frame = cap.read()
        if ret == True:
            i += 1 
    # Sort du while quand il n'y a plus de frame a afficher
        else: 
            break
    return i

# Fonction qui fait une capture d'image de la video sur l'image i-1
def capture_image(video,n_image,dossier) :
    cap = cv2.VideoCapture(video)
    # Check si le flux est ouvert ou pas 
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    i = 0

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            # Enregistre l'image i - 1 et l'enregistre dans le bon dossier 
            if i == n_image - 1  :
                nb = len(os.listdir("images/"+dossier+"_photo/"))
                cv2.imwrite("images/"+dossier+"_photo/"+ str(nb+1)+".png",frame)
                print("images/"+dossier+"_photo/"+ str(nb+1)+".png")
            i += 1

        else: 
            break     

# Fonction qui modifie une image, pour epaissir la base de données
# Effectue une rotation d'une image de 10° a D et a G
def modif_image(image,dossier) :
    # Recupere le nombre d'image dans le dossier, pour enregistrer les nouvelles images
    # sans ecraser les images existantes
    i = len(os.listdir("images/"+dossier+"_photo/"))
    img = cv2.imread(image)
    rows,cols = img.shape[:2]
    # Premiere modif de 10°
    M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+1)+".png",dst)
    # Deuxieme modif de -10°
    M2 = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
    dst2 = cv2.warpAffine(img,M2,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+2)+".png",dst2)
    # Decale l'image en bas a droite de 50 pixels a droite et 50 pixels en bas
    M3 = np.float32([[1,0,50],[0,1,50]])
    dst3 = cv2.warpAffine(img,M3,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+3)+".png",dst3)
    # Decale l'image en haut a gauche de 50 pixels a gauche et 50 pixels en haut
    M4 = np.float32([[1,0,-50],[0,1,-50]])
    dst4 = cv2.warpAffine(img,M4,(cols,rows))
    cv2.imwrite("images/"+dossier+"_photo/"+str(i+4)+".png",dst4)


# Focntion qui creé des images pour un dossier specifique (donc une emotion)
def creation_image (dossier) :
    # Dans un dossier, recupere tout les fichiers au format .mpeg
    liste = glob.glob("images/"+dossier+"/*.mpeg")
    len_liste = len(liste)
    print ("Nombre de videos dans le dossier : ", len_liste)
    i = 0
    # Utilise capture_image pour transformer toute les videos en image
    while (i<len_liste) : 
        nb = nombre_image(liste[i])
        capture_image(liste[i],nb,dossier)
        i += 1
    i = 0
    max = len(os.listdir("images/"+dossier+"_photo/"))
    # Affiche le taux de conversion des videos en images
    print ("Nombre d'image sur le nombre de video du dossier : ", max,"/",len_liste, " taux : ", (max/len_liste)*100)
    # Effectue les modifs sur toute les images nouvellements creé
    while (i<max) :
        modif_image("images/"+dossier+"_photo/"+str(i+1)+".png",dossier)
        print ("Modif effectue sur l'image ", i+1 )
        i += 1

# Fonction qui boucle creation_image sur tout les dossiers (donc sur les 6 emotions)  
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