README
======

Ceci est le ReadMe de notre projet

-------------

Objectif
--------

Le projet permet de lancer le flux d'une webcam, une image ou une vidéo, et lorsque le programme reconnait un visage sur ce flux, d'analyser les émotions présentes (Happiness ou Anger).

Installation
-------------
> **Note**

> - Le programme a été realisé sous python 3.5, il ne fonctionne donc qu'avec cette version

Les programmes nécésitent les librairies OpenCV et TensorFlow ainsi que sa surcouche, Keras </br>

OpenCV : https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/ </br>

Tensorflow : https://www.tensorflow.org/install/ </br>

Keras : https://keras.io/#installation </br>

Pour compiler notre programme : </br>
- Si vous n'avez pas le fichier "modele.json" : $python3.5 image_learning.py </br>

- Si vous avez déjà le fichier "modele.json" : $python3.5 main_image.py </br> 
	Arguments à passer en parametres :
	 -s'il s'agit d'une image : -i pathImage ou --image pathImage
	 -s'il s'agit d'une vidéo : -v pathVidéo ou --video pathVidéo
	Si aucun argument n'est spécifié, cela lancera la webcam par défaut.
	

Programme
---------

Comme vous le voyez il y a beaucoup de fonctions. Malgré tout, seulement deux sont a utiliser </br>

La fonction image_learning.py : </br>
C'est la fonction qui fait tourner le réseau de neurones (il se peut qu'elle mette un certains temps a se terminer)

La fonction main_image.py : </br>
C'est la fonction "main", il lui faut les résultats du réseau de neurones qui sont sous la forme "modele.json" pour fonctionner. Elle va démarrer la webcam par defaut et commencer ses analyses.

Divers
------

Les autres fonctions sont soit des tests ou des modéles (classifier_keras.py), soit des fonctions accessoires. Par exemple, creation_image.py est la fonction qui nous a permis de transformer toute les vidéos en images et, en les modifiants légèrement, d'étendre notre base de données. De même test_network.py est un des tests d'amelioration du réseau de neurones. Il s'agissait d'un méta-réseau qui en testait plusieurs avec differentes configuration.

Résultats
---------
Le réseau de neurones est capable d'interpréter le type d'émotion sur l'image : soit Anger soit Happiness. 


Auteur
------
MEDJEDEL Nour El Houda p1411470 </br>
POITEVIN Louis p1410541 </br>
DUMENIL Thibault p1306146 </br>

