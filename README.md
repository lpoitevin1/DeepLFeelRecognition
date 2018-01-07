README
======

Ceci est le ReadMe de notre projet

-------------

Configuration
-------------
> **Note**

> - Le programme a été realisé sous python 3.5, il ne fonctionne donc qu'avec cette version

Les programmes nécésite les librairies OpenCV et TensorFlow ainsi que sa surcouche, Keras </br>
OpenCV : https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/ </br>

Tensorflow : https://www.tensorflow.org/install/ </br>

Keras : https://keras.io/#installation </br>

Programme
---------

Comme vous le voyez il y a beaucoup de fonctions. Malgré tout, seulement deux sont a utiliser </br>

La fonction image_learning.py : </br>
C'est la fonction qui fait tourner le réseau de neurones (il se peut qu'elle mette un certains temps a se terminer)

La fonction main_image.py : </br>
C'est la fonction "main", il lui faut les résultats du réseau de neurones qui sont sous la forme "modele.json" pour fonctionner. Elle va démarrer la webcam par defaut et commencer ses analyses.

Divers
------

Les autres fonctions sont soit des tests ou des modéles (classifier_keras.py), soit des fonctions accessoires. Par exemple, creation_image.py est la fonction qui nous a permis de transformer toute les vidéos en images et, en les modifiants légèrement, d'étendre notre base de données.

Auteur
------
MEDJEDEL Nour El Houda p1411470
POITEVIN Louis p1410541
DUMENIL Thibault p1306146