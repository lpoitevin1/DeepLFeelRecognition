#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

# comparaison entre les points sur les photos et les points tirés de la webcam
# via un systeme de difference de chaque longueur & 
# une addition de ces differences

def comparaison2 (point,liste) :
    l = len(liste)
    i=0
    a = 0
    y =0
    pointcomp = []
    print ("taille de la liste d'image : ", l)
    while (i<l) :
        pointcomp = liste[i]
        test = 0
        j = 0
        '''print ("Nombre d'element dans point : ",len(point)) '''
        print ("Nombre d'element dans pointcomp : ",len(pointcomp))
        while (j<len(pointcomp)) :
            test += (abs(pointcomp[j]-point[j]))
            j += 1
        print ("valeur de la variable test de l'image", i+1," : ",test)  
        if (a==0):
            a = test
            y = i
        elif (a > test) :
            a = test
            y = i
        i += 1
    return (y+1)