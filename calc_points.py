#!/usr/bin/python3.5
# -*-coding:Utf-8 -*

#packages
import math

# Calcule de la distance entre tout les points
def calc_points (shape,w) :
	i = 0
	''' liste_point = [37,46,40,43,28,34,49,55,28,52,28,58]
	print (liste_point)
	point = []
	while (i<len(liste_point)) :
		(x0,y0) = shape[liste_point[i]]
		(x1,y1) = shape[liste_point[i+1]]
		a = math.sqrt(math.pow((x1-x0),2)+math.pow((y1-y0),2))
		point.append(a/w)
		i += 2
	print (point)
	return point '''
	shape2 = shape
	point = []
	while (i<len(shape)-1) :
		(x0,y0) = shape[i]
		j = 0
		while (j<len(shape2)) :
			if (j == i) :
				j +=1
			(x1,y1) = shape2[j]
			a = math.sqrt(math.pow((x1-x0),2)+math.pow((y1-y0),2))
			print ("shape", i, " et shape2", j, " : ", a)
			point.append(a/w)
			j += 1
		i += 1
    
	return point
'''	(x1,y1) = shape [x]
	(x2,y2) = shape [y]
	a = math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))
	return a/w '''