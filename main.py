""""
Este programa sirve para re pintar una imagen
"""


import os 
import sys
import cv2
import numpy as np

# k means classification
K = int(sys.argv[1])
inputName = sys.argv[2]
img = cv2.imread(inputName)
imgCL = np.float32(img.reshape((-1,3)))
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,15,1.0)

print(K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
ret,lab,center = cv2.kmeans(imgCL,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

print('Accuracy: ',ret)

center = np.uint8(center)
res=center[lab.flatten()]

# Se incremento el tamaño de la imagen

res2 = res.reshape((img.shape))
res3 = cv2.resize(res2,(640,480))
# Mostramos la imagen:
cv2.imshow('Output',res3)

cv2.waitKey()

