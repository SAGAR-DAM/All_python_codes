# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:29:30 2023

@author: sagar
"""

from skimage import io, color
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
import numpy as np

import matplotlib
matplotlib.rcParams['figure.dpi']=700 # highres display


image=io.imread("D:\\Codes\\complex field sets\\quadro dragon 12.png")
image=color.rgba2rgb(image)
image=img_as_ubyte(image)

def grayscale(point):
    val=(point[0])**2+(point[1])**2+(point[2])**2
    return(val)

X=image.shape[0]
Y=image.shape[1]
print(image.shape)

plt.imshow(image)
plt.axis('off')
plt.show()

print(X)
print(Y)

for i in range(X):
    for j in range(Y):
        if(image[i][j][0]==255 and image[i][j][1]==255 and image[i][j][2]==255):
            image[i][j]=[49,49,49]

for i in range(X):
    for j in range(Y):
        if(image[i][j][0]==255 and (image[i][j][1]>80 or image[i][j][2]>80)):
            image[i][j]=[255,0,0]
        elif(image[i][j][1]>100 and (image[i][j][0]<100 or image[i][j][2]<100)):
            image[i,j]=[4,140,4]
        elif(image[i][j][2]>230 and (image[i][j][0]>80 or image[i][j][1]>80)):
            image[i][j]=[0,0,255]
        elif(image[i][j][0]==image[i][j][1] and image[i][j][0]==image[i][j][2] and image[i][j][0]>=50):
            image[i][j]==[0,0,0]
            
for i in range(X):
    for j in range(Y):
        if(image[i][j][0]==49 and image[i][j][1]==49 and image[i][j][2]==49):
            image[i][j]=[255,255,255]

plt.imshow(image)
plt.axis('off')
plt.show()

