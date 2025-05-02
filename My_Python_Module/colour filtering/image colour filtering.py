# -*- coding: utf-8 -*-
"""
Created : Sagar Dam

This code takes some image and colour and gives the coloured filter output of the image. 
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io

'''     INPUTS     '''

image = io.imread("D:\\Images\\Wedding photos\\Asirbad\\_DSC9866.JPG")
colour='cyan'   # Inputs: red, green, blue, yellow, magenta, cyan


#print(image.shape)

def colour_filtering(image,colour):
    matrix=[[[0,0,0]]*image.shape[1]]*image.shape[0]
    matrix=np.array(matrix)
    #print(matrix.shape)
    
    if(colour=='red'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=red
                matrix[i,j][1]=0
                matrix[i,j][2]=0    
                
    elif(colour=='green'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=0
                matrix[i,j][1]=green
                matrix[i,j][2]=0
                
    elif(colour=='blue'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=0
                matrix[i,j][1]=0
                matrix[i,j][2]=blue
                
    elif(colour=='yellow'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=red
                matrix[i,j][1]=green
                matrix[i,j][2]=0
                
    elif(colour=='magenta'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=red
                matrix[i,j][1]=0
                matrix[i,j][2]=blue
                
    elif(colour=='cyan'):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                red=image[i][j][0]
                green=image[i][j][1]
                blue=image[i][j][2]
                #matrix[i,j]=y
                matrix[i,j][0]=0
                matrix[i,j][1]=green
                matrix[i,j][2]=blue
    
    return(matrix)

matrix=colour_filtering(image, colour)

plt.figure(figsize=(30,20))
plt.axis('off')
plt.imshow(matrix)