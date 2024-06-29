# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:27:03 2023

@author: sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

image=cv2.imread("D:\\Codes\\image processing\\Histogram based image segmentation\\test.jpg",1)

image=cv2.line(image,(0,20),(40,0),(0,0,255),5)
image=cv2.arrowedLine(image,(0,200),(200,200),(255,0,0),1)
image=cv2.rectangle(image,(0,0),(100,100),(0,255,0),1)
image=cv2.circle(image,(100,100),50,(0,255,255),-1)


#cv2.imshow('lena',image)
#cv2.waitKey(0)
#cv2.destroyWindow()

plt.imshow(image)
plt.axis('off')

for v in dir():
    exec('del '+ v)
    del v