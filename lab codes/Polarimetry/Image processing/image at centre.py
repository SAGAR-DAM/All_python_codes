# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:02:29 2023

@author: sagar
"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import color,io,img_as_ubyte,feature
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.optimize import curve_fit
from skimage.draw import line

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


image_path="D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\17Aug23_par\\par_splitted_image\\P_110_br.png"
image=io.imread(image_path)
#image=np.roll(np.roll(image,-100,axis=0),-100,axis=1)
cut_radius=200

def brightest(image):
    # Find the coordinates of the brightest point using the corner_peaks function
    if(image.ndim==2):
        coords = feature.corner_peaks(np.abs(image), min_distance=10)
    elif(image.ndim==3):
        if(image.shape[2]==3):
            coords = feature.corner_peaks(np.abs(image), min_distance=10)
        elif(image.shape[2]==4):
            coords = feature.corner_peaks(np.abs(image), min_distance=10)
    return(coords[0])


def make_at_centre(image):
    global cut_radius
    point=brightest(image)
    print(point)
    
    X=image.shape[0]
    Y=image.shape[1]
    
    base_noise=np.mean(image[3*X//4:,:Y//4])
    #base_noise_max=np.max(image[3*X//4:,:Y//4])
    
    image2 = np.asarray([[base_noise]*Y]*X)
    #image2=np.random.uniform(low=0,high=base_noise_max,size=(X,Y))
    
    for i in range(point[0]-cut_radius-1,point[0]+cut_radius+1):
        for j in range(point[1]-cut_radius-1,point[1]+cut_radius+1):
            if((i-point[0])**2+(j-point[1])**2<=cut_radius**2):
                dx=i-point[0]
                dy=j-point[1]
                image2[512+dx,512+dy]=image[i,j]
    return(image2)

plt.imshow(image)
plt.axis("off")
plt.show()


image2=make_at_centre(image)
#image2=image2[512-cut_radius-50:512+cut_radius+50,512-cut_radius-50:512+cut_radius+50]
image2=image2[256:3*256,256:3*256]
plt.imshow(image2)
plt.axis('off')
plt.show()