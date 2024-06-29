# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 08:52:20 2023

@author: sagar
"""

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

brightest_point=brightest(image)

image=np.roll(np.roll(image,-(brightest_point[0]-512),axis=0),-(brightest_point[1]-512),axis=1)
X=image.shape[0]
Y=image.shape[1]
image=image[X//5:4*X//5,Y//5:4*Y//5]

plt.imshow(image)
plt.axis("off")
plt.show()

