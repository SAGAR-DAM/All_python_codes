# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:40:05 2023

@author: mrsag
"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io,color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np


image=plt.imread('D:\\Codes\\image processing\\image entropy\\qrcode-big-content-2edited.png')
image=color.rgba2rgb(image)
image=color.rgb2gray(image)    # as image entropy only works with grayscale image
print(image.ndim)              # if image.ndim=2 then only entropy function works

entropy_image=entropy(image, disk(14)) # creates a new image from the given image with different entropy (disorder) regions separated out by different colours
threshold=threshold_otsu(entropy_image)     # gives the best value for the pixels on the boundary of the segments of the entropy image

binary=(entropy_image >= threshold)   # creates a new image for which the pixel_value(entropy_image)<=threshold to give the proper segmentation 

print(threshold)
plt.imshow(entropy_image)
plt.axis('off')
plt.show()

plt.imshow(binary)
plt.axis('off')
plt.show()

print("Total number of dark pixels: ",np.sum(binary==0))
print("Total number of bright pixels: ",np.sum(binary==1))

#print(binary.shape)
#print(image)
print(np.max(entropy_image))