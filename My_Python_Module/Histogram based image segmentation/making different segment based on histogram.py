# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:14:49 2023

@author: sagar
"""

''' MAKING DIFFERNET SEGMENT BASED ON HISTOGRAM '''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,img_as_ubyte


image_path="D:\\Codes\\image processing\\Histogram based image segmentation\\test.jpg"
image=img_as_ubyte(io.imread(image_path,as_gray=True))

print(image.shape)

plt.imshow(image)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

segm1=(image <= 50)
segm2=(image <= 150) & (image>50)
segm3=(image <=200)  & (image>150)
segm4=(image>200)

print(segm1)

all_segments=np.zeros((image.shape[0],image.shape[1],3))

all_segments[segm1]=[1,0,0]
all_segments[segm2]=[0,1,0]
all_segments[segm3]=[0,0,1]
all_segments[segm4]=[1,1,0]

plt.imshow(all_segments)
plt.axis('off')
plt.show()
