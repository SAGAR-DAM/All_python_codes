# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:14:13 2023

RESIZE IMAGE BY NUMBER OF PIXELS AND SWIRL 

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean, swirl
from skimage.filters import roberts, sobel, scharr, prewitt

# Load the image
image_path='D:\\pictures\\5.png'
image = img_as_ubyte(io.imread(image_path,as_gray=True))

rescaled_image=rescale(image, 1.0/2.0, anti_aliasing=True)   #rescales image by given factor in both the axis
resized_image=resize(image,(150,200))    # resizes image in both axis with given number of pixels
#downscaled_image=downscale_local_mean(image, (4,3))   # resizes image using the average of each (4,3) from (0,0) 
swirled_image = swirl(image, rotation=0, strength=100, radius=300)


print("Image shape: ",image.shape)
print("Rescaled image shape: ",rescaled_image.shape)
print("Resized image shape: ",resized_image.shape)
print("Swirled image shape: ",swirled_image.shape)


plt.figure()
plt.axis('off')
plt.set_cmap('gray')
#plt.imshow(image)
#plt.imshow(rescaled_image)
#plt.imshow(resized_image)
#plt.imshow(downscaled_image)
plt.imshow(swirled_image)
plt.show()

