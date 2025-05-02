# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 13:40:25 2023

IMAGE EDGE FILTERING

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte,color
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny

# Load the image
#image_path="D:\\Codes\\image processing\\edge detection\\5.jpeg"
image_path="D:\\Images\\Wedding photos\\Asirbad\\_DSC9892.JPG"
main_image = img_as_ubyte(io.imread(image_path))
image = img_as_ubyte(color.rgb2gray(main_image))

image_roberts=roberts(image)
image_sobel=sobel(image)
image_scharr=scharr(image)
image_prewitt=prewitt(image)
image_canny=canny(image, sigma=0.5)

print("Image shape: ",image.shape)
print("Roberts image shape: ",image_roberts.shape)
print("Sobel image shape: ",image_sobel.shape)
print("Scharr image shape: ",image_scharr.shape)
print("Prewitt image shape: ",image_prewitt.shape)
print("Canny image shape: ",image_canny.shape)






plt.imshow(main_image)
plt.axis('off')
plt.title("Main image")
plt.show()
 
plt.imshow(image_roberts)
plt.axis('off')
plt.title("roberts")
plt.show()

plt.imshow(image_sobel)
plt.axis('off')
plt.title("sobel")
plt.show()


plt.imshow(image_scharr)
plt.axis('off')
plt.title("scharr")
plt.show()

plt.imshow(image_prewitt)
plt.axis('off')
plt.title("prewitt")
plt.show()

plt.imshow(image_canny)
plt.axis('off')
plt.title("canny")
plt.show()