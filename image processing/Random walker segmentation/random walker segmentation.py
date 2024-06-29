# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:35:49 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,img_as_ubyte, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
from skimage.segmentation import random_walker


image_path="D:\\Codes\\image processing\\Random walker segmentation\\plot_random_walker_segmentation.png"
#image_path="D:\\Codes\\image processing\\Random walker segmentation\\B5EhD.png"

image=img_as_float(io.imread(image_path,as_gray=True))
print(image.shape)

'''                                          # This part was for the test image: Ship.png
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if(image[i][j]>=0.8 or image[i][j]<=0.2):
            image[i][j]=0
          
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if(image[i][j]>=0.8):
            image[i][j]=0.4+0.2*(image[i][j]-0.8)/0.2
        elif(image[i][j]<=0.2):
            image[i][j]=0.6+0.2*(image[i][j])/0.2
'''


plt.imshow(image)
plt.axis("off")
plt.title("Main image")
plt.set_cmap('gray')
plt.show()

plt.hist(image.flat, bins=100, range=(0,1))
plt.title("Histogram of the main image")
plt.show()


patch_kw=dict(patch_size=5, patch_distance=6, multichannel=False)
sigma_est=np.mean(estimate_sigma(image, multichannel=False))
denoise=denoise_nl_means(image, h=4.5*sigma_est, fast_mode=True, **patch_kw)

plt.imshow(denoise)
plt.axis("off")
plt.title("Denoised image")
plt.show()

plt.hist(denoise.flat, bins=100, range=(0,1))
plt.title("Histogram of the denoised image")
plt.show()

eq_image=exposure.equalize_adapthist(denoise)

plt.imshow(eq_image)
plt.axis("off")
plt.title("equated image")
plt.show()

plt.hist(eq_image.flat, bins=100, range=(0,1))
plt.title("Histogram of the equated image")
plt.show()

new_image=np.zeros((eq_image.shape[0],eq_image.shape[1],3))

segm1=(eq_image>=0.19)&(eq_image<=0.5)
segm2=(eq_image<0.19)
segm3=(eq_image>0.5)


new_image[segm1]=[0,1,0]    # showing all segments
new_image[segm2]=[0,1,0]
new_image[segm3]=[0,0,1]

plt.imshow(new_image)
plt.axis('off')
plt.title("segmentation of the equated image")
plt.show()


markers=np.zeros(image.shape,dtype=np.uint)
markers[(denoise>0.2) & (denoise<=0.4)]=0
markers[(denoise>0.6) & (denoise<=0.9)]=1
random_walked_image=random_walker(eq_image, markers, beta=10, mode='bf')

plt.imshow(random_walked_image)
plt.axis('off')
plt.title("Random walker segmented image \n of the denoised image")
plt.show()


#plt.hist(random_walked_image.flat,bins=100, range=(0,3))
#plt.show()