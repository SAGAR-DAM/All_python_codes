# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:14:49 2023

@author: sagar
"""

'''  DENOISING AND SEGMENTATION BASED ON HISTOGRAM '''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,img_as_ubyte,img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy import ndimage as nd

image_path="D:\\Codes\\image processing\\Histogram based image segmentation\\test.jpg"
image=img_as_float(io.imread(image_path,as_gray=True))

print(image.shape)
plt.imshow(image)
plt.set_cmap('gray')
plt.axis('off')
plt.title("Main image")
plt.show()


patch_kw=dict(patch_size=10, patch_distance=6, multichannel=False)
sigma_est=np.mean(estimate_sigma(image, multichannel=False))
denoise=denoise_nl_means(image, h=10.15*sigma_est, fast_mode=True, **patch_kw)

denoise_ubyte=img_as_ubyte(denoise)

plt.imshow(denoise_ubyte)
plt.set_cmap('gray')
plt.axis('off')
plt.title("Denoised image")
plt.show()


plt.hist(img_as_ubyte(image).flat,bins=100, range=(0,255))
plt.title("Histogram of the Grayscale of main image")
plt.show()


plt.hist(denoise_ubyte.flat, bins=100, range=(0,255))
plt.title("Histogram of the denoised image")
plt.show()

''' Segmentation of the main image without the filtering and showing the segments'''

image_ubyte=img_as_ubyte(image)
seg1=(image_ubyte <= 64)
seg2=(image_ubyte <= 128) & (image_ubyte>64)
seg3=(image_ubyte <=192)  & (image_ubyte>128)
seg4=(image_ubyte>192)

#print(seg1)

all_segments_main_image=np.zeros((image.shape[0],image.shape[1],3))

all_segments_main_image[seg1]=[1,0,0]    # showing all segments
all_segments_main_image[seg2]=[0,1,0]
all_segments_main_image[seg3]=[0,0,1]
all_segments_main_image[seg4]=[1,1,0]

plt.imshow(all_segments_main_image)
plt.title("Segmented image of the main image")
plt.axis('off')
plt.show()

segm1=(denoise_ubyte <= 64)
segm2=(denoise_ubyte <= 128) & (denoise_ubyte>64)
segm3=(denoise_ubyte <=192)  & (denoise_ubyte>128)
segm4=(denoise_ubyte>192)

print(segm1)

all_segments=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))

all_segments[segm1]=[1,0,0]    # showing all segments
all_segments[segm2]=[0,1,0]
all_segments[segm3]=[0,0,1]
all_segments[segm4]=[1,1,0]

plt.imshow(all_segments)
plt.title("Segmented image of the denoised image")
plt.axis('off')
plt.show()


all_segments[segm1]=[1,0,0]    # showing segment 1
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.title("Binary image with range: 0<grayscale<=64")
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]    # shoeing segment 2
all_segments[segm2]=[0,1,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.title("Binary image with range: 64<grayscale<=128")
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]   # showing segment 3
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,1]
all_segments[segm4]=[0,0,0]

plt.imshow(all_segments)
plt.title("Binary image with range: 128<grayscale<=192")
plt.axis('off')
plt.show()


all_segments[segm1]=[0,0,0]   # showing segment 4
all_segments[segm2]=[0,0,0]
all_segments[segm3]=[0,0,0]
all_segments[segm4]=[1,1,0]

plt.imshow(all_segments)
plt.title("Binary image with range: 192<grayscale<=255")
plt.axis('off')
plt.show()


''' Filtering with scipy and segmentation of the same '''

segm1_opened=nd.binary_opening(segm1, np.ones((3,3)))
segm1_closed=nd.binary_closing(segm1_opened, np.ones((3,3)))

segm2_opened=nd.binary_opening(segm2, np.ones((3,3)))
segm2_closed=nd.binary_closing(segm2_opened, np.ones((3,3)))

segm3_opened=nd.binary_opening(segm3, np.ones((3,3)))
segm3_closed=nd.binary_closing(segm3_opened, np.ones((3,3)))

segm4_opened=nd.binary_opening(segm4, np.ones((3,3)))
segm4_closed=nd.binary_closing(segm4_opened, np.ones((3,3)))

all_segments_cleaned=np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))

all_segments_cleaned[segm1]=[1,0,0]    # showing all segments
all_segments_cleaned[segm2]=[0,1,0]
all_segments_cleaned[segm3]=[0,0,1]
all_segments_cleaned[segm4]=[1,1,0]

plt.imshow(all_segments_cleaned)
plt.axis('off')
plt.title("Segmented image after scipy filtering")
plt.show()


