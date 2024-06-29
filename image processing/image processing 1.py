'''
Image processing using python
Using basic image processing tool of skimage

'''


import numpy as np
import matplotlib.pyplot as plt
from skimage import io    # image processing tool

my_image=io.imread("D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_33.png")   # calling an image
#print(my_image)     #printing the array of the image

my_image_dim=np.asarray(np.zeros(3),dtype=int)
for i in range(3):
    my_image_dim[i]=my_image.shape[i]
print(my_image_dim)

random_image=np.array(np.random.random([500,500])*20,dtype=int)        # creating some random scalar image
#random_image=np.array(np.random.random([500,500,3])*255,dtype=int)     # creating some random RGB image
#random_image=np.array(np.random.random([500,500,4])*255,dtype=int)     # creating some random RGB image with transparency

#random_image=my_image+np.array(np.random.random(my_image_dim)*79.0,dtype=int)    # creating some random image with given image dimension

#plt.imshow(my_image)
plt.imshow(random_image)