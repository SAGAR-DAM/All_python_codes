# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:36:35 2023

@author: mrsag
"""
'''
THIS CODE FINDS SHAPES ON AN IMAGE DEPENDING ON THE BACKGROUND CONTRAST RATIO TO AND DRAWS
CONTOURS OVER THE SHAPE 
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,measure,color
from skimage.transform import resize


# Construct some test data
image_path="D:\\Codes\\image processing\\BeFunky-collage.jpg"    
image=io.imread(image_path)          #opening the image
#r=color.rgba2rgb(r)                 # converting 4 colour to 3 colour
image=color.rgb2gray(image)          # converting 3 colour image to greyscale
image=resize(image,(400,400))

#image=image[10:(image.shape[0]-10),10:(image.shape[1]-10)]  # when the image has unnecessary margins
# Find contours at a constant value of 0.8
contours = measure.find_contours(image, 0.2)    # finding contours at proper contrast value
print(np.array(contours).shape)

# Display the image and plot all contours found


fig, (ax1,ax2) = plt.subplots(1,2)    
ax2.imshow(image, cmap=plt.cm.gray)

for i in range(np.array(contours).shape[0]):
    ax2.plot(contours[i][:,1],contours[i][:,0],linewidth=1)
    plt.text(contours[i][0][0],contours[i][0][1],'%d'%i,color='yellow')
    
ax2.axis('off')
ax2.set_xticks([])
ax2.set_yticks([])

ax1.imshow(image)
ax1.axis('off')
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()

#print(image.shape)