# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:59:41 2023

@author: sagar
"""

import numpy as np
from skimage import io,color,img_as_ubyte
from skimage.feature import blob_dog
import matplotlib.pyplot as plt
import cv2

import matplotlib
matplotlib.rcParams["figure.dpi"]=300


def draw_on_image(image,x,y):
    image=cv2.line(image,(x[1],x[0]),(y[1],y[0]),(255,0,0),3)
    return(image)

def linecut_function(start_point,end_point,image,image2):
    # Use the numpy function linspace to get the coordinates of the line
    num=round(np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2))
    x, y = np.linspace(start_point[0], end_point[0], num), np.linspace(start_point[1], end_point[1], num)
    image2=draw_on_image(image2, start_point,end_point)
    # Get the grayscale values along the line
    gray_values = image[x.astype(int),y.astype(int)]
    linecut=[]
    for i in range(len(gray_values)):
        linecut_value=gray_values[i]
        linecut.append(linecut_value)
        
    return(np.array(linecut),image2)

filename="D:\\Codes\\lab codes\\Niladri CR 39\\SNAP-134309-0105.tif"
image=io.imread(filename,as_gray=True)

plt.imshow(image)
plt.axis("off")
plt.show()

image2=np.asarray(image>0.55)
image3=color.gray2rgb(img_as_ubyte(image2))

angle=np.linspace(0,3*np.pi/4,4)
# Detect blobs (spheres) in the image
blobs_dog = blob_dog(image2, max_sigma=20, threshold=0.00)

# Display the original image
plt.imshow(image2, cmap='gray', interpolation='nearest')
Y=image2.shape[0]
X=image2.shape[1]


# Plot the detected blobs
total_loop_length=len(blobs_dog)
loop_index=1
radii=[]
threshold_radius=3

for j in range(total_loop_length):
    y, x, r = blobs_dog[j]
    a=2.5
    if x>a*r and (X-x)>a*r and y>a*r and (Y-y)>a*r and r>=threshold_radius:
        radius=[]
        for i in range(len(angle)):
            rad=0
            start_point=[int(y+a*r*np.sin(angle[i])),int(x-a*r*np.cos(angle[i]))]
            end_point=[int(y-a*r*np.sin(angle[i])),int(x+a*r*np.cos(angle[i]))]
            linecut,image3=linecut_function(start_point,end_point,image2,image3)
            rad=sum(linecut)
            radius.append(rad)
        
        if(np.mean(radius)>threshold_radius):
            radii.append(np.mean(radius))
            image3=cv2.circle(image3,(int(x),int(y)),int(a*r),(0,255,255),3)
        
    print(f"progress: {loop_index*100/total_loop_length} %")
    loop_index+=1
    
plt.imshow(image3)
plt.axis("off")
plt.show()

print(radii)
print(len(radii))

plt.hist(radii,bins=int(max(radii)))
plt.xlabel("radius of the dip")
plt.ylabel("Histogram of the radius")
plt.show()


for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()