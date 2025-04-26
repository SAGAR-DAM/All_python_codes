# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:37:48 2025

@author: mrsag
"""

import numpy as np
from skimage import io, transform,color,img_as_ubyte,img_as_float
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from skimage.draw import line
import math

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def moving_avg(arr,n):
    window_size = n
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:
    
    	# Calculate the average of current window
    	window_average = round(np.sum(arr[
    	i:i+window_size]) / window_size, 2)
    	
    	# Store the average of current
    	# window in moving average list
    	moving_averages.append(window_average)
    	
    	# Shift window to right by one position
    	i += 1
    return(moving_averages)


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    arr1 = np.array(arr1)
    
    return(arr1)



def replace_zeros(image, small_value=1.0):
    img_array = img_as_float(image)  # Convert image to float format (if not already)
    img_array[img_array <= 1] = small_value  # Replace zero values
    return img_array


def draw_on_image(image,x,y):
    imagergb = color.gray2rgb(img_as_ubyte(image))
    rr, cc = line(x[0],x[1],y[0],y[1])
    line_width = 3
    for i in range(-line_width//2, line_width//2 + 1):
        imagergb[np.clip(rr + i, 0, np.array(image).shape[0] - 1), np.clip(cc + i, 0, np.array(image).shape[1] - 1)] = [255,0,0]  # Set the color of the line

    #image2[rr, cc] = [1]  # Set the color of the line (red in this example)
    return(imagergb)

################################################


def draw_on_image_pillow(image,x,y):
    draw = ImageDraw.Draw(image)
    draw.line((x[1],x[0],y[1],y[0]), fill=(255, 0, 0), width=2)
    return(image)

##################################################

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


##################################################



# Function to draw the line and faint red band on the image
def draw_on_image_with_band(image, start, end, band=5):
    if len(image.shape) == 2:
        image_rgb = np.repeat(image[:, :, np.newaxis], 3, axis=2)  # Convert grayscale to RGB
    else:
        image_rgb = image.copy()

    overlay = image_rgb.copy()

    rr, cc = line(start[0], start[1], end[0], end[1])
    line_width = 4

    for i in range(-line_width//2, line_width//2 + 1):
        image_rgb[np.clip(rr + i, 0, np.array(image).shape[0] - 1), np.clip(cc + i, 0, np.array(image).shape[1] - 1)] = [255,0,0]  # Set the color of the line

    # Compute angle of the line
    dx, dy = end[1] - start[1], end[0] - start[0]
    if(dx!=0):
        angle = np.arctan2(dy, dx)
    else:
        angle = np.pi/2

    # Compute perpendicular direction
    perp_dx = np.cos(angle + np.pi / 2)
    perp_dy = np.sin(angle + np.pi / 2)

    # Draw the faint red band
    for i in range(len(rr)):
        for j in range(-band, band + 1):
            px = np.clip(int(rr[i] + j * perp_dy), 0, image.shape[0] - 1)
            py = np.clip(int(cc[i] + j * perp_dx), 0, image.shape[1] - 1)
            overlay[px, py] = [100, 100, 0]  # Faint red band

    # Blend overlay with original image
    alpha = 0.3
    image_rgb = (1 - alpha) * image_rgb + alpha * overlay
    return np.clip(image_rgb, 0, 255).astype(np.uint8)


##########################################################


# Function to compute the linecut with band averaging
def linecut_function_with_band(start, end, image, image2, band=5):
    num = round(np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2))

    if num == 0:
        return np.array([]), image2

    x, y = np.linspace(start[0], end[0], num), np.linspace(start[1], end[1], num)

    # Compute angle of the line
    dx, dy = end[1] - start[1], end[0] - start[0]
    
    if(dx!=0):
        angle = np.arctan2(dy, dx)
    else:
        angle = np.pi/2

    # Compute perpendicular direction
    perp_dx = np.cos(angle + np.pi / 2)
    perp_dy = np.sin(angle + np.pi / 2)

    linecut = []
    for i in range(len(x)):
        band_pixels = []
        for j in range(-band, band + 1):
            px = np.clip(int(x[i] + j * perp_dy), 0, image.shape[0] - 1)
            py = np.clip(int(y[i] + j * perp_dx), 0, image.shape[1] - 1)
            band_pixels.append(image[px, py])

        linecut.append(np.mean(band_pixels))

    image2 = draw_on_image_with_band(image2, start, end, band)
    return np.array(linecut), image2

# Load the image
image_path = r"D:\data Lab\ELI-NP March 2025\UPM50018_Kumar_Ouside_CsI\CsI array\01042025\Image__2025-04-01__14-01-38.tiff"
image = io.imread(image_path)
image_2 = Image.open(image_path)
image_2 = image_2.transpose(Image.FLIP_LEFT_RIGHT)

image = np.flip(image,axis=1)

# plt.imshow(np.log(1+image),cmap="jet")
# plt.show()

upper_point = 288
lower_point = 1098
p1 = [upper_point,15*find_index(point_avg(image[upper_point,:],15)[:-1],max(point_avg(image[upper_point,:],15)[:-1]))]
p2 = [lower_point,15*find_index(point_avg(image[lower_point,:],15)[:-1],max(point_avg(image[lower_point,:],15)[:-1]))]

# Rotate the image by requirement
theta = np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
rotated_image = img_as_ubyte(transform.rotate(image,-math.degrees(theta)))


blob_width = 45
image = rotated_image[upper_point-blob_width:lower_point+blob_width,30:]

noise_image = rotated_image[lower_point:,30:]
linecut_noise,image2 = linecut_function_with_band([noise_image.shape[0]//2,0],[noise_image.shape[0]//2,noise_image.shape[1]],noise_image,noise_image,band=noise_image.shape[0]//2)


# Plot the linecut graph
avging_length = 5
offset = 14


# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot the image with the drawn line and band
ax[0].imshow(np.log(image2[:, :, 0] + image2[:, :, 1] + image2[:, :, 2]), cmap="inferno")
ax[0].set_title("Image with Line and Band")
ax[0].axis("off")


ax[1].plot(offset+avging_length*np.arange(0,len(point_avg(linecut_noise[offset:],avging_length))),point_avg(linecut_noise[offset:],avging_length), color="blue")
ax[1].set_title("Linecut with Band Averaging")
ax[1].set_xlabel("Position along the line")
# ax[1].set_yscale("log")
ax[1].grid(which="both",color="k",lw=0.5)
ax[1].set_ylabel("Average Intensity")

# Adjust layout and show both plots together
plt.tight_layout()
plt.show()


def plot_linecut_with_noise(image2,linecut,linecut_noise):
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the image with the drawn line and band
    ax[0].imshow(np.log(image2[:, :, 0] + image2[:, :, 1] + image2[:, :, 2])>2, cmap="inferno")
    ax[0].set_title("Image with Line and Band")
    ax[0].axis("off")
    
    x = offset+avging_length*np.arange(0,len(point_avg(linecut[offset:],avging_length)))
    y = point_avg(linecut[offset:],avging_length)
    y_noise = point_avg(linecut_noise[offset:],avging_length)
    
    ax[1].plot(x,y, color="blue")
    ax[1].fill_between(x,y-y_noise,y+y_noise,color="red",alpha=0.3)
    ax[1].set_title("Linecut with Band Averaging")
    ax[1].set_xlabel("Position along the line")
    # ax[1].set_yscale("log")
    ax[1].grid(which="both",color="k",lw=0.5)
    ax[1].set_ylabel("Average Intensity")
    
    # Adjust layout and show both plots together
    plt.tight_layout()
    plt.show()




layers = np.arange(25,image.shape[0],54)

for layer in layers:
    linecut_orientation="h"
    
    if(linecut_orientation=="v"):
        p1 = [0,layer]
        p2 = [image.shape[0]-1,layer]
    elif(linecut_orientation=="h"):
        p1 = [layer,0]
        p2 = [layer,image.shape[1]-1]
        
    linecut,image2 = linecut_function_with_band(p1,p2,image,image,band=15)
    plot_linecut_with_noise(image2, linecut, linecut_noise)