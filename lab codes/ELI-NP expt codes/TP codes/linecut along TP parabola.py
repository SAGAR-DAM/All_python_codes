# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:55:40 2025

@author: mrsag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.integrate as integrate
from Curve_fitting_with_scipy import polynomial_fit as pft
import glob

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


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)

image = io.imread("D:\\data Lab\\ELI-NP March 2025\\20-03-2025\\TP data\\IP_TP_20_03_2025_14_40_scan2-[Phosphor]_cutted.tif")
a = 165
b = 2100
c = 250
d = 900
image = image[a:b,c:d]
noise = np.mean(image[:100,:100])


# image = np.flip(image, axis=1)
# image = np.flip(image, axis=0)
image = np.flipud(image.T)


imagelog = (np.log(image))
log_image_cutoff = 7
imagelog = imagelog > log_image_cutoff

# plt.imshow(imagelog)
# plt.show()

# # Given points (x, y) through which the parabola should pass
# points = np.array([[900-c, 165-a], [812-c,950-a], [720-c,1209-a], [371-c,1935-a], [470-c,1768-a], [276-c, 2058-a]])  # Example points


# points = [[image.shape[0]-(900-c), 165-a]]
points = [[image.shape[0]-(d-c), a-a]]

for i in range(700,1900,50):
    max_x_index = find_index(image[:,i],np.max(image[:,i]))
    x0,y0 = max_x_index, i
    points.append([x0,y0])
points.append([image.shape[0]-(276-c), 2058-a])

points = np.array(points)
# Fit a parabola (y = ax^2 + bx + c)
x = points[:, 1]
y = points[:, 0]
coeffs = np.polyfit(x, y, 2)  # Quadratic fit
coeffs[2] = 0


# Generate smooth parabola points
x_vals = np.linspace(min(x), max(x), 100)
y_vals = np.polyval(coeffs, x_vals)

# Define the axis range (e.g., X: 0 to 10, Y: 20 to 30)
x_min, x_max = 0, 10
y_min, y_max = 20, 30

plt.imshow(imagelog,cmap="inferno",origin="lower")# ,extent=[x_min, x_max, y_min, y_max])
plt.plot(x_vals, y_vals, color='red', linewidth=1)  # Draw parabola
plt.scatter(x, y, color='c', s=20)  # Mark given points
# plt.colorbar()
plt.show()


def parabolic_curve(x, a, b, c):
    """Returns y values for a parabola y = ax^2 + bx + c."""
    return a * x**2 + b * x + c

def extract_linecut(image, a, b, c, x_range=None, num_points=1000,logimage=False,log_image_cutoff=7, plot_logscale=False):
    """Extracts pixel intensity values along a parabolic curve."""
    
    h, w = image.shape
    
    imagelog = (np.log(image))
    imagelog = imagelog>log_image_cutoff

    # Define x range
    if x_range is None:
        x_range = (0, w-1)

    # Generate x values
    x_values = np.linspace(x_range[0], x_range[1], num_points).astype(int)
    y_values = parabolic_curve(x_values, a, b, c).astype(int)

    # Filter valid points
    valid_idx = (y_values >= 0) & (y_values < h)
    x_values, y_values = x_values[valid_idx], y_values[valid_idx]

    intensities = image[y_values, x_values]


        
    # Plot the results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Show original image with curve
    axes[0].imshow(imagelog, cmap="gray",origin="lower")
    axes[0].plot(x_values, y_values, 'r-', linewidth=2)
    axes[0].set_title("Parabolic Linecut on Image")
    axes[0].set_xlabel("x axis"+"\n=================================================================================")
    

    # Plot intensity profile
    axes[1].plot(x_values, intensities, 'b-', lw=1)
    axes[1].plot(point_avg(x_values,20),point_avg(intensities,20),"k-",lw=4)
    axes[1].axhline(y=noise+max(intensities)/1e2, color='r', linestyle='--', lw=2)
    if(plot_logscale):
        axes[1].set_yscale("log")
    axes[1].set_title("Intensity Profile Along Curve")
    axes[1].set_xlabel(r"Pixel Index Along $\vec v\times \vec B$ (x axis of parabola)")
    axes[1].set_ylabel("Intensity")
    axes[1].set_ylim(min(intensities)/1.1,max(intensities)*1.1)
    axes[1].set_xlim(0,image.shape[1])
    axes[1].grid(which="both", lw=0.5, color="k")


    plt.show()


extract_linecut(image, a=coeffs[0], b=coeffs[1], c=coeffs[2], num_points=image.shape[1],logimage = False, log_image_cutoff=log_image_cutoff,plot_logscale=True)  # Example parabola
