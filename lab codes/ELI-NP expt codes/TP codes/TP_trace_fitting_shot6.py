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

imagepath = r"D:\data Lab\ELI-NP March 2025\01_04_2025\separated out TP images\01042025_TP_19_31_scan_2-[Phosphor]_Shot6.tif"
image = io.imread(imagepath)
a = 10
b = image.shape[0]
c = 0
d = 645
image = image[a:b,c:d]

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

points = [[0, 0]]

for i in range(650,1850,50):
    max_x_index = find_index(image[:,i],np.max(image[:,i]))
    x0,y0 = max_x_index, i
    points.append([x0,y0])
    
# points.append([image.shape[0]-(276-c), 2058-a])

points = np.array(points)
# Fit a parabola (y = ax^2 + bx + c)
x = points[:, 1]
y = points[:, 0]
coeffs = np.polyfit(x, y, 2)  # Quadratic fit


# Generate smooth parabola points
x_vals = np.linspace(min(x), max(x), 100)
y_vals = np.polyval(coeffs, x_vals)

# Define the axis range (e.g., X: 0 to 10, Y: 20 to 30)
x_min, x_max = 0, 10
y_min, y_max = 20, 30

plt.imshow(imagelog,cmap="inferno",origin="lower")# ,extent=[x_min, x_max, y_min, y_max])
plt.plot(x_vals, y_vals, color='red', linewidth=1)  # Draw parabola
plt.scatter(x, y, color='c', s=20)  # Mark given points
plt.title(imagepath[-45:])
# plt.colorbar()
plt.show()

plt.plot(x_vals, y_vals, color='red', linewidth=2)  # Draw parabola
plt.scatter(x, y, color='b', s=20)  # Mark given points
plt.title(imagepath[-45:])
plt.show()


