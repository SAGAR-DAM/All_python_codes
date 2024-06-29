# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:59:31 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib
import time

matplotlib.rcParams['figure.dpi'] = 500  # highres display

x = np.linspace(0, 3,201)
y = np.linspace(0, 3,201)
X, Y = np.meshgrid(x, y)

period = 0.2
amplitude = 10

plasma_box_thickness = 0.5

def f(t):
    return t**2

def periodic_function(t,a):
    t = t % period
    return(amplitude*f(t-a))

def plasma_density(x,y):
    if(y<plasma_box_thickness+periodic_function(x,0.1)):
        return(0.999)
    else:
        return(0)
    
plasma_density=np.vectorize(plasma_density)

dense=plasma_density(X,Y)

plt.imshow(dense,cmap="hot",extent=[min(x),max(x),min(y),max(y)])
plt.title("Density map of electron", fontname="Times New Roman", fontsize=10)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()