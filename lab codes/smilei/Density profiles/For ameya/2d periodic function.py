# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:38:40 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib
import time

matplotlib.rcParams['figure.dpi'] = 500  # highres display

x = np.linspace(0,10,501)
y = np.linspace(0,10,501)
X, Y = np.meshgrid(x, y)

period_x = 1
period_y = 1


def circle(x, y):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return (r2<0.25**2)

# circle = np.vectorize(circle)

def square(x,y):
    return((x>0.25)*(x<0.75)*(y>0.25)*(y<0.75))

def gaussian(x,y):
    r2=(x-0.5)**2+(y-0.5)**2
    return(np.exp(-r2))

def periodic(x,y):
    x = x % period_x
    y = y % period_y
    
    return(square(x,y))

periodic=np.vectorize(periodic)

image=periodic(X,Y)

plt.imshow(image,extent=[min(x),max(x),min(y),max(y)],cmap="hot")
plt.show()
