# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:14:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

def cornu_spiral(t):
    x, y = fresnel(t / np.sqrt(np.pi))
    return x, y

a=20
t_values = np.linspace(-a * np.sqrt(np.pi), a * np.sqrt(np.pi), int(a)*500+1)
x, y = cornu_spiral(t_values)

plt.figure(figsize=(8, 6))
plt.plot(x, y,'r-',linewidth=10)
plt.plot(x,y,'ko-',markersize=2,linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cornu Spiral')
plt.grid(True)
plt.axis('equal')
plt.show()