# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:15:02 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 500  # high-res display

period = 1

def f(t):
    return t**2

def periodic_function(t,a):
    t = t % period
    return(f(t-a))

t = np.linspace(-5,5 , 1000)
y = periodic_function(t,0.5)

plt.plot(t, y)
plt.show()