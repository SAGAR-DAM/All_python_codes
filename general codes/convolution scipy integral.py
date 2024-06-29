# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:29:12 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

x=np.linspace(-15,15,601)

def f(x):
    return(1*(abs(x)<5))

def g(x):
    return np.exp(-x**2)

conv=np.zeros(len(x))

def convolution(t):
    result, error = integrate.quad(lambda x:f(t-x)*g(x),-np.inf,np.inf)
    return(result)

convolution=np.vectorize(convolution)
conv = convolution(x)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, f(x), label='f(x)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, g(x), label='g(x)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, conv, label='Convolution of f and g')
plt.legend()

plt.tight_layout()
plt.show()

'''
def fg(x,a):
    return(f(x)*g(x-a))

plt.plot(x,fg(x,4))
plt.show()
'''