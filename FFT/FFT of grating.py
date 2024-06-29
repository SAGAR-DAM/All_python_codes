# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 22:55:44 2023

@author: mrsag
"""

# gratings.py

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,1001)

X, Y = np.meshgrid(x, x)

wavelength = 100
angle = np.pi/1000
'''
grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength)  + np.sin(
        2*np.pi*(X*np.cos(angle) - Y*np.sin(angle)) / wavelength)
'''
grating=np.sin(X)*np.sin(5*Y)
plt.set_cmap("gray")

plt.subplot(121)
plt.imshow(grating,cmap='hot')
#plt.colorbar()

# Calculate Fourier transform of grating
ft = np.fft.fftshift(grating)
#ft = np.fft.fft2(grating)
ft = np.fft.fft2(ft)
ft = np.abs(np.fft.fftshift(ft))

plt.subplot(122)
plt.imshow(abs(ft),cmap='gray')
plt.colorbar()
plt.xlim([480, 520])
plt.ylim([520, 480])  # Note, order is reversed for y
plt.show()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__