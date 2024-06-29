# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 22:43:41 2023

@author: mrsag
"""

# gratings.py

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-500, 501, 1)

X, Y = np.meshgrid(x, x)

wavelength = 50
angle = np.pi / 4
grating = np.sin(2*np.pi*(X*np.cos(angle) - Y*np.sin(angle)) / wavelength) + np.sin(2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength)

plt.set_cmap("gray")
plt.imshow(grating)
plt.show()