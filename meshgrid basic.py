# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:49:09 2023

@author: mrsag
"""
import numpy as np
nx, ny = (5,3)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 2, ny)
xv, yv = np.meshgrid(x, y)

print(xv)
print(yv)
z=xv+1j*yv

import matplotlib.pyplot as plt
plt.plot(xv, yv, marker='o')
plt.show()

print(z[0,1])

xv=[[1,2,3],[3,2,5],[4,5,6]]
yv=[[1,2,3],[4,5,6],[-1,0,-8]]
plt.plot(xv,yv,marker='o')
