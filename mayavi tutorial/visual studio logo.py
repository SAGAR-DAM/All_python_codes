# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:56:06 2025

@author: mrsag
"""

import numpy as np
from mayavi import mlab

# Parameters
n_points = 300      # number of points along the path
width = 0.5         # half-width of the ribbon
R = 3               # size of the loop
twists = 1          # number of half-twists in the ribbon

# Parameter along the path
t = np.linspace(0, 2.1 * np.pi, n_points)

# Infinity (figure-eight) path
x = R * np.sin(t)
y = R * np.sin(t) * np.cos(t)
z = R * np.cos(t)

# Tangent vectors
dx = np.gradient(x)
dy = np.gradient(y)
dz = np.gradient(z)
tangent = np.array([dx, dy, dz])
tangent /= np.linalg.norm(tangent, axis=0)

# Fixed up vector to construct normal and binormal
up = np.array([0, 1, 0])
normals = np.cross(tangent.T, up)
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
binormals = np.cross(tangent.T, normals)

# Twisting ribbon: define two edges of the ribbon
theta = twists * t  # twist angle along the path
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

# Left and right edges of the ribbon
X = np.zeros((2, n_points))
Y = np.zeros((2, n_points))
Z = np.zeros((2, n_points))

for i in range(n_points):
    offset = width * (cos_theta[i] * normals[i] + sin_theta[i] * binormals[i])
    X[0, i] = x[i] + offset[0]
    Y[0, i] = y[i] + offset[1]
    Z[0, i] = z[i] + offset[2]
    
    X[1, i] = x[i] - offset[0]
    Y[1, i] = y[i] - offset[1]
    Z[1, i] = z[i] - offset[2]

# Display twisted ribbon
mlab.figure(bgcolor=(1, 1, 1))
mlab.mesh(X, Y, Z, colormap="Purples")  # Visual Studio-ish purple
mlab.show()
