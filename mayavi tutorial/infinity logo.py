# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:47:56 2025

@author: mrsag
"""

import numpy as np
from mayavi import mlab

# Parameters
n_points = 400  # number of points around the figure-eight path
n_circle = 40   # number of points around the circular cross-section
R = 3           # radius of the figure-eight loops
r = 0.5         # radius of the tube

# Parameter along the main path
t = np.linspace(0, 2 * np.pi, n_points)

# Infinity-shaped (figure-eight) path in 3D
x = R * np.sin(t)
y = R * np.sin(t) * np.cos(t)
z = R * np.cos(t)

# Compute Frenet frame (approximate)
dx = np.gradient(x)
dy = np.gradient(y)
dz = np.gradient(z)

# Normalize tangent vectors
tangent = np.array([dx, dy, dz])
tangent /= np.linalg.norm(tangent, axis=0)

# Choose arbitrary normal using cross product with a fixed vector
up = np.array([0, 0, 1])
normals = np.cross(tangent.T, up)
normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

# Compute binormal as cross product of tangent and normal
binormals = np.cross(tangent.T, normals)

# Create the tube surface
theta = np.linspace(0, 2 * np.pi, n_circle)
circle_x = r * np.cos(theta)
circle_y = r * np.sin(theta)

# Allocate mesh arrays
X = np.zeros((n_circle, n_points))
Y = np.zeros((n_circle, n_points))
Z = np.zeros((n_circle, n_points))

# Sweep circle around the path
for i in range(n_points):
    cx = x[i]
    cy = y[i]
    cz = z[i]

    n = normals[i]
    b = binormals[i]

    # Create circle in the normal-binormal plane
    X[:, i] = cx + circle_x * n[0] + circle_y * b[0]
    Y[:, i] = cy + circle_x * n[1] + circle_y * b[1]
    Z[:, i] = cz + circle_x * n[2] + circle_y * b[2]

# Plot the mesh
mlab.figure(bgcolor=(1, 1, 1))
mlab.mesh(X, Y, Z,representation="wireframe")
mlab.show()
