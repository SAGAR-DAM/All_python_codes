# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:03:08 2025

@author: mrsag
"""

import numpy as np
from mayavi import mlab

# Parametric equations for the Klein bottle
def klein_bottle(u, v):
    # Parameters for the Klein bottle
    r = 4 * (1 - np.cos(u) / 2)
    
    # x, y, z coordinates
    x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(u) * np.cos(v)
    y = 16 * np.sin(u) + r * np.sin(u) * np.cos(v)
    z = r * np.sin(v)
    
    # Adjust for the "handle" part of the Klein bottle
    mask = (u > np.pi)
    x[mask] = 6 * np.cos(u[mask]) * (1 + np.sin(u[mask])) + r[mask] * np.cos(v[mask] + np.pi)
    y[mask] = 16 * np.sin(u[mask])
    z[mask] = r[mask] * np.sin(v[mask])
    
    return x, y, z

# Create the mesh grid for parameters u and v
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)

# Compute the coordinates
x, y, z = klein_bottle(u, v)

# Create a Mayavi figure
mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))

# Plot the surface
mlab.mesh(x, y, z, colormap='jet', opacity=0.5, representation="surface")

# Add a colorbar
mlab.colorbar(title="Klein Bottle Surface", orientation='vertical')

# Set up axes
axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z', color=(0, 0, 0))
axes.title_text_property.color = (0, 0, 0)  # Black title text
axes.label_text_property.color = (0, 0, 0)  # Black label text

# Add a title
mlab.title("3D Klein Bottle", color=(0, 0, 0))

# Adjust the view for better visualization
# mlab.view(azimuth=45, elevation=60, distance=30)

# Show the plot
mlab.show()