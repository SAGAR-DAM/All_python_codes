# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:27:04 2025
@author: mrsag
"""

from mayavi import mlab
import numpy as np

# Create a disk (filled circle) at a given height
def create_disk(radius=10, z_height=0, resolution=100, color=(0.6, 0.6, 0.6), x0=0, y0=0):
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta) + x0
    y = radius * np.sin(theta) + y0
    z = np.full_like(x, z_height)

    # Add center point
    x = np.append(x, x0)
    y = np.append(y, y0)
    z = np.append(z, z_height)

    # Create triangle fan
    n = len(theta)
    triangles = [[i, (i + 1) % n, n] for i in range(n)]
    triangles = np.array(triangles)

    return mlab.triangular_mesh(x, y, z, triangles, color=color)

# Create the base plate (disk with thickness)
def create_base(radius=10, height=0.2):
    # Side
    phi, z = np.mgrid[0:2*np.pi:200j, 0:height:2j]
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    mlab.mesh(x, y, z, color=(0.6, 0.6, 0.6))

    # Faces
    create_disk(radius, z_height=0, color=(0.6, 0.6, 0.6))
    create_disk(radius, z_height=height, color=(0.6, 0.6, 0.6))

# Create a single solid pillar (side + top/bottom disks)
def create_pillar(x0, y0, radius=0.3, height=2.0, resolution=30):
    # Side
    phi, z = np.mgrid[0:2*np.pi:resolution*1j, 0:height:2j]
    x = radius * np.cos(phi) + x0
    y = radius * np.sin(phi) + y0
    mlab.mesh(x, y, z)

    # Caps
    create_disk(radius, z_height=0, color=(0.2, 0.2, 0.7), x0=x0, y0=y0)
    create_disk(radius, z_height=height, color=(0.2, 0.2, 0.7), x0=x0, y0=y0)

# Initialize the figure
mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

# Draw the base plate
create_base(radius=2, height=1.2)

# Create nanopillars in a grid
n_pillars = 10
spacing = 0.4
pillar_radius = 0.05
pillar_height = 2.2
offset = -(n_pillars - 1) / 2 * spacing

for i in range(n_pillars):
    for j in range(n_pillars):
        x = offset + i * spacing
        y = offset + j * spacing
        if np.sqrt(x**2 + y**2) < 1.7:
            create_pillar(x, y, radius=pillar_radius, height=pillar_height)

mlab.view(azimuth=45, elevation=75, distance=30)
mlab.show()
