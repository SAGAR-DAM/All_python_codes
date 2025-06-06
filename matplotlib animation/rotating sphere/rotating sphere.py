# -*- coding: utf-8 -*-
"""
Created on Mon May  5 21:15:26 2025

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

# Generate sphere surface
phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)



# Apply color gradient: red at poles, blue at equator
norm = colors.Normalize(vmin=-1, vmax=1)
facecolors = cm.jet(norm(z))

# Plotting
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface with gradient color
ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1,
                linewidth=0, antialiased=True, shade=False,alpha=0.3)

ax.plot_wireframe(1.02*x, 1.02*y, 1.02*z, color='b', linewidth=0.5, rstride=5, cstride=5)

# Point (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
x_end = 1 / np.sqrt(3)+0.05
y_end = 1 / np.sqrt(3)+0.05
z_end = 1 / np.sqrt(3)+0.05

# Draw a solid black line from center (0, 0, 0) to (x_end, y_end, z_end)
ax.plot([0, x_end], [0, y_end], [0, z_end], color='k', linewidth=2)
ax.plot([0, 0.71], [0, 0.71], [0, 0], color='k', linewidth=2)
ax.plot([x_end], [y_end], [z_end], color='r', marker='o', markersize=10)


# Draw the line along the equator from 0 to 45 degrees (theta: 0 to pi/4)
theta_line = np.linspace(0, np.pi/4, 20)
x_eq = np.cos(theta_line)*1.05
y_eq = np.sin(theta_line)*1.05
z_eq = np.zeros_like(theta_line)

ax.plot(x_eq, y_eq, z_eq, color='k', linewidth=2)
ax.plot(0.4*x_eq, 0.4*y_eq, 0.4*z_eq, color='k', linewidth=2)
ax.text(x_eq[5]*0.7, y_eq[5]*0.5, z_eq[10]*0.7, s=r'$2\psi$', color='k', fontsize=20)


phi_line = np.linspace(0.955,np.pi/2,20)
x_lat = np.sin(phi_line)*np.cos(np.pi/4)*1.05
y_lat = np.sin(phi_line)*np.cos(np.pi/4)*1.05
z_lat = np.cos(phi_line)*1.05

ax.plot(x_lat, y_lat, z_lat, color='k', linewidth=2)
ax.plot(0.4*x_lat, 0.4*y_lat, 0.4*z_lat, color='k', linewidth=2)
ax.text(x_lat[5]*1.1, y_lat[5]*1.1, z_lat[10]*1.1, s=r'$2\chi$', color='k', fontsize=20)

# Plot 3D axes
axis_length = 2  # extend beyond sphere

# X-axis (red)
ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', linewidth=2,arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, -axis_length, 0, 0, color='r', linewidth=2,arrow_length_ratio=0.1)
ax.text(axis_length, 0, 0, 'S1', color='r', fontsize=18)

# Y-axis (green)
ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', linewidth=2,arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, -axis_length, 0, color='g', linewidth=2,arrow_length_ratio=0.1)
ax.text(0, axis_length, 0, 'S2', color='g', fontsize=18)

# Z-axis (blue)
ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', linewidth=2,arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, -axis_length, color='b', linewidth=2,arrow_length_ratio=0.1)
ax.text(0, 0, axis_length, 'S3', color='b', fontsize=18)

ax.quiver(1.1,0,0,0,0.3,0,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(1.1,0,0,0,-0.3,0,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(-1.1,0,0,0,0,0.3,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(-1.1,0,0,0,0,-0.3,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(0,1.1,0,0.25,0,0.25,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(0,1.1,0,-0.25,0,-0.25,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(0,-1.1,0,0.25,0,-0.25,color='k', linewidth=2,arrow_length_ratio=0.4)
ax.quiver(0,-1.1,0,-0.25,0,0.25,color='k', linewidth=2,arrow_length_ratio=0.4)

# Circle parameters
r_top = 0.2  # radius
z_center_top= 1.1
theta_top = np.linspace(0, 2 * np.pi, 20)

# Since it's in the XZ-plane (y is constant)
x_top = r_top * np.cos(theta_top)
y_top = np.zeros_like(theta_top)  # constant y
z_top = z_center_top + r_top * np.sin(theta_top)

ax.plot(x_top, y_top, z_top, color='k', linewidth=2)  # 'k' = black
ax.plot(x_top, y_top, -z_top, color='k', linewidth=2)  # 'k' = black
i=0
ax.quiver(x_top[i],y_top[i],z_top[i],0,0,-0.05,color='k', linewidth=2,arrow_length_ratio=3)
ax.quiver(x_top[i],y_top[i],-z_top[i],0,0,0.05,color='k', linewidth=2,arrow_length_ratio=3)


# Define the center of the sphere
x0 = 1 / np.sqrt(3)
y0 = 1 / np.sqrt(3)
z0 = 1 / np.sqrt(3)

# Normal vector (the direction perpendicular to the plane)
n = np.array([1, 1, 1])

# Define arbitrary vector not parallel to n (we can use (1, 0, -1))
v1 = np.array([1, 0, -1])

# Calculate the second tangent vector by taking the cross product of n and v1
v2 = np.cross(n, v1)

# Normalize the tangent vectors
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# Ellipse semi-axes
r1_ellipse = 0.3  # radius along the first tangent direction
r2_ellipse = 0.15  # radius along the second tangent direction
theta_ellipse = np.linspace(0, 2 * np.pi, 20)

# Parametrize the ellipse in the tangent plane
# Parametric equations for the ellipse
x_ellipse = x0*1.1 + r1_ellipse * np.cos(theta_ellipse) * v2[0] + r2_ellipse * np.sin(theta_ellipse) * v1[0]
y_ellipse = y0*1.1 + r1_ellipse * np.cos(theta_ellipse) * v2[1] + r2_ellipse * np.sin(theta_ellipse) * v1[1]
z_ellipse = z0*1.1 + r1_ellipse * np.cos(theta_ellipse) * v2[2] + r2_ellipse * np.sin(theta_ellipse) * v1[2]

ax.plot(x_ellipse, y_ellipse, z_ellipse, color='k', linewidth=2)
ax.quiver(x_ellipse[-5],y_ellipse[-5],z_ellipse[-5],0.05,-0.07,0.05,color='k', linewidth=2,arrow_length_ratio=2)

# Clean up
ax.set_box_aspect([1, 1, 1])  # make aspect ratio equal
ax.axis('off')
plt.show()
