# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


# Set up the grid for 3D plotting
grid_size = 200  # Grid resolution
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)  # Radial distance from center
theta = np.pi/4

# Gaussian mode function
def gaussian_mode(amplitude):
    return amplitude * np.exp(-R**2)

def laser(x,t):
    return (np.exp(-(x-t)**2/0.1) * np.cos(100 * x))**2*np.sign(-t)

# Set up the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# amplitudes = -velocity
# amplitudes = -critical_surf_pos*np.diff(delay)[0]*1e-12*1e4
amplitudes = np.linspace(3,-3,100)
amplitudes = (amplitudes<0)*amplitudes

amplitudes = 3*amplitudes/(max(abs(amplitudes)))
delay = np.linspace(-10,10,len(amplitudes))

# Initialize the colormap normalization
norm = TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)#vmin=-max(abs(amplitudes)), vcenter=0, vmax=max(abs(amplitudes)))

# Initialize the surface plot with the first frame
Z = gaussian_mode(amplitudes[0])
surf = ax.plot_surface(X, Y, Z, cmap='seismic_r', norm=norm, edgecolor='k', linewidth=0.05, alpha=0.7)
# fig.colorbar(surf, ax=ax, aspect=20)
ax.set_zlim(-3,3)#-max(abs(amplitudes)) * 1.1, max(abs(amplitudes)) * 1.1)  # Set z-axis limits based on amplitude range
# ax.set_title(r"velocity (in cm/s) map for I = $1.3\times 10^{18} W/cm^2$")
ax.set_title(r"Critical surface (in $\mu m$) position for I = $1.3\times 10^{18} W/cm^2$")
# Reapply axis settings in update
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Update function for animation
def update(frame):
    ax.clear()

    if(delay[frame]<=0):
        y = np.linspace(-3, 0, 1001)
        z = laser(y,t=delay[frame])
        x = np.zeros(len(z))
        #theta = np.pi/4#np.arccot(max(abs(amplitudes))/max(y))
        yp = np.cos(theta) * y + np.sin(theta) * z
        zp = -np.sin(theta) * y + np.cos(theta) * z
    else:
        y = np.linspace(0, 3, 1001)
        z = laser(y,t=delay[frame])
        x = np.zeros(len(z))
        #theta = np.pi/4#np.arccot(max(abs(amplitudes))/max(y))
        yp = np.cos(theta) * y + np.sin(theta) * z
        zp = np.sin(theta) * y - np.cos(theta) * z
    Z = gaussian_mode(amplitudes[frame])
    ax.plot_surface(X, Y, Z, cmap='seismic_r', norm=norm, edgecolor='k', linewidth=0.05, alpha=0.7)
    ax.plot(x,yp,zp, color='blue', lw=0.5,alpha=1)
    ax.set_zlim(-3,3)#-max(abs(amplitudes)) * 1.1, max(abs(amplitudes)) * 1.1)  # Reapply z-axis limits
    # ax.set_title(r"velocity (in cm/s) map for I = $1.3\times 10^{18} W/cm^2$"+f"\ndelay = {delay[frame]:.2f} ps")
    ax.set_title(r"Critical surface (in $\mu m$) position for I = $1.3\times 10^{18} W/cm^2$"+f"\ndelay = {delay[frame]:.2f} ps")
    # Reapply axis settings in update
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    return ax,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(amplitudes), interval=100, blit=False)
# Save the animation as a GIF
# ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\codes\\crit_surf_ani_I_1_18_avg_scan1_2_with_laser.gif", writer='pillow', fps=10)
# Show the plot
plt.show()

