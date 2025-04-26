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

c = 0.3   #in mm/ps

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


#################################################################
#################################################################
#################################################################
#################################################################
''' pos 17 scan 1'''
#################################################################
#################################################################
#################################################################
#################################################################
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))

peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)


peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

for i in range(0,len(files_17),2):
    f = open(files_17[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks1.append(parameters[1])
    

for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
        except:
            peaks1[i] = 410.4

peaks1 = moving_average(peaks1,4)

for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            if((peaks1[i-1]<412 and peaks1[i+1]>409) or (peaks1[i+1]<412 and peaks1[i+1]>409)):
                peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
            else:
                peaks1[i] = 410.4
        except:
            peaks1[i] = 410.4

delay1 = delay1[0:len(peaks1)]

#################################################################
#################################################################
#################################################################
#################################################################
''' pos 17 scan 2'''
#################################################################
#################################################################
#################################################################
#################################################################
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 2\\*.txt"))



peaks2 = []
delay2 = np.linspace(12,15,len(files_17)//2)-12.65
delay2 = 2*delay2/c
delay2 = np.around(delay2, decimals=3)

for i in range(1,len(files_17),2):
    f = open(files_17[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks2.append(parameters[1])
    
    
for i in range(len(peaks2)):
    if (peaks2[i]<409 or peaks2[i]>412):
        try:
            peaks2[i] = (peaks2[i+1]+peaks2[i-1])/2
        except:
            peaks2[i] = 410.3

peaks2 = moving_average(peaks2,4)

for i in range(len(peaks2)):
    if (peaks2[i]<409 or peaks2[i]>412):
        try:
            if((peaks2[i-1]<412 and peaks2[i+1]>409) or (peaks2[i+1]<412 and peaks2[i+1]>409)):
                peaks2[i] = (peaks2[i+1]+peaks2[i-1])/2
            else:
                peaks2[i] = 410.3
        except:
            peaks2[i] = 410.3

delay2 = delay2[0:len(peaks2)]


def calc_vel(w, w0):
    c_norm = 3e10
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

v1 = calc_vel(w = peaks1, w0 = 410.4)
v1_uerr = calc_vel(w = peaks1+0.064, w0 = 410.4)-v1
v1_lerr = v1-calc_vel(w = peaks1-0.064, w0 = 410.4)




v2 = calc_vel(w = peaks2, w0 = 410.3)
v2_uerr = calc_vel(w = peaks2+0.064, w0 = 410.3)-v2
v2_lerr = v2-calc_vel(w = peaks2-0.064, w0 = 410.3)



velocity = (v1+v2)/2


critical_surf_pos = np.cumsum(velocity)



# Set up the grid for 3D plotting
grid_size = 200  # Grid resolution
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)  # Radial distance from center

# Gaussian mode function
def gaussian_mode(amplitude):
    return amplitude * np.exp(-R**2)

# Set up the 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

amplitudes = -velocity
# amplitudes = -critical_surf_pos*np.diff(delay1)[0]*1e-12*1e4
# Initialize the colormap normalization
norm = TwoSlopeNorm(vmin=-max(abs(amplitudes)), vcenter=0, vmax=max(abs(amplitudes)))

# Initialize the surface plot with the first frame
Z = gaussian_mode(amplitudes[0])
surf = ax.plot_surface(X, Y, Z, cmap='seismic_r', norm=norm, edgecolor='k', linewidth=0.05, alpha=0.7)
fig.colorbar(surf, ax=ax, aspect=20)
ax.set_zlim(-max(abs(amplitudes)) * 1.1, max(abs(amplitudes)) * 1.1)  # Set z-axis limits based on amplitude range
ax.set_title(r"velocity (in cm/s) map for I = $1.3\times 10^{18} W/cm^2$")
# ax.set_title(r"Critical surface (in $\mu m$) position for I = $1.3\times 10^{18} W/cm^2$")
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
    Z = gaussian_mode(amplitudes[frame])
    ax.plot_surface(X, Y, Z, cmap='seismic_r', norm=norm, edgecolor='k', linewidth=0.05, alpha=0.7)
    ax.set_zlim(-max(abs(amplitudes)) * 1.1, max(abs(amplitudes)) * 1.1)  # Reapply z-axis limits
    ax.set_title(r"velocity (in cm/s) map for I = $1.3\times 10^{18} W/cm^2$"+f"\ndelay = {delay1[frame]} ps")
    # ax.set_title(r"Critical surface (in $\mu m$) position for I = $1.3\times 10^{18} W/cm^2$"+f"\ndelay = {delay1[frame]} ps")
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
# ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\codes\\velocity_ani_I_1_18_avg_scan1_2.gif", writer='pillow', fps=10)
# Show the plot
plt.show()





