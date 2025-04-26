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

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []


for i in range(1,len(files),2):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 400)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 395.35

peaks = moving_average(peaks,10)

for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            if((peaks[i-1]<395 and peaks[i+1]>396) or (peaks[i+1]<395 and peaks[i+1]>396)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 395.35
        except:
            peaks[i] = 395.35

delay = delay[0:len(peaks)]


peaks = peaks[find_index(delay,-5):find_index(delay, 18)]
delay = delay[find_index(delay,-5):find_index(delay, 18)]

def calc_vel(w, w0):
    c_norm = 3e10
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

lambda0 = np.mean(peaks[0:5])


velocity = calc_vel(np.array(peaks), lambda0)
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

# amplitudes = -velocity
amplitudes = -critical_surf_pos*np.diff(delay)[0]*1e-12*1e4
# Initialize the colormap normalization
norm = TwoSlopeNorm(vmin=-max(abs(amplitudes)), vcenter=0, vmax=max(abs(amplitudes)))

# Initialize the surface plot with the first frame
Z = gaussian_mode(amplitudes[0])
surf = ax.plot_surface(X, Y, Z, cmap='seismic_r', norm=norm, edgecolor='k', linewidth=0.05, alpha=0.7)
fig.colorbar(surf, ax=ax, aspect=20)
ax.set_zlim(-max(abs(amplitudes)) * 1.1, max(abs(amplitudes)) * 1.1)  # Set z-axis limits based on amplitude range
# ax.set_title(r"velocity (in cm/s) map for I = $3\times 10^{17} W/cm^2$")
ax.set_title(r"Critical surface (in $\mu m$) position for I = $3\times 10^{17} W/cm^2$")
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
    # ax.set_title(r"velocity (in cm/s) map for I = $3\times 10^{17} W/cm^2$"+f"\ndelay = {delay[frame]} ps")
    ax.set_title(r"Critical surface (in $\mu m$) position for I = $3\times 10^{17} W/cm^2$"+f"\ndelay = {delay[frame]} ps")
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
# ani.save('D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\crit_surf animation I_3_17.gif', writer='pillow', fps=10)
# Show the plot
plt.show()





