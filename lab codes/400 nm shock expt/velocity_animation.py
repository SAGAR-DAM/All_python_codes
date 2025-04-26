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
    filtered_signal = fftconvolve(signal, window, mode='valid')
    
    return filtered_signal


def find_w0_std(filepath):
    files = glob.glob(filepath)
    p = []

    
    for i in range(len(files)):
        f = open(files[i])
        r=np.loadtxt(f,skiprows=17,comments='>')
        
        wavelength = r[:,0]
        intensity = r[:,1]
        intensity /= max(intensity)
        
        minw = find_index(wavelength, 410)
        maxw = find_index(wavelength, 420)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw] 
        
        intensity -= np.mean(intensity[0:50])
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        
        error = np.std(fit_I-intensity)
        
        if(error<0.03 and max(fit_I)>0.2):
            p.append(parameters[1])
    #         plt.plot(wavelength, intensity)
    #         plt.plot(wavelength, fit_I,'-')
            
    # plt.title(filepath[85:-5])
    # plt.show()
            
    if(len(p)>0):
        peak_w0 = np.mean(p)
        std_w = np.std(p)
        
    return peak_w0, std_w



pr_only = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\pr only\\*.txt"    

peak_pr_only, std_pr_only = find_w0_std(pr_only)

delays = np.array([15.15,15.9,16.05,16.2,16.35,16.5,16.65,16.8,16.95,17.1,17.25,17.4,17.55,17.7,17.85,18,18.15,18.3,18.45,18.6,18.75,18.9,19.2,19.5,19.8,20.1,20.85,21.3,21.75])
time_delay = (delays-17.4)*2/c

peaks = []
errors = []

for i in range(len(delays)):
    delay = delays[i]
    print(delay)
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\{delay}\\*.txt"
    
    peak, std = find_w0_std(filepath)
    
    peaks.append(peak)
    errors.append(max([std,std_pr_only]))
    
peaks = np.array(peaks)

peaks = moving_average(signal=peaks, window_size=3)
time_delay = moving_average(signal=time_delay, window_size=3)

def calc_vel(w, w0):
    c_norm = 3e10
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

velocity = calc_vel(w = peaks, w0 = peak_pr_only)
critical_surf_pos = np.cumsum(velocity)


delay = time_delay
# Set up the grid for 3D plotting
grid_size = 200  # Grid resolution
x = np.linspace(-3, 3, grid_size)
y = np.linspace(-3, 3, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)  # Radial distance from center

# Gaussian mode function
def gaussian_mode(amplitude):
    return amplitude * np.exp(-R**2)

def laser(x,t):
    return 2*np.exp(-(x-t)**2/0.1) * np.cos(100 * x)

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
# ax.set_title(r"velocity (in cm/s) map for I = $6.1\times 10^{18} W/cm^2$")
ax.set_title(r"Critical surface (in $\mu m$) position for I = $6.1\times 10^{18} W/cm^2$")
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
    # ax.set_title(r"velocity (in cm/s) map for I = $6.1\times 10^{18} W/cm^2$"+f"\ndelay = {delay[frame]:.2f} ps")
    ax.set_title(r"Critical surface (in $\mu m$) position for I = $6.1\times 10^{18} W/cm^2$"+f"\ndelay = {delay[frame]:.2f} ps")
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
# ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\codes\\crit_surf_animation_I_6_18.gif", writer='pillow', fps=10)
# Show the plot
plt.show()







