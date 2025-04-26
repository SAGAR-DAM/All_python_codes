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
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.dpi'] = 300  # high-res display

c = 0.3  # in mm/ps

def find_index(array, value):
    absolute_diff = np.abs(array - value)
    index = np.argmin(absolute_diff)
    return index

def moving_average(signal, window_size):
    window = np.ones(window_size) / float(window_size)
    filtered_signal = fftconvolve(signal, window, mode='same')
    return filtered_signal

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")

delay = np.linspace(9.5, 13.5, len(files)//2) - 10.5
delay = 2 * delay / c
delay = np.around(delay, decimals=3)

fit_I_arr = []
raw_data_arr = []
peaks = []

min_wavelength = 390
max_wavelength = 399

for i in range(1, len(files), 2):
    f = open(files[i])
    r = np.loadtxt(f, skiprows=17, comments='>')
    
    wavelength = r[:, 0]
    intensity = r[:, 1]
    
    intensity -= np.mean(intensity[0:50])
    minw = find_index(wavelength, min_wavelength)
    maxw = find_index(wavelength, max_wavelength)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw]
    
    intensity /= max(intensity)
    fit_I, parameters, string = Gf.Gaussfit(wavelength, intensity)
    raw_data_arr.append(intensity)
    fit_I_arr.append(fit_I)
    peaks.append(parameters[1])

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

t0_index = find_index(delay, 0)
lambda0 = np.mean(peaks[0:5])
pr_only_w = np.ones(10) * lambda0
pr_only = np.linspace(0, 1, len(pr_only_w))

def f(x, t):
    return (0.5 * np.exp(-20 * (x - t)**2) * np.cos(100 * (x - t)))**2 * np.sign(-t)

theta = np.pi / 10

line1, = plt.plot(wavelength, raw_data_arr[t0_index], 'b-', lw=2, label="Raw Data")
line2, = plt.plot(wavelength, fit_I_arr[t0_index], lw=0.5, color='k', label="Fit")
line3, = plt.plot(pr_only_w, pr_only, 'g--', lw=1, label="pr_only")
line, = plt.plot([], [], lw=0.5, color='red', label="Laser")

# Create an inset axis in the top-left corner for the image
img = mpimg.imread("D:\\for ppt\\3e17 energy plot.png")
inset_ax = fig.add_axes([0.1, 0.45, 0.3, 0.4])  # Adjust position and size as needed
inset_ax.imshow(img, aspect='auto')
inset_ax.axis('off')  # Hide the axis for the inset image

plt.legend()
ax.set_xlim(min_wavelength, max_wavelength)
ax.set_ylim(-0.2, 1.2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.grid(lw=0.2, color="k")

# Set up colormap normalization from -50 to 50
norm = Normalize(vmin=395, vmax=396.2)
colormap = plt.cm.jet  # Use the 'jet' colormap for the desired color transition

# Update function for animation
def update(delay_index):
    peak = peaks[delay_index]

    # Update line data
    line1.set_xdata(wavelength)
    line1.set_ydata(raw_data_arr[delay_index])
    
    line2.set_xdata(wavelength)
    line2.set_ydata(fit_I_arr[delay_index])
    
    t = delay[delay_index]
    if t < 0:
        x = np.linspace(-50, 0, 10001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = -np.sin(theta) * x + np.cos(theta) * y
    else:
        x = np.linspace(0, 50, 10001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = np.sin(theta) * x - np.cos(theta) * y
        
    line.set_ydata(yp)
    line.set_xdata(xp + lambda0)
    
    # Update line color based on the slider value
    color = colormap(norm(peak))
    line1.set_color(color)
    line2.set_color("k")
    line.set_color("red")

    ax.set_title(r"Spectrum for I = $3.1\times 10^{17} W/cm^2$" + f"\nSpectrum at delay: {delay[delay_index]} ps;  Peak wavelength: {peak:.3f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")

    # Update the legend to reflect the new color of line1
    ax.legend([line1, line2, line3, line], ["Raw Data", "Fit", "pr_only", "Laser"], loc="upper right")
    fig.canvas.draw_idle()

# Create and save animation as a GIF
ani = FuncAnimation(fig, update, frames=len(delay), interval=100)
ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\spectrum_animation.gif", writer="pillow", fps=10)

plt.show()
