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

c = 0.3   # in mm/ps
min_wavelength = 410
max_wavelength = 420

def find_index(array, value):
    absolute_diff = np.abs(array - value)
    index = np.argmin(absolute_diff)
    return index

def find_w0_std(filepath):
    files = glob.glob(filepath)
    p = []
    
    for i in range(len(files)):
        f = open(files[i])
        r = np.loadtxt(f, skiprows=17, comments='>')
        
        wavelength = r[:, 0]
        intensity = r[:, 1]
        intensity /= max(intensity)
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw]
        
        intensity -= np.mean(intensity[0:50])
        
        fit_I, parameters, string = Gf.Gaussfit(wavelength, intensity)
        
        error = np.std(fit_I - intensity)
        
        if error < 0.03 and max(fit_I) > 0.2:
            p.append(parameters[1])
    
    if len(p) > 0:
        peak_w0 = np.mean(p)
        std_w = np.std(p)
        return peak_w0, std_w
    return None, None

pr_only = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\pr only\\*.txt"    
peak_pr_only, std_pr_only = find_w0_std(pr_only)

delays = np.array([15.15, 15.9, 16.05, 16.2, 16.35, 16.5, 16.65, 16.8, 16.95, 17.1, 17.25, 17.4, 17.55, 17.7, 17.85, 18, 18.15, 18.3, 18.45, 18.6, 18.75, 18.9, 19.2, 19.5, 19.8, 20.1, 20.85, 21.3, 21.75])
time_delay = (delays - 17.4) * 2 / c

peaks = []
fit_I_arr = []
raw_data_arr = []
errors = []

for i in range(len(delays)):
    delay = delays[i]
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\{delay}\\*.txt"
    
    peak, std = find_w0_std(filepath)
    
    peaks.append(peak)
    errors.append(max([std, std_pr_only]))

peaks = np.array(peaks)

for i in range(len(delays)):
    delay = delays[i]
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\{delay}\\*.txt"
    files = glob.glob(filepath)
    raw = []
    fit = []
    counter = 0
    for j in range(len(files)):
        f = open(files[j])
        r = np.loadtxt(f, skiprows=17, comments='>')
        
        wavelength = r[:, 0]
        intensity = r[:, 1]
        intensity -= np.mean(intensity[0:200])
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw]
        intensity /= max(intensity)
        
        fit_I, parameters, string = Gf.Gaussfit(wavelength, intensity)
        error = np.std(fit_I - intensity)
        
        if error < 0.03 and max(fit_I) > 0.2:
            raw.append(intensity)
            fit.append(fit_I)
            counter += 1
            
    raw = np.array(raw)
    fit = np.array(fit)
    
    raw = np.sum(raw, axis=0)
    fit = np.sum(fit, axis=0)
    
    raw /= counter
    fit /= counter
    
    raw_data_arr.append(raw)
    fit_I_arr.append(fit)   

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

t0_index = find_index(time_delay, 0)
lambda0 = peak_pr_only
pr_only_w = np.ones(10) * lambda0
pr_only = np.linspace(0, 1, len(pr_only_w))

def f(x, t):
    return (0.5 * np.exp(-20 * (x - t)**2) * np.cos(100 * (x - t)))**2 * np.sign(-t)

theta = np.pi / 10

# Prepares the lines for the first frame
line1, = plt.plot([], [], 'b-', lw=2, label="Raw Data")
line2, = plt.plot([], [], lw=0.5, color='k', label="Fit")
line3, = plt.plot(pr_only_w, pr_only, 'g--', lw=1, label="pr_only")
line, = plt.plot([], [], lw=0.5, color='red', label="Laser")

# Create an inset axis in the top-left corner for the image
img = mpimg.imread("D:\\for ppt\\6e18 energy plot.png")
inset_ax = fig.add_axes([0.1, 0.45, 0.3, 0.4])  # Adjust position and size as needed
inset_ax.imshow(img, aspect='auto')
inset_ax.axis('off')  # Hide the axis for the inset image

plt.legend()
ax.set_xlim(412, 419)
ax.set_ylim(-0.2, 1.2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.grid(lw=0.2, color="k")

# Update function for animation
def update(frame):
    delay_index = frame
    peak = peaks[delay_index]
    
    # Update line data
    line1.set_xdata(wavelength)
    line1.set_ydata(raw_data_arr[delay_index])
    
    line2.set_xdata(wavelength)
    line2.set_ydata(fit_I_arr[delay_index])
    
    t = time_delay[delay_index]
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
    
    # Update line color based on the peak value
    norm = Normalize(vmin=415.5, vmax=415.7)
    colormap = plt.cm.jet
    color = colormap(norm(peak))
    line1.set_color(color)
    line2.set_color("k")

    ax.set_title(r"Spectrum for I = $6\times 10^{18} W/cm^2$" + f"\nSpectrum at delay: {time_delay[delay_index]:.2f} ps;  Peak wavelength: {peak:.3f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")

# Create and save animation as a GIF
ani = FuncAnimation(fig, update, frames=len(time_delay), interval=100)
ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\codes\\spectrum_animation.gif", writer="pillow", fps=3)

plt.show()
