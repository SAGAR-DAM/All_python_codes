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
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from matplotlib import cm

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.dpi'] = 300  # highres display

c = 0.3  # in mm/ps

def find_index(array, value):
    absolute_diff = np.abs(array - value)
    index = np.argmin(absolute_diff)
    return index

def moving_average(signal, window_size):
    window = np.ones(window_size) / float(window_size)
    filtered_signal = fftconvolve(signal, window, mode='same')
    return filtered_signal

# Load data
min_wavelength = 405
max_wavelength = 415
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))

peaks1 = []
delay1 = np.linspace(12, 15, len(files_17) // 2) - 12.65
delay1 = 2 * delay1 / c
delay1 = np.around(delay1, decimals=3)

fit_I_arr_1 = []
raw_data_arr_1 = []

for i in range(0, len(files_17), 2):
    f = open(files_17[i])
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
    raw_data_arr_1.append(intensity)
    fit_I_arr_1.append(fit_I)
    peaks1.append(parameters[1])

# Post-processing peaks
for i in range(len(peaks1)):
    if (peaks1[i] < 409 or peaks1[i] > 412):
        try:
            peaks1[i] = (peaks1[i + 1] + peaks1[i - 1]) / 2
        except:
            peaks1[i] = 410.4

peaks1 = moving_average(peaks1, 4)

for i in range(len(peaks1)):
    if (peaks1[i] < 409 or peaks1[i] > 412):
        try:
            if ((peaks1[i - 1] < 412 and peaks1[i + 1] > 409) or (peaks1[i + 1] < 412 and peaks1[i + 1] > 409)):
                peaks1[i] = (peaks1[i + 1] + peaks1[i - 1]) / 2
            else:
                peaks1[i] = 410.4
        except:
            peaks1[i] = 410.4

delay1 = delay1[0:len(peaks1)]

# Load second scan data
files_17_scan2 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 2\\*.txt"))
fit_I_arr_2 = []
raw_data_arr_2 = []
peaks2 = []
delay2 = np.linspace(12, 15, len(files_17_scan2) // 2) - 12.65
delay2 = 2 * delay2 / c
delay2 = np.around(delay2, decimals=3)

for i in range(1, len(files_17_scan2), 2):
    f = open(files_17_scan2[i])
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
    raw_data_arr_2.append(intensity)
    fit_I_arr_2.append(fit_I)
    peaks2.append(parameters[1])

# Post-processing for peaks in scan 2
for i in range(len(peaks2)):
    if (peaks2[i] < 409 or peaks2[i] > 412):
        try:
            peaks2[i] = (peaks2[i + 1] + peaks2[i - 1]) / 2
        except:
            peaks2[i] = 410.3

peaks2 = moving_average(peaks2, 4)

for i in range(len(peaks2)):
    if (peaks2[i] < 409 or peaks2[i] > 412):
        try:
            if ((peaks2[i - 1] < 412 and peaks2[i + 1] > 409) or (peaks2[i + 1] < 412 and peaks2[i + 1] > 409)):
                peaks2[i] = (peaks2[i + 1] + peaks2[i - 1]) / 2
            else:
                peaks2[i] = 410.3
        except:
            peaks2[i] = 410.3

# Set up plot
img = plt.imread("D:\\for ppt\\1e18 energy plot.png")
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
t0_index = find_index(delay1, 0)

lambda0 = 410.35
pr_only_w = np.ones(10) * lambda0
pr_only = np.linspace(0, 1, len(pr_only_w))

def f(x, t):
    return (0.5 * np.exp(-20 * (x - t) ** 2) * np.cos(100 * (x - t))) ** 2 * np.sign(-t)

theta = np.pi / 10
t0 = delay1[t0_index]

if(t0 <= 0):
    x = np.linspace(-50, 0, 10001)
    y = f(x, t0)
    xp = np.cos(theta) * x + np.sin(theta) * y
    yp = -np.sin(theta) * x + np.cos(theta) * y
else:
    x = np.linspace(0, 50, 10001)
    y = f(x, t0)
    xp = np.cos(theta) * x + np.sin(theta) * y
    yp = -np.sin(theta) * x + np.cos(theta) * y

line1, = plt.plot(wavelength, raw_data_arr_1[t0_index], 'b-', lw=2, label="Scan 1")
line2, = plt.plot(wavelength, raw_data_arr_2[t0_index], 'b-', lw=2, label="Scan 2")
line3, = plt.plot(wavelength, fit_I_arr_1[t0_index], lw=0.5, color='k', label="Fit 1")
line4, = plt.plot(wavelength, fit_I_arr_2[t0_index], lw=0.5, color='b', label="Fit 2")
line5, = plt.plot(pr_only_w, pr_only, 'g--', lw=1, label="pr_only")
line_laser, = plt.plot(xp + lambda0, yp + 200, lw=0.5, color='red')

# Create an inset axis for the image
inset_ax = fig.add_axes([0.12, 0.45, 0.3, 0.4])
inset_ax.imshow(img, aspect='auto', alpha=0.7)
inset_ax.axis('off')  # Hide the axis for the inset image

plt.legend()
ax.set_xlim(min_wavelength, max_wavelength)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.grid(color="k", lw=0.2)

# Create animation
num_frames = len(delay1)
def update(frame):
    t_index = frame % len(delay1)
    peak1 = peaks1[t_index]
    peak2 = peaks2[t_index]

    # Update line data
    line1.set_ydata(raw_data_arr_1[t_index])
    line2.set_ydata(raw_data_arr_2[t_index])
    line3.set_ydata(fit_I_arr_1[t_index])
    line4.set_ydata(fit_I_arr_2[t_index])
    
    t = delay1[t_index]
    if(t < 0):
        x = np.linspace(-50, 0, 10001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = -np.sin(theta) * x + np.cos(theta) * y
    else:
        x = np.linspace(0, 50, 10001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = np.sin(theta) * x - np.cos(theta) * y
        
    line_laser.set_ydata(yp)
    line_laser.set_xdata(xp + lambda0)

    # Update title
    ax.set_title(r"Spectrum for I = $1\times 10^{18} W/cm^2$" + f"\nspectrum at delay: {delay1[t_index]} ps;  peak wavelength: scan1= {peak1:.3f} nm, scan2 = {peak2:.3f} nm")
    fig.canvas.draw_idle()

# Create FuncAnimation
ani = FuncAnimation(fig, update, frames=num_frames, interval=100)

# Save as GIF
ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\pos_17\\codes\\spectrum_animation.gif", writer="pillow", fps=3)

plt.show()
