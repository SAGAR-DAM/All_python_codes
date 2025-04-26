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
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

c = 0.3   #in mm/ps
min_wavelength=410
max_wavelength=420

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
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
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
fit_I_arr = []
raw_data_arr = []
errors = []

for i in range(len(delays)):
    delay = delays[i]
    print(delay)
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\{delay}\\*.txt"
    
    peak, std = find_w0_std(filepath)
    
    peaks.append(peak)
    errors.append(max([std,std_pr_only]))
    
peaks = np.array(peaks)

for i in range(len(delays)):
    delay = delays[i]
    print(delay)
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\30 th april 2024\\delays\\{delay}\\*.txt"
    files = glob.glob(filepath)
    raw = []
    fit = []
    counter = 0
    for i in range(0,len(files)):
        f = open(files[i])
        r=np.loadtxt(f,skiprows=17,comments='>')
        
        wavelength = r[:,0]
        intensity = r[:,1]
        intensity -= np.mean(intensity[0:200])
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw] 
        intensity /= max(intensity)
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        error = np.std(fit_I-intensity)
        
        if(error<0.03 and max(fit_I)>0.2):
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
pr_only_w = np.ones(10)*lambda0
pr_only = np.linspace(0,1,len(pr_only_w))


line1, = plt.plot(wavelength, raw_data_arr[t0_index], 'b-', lw=2, label="raw data")
line2, = plt.plot(wavelength, fit_I_arr[t0_index], lw=0.5, color='k', label="Fit")
line3,  = plt.plot(pr_only_w, pr_only,  'g--' ,lw=1, label= "pr_only")


# Create an inset axis in the top-left corner for the image
img = plt.imread("D:\\for ppt\\6e18 energy plot.png")
inset_ax = fig.add_axes([0.1, 0.45, 0.3, 0.4])  # Adjust position and size as needed
inset_ax.imshow(img, aspect='auto')
inset_ax.axis('off')  # Hide the axis for the inset image

plt.legend()
#ax.set_aspect('equal', adjustable='box')
axcolor = 'lightgoldenrodyellow'
ax.set_xlim(412, 419)

ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.grid(lw=0.2,color="k")

# Create slider for t
ax_t = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
slider_t = Slider(ax_t, 'delay', valmin=0, valmax=len(time_delay)-1, valinit=t0_index, valstep=1)

# Set up colormap normalization from -50 to 50
norm = Normalize(vmin=415.5, vmax=415.7)
colormap = plt.cm.jet  # Use the 'jet' colormap for the desired color transition

def update(val):
    delay_index = slider_t.val
    peak = peaks[delay_index]

    # Update line data
    line1.set_xdata(wavelength)
    line1.set_ydata(raw_data_arr[delay_index])
    
    line2.set_xdata(wavelength)
    line2.set_ydata(fit_I_arr[delay_index])
    
    
    # Update line color based on the slider value
    color = colormap(norm(peak))  # Get color from colormap
    line1.set_color(color)
    line2.set_color("k")

    ax.set_title(r"Spectrum for I = $6\times 10^{18} W/cm^2$"+f"\nspectrum at delay: {time_delay[delay_index]:.2f} ps;  peak wavelength: {peak:.3f} nm")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")

    # Update the legend to reflect the new color of line1
    ax.legend([line1, line2, line3], ["Raw Data", "Fit", "pr_only"])
    fig.canvas.draw_idle()

slider_t.on_changed(update)

plt.show()

