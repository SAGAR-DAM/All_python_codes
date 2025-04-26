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
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from matplotlib import cm

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
min_wavelength = 405
max_wavelength = 415



files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))

peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)


fit_I_arr_1 = []
raw_data_arr_1 = []
peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

for i in range(0,len(files_17),2):
    f = open(files_17[i])
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
    raw_data_arr_1.append(intensity)
    fit_I_arr_1.append(fit_I)
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


fit_I_arr_2 = []
raw_data_arr_2 = []
peaks2 = []
delay2 = np.linspace(12,15,len(files_17)//2)-12.65
delay2 = 2*delay2/c
delay2 = np.around(delay2, decimals=3)

for i in range(1,len(files_17),2):
    f = open(files_17[i])
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
    raw_data_arr_2.append(intensity)
    fit_I_arr_2.append(fit_I)
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
            
        
        
#####################################################################
#####################################################################
#####################################################################
#####################################################################
img = plt.imread("D:\\for ppt\\1e18 energy plot.png")

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
t0_index = find_index(delay1, 0)

lambda0 = 410.35 #np.mean(peaks[0:5])
pr_only_w = np.ones(10)*lambda0
pr_only = np.linspace(0,1,len(pr_only_w))

def f(x, t):
    return (0.5*np.exp(-20*(x - t)**2) * np.cos(100 * (x - t)))**2*np.sign(-t)


theta = np.pi / 10

t0 = delay1[t0_index]
if(t0<=0):
    x = np.linspace(-5, 0, 1001)
    y = f(x, t0)
    xp = np.cos(theta) * x + np.sin(theta) * y
    yp = -np.sin(theta) * x + np.cos(theta) * y
else:
    x = np.linspace(0, 5, 1001)
    y = f(x, t0)
    xp = np.cos(theta) * x + np.sin(theta) * y
    yp = -np.sin(theta) * x + np.cos(theta) * y
    
    
line1, = plt.plot(wavelength, raw_data_arr_1[t0_index], 'b-', lw=2, label="Scan 1")
line2, = plt.plot(wavelength, raw_data_arr_2[t0_index], 'b-', lw=2, label="Scan 2")
line3, = plt.plot(wavelength, fit_I_arr_1[t0_index], lw=0.5, color='k', label="Fit 1")
line4, = plt.plot(wavelength, fit_I_arr_2[t0_index], lw=0.5, color='b', label="Fit 2")
line5,  = plt.plot(pr_only_w, pr_only,  'g--' ,lw=1, label= "pr_only")
line, = plt.plot(xp+lambda0, yp+200, lw=0.5, color='red')

# Create an inset axis in the top-left corner for the image
inset_ax = fig.add_axes([0.12, 0.45, 0.3, 0.4])  # Adjust position and size as needed
inset_ax.imshow(img, aspect='auto', alpha=0.7)
inset_ax.axis('off')  # Hide the axis for the inset image


plt.legend()

#ax.set_aspect('equal', adjustable='box')
axcolor = 'lightgoldenrodyellow'
ax.set_xlim(min_wavelength, max_wavelength)
ax.set_ylim(-0.2,1.2)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("Normalized Intensity")
ax.grid(color="k", lw=0.2)  # Apply grid directly to ax

# Create slider for t
ax_t = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
slider_t = Slider(ax_t, 'delay', valmin=0, valmax=len(delay1)-2, valinit=t0_index, valstep=1)

# Set up colormap normalization from -50 to 50
norm = Normalize(vmin=410.2, vmax=410.9)
colormap = plt.cm.jet  # Use the 'jet' colormap for the desired color transition

def update(val):
    delay_index = slider_t.val
    peak1 = peaks1[delay_index]
    peak2 = peaks2[delay_index]

    # Update line data
    line1.set_xdata(wavelength)
    line1.set_ydata(raw_data_arr_1[delay_index])
    
    line2.set_xdata(wavelength)
    line2.set_ydata(raw_data_arr_2[delay_index])
    
    # Update line data
    line3.set_xdata(wavelength)
    line3.set_ydata(fit_I_arr_1[delay_index])
    
    line4.set_xdata(wavelength)
    line4.set_ydata(fit_I_arr_2[delay_index])
    
    t = delay1[delay_index]
    if(t<0):
        x = np.linspace(-5, 0, 1001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = -np.sin(theta) * x + np.cos(theta) * y
    else:
        x = np.linspace(0, 5, 1001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = np.sin(theta) * x - np.cos(theta) * y
        
    line.set_ydata(yp)
    line.set_xdata(xp+lambda0)
    
    # Update line color based on the slider value
    color1 = colormap(norm(peak1))  # Get color from colormap
    color2 = colormap(norm(peak2))  # Get color from colormap
    line1.set_color(color1)
    line2.set_color(color2)
    line3.set_color("k")
    line4.set_color("b")

    ax.set_title(r"Spectrum for I = $1\times 10^{18} W/cm^2$"+f"\nspectrum at delay: {delay1[delay_index]} ps;  peak wavelength: scan1= {peak1:.3f} nm, scan2 = {peak2:.3f} nm")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    # Update the legend to reflect the new color of line1
    ax.legend([line1, line2, line3, line4, line5], ["Scan 1", "Scan 2", "Fit 1", "Fit2", "pr_only"], loc="upper right")
    fig.canvas.draw_idle()

slider_t.on_changed(update)

# Create animation
# num_frames = len(delay)
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100)

# # Save as GIF
# ani.save("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\spectrum animation.gif", writer="pillow", fps=10)


plt.show()
