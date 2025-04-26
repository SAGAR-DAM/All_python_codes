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
import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

c = 0.3   #in mm/ps

def moving_average(signal, window_size):
    # Define the window coefficients for the moving average
    window = np.ones(window_size) / float(window_size)
    
    # Apply the moving average filter using fftconvolve
    filtered_signal = fftconvolve(signal, window, mode='same')
    
    return filtered_signal


files1 = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14 th march 2024\\Scan4\\s*.txt")
files2 = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14 th march 2024\\Scan4\\e*.txt")
files = files1+files2


delay = np.linspace(10,14,len(files)//2)-11.35
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


# for i in range(1,len(files)-2,2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files)-2,2):
    # if(i<=161):
    #     f = open(files[i])
    # else:
    #     f = open(files[i-1])
  
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')


    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i//2]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<412 or peaks[i]>414):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 413.2

peaks = moving_average(peaks,2)

for i in range(len(peaks)):
    if (peaks[i]<412 or peaks[i]>414):
        try:
            if((peaks[i-1]<412 and peaks[i+1]>414) or (peaks[i+1]<412 and peaks[i+1]>414)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 413.2
        except:
            peaks[i] = 413.2
plt.plot(delay[0:len(peaks)],peaks, 'ro-')
#plt.xlim(-10,8)
plt.ylim(412.7,413.8)
plt.xlabel("delay (ps)")
plt.ylabel("peak wavelength (nm)")
plt.grid(lw = 1, color = "black")
plt.title("Peak Wavelength")
plt.show()

