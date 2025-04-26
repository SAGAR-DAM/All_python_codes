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
''' pos 7 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_7 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_7\\scan 1\\*.txt"))

# peaks = []
# delay = np.linspace(12,15,len(files_7)//2)-12.65
# delay = 2*delay/c
# delay = np.around(delay, decimals=3)

# for i in range(1,len(files_7),2):
#     f = open(files_7[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks = []
delay = np.linspace(12,15,len(files_7)//2)-12.65
delay = 2*delay/c
delay = np.around(delay, decimals=3)

for i in range(1,len(files_7),2):
    f = open(files_7[i])
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
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<409 or peaks[i]>412):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 410.4

peaks = moving_average(peaks, 4)

for i in range(len(peaks)):
    if (peaks[i]<409 or peaks[i]>412):
        try:
            if((peaks[i-1]<412 and peaks[i+1]>409) or (peaks[i+1]<412 and peaks[i+1]>409)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 410.4
        except:
            peaks[i] = 410.4

delay = delay[0:len(peaks)]

plt.plot(delay[0:len(peaks)-1],peaks[0:len(peaks)-1], 'ro-')
plt.title("Doppler shift")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
#plt.xlim(-2,max(delay))
#plt.ylim(395,396)
plt.title("Peak Wavelength")
plt.grid(lw = 1, color = "black")
plt.show()

plt.plot(delay[0:len(peaks)-1],peaks[0:len(peaks)-1]-410.4, 'ro')
plt.plot(delay[0:len(peaks)-1],peaks[0:len(peaks)-1]-410.4, 'k-')
plt.title("Doppler shift")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
# plt.xlim(-5,18)
# plt.ylim(-0.125,0.5)
plt.title("Doppler Shift for pos 7")
plt.grid(lw = 1, color = "black")
plt.show()