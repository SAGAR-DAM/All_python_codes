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
import random


import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display




c = 0.3   #in mm/ps
min_wavelength=400
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




# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))

def plot_file(file1,labelname = "pr_only"):
    f = open(file1)
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))
    
    # r = (r+r1)
    wavelength = r[:,0]
    intensity = r[:,1]
    
    
    intensity -= np.mean(intensity[0:200])
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    
    plt.plot(wavelength, intensity,label=labelname)
    # plt.plot(wavelength, fit_I, label=labelname)
    
    peak = np.mean(wavelength[intensity>0.95])
    
    return peak

delays = np.array(sorted([30.4,29.65,31.15,31.9,32.65,33.4,28.9,30.7,32.95,33.7,34.9,35.65,32.2]))
time_delays = (delays-30.4)*2/c
peaks = []

file_pr_only = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0063.txt"
peak_pr_only = plot_file(file_pr_only,labelname="pr only")

file_28_9 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0114.txt"
peaks.append(plot_file(file_28_9,labelname="-10 ps"))

file_29_65 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0076.txt"
peaks.append(plot_file(file_29_65,labelname="-5 ps"))

file_30_4 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0070.txt"
peaks.append(plot_file(file_30_4,labelname="0 ps"))

file_30_7 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0142.txt"
peaks.append(plot_file(file_30_7,labelname="2 ps"))

file_31_15 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0088.txt"
peaks.append(plot_file(file_31_15,labelname="5 ps"))

file_31_9 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0097.txt"
peaks.append(plot_file(file_31_9,labelname="10 ps"))

file_32_2 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0164.txt"
peaks.append(plot_file(file_32_2,labelname="12 ps"))

file_32_65 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0107.txt"
peaks.append(plot_file(file_32_65,labelname="15 ps"))

file_32_95 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0169.txt"
peaks.append(plot_file(file_32_95,labelname="17 ps"))

file_33_4 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0110.txt"
peaks.append(plot_file(file_33_4,labelname="20 ps"))

file_33_7 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0174.txt"
peaks.append(plot_file(file_33_7,labelname="22 ps"))

file_34_9 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0188.txt"
peaks.append(plot_file(file_34_9,labelname="30 ps"))

file_35_65 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0193.txt"
peaks.append(plot_file(file_35_65,labelname="35 ps"))

plt.xlim(407.5,409.5)
plt.legend()
plt.show()

peaks = np.array(peaks)
plt.plot(time_delays, peaks-np.mean(peaks[0:2]),'ko-')
plt.grid(lw=0.5,color='b')
plt.xlabel("delay (ps)")
plt.ylabel("spectral shift (nm)")
plt.title("Pu-Pr delay vs probe doppler shift")
# plt.ylim(-0.1,0.15)
plt.show()


# peaks = np.array(peaks)
# plt.plot(time_delays, peaks-peak_pr_only,'ko-')
# plt.grid(lw=0.5,color='b')
# plt.xlabel("delay (ps)")
# plt.ylabel("spectral shift (nm)")
# plt.title("Pu-Pr delay vs probe doppler shift")
# # plt.ylim(-0.1,0.15)
# plt.show()