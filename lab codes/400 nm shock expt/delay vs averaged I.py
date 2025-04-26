# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:54:41 2025

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


# delays = np.array(sorted([30.4,29.65,31.15,31.9,32.65,33.4,28.9,30.7,31.45]))#,32.2,32.95,33.7,34.15,34.9,35.65]))
delays = np.array(sorted([30.4,31.15,31.9,32.65,33.4,30.7]))#,32.2,32.95,33.7,34.15,34.9,35.65]))

time_delays = (delays-30.4)*2/c



delay = "pr only"

files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\all delays\\{delay}\\*.txt")
f = open(files[0])
r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

wavelength = r[:,0]
fullrange = len(wavelength)

w=np.array(wavelength)
I=np.zeros(fullrange)

for file in files[-4:]:
    f = open(file)
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

    wavelength = r[:,0]
    intensity = r[:,1]


    intensity -= np.mean(intensity[0:200])
    intensity /= max(intensity)
    
    I += intensity/len(files[-4:])
    
minw = find_index(w, 406)
maxw = find_index(w, 410)
wavelength = w[minw:maxw]
I = I[minw:maxw]


fit_I,parameters,string = Gf.Gaussfit(wavelength, I)
# pr_only = parameters[1]
pr_only = wavelength[find_index(I, max(I))]


# max_I_index = find_index(I, max(I))
# wavelength = np.insert(wavelength,max_I_index,wavelength[max_I_index])
# wavelength = np.insert(wavelength,max_I_index,wavelength[max_I_index])

# I = np.insert(I, max_I_index, 0)
# I = np.insert(I, max_I_index, max(I))

plt.plot(wavelength,I,'-',label="pr_only ",lw=1)
# plt.plot(wavelength,fit_I,label="fit")
# plt.title(f"{delay}")



peaks = []

for i in range(len(delays)):
    delay = delays[i]

    files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\all delays\\{delay}\\*.txt")
    f = open(files[0])
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

    wavelength = r[:,0]
    fullrange = len(wavelength)

    w=np.array(wavelength)
    I=np.zeros(fullrange)

    for file in files:
        f = open(file)
        r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

        wavelength = r[:,0]
        intensity = r[:,1]


        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        
        I += intensity/len(files)
        
    minw = find_index(w, 406)
    maxw = find_index(w, 410)
    wavelength = w[minw:maxw]
    I = I[minw:maxw]


    fit_I,parameters,string = Gf.Gaussfit(wavelength, I)
    # peaks.append(parameters[1])
    peaks.append(wavelength[find_index(I, max(I))])

    
    
    # max_I_index = find_index(I, max(I))
    # wavelength = np.insert(wavelength,max_I_index,wavelength[max_I_index])
    # wavelength = np.insert(wavelength,max_I_index,wavelength[max_I_index])

    # I = np.insert(I, max_I_index, 0)
    # I = np.insert(I, max_I_index, max(I))
    
    
    plt.plot(wavelength,I,'-',label=f"{time_delays[i]:.0f} ps", lw=0.5)
    # plt.plot(wavelength,fit_I,label="fit")
    # plt.title(f"{delay}")

plt.legend()
plt.xlim(407.5,409.5)
plt.grid(lw=0.5, color="k")
plt.xlabel("wavelength (nm)")
plt.ylabel("Normalized intensity")
plt.show()

peaks = np.array(peaks)
plt.plot(time_delays,peaks-pr_only)
plt.show()

