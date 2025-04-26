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

delays = np.array(sorted([30.4,29.65,31.15,31.9,32.65,33.4,28.9,30.7,31.45,32.2,32.95,33.7,34.15,34.9,35.65]))
time_delays = (delays-30.4)*2/c

index=find_index(delays,34.9)
delay = str(delays[index])
print(f"retro: {delays[index]}, time delay: {time_delays[index]:.2f} ps")

# delay = "pr only"
files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\all delays\\{delay}\\*.txt")

for file in files:
    f = open(file)
    r=np.array(np.loadtxt(f,skiprows=17,comments='>'))

    wavelength = r[:,0]
    intensity = r[:,1]


    intensity -= np.mean(intensity[0:200])
    intensity /= max(intensity)

    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)

    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 



    plt.plot(wavelength, intensity, label = f"{delay}\n"+file[-10:],lw=1)
    if(delay != "pr only"):
        plt.title(f"retro: {delay}    delay: {time_delays[index]}\n"+file[-10:])
    else:
        plt.title(f"retro: {delay}  \n"+file[-10:])

    plt.xlim(407,410)
    plt.grid(lw=0.5,color="k")
    plt.legend()
plt.show()





