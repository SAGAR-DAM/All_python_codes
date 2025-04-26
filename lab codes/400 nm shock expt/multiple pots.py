# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:18:11 2025

@author: mrsag
"""

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



file1 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0063.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


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



plt.plot(wavelength, intensity,label="pr on;ly")
# plt.title(f"range: {wavelength[-1]-wavelength[0]}")




file2 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0067.txt"


f = open(file2)
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength = r[:,0]
intensity = r[:,1]


intensity -= np.mean(intensity[0:200])
intensity /= max(intensity)

minw = find_index(wavelength, 400)
maxw = find_index(wavelength, 420)

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 



plt.plot(wavelength, intensity,label="0ps")
# plt.title(f"range: {wavelength[-1]-wavelength[0]}")




file3 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0092.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


f = open(file3)
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



plt.plot(wavelength, intensity,label="5ps")




file4 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0093.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


f = open(file4)
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



plt.plot(wavelength, intensity, label="10ps")



file5 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0082.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


f = open(file5)
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



plt.plot(wavelength, intensity,label="-5ps")


file6 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0114.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


f = open(file6)
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



plt.plot(wavelength, intensity,label="-10ps")


plt.legend()
plt.show()



file7 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0114.txt"
# r1 = np.array(np.loadtxt("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\a_0031.txt",skiprows=17,comments='>'))


f = open(file6)
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



plt.plot(wavelength, intensity,label="-10ps")


plt.legend()
plt.show()