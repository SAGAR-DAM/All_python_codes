# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:34:56 2024

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

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


file1 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\s_261.txt"

f = open(file1)
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength = r[:,0]
intensity = r[:,1]
intensity /= max(intensity)

minw = find_index(wavelength, 400)
maxw = find_index(wavelength, 420)

wavelength = wavelength[minw:maxw]
intensity = intensity[minw:maxw] 

intensity -= np.mean(intensity[0:50])
intensity /= max(intensity)
fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)

plt.plot(wavelength, intensity, 'ro', markersize=2)
plt.plot(wavelength, fit_I,'r-', label = "pump-probe ")
#plt.show()

file2 = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\pr only\\s_003.txt"

f = open(file2)
r=np.loadtxt(f,skiprows=17,comments='>')

wavelength1 = r[:,0]
intensity1 = r[:,1]
intensity1 /= max(intensity1)

minw = find_index(wavelength1, 400)
maxw = find_index(wavelength1, 420)

wavelength1 = wavelength1[minw:maxw]
intensity1 = intensity1[minw:maxw] 

intensity1 -= np.mean(intensity1[0:50])
intensity1 /= max(intensity1)
fit_I1,parameters1,string1 = Gf.Gaussfit(wavelength1, intensity1)

plt.plot(wavelength1, intensity1,'bo',markersize=2)
plt.plot(wavelength1, fit_I1,'b-', label = 'probe only')

plt.plot(wavelength, intensity-intensity1, 'go', markersize=3)
plt.plot(wavelength, fit_I-fit_I1,'k-', label = 'subtracted signal')
plt.legend()
plt.title("Subtracted signal (pupr - pr_only)"+"\n"+"delay: 7.5 ps")
plt.ylabel("Intensity normalized")
plt.xlabel("Wavelength (nm)"+"\n Shows a clear RED shift")
plt.show()


