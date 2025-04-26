# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import glob
from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
import pandas as pd

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


c = 0.3   #in mm/ps
# Replace 'your_file.csv' with the path to your CSV file
file_path = r"D:\data Lab\400 vs 800 doppler experiment\400 pump 400 probe\14th Feb 2024\good reflectivity\Wed Feb 14 15_56_19 2024\MeasLog.csv"

# Read the CSV file
df = pd.read_csv(file_path)

pd_signal = np.array(df["CH2 - PK2Pk"])
norm_factor = np.array(df["CH1 - PK2Pk"])
norm_signal = pd_signal/norm_factor

delay = np.linspace(9.5,13.5,len(pd_signal))-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

maxw = find_index(delay,18)
shift = 10
# plt.plot(pd_signal/np.max(pd_signal))
plt.plot(delay[2:maxw],(norm_signal/np.max(norm_signal))[2+shift:maxw+shift],"r-")
plt.grid(lw=0.5,color="k")
plt.xlabel("delay")
plt.ylabel("Normalized pd signal (reflectivity)")
plt.xlim(-5,18)
plt.title("Plasma reflectivity vs delay")
plt.show()