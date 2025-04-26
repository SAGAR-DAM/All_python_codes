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

filepath = "D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\23 rd april 2024\\pr only\\*.txt"
files = glob.glob(filepath)
peaks_pr_only = [[] for _ in range(15)]


for i in range(len(files)):
    f = open(files[i])
    r = np.loadtxt(f)
    
    w = np.zeros(shape=(15,3648))
    I = np.zeros(shape=(15,3648))
    
    for j in range(15):
        w[j]=r[:,2*j]
        I[j]=r[:,2*j+1]
        
        I[j] -= np.mean(I[j][0:200])
        I[j] /= max(I[j])
        
        minw = find_index(w[j],405)
        maxw = find_index(w[j],420)
        
        wavelength = w[j][minw:maxw]
        intensity = I[j][minw:maxw]
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        error = np.std(fit_I-intensity)
        print(f"fitting error:  {error}")
        
        if(error<0.03):
            peaks_pr_only[j].append(parameters[1])
            
            # plt.plot(wavelength,intensity,'o', markersize = 1)
            # plt.plot(wavelength,fit_I,'-', label = fr'{j} , $\epsilon$: {error: .2f}')
        
        
#     plt.legend()
#     plt.title('probe only')
#     plt.xlabel("wavelength (nm)")
#     plt.ylabel("Normalized I")
#     plt.show()
        
for i in range(15):
    if (peaks_pr_only[i]!=[]):
        plt.plot(peaks_pr_only[i], label=f"{i}, std: {np.std(peaks_pr_only[i]): .3f} nm")
        print(len(peaks_pr_only[i]))
    
plt.legend()
plt.title("Probe only in different positions")
plt.show()



delays = [12.25,12.55,12.85,13,13.9,13.15,13.45,13.75,14.2,14.05,14.5,14.8,14.35,14.65,14.95,15.1,15.4,15.7,15.25,15.55,15.85,16,16.3,16.3,16.15,16.45,16.75,17.05,17.5,17.35,17.65,17.95,18.25]

delays.sort()
print(delays)
delays = np.array(delays)
time_delay = 2*(delays-13.6)/0.3


spec = 4

peaks = []
errors = []
delay_allowed = []

for d in range(len(delays)):
    delay = delays[d]
    
    filepath = f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\23 rd april 2024\\{delay}\\*.txt"
    files = glob.glob(filepath)
    
    p = []
    
    for i in range(len(files)):
        f = open(files[i])
        r = np.loadtxt(f)
        
        w = r[:,2*spec]
        I = r[:,2*spec+1]
        
        I -= np.mean(I[0:200])
        I /= max(I)
        
        minw = find_index(w,405)
        maxw = find_index(w,420)
        
        wavelength = w[minw:maxw]
        intensity = I[minw:maxw]
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        error = np.std(fit_I-intensity)
        
        if(error<0.03):
            p.append(parameters[1])
    
    if(len(p)>0):
        peaks.append(np.mean(p))
        errors.append(np.std(p)+np.std(peaks_pr_only[spec]))
        delay_allowed.append(time_delay[d])
    
print((delay_allowed))
print(len(peaks))

# plt.plot(delay_allowed, peaks, 'o-')
plt.plot(delay_allowed, np.array(peaks)-np.mean(peaks_pr_only[spec]),'o-' )
plt.plot(delay_allowed, np.array(peaks)-np.mean(peaks_pr_only[spec]),'ro' )
plt.errorbar(delay_allowed, np.array(peaks)-np.mean(peaks_pr_only[spec]) ,yerr = errors, lw=0, elinewidth=1, capsize=2, color = 'k')
plt.show()
    
    
    
