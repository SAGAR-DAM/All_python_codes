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




#foldernames = ["28_95","30_45","31_20","31_95","32_70","33_45","34_20","34_95"]
foldernames = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\16102024\\Different delays\\*")
foldernames.remove("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\16102024\\Different delays\\pr_only")
#print(foldernames)

min_wavelength = 405
max_wavelength = 412

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\16102024\\Different delays\\pr_only\\*.txt")

w0_arr = []
for i in range(len(files)):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    
    minw = find_index(wavelength, min_wavelength)
    maxw = find_index(wavelength, max_wavelength)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw]
    
    if(max(intensity)>500):
        intensity /= max(intensity)
        intensity -= np.mean(intensity[0:50])
        
        try:
            fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
            w0_arr.append(parameters[1])
            
            # plt.plot(wavelength, intensity, '-', label = "data")
            # # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
            # plt.title(f"Probe only {i}")
            # # plt.xlim(390,420)
            # plt.xlabel("Wavelength (nm)\n"+f"Peak:  {parameters[1]:.3f} nm")
            # plt.ylabel("Intensity (arb unit)")
            
        except:
            pass
    plt.show()   
w0 = np.mean(w0_arr)
w0_zitter = np.std(w0_arr)

print(f"Probe only wavelenth: {w0:.3f} nm;  Std: {w0_zitter:.3f} nm")





peaks = []
zitter = []
for foldername in foldernames:
    files = glob.glob(foldername+"\\*.txt")
    w_arr = []
    for i in range(len(files)):
        f = open(files[i])
        r=np.loadtxt(f,skiprows=17,comments='>')
        
        wavelength = r[:,0]
        intensity = r[:,1]
        
        minw = find_index(wavelength, min_wavelength)
        maxw = find_index(wavelength, max_wavelength)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw]
        
        if(max(intensity>500)):
            intensity /= max(intensity)
            intensity -= np.mean(intensity[0:50])
            
            try:
                fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
                w_arr.append(parameters[1])
                
                plt.plot(wavelength, intensity, '-', label = "data")
                # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
                plt.title(f"folder: {foldername[-10:]}; file: {files[i][-11:]}")
                # plt.xlim(390,420)
                plt.xlabel("Wavelength (nm)\n"+f"Peak:  {parameters[1]:.3f} nm")
                plt.ylabel("Intensity (arb unit)")
                # plt.show()  
            except:
                pass
            
    w = np.mean(w_arr)
    w_zitter = np.std(w_arr)
    
    peaks.append(w)
    zitter.append(w_zitter)

    print(f"delay: {foldername[-10:]}:  wavelenth: {w:.3f} nm;  Std: {w_zitter:.3f} nm")
    
    
# plt.plot(np.array(peaks)-w0,"ko-")
# plt.show()
