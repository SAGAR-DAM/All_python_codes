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

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\pr only\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

for i in range(0,len(files)):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 400)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 

    intensity -= np.mean(intensity[0:50])
    

    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    #peaks.append(parameters[1])
    
    if(max(fit_I)>0.1):
        #if(parameters[1]>395.75):
        if(len(peaks)==0 or parameters[1]!=peaks[-1]):
            peaks.append(parameters[1])
    
        plt.plot(wavelength, intensity, 'o', markersize = 1)
        plt.plot(wavelength, fit_I, '-')
        
plt.plot(np.ones(10)*np.mean(peaks),np.linspace(0, 1 ,10), 'g--', label = f"Mean probe wavelength = {np.mean(peaks) :.2f} nm")
plt.title("Probe only wavelengths"+"\n"+r"Low energy: I ~ $3\times 10^{17} W/cm^2$")
plt.xlim(392,400)
plt.xlabel("Wavelength (nm)\n"+f"std:  {np.std(peaks) :.3f} nm")
plt.ylabel("Intensity")
plt.legend()
plt.show() 

for i in range(0,len(files)):
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 400)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    
    if(max(fit_I)>0.1):
        #if(parameters[1]>395.75):
        if(len(peaks)==0 or parameters[1]!=peaks[-1]):
            peaks.append(parameters[1])
    
        # plt.plot(wavelength, intensity, 'r-', label = "data")
        # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
        # plt.title(files[i][-9:])#+"\n"+f"Delay: {delay[i]}")
        # #plt.xlim(400,420)
        # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
        # plt.ylabel("Intensity")
        # plt.show()  
   
    
# for i in range(len(peaks)):
#     if (peaks[i]<395 or peaks[i]>396):
#         try:
#             peaks[i] = (peaks[i+1]+peaks[i-1])/2
#         except:
#             peaks[i] = 395.35

# peaks = moving_average(peaks,10)

# for i in range(len(peaks)):
#     if (peaks[i]<395 or peaks[i]>396):
#         try:
#             if((peaks[i-1]<395 and peaks[i+1]>396) or (peaks[i+1]<395 and peaks[i+1]>396)):
#                 peaks[i] = (peaks[i+1]+peaks[i-1])/2
#             else:
#                 peaks[i] = 395.35
#         except:
#             peaks[i] = 395.35

delay = delay[0:len(peaks)]

plt.plot(peaks, 'b-')
plt.plot(peaks, 'ro', label='Pr only')
plt.plot(np.ones(len(peaks))*np.mean(peaks), 'g-', label='Mean')
plt.legend()
plt.title("Doppler shift")
plt.xlabel(f"Stdev: {np.std(peaks): .3f} nm")
plt.ylabel("Pr only Peak wavelength (nm)")
#plt.xlim(-2,max(delay))
# plt.ylim(np.mean(peaks)-0.5,np.mean(peaks)+0.5)
plt.title("Pr only Peak Wavelength")
plt.grid(lw = 1, color = "black")
plt.show()

print(f"std:  {np.std(peaks)}")
# plt.plot(delay[0:len(peaks)],peaks-395.3, 'ro')
# plt.plot((delay[0:len(peaks)])[peaks-395.3<=0],peaks[peaks-395.3<=0]-395.3, 'bo')
# plt.plot(delay[0:len(peaks)],peaks-395.3, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay (ps)")
# plt.ylabel("Doppler Shift")
# plt.xlim(-5,18)
# plt.ylim(-0.125,0.5)
# plt.title("Doppler shift Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()


# peaks = peaks[find_index(delay,-5):find_index(delay, 18)]
# delay = delay[find_index(delay,-5):find_index(delay, 18)]

# lambda0 = np.mean(peaks[0:5])


# velocity =[]
# for i in range(len(peaks)):
#     v = (peaks[i]**2-lambda0**2)/(peaks[i]**2+lambda0**2)
#     velocity.append(v)
    
# blue_v = []
# red_v = []

# blue_delay = []
# red_delay= []

# for i in range(len(velocity)):
#     if(velocity[i]<=0):
#         blue_delay.append(delay[i])
#         blue_v.append(velocity[i])
        
#     else:
#         red_delay.append(delay[i])
#         red_v.append(velocity[i])

# plt.plot(delay,velocity, 'ro')
# plt.plot(delay,velocity, 'k-')
# plt.title("Doppler Velocity")
# plt.xlabel("probe delay (ps)")
# plt.ylabel("v/c")
# plt.xlim(-5,18)
# #plt.ylim(-0.125,0.5)
# plt.grid(lw = 0.5, color = "black")
# plt.show()

# plt.plot(blue_delay,blue_v, 'bo')
# plt.plot(red_delay,red_v, 'ro')
# plt.plot(delay,velocity, 'k-')
# plt.title("Doppler Velocity of critical surface"+"\n"+"Energy on target: 30 mJ 400 nm (p pol)")
# plt.xlabel("probe delay (ps)")
# plt.ylabel("v/c")
# plt.xlim(-5,18)
# #plt.ylim(-0.125,0.5)
# plt.grid(lw = 0.5, color = "black")
# plt.show()
