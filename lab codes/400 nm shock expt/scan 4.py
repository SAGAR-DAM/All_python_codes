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
mpl.rcParams['font.weight'] = 'bold'
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

files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14th Feb 2024\\Spectrum\\run7\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run9_70%_20TW_ret_11-15_250fs\\*.txt")
#files = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\800 pump 400 probe\\5th feb 2024\\spectrum\\5Feb23_Doppler_FS_Front\\Run8_30%_20TW_ret_11-15_250fs\\*.txt")

delay = np.linspace(9.5,13.5,len(files)//2)-10.5
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

# for i in range(1,len(files),2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 399)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files),2):
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
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 395.35

peaks = moving_average(peaks,10)

for i in range(len(peaks)):
    if (peaks[i]<395 or peaks[i]>396):
        try:
            if((peaks[i-1]<395 and peaks[i+1]>396) or (peaks[i+1]<395 and peaks[i+1]>396)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 395.35
        except:
            peaks[i] = 395.35

delay = delay[0:len(peaks)]

# plt.plot(delay[0:len(peaks)],peaks, 'ro')
# plt.title("Doppler shift")
# plt.xlabel("probe delay (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay))
# plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()


blue_peaks = []
blue_delay = []
for i in range(len(peaks)):
    if(peaks[i]-395.3<=0):
        blue_peaks.append(peaks[i])
        blue_delay.append(delay[i])
        
plt.plot(delay[0:len(peaks)],peaks-395.3, 'ro')
plt.plot(blue_delay, np.array(blue_peaks)-395.3, 'bo')
plt.plot(delay[0:len(peaks)],peaks-395.3, 'k-')
plt.title("Doppler shift")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
plt.text(x=6, y=0.1, s="low energy"+"\n"+"~30mJ")
# plt.xlim(-5,18)
# plt.ylim(-0.125,0.5)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()


peaks = peaks[find_index(delay,-5):find_index(delay, 18)]
delay = delay[find_index(delay,-5):find_index(delay, 18)]

lambda0 = np.mean(peaks[0:5])


velocity =[]
for i in range(len(peaks)):
    v = (peaks[i]**2-lambda0**2)/(peaks[i]**2+lambda0**2)
    velocity.append(v)
    
blue_v = []
red_v = []

blue_delay = []
red_delay= []

for i in range(len(velocity)):
    if(velocity[i]<=0):
        blue_delay.append(delay[i])
        blue_v.append(velocity[i])
        
    else:
        red_delay.append(delay[i])
        red_v.append(velocity[i])

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











files1 = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14 th march 2024\\Scan4\\s*.txt")
files2 = glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\14 th march 2024\\Scan4\\e*.txt")
files = files1+files2


delay = np.linspace(10,14,len(files)//2)-11.35
delay = 2*delay/c
delay = np.around(delay, decimals=3)

peaks = []

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


# for i in range(1,len(files)-2,2):
#     f = open(files[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i]}")
#     plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

for i in range(1,len(files)-2,2):
    # if(i<=161):
    #     f = open(files[i])
    # else:
    #     f = open(files[i-1])
  
    f = open(files[i])
    r=np.loadtxt(f,skiprows=17,comments='>')


    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 390)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"Delay: {delay[i//2]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks)):
    if (peaks[i]<412 or peaks[i]>414):
        try:
            peaks[i] = (peaks[i+1]+peaks[i-1])/2
        except:
            peaks[i] = 413.2

peaks = moving_average(peaks,5)

for i in range(len(peaks)):
    if (peaks[i]<412 or peaks[i]>414):
        try:
            if((peaks[i-1]<412 and peaks[i+1]>414) or (peaks[i+1]<412 and peaks[i+1]>414)):
                peaks[i] = (peaks[i+1]+peaks[i-1])/2
            else:
                peaks[i] = 413.2
        except:
            peaks[i] = 413.2
            
blue_peaks = []
blue_delay = []
delay += 5
lambda0 = 413.4

for i in range(len(peaks)):
    if(peaks[i]-lambda0<=0):
        blue_peaks.append(peaks[i])
        blue_delay.append(delay[i])

plt.plot(delay[0:len(peaks)],-(peaks-lambda0), 'o-', color='orange')
plt.plot(blue_delay, -(np.array(blue_peaks)-lambda0), 'o', color='purple')
plt.plot(delay[0:len(peaks)],-(peaks-lambda0), 'g-')
#plt.xlim(-10,8)
#plt.ylim(412.7,413.8)
# plt.xlabel("delay (ps)")
# plt.ylabel("peak wavelength (nm)")
plt.grid(lw = 0.5, color = "black")
plt.text(x=14, y=-(peaks[len(peaks)//2]-lambda0)*3/4, s="high energy"+"\n"+"~70mJ")
# plt.title("Peak Wavelength")
plt.show()

