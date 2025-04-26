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

def calc_vel(w, w0):
    c_norm = 1
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v



#################################################################
#################################################################
#################################################################
#################################################################
''' pos 17 '''
#################################################################
#################################################################
#################################################################
#################################################################
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))

peaks17 = []
delay17 = np.linspace(12,15,len(files_17)//2)-12.65
delay17 = 2*delay17/c
delay17 = np.around(delay17, decimals=3)

# for i in range(0,len(files_17),2):
#     f = open(files_17[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks17.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay17: {delay17[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks17 = []
delay17 = np.linspace(12,15,len(files_17)//2)-12.65
delay17 = 2*delay17/c
delay17 = np.around(delay17, decimals=3)

for i in range(0,len(files_17),2):
    f = open(files_17[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks17.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay17: {delay17[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks17)):
    if (peaks17[i]<409 or peaks17[i]>412):
        try:
            peaks17[i] = (peaks17[i+1]+peaks17[i-1])/2
        except:
            peaks17[i] = 410.4

peaks17 = moving_average(peaks17,4)

for i in range(len(peaks17)):
    if (peaks17[i]<409 or peaks17[i]>412):
        try:
            if((peaks17[i-1]<412 and peaks17[i+1]>409) or (peaks17[i+1]<412 and peaks17[i+1]>409)):
                peaks17[i] = (peaks17[i+1]+peaks17[i-1])/2
            else:
                peaks17[i] = 410.4
        except:
            peaks17[i] = 410.4

delay17 = delay17[0:len(peaks17)]

# plt.plot(delay17[0:len(peaks17)],peaks17, 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay17 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay17))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay17[0:len(peaks17)-1],peaks17[0:len(peaks17)-1]-410.4, 'o', label = 'pos 17')
plt.plot(delay17[0:len(peaks17)-1],peaks17[0:len(peaks17)-1]-410.4, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay17 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 17")
# plt.grid(lw = 1, color = "black")
# plt.show()




#################################################################
#################################################################
#################################################################
#################################################################
''' pos 6 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_1 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_1\\scan 1\\*.txt"))

peaks6 = []
delay6 = np.linspace(12,15,len(files_1)//2)-12.65
delay6 = 2*delay6/c
delay6 = np.around(delay6, decimals=3)

# for i in range(1,len(files_1),2):
#     f = open(files_1[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks6.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay6: {delay6[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks6 = []
delay6 = np.linspace(12,15,len(files_1)//2)-12.65
delay6 = 2*delay6/c
delay6 = np.around(delay6, decimals=3)

for i in range(1,len(files_1),2):
    f = open(files_1[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks6.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay6: {delay6[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks6)):
    if (peaks6[i]<409 or peaks6[i]>412):
        try:
            peaks6[i] = (peaks6[i+1]+peaks6[i-1])/2
        except:
            peaks6[i] = 410.5

peaks6 = moving_average(peaks6, 4)

for i in range(len(peaks6)):
    if (peaks6[i]<409 or peaks6[i]>412):
        try:
            if((peaks6[i-1]<412 and peaks6[i+1]>409) or (peaks6[i+1]<412 and peaks6[i+1]>409)):
                peaks6[i] = (peaks6[i+1]+peaks6[i-1])/2
            else:
                peaks6[i] = 410.5
        except:
            peaks6[i] = 410.5

delay6 = delay6[0:len(peaks6)]

# plt.plot(delay6[0:len(peaks6)-1],peaks6[0:len(peaks6)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay6 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay6))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay6[0:len(peaks6)-1],peaks6[0:len(peaks6)-1]-410.5, 'o', label = 'pos 6')
plt.plot(delay6[0:len(peaks6)-1],peaks6[0:len(peaks6)-1]-410.5, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay6 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 6")
# plt.grid(lw = 1, color = "black")
# plt.show()





#################################################################
#################################################################
#################################################################
#################################################################
''' pos 7 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_7 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_7\\scan 1\\*.txt"))

# peaks7 = []
# delay7 = np.linspace(12,15,len(files_7)//2)-12.65
# delay7 = 2*delay7/c
# delay7 = np.around(delay7, decimals=3)

# for i in range(1,len(files_7),2):
#     f = open(files_7[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks7.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay7: {delay7[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks7 = []
delay7 = np.linspace(12,15,len(files_7)//2)-12.65
delay7 = 2*delay7/c
delay7 = np.around(delay7, decimals=3)

for i in range(1,len(files_7),2):
    f = open(files_7[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks7.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay7: {delay7[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks7)):
    if (peaks7[i]<409 or peaks7[i]>412):
        try:
            peaks7[i] = (peaks7[i+1]+peaks7[i-1])/2
        except:
            peaks7[i] = 410.4

peaks7 = moving_average(peaks7, 4)

for i in range(len(peaks7)):
    if (peaks7[i]<409 or peaks7[i]>412):
        try:
            if((peaks7[i-1]<412 and peaks7[i+1]>409) or (peaks7[i+1]<412 and peaks7[i+1]>409)):
                peaks7[i] = (peaks7[i+1]+peaks7[i-1])/2
            else:
                peaks7[i] = 410.4
        except:
            peaks7[i] = 410.4

delay7 = delay7[0:len(peaks7)]

# plt.plot(delay7[0:len(peaks7)-1],peaks7[0:len(peaks7)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay7 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay7))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay7[0:len(peaks7)-1],peaks7[0:len(peaks7)-1]-410.4, 'o', label = 'pos 7')
plt.plot(delay7[0:len(peaks7)-1],peaks7[0:len(peaks7)-1]-410.4, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay7 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 7")
# plt.grid(lw = 1, color = "black")
# plt.show()




#################################################################
#################################################################
#################################################################
#################################################################
''' pos 11 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_11 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_11\\scan 1\\*.txt"))

# peaks11 = []
# delay11 = np.linspace(12,15,len(files_11)//2)-12.65
# delay11 = 2*delay11/c
# delay11 = np.around(delay11, decimals=3)

# for i in range(1,len(files_11),2):
#     f = open(files_11[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks11.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay11: {delay11[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks11 = []
delay11 = np.linspace(12,15,len(files_11)//2)-12.65
delay11 = 2*delay11/c
delay11 = np.around(delay11, decimals=3)

for i in range(1,len(files_11),2):
    f = open(files_11[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks11.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay11: {delay11[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks11)):
    if (peaks11[i]<409 or peaks11[i]>412):
        try:
            peaks11[i] = (peaks11[i+1]+peaks11[i-1])/2
        except:
            peaks11[i] = 410.45

peaks11 = moving_average(peaks11,5)

for i in range(len(peaks11)):
    if (peaks11[i]<409 or peaks11[i]>412):
        try:
            if((peaks11[i-1]<412 and peaks11[i+1]>409) or (peaks11[i+1]<412 and peaks11[i+1]>409)):
                peaks11[i] = (peaks11[i+1]+peaks11[i-1])/2
            else:
                peaks11[i] = 410.45
        except:
            peaks11[i] = 410.45

delay11 = delay11[0:len(peaks11)]

# plt.plot(delay11[0:len(peaks11)-1],peaks11[0:len(peaks11)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay11 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay11))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay11[0:len(peaks11)-1],peaks11[0:len(peaks11)-1]-410.45, 'o', label = 'pos 11')
plt.plot(delay11[0:len(peaks11)-1],peaks11[0:len(peaks11)-1]-410.45, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay11 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 11")
# plt.grid(lw = 1, color = "black")
# plt.show() 




#################################################################
#################################################################
#################################################################
#################################################################
''' pos 1 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_1 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_1\\scan 1\\*.txt"))

# peaks1 = []
# delay1 = np.linspace(12,15,len(files_10)//2)-12.65
# delay1 = 2*delay1/c
# delay1 = np.around(delay1, decimals=3)

# for i in range(1,len(files_10),2):
#     f = open(files_10[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks1.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay1: {delay1[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks1 = []
delay1 = np.linspace(12,15,len(files_1)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

for i in range(1,len(files_1),2):
    f = open(files_1[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks1.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay1: {delay1[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
        except:
            peaks1[i] = 410.5

peaks1 = moving_average(peaks1,4)

for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            if((peaks1[i-1]<412 and peaks1[i+1]>409) or (peaks1[i+1]<412 and peaks1[i+1]>409)):
                peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
            else:
                peaks1[i] = 410.5
        except:
            peaks1[i] = 410.5

delay1 = delay1[0:len(peaks1)]

# plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay1 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay1))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.5, 'o', label = "pos 1")
plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.5, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay1 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 1")
# plt.grid(lw = 1, color = "black")
# plt.show() 




#################################################################
#################################################################
#################################################################
#################################################################
''' pos 13 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_13 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_13\\scan 2\\*.txt"))

# peaks13 = []
# delay13 = np.linspace(12,15,len(files_13)//2)-12.65
# delay13 = 2*delay13/c
# delay13 = np.around(delay13, decimals=3)

# for i in range(1,len(files_13),2):
#     f = open(files_13[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks13.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay13: {delay13[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks13 = []
delay13 = np.linspace(12,15,len(files_13)//2)-12.65
delay13 = 2*delay13/c
delay13 = np.around(delay13, decimals=3)

for i in range(1,len(files_13),2):
    f = open(files_13[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks13.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay13: {delay13[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks13)):
    if (peaks13[i]<409 or peaks13[i]>412):
        try:
            peaks13[i] = (peaks13[i+1]+peaks13[i-1])/2
        except:
            peaks13[i] = 410.0

peaks13 = moving_average(peaks13,1)

for i in range(len(peaks13)):
    if (peaks13[i]<409 or peaks13[i]>412):
        try:
            if((peaks13[i-1]<412 and peaks13[i+1]>409) or (peaks13[i+1]<412 and peaks13[i+1]>409)):
                peaks13[i] = (peaks13[i+1]+peaks13[i-1])/2
            else:
                peaks13[i] = 410.0
        except:
            peaks13[i] = 410.0

delay13 = delay13[0:len(peaks13)]

# plt.plot(delay13[0:len(peaks13)-1],peaks13[0:len(peaks13)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay13 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay13))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

# plt.plot(delay13[0:len(peaks13)-1],peaks13[0:len(peaks13)-1]-410.0, 'o', label = 'pos 13')
# plt.plot(delay13[0:len(peaks13)-1],peaks13[0:len(peaks13)-1]-410.0, 'k-')
# # plt.title("Doppler shift")
# plt.xlabel("probe delay13 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 16")
# plt.grid(lw = 1, color = "black")
# plt.show() 



#################################################################
#################################################################
#################################################################
#################################################################
''' pos 16 '''
#################################################################
#################################################################
#################################################################
#################################################################

files_16 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_16\\scan 1\\*.txt"))

# peaks16 = []
# delay16 = np.linspace(12,15,len(files_10)//2)-12.65
# delay16 = 2*delay16/c
# delay16 = np.around(delay16, decimals=3)

# for i in range(1,len(files_10),2):
#     f = open(files_10[i])
#     r=np.loadtxt(f,skiprows=17,comments='>')
    
#     wavelength = r[:,0]
#     intensity = r[:,1]
#     intensity /= max(intensity)
    
#     minw = find_index(wavelength, 392)
#     maxw = find_index(wavelength, 420)
    
#     wavelength = wavelength[minw:maxw]
#     intensity = intensity[minw:maxw] 

#     intensity -= np.mean(intensity[0:50])
    
#     fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
#     peaks16.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay16: {delay16[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks16 = []
delay16 = np.linspace(12,15,len(files_16)//2)-12.65
delay16 = 2*delay16/c
delay16 = np.around(delay16, decimals=3)

for i in range(1,len(files_16),2):
    f = open(files_16[i])
    r=np.loadtxt(f,skiprows=17,comments='>')
    
    wavelength = r[:,0]
    intensity = r[:,1]
    intensity /= max(intensity)
    
    minw = find_index(wavelength, 400)
    maxw = find_index(wavelength, 420)
    
    wavelength = wavelength[minw:maxw]
    intensity = intensity[minw:maxw] 
    
    intensity -= np.mean(intensity[0:50])
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks16.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay16: {delay16[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks16)):
    if (peaks16[i]<409 or peaks16[i]>412):
        try:
            peaks16[i] = (peaks16[i+1]+peaks16[i-1])/2
        except:
            peaks16[i] = 410.30

peaks16 = moving_average(peaks16,4)

for i in range(len(peaks16)):
    if (peaks16[i]<409 or peaks16[i]>412):
        try:
            if((peaks16[i-1]<412 and peaks16[i+1]>409) or (peaks16[i+1]<412 and peaks16[i+1]>409)):
                peaks16[i] = (peaks16[i+1]+peaks16[i-1])/2
            else:
                peaks16[i] = 410.30
        except:
            peaks16[i] = 410.30

delay16 = delay16[0:len(peaks16)]

# plt.plot(delay16[0:len(peaks16)-1],peaks16[0:len(peaks16)-1], 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay16 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay16))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay16[0:len(peaks16)-1],peaks16[0:len(peaks16)-1]-410.30, 'o', label = 'pos 16')
plt.plot(delay16[0:len(peaks16)-1],peaks16[0:len(peaks16)-1]-410.30, 'k-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay16 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# # plt.xlim(-5,18)
# # plt.ylim(-0.125,0.5)
# plt.title("Doppler Shift for pos 16")


plt.xlabel("probe delay16 (ps)")
plt.ylabel("doppler shift (nm)")
plt.legend(fontsize = 10)
plt.title("Doppler Shift")
plt.ylim(-0.2,0.6)
plt.grid(lw = 1, color = "black")


# Set background colors based on y-values
# Generate sample data
x = np.linspace(-5, 15, 100)
y1 = np.ones(len(x))*0.8
y2 = -y1*0.25


# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
plt.show() 




v17 = calc_vel(w = peaks17, w0 = 410.4)
v17_uerr = calc_vel(w = peaks17+0.064, w0 = 410.4)-v17
v17_lerr = v17-calc_vel(w = peaks17-0.064, w0 = 410.4)

plt.plot(delay17, v17,'o', label='pos 17')
plt.errorbar(x=delay17, y=v17, yerr=[v17_lerr,v17_uerr], elinewidth=0.3, capsize=2, capthick=0.3, color = 'k')
#plt.plot(delay17, v17, 'k-')


v6 = calc_vel(w = peaks6, w0 = 410.5)
v6_uerr = calc_vel(w = peaks6+0.064, w0 = 410.5)-v6
v6_lerr = v6-calc_vel(w = peaks6-0.064, w0 = 410.5)

plt.plot(delay6, v6,'o', label='pos 6')
plt.errorbar(x=delay6, y=v6, yerr=[v6_lerr,v6_uerr], elinewidth=0.3, capsize=2, capthick=0.3, color = 'k')
#plt.plot(delay6, v6, 'k-')



v7 = calc_vel(w = peaks7, w0 = 410.4)
v7_uerr = calc_vel(w = peaks7+0.064, w0 = 410.4)-v7
v7_lerr = v7-calc_vel(w = peaks7-0.064, w0 = 410.4)

plt.plot(delay7, v7,'o', label='pos 7')
plt.errorbar(x=delay7, y=v7, yerr=[v7_lerr,v7_uerr], elinewidth=0.3, capsize=2, capthick=0.3, color = 'k')
#plt.plot(delay7, v7, 'k-')



v11 = calc_vel(w = peaks11, w0 = 410.45)
v11_uerr = calc_vel(w = peaks11+0.064, w0 = 410.45)-v11
v11_lerr = v11-calc_vel(w = peaks11-0.064, w0 = 410.45)

plt.plot(delay11, v11,'o', label='pos 11')
plt.errorbar(x=delay11, y=v11, yerr=[v11_lerr,v11_uerr], elinewidth=0.3, capsize=2, capthick=0.3,color = 'k')
#plt.plot(delay11, v11, 'k-')



v1 = calc_vel(w = peaks1, w0 = 410.5)
v1_uerr = calc_vel(w = peaks1+0.064, w0 = 410.5)-v1
v1_lerr = v1-calc_vel(w = peaks1-0.064, w0 = 410.5)

plt.plot(delay1, v1,'o', label='pos 1')
plt.errorbar(x=delay1, y=v1, yerr=[v1_lerr,v1_uerr], elinewidth=0.3, capsize=2, capthick=0.3, color = 'k')
#plt.plot(delay1, v1, 'k-')



v16 = calc_vel(w = peaks16, w0 = 410.3)
v16_uerr = calc_vel(w = peaks16+0.064, w0 = 410.3)-v16
v16_lerr = v16-calc_vel(w = peaks16-0.064, w0 = 410.3)

plt.plot(delay16, v16,'o', label='pos 16')
plt.errorbar(x=delay16, y=v16, yerr=[v16_lerr,v16_uerr], elinewidth=0.3, capsize=2, capthick=0.3, color = 'k')
#plt.plot(delay16, v16, 'k-')

plt.xlim(-5,14.7)
plt.grid(lw = 1, color = "black")
plt.legend()
plt.title("Doppler veolcity for differnet spatial location")
plt.xlabel("Delay (ps)")
plt.ylabel("v/c")

# Set background colors based on y-values
# Generate sample data
x = np.linspace(-5, 15, 100)
y1 = np.ones(len(x))*0.0015
y2 = -np.ones(len(x))*0.0005


# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
plt.show()