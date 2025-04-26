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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

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


#################################################################
#################################################################
#################################################################
#################################################################
''' pos 17 scan 1'''
#################################################################
#################################################################
#################################################################
#################################################################
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 1\\*.txt"))

peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

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
#     peaks1.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay1: {delay1[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks1 = []
delay1 = np.linspace(12,15,len(files_17)//2)-12.65
delay1 = 2*delay1/c
delay1 = np.around(delay1, decimals=3)

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
    intensity /= max(intensity)
    
    fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
    peaks1.append(parameters[1])
    
    plt.plot(wavelength, intensity)
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay1: {delay1[i]}")
    plt.xlim(408,414)
    plt.xlabel("Wavelength")#"\n"+f"Peak:  {parameters[1]}")
    plt.ylabel("Intensity")
plt.show()    
    
for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
        except:
            peaks1[i] = 410.4

peaks1 = moving_average(peaks1,4)

for i in range(len(peaks1)):
    if (peaks1[i]<409 or peaks1[i]>412):
        try:
            if((peaks1[i-1]<412 and peaks1[i+1]>409) or (peaks1[i+1]<412 and peaks1[i+1]>409)):
                peaks1[i] = (peaks1[i+1]+peaks1[i-1])/2
            else:
                peaks1[i] = 410.4
        except:
            peaks1[i] = 410.4

delay1 = delay1[0:len(peaks1)]

shifts1 = peaks1-410.4

# plt.plot(delay1[0:len(peaks1)],peaks1, 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay1 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay1))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()

plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.4, 'ro')
plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.4, 'k-')
plt.errorbar(x=delay1[0:len(peaks1)-1],y=peaks1[0:len(peaks1)-1]-410.4, yerr=np.ones(len(peaks1[0:len(peaks1)-1]))*0.074,lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(x=delay1[0:len(peaks1)-1],y=peaks1[0:len(peaks1)-1]-410.4, yerr=np.ones(len(peaks1[0:len(peaks1)-1]))*0.074,lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot((delay1[0:len(peaks1)])[peaks1-410.4<=0],peaks1[peaks1-410.4<=0]-410.4, 'bo')
plt.plot(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.4,'ko-', markersize=3,label = 'scan 1')
plt.fill_between(delay1[0:len(peaks1)-1],peaks1[0:len(peaks1)-1]-410.4-np.ones(len(peaks1[0:len(peaks1)-1]))*0.074,peaks1[0:len(peaks1)-1]-410.4+np.ones(len(peaks1[0:len(peaks1)-1]))*0.074,color="k",alpha=0.3)
plt.title("Doppler shift")
plt.xlabel("probe delay (ps)")
plt.ylabel("Peak wavelength (nm)")
plt.xticks()
plt.yticks()
# plt.xlim(-5,18)
# plt.ylim(-0.125,0.5)
plt.title("Doppler Shift for pos 17")
plt.grid(lw = 1, color = "black")





#################################################################
#################################################################
#################################################################
#################################################################
''' pos 17 scan 2'''
#################################################################
#################################################################
#################################################################
#################################################################
files_17 = sorted(glob.glob("D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\04 th april 2024\\pos_17\\scan 2\\*.txt"))

# peaks2 = []
# delay2 = np.linspace(12,15,len(files_17)//2)-12.65
# delay2 = 2*delay2/c
# delay2 = np.around(delay2, decimals=3)

# for i in range(1,len(files_17),2):
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
#     peaks2.append(parameters[1])
    
#     plt.plot(wavelength, intensity, 'r-', label = "data")
#     plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
#     #plt.title(files[i][-9:]+"\n"+f"delay2: {delay2[i]}")
#     #plt.xlim(400,420)
#     plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
#     plt.ylabel("Intensity")
#     plt.show()  

peaks2 = []
delay2 = np.linspace(12,15,len(files_17)//2)-12.65
delay2 = 2*delay2/c
delay2 = np.around(delay2, decimals=3)

for i in range(1,len(files_17),2):
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
    peaks2.append(parameters[1])
    
    # plt.plot(wavelength, intensity, 'r-', label = "data")
    # plt.plot(wavelength, fit_I, 'b-', label = " Gauss fit")
    # plt.title(files[i][-9:]+"\n"+f"delay2: {delay2[i]}")
    # plt.xlim(400,420)
    # plt.xlabel("Wavelength\n"+f"Peak:  {parameters[1]}")
    # plt.ylabel("Intensity")
    # plt.show()    
    
for i in range(len(peaks2)):
    if (peaks2[i]<409 or peaks2[i]>412):
        try:
            peaks2[i] = (peaks2[i+1]+peaks2[i-1])/2
        except:
            peaks2[i] = 410.3

peaks2 = moving_average(peaks2,4)

for i in range(len(peaks2)):
    if (peaks2[i]<409 or peaks2[i]>412):
        try:
            if((peaks2[i-1]<412 and peaks2[i+1]>409) or (peaks2[i+1]<412 and peaks2[i+1]>409)):
                peaks2[i] = (peaks2[i+1]+peaks2[i-1])/2
            else:
                peaks2[i] = 410.3
        except:
            peaks2[i] = 410.3

delay2 = delay2[0:len(peaks2)]
shifts2 = peaks2-410.3

# plt.plot(delay2[0:len(peaks2)],peaks2, 'ro-')
# plt.title("Doppler shift")
# plt.xlabel("probe delay2 (ps)")
# plt.ylabel("Peak wavelength (nm)")
# #plt.xlim(-2,max(delay2))
# #plt.ylim(395,396)
# plt.title("Peak Wavelength")
# plt.grid(lw = 1, color = "black")
# plt.show()


plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3, 'rd')
plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3, 'k-')
plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot((delay2[0:len(peaks2)])[peaks2-410.3<=0],peaks2[peaks2-410.3<=0]-410.3, 'bd')
plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'kd', markersize=3)
plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'gd-', markersize=3,label = 'scan 2')
plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'gd', markersize=3)
plt.fill_between(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3-np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,peaks2[0:len(peaks2)-1]-410.3+np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,color="k",alpha=0.3)

# plt.plot(delay2[0:len(peaks2)],peaks2[0:len(peaks2)]-410.3, 'go', label = 'scan 2')
# plt.plot(delay2[0:len(peaks2)],peaks2[0:len(peaks2)]-410.3, 'k-')
# plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,color='k', capsize=1, linewidth=0.5)
plt.title("Doppler shift")
plt.xlabel("Probe delay (ps)")
plt.legend()
plt.ylabel("Doppler Shift (nm)")
plt.xlim(-5,15.5)
plt.ylim(-0.15,0.53)
plt.title("Doppler Shift\n"+r"Intensity: $1.3\times 10^{18}\ W/cm^2$")
plt.grid(lw = 1, color = "black")

# Generate sample data
x = np.linspace(-5, 15.5, 100)
y1 = np.ones(len(x))*0.53
y2 = -np.ones(len(x))*0.15

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
plt.xticks()
plt.yticks()
plt.show()



# def calc_vel(w, w0):
#     c_norm = -3e10
#     v = (w**2-w0**2)/(w**2+w0**2)*c_norm
#     return v


def calc_vel(w, w0):
    c_norm = 3e10
    v = -0.5*(w-w0)/w*c_norm
    return v

# def calc_vel(w, w0):
#     c_norm = 3e10
#     v = -0.5*(w-w0)/w*c_norm
#     return v


v1 = calc_vel(w = peaks1, w0 = 410.4)
v1_uerr = np.abs(calc_vel(w = peaks1+0.064, w0 = 410.4)-v1)
v1_lerr = np.abs(v1-calc_vel(w = peaks1-0.064, w0 = 410.4))

red_delay1 = delay1[shifts1>0]
red_v1 = v1[shifts1>0]

blue_delay1 = delay1[shifts1<0]
blue_v1 = v1[shifts1<0]

plt.plot(delay1, v1, 'ro')
plt.errorbar(x=delay1, y=v1, yerr=[v1_lerr,v1_uerr],  elinewidth=0.5, capsize=2, capthick=0.5, color = 'k')
plt.plot(delay1, v1, 'k-')

v2 = calc_vel(w = peaks2, w0 = 410.3)


blue_delay2 = delay2[shifts2<0]
blue_v2 = v2[shifts2<0]


v2_uerr = np.abs(calc_vel(w = peaks2+0.064, w0 = 410.3)-v2)
v2_lerr = np.abs(v2-calc_vel(w = peaks2-0.064, w0 = 410.3))


plt.plot(delay2, v2, 'go')
plt.errorbar(x=delay2, y=v2, yerr=[v2_lerr,v2_uerr],  elinewidth=0.5, capsize=2, capthick=0.5, color = 'k')
plt.plot(delay2, v2, 'k-')

plt.title("velocity (v/c) for pos 17")
plt.grid(lw = 0.5, color = "black")
plt.xlabel("""Probe delay (ps)""")
plt.ylabel(r"$\beta$ (v/c)")

# Generate sample data
x = np.linspace(min(delay1), max(delay1), 100)
y1 = np.ones(len(x))*(max(v2)+max(v2_uerr))
y2 = np.ones(len(x))*(min(v2)-min(v2_lerr))

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='r', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='b', alpha=0.3)
plt.xlim(-5,15.3)
plt.xticks()
plt.yticks()
plt.show()



# ############################################################################
# ############################################################################

# v1 = calc_vel(w = peaks1, w0 = 410.4)
# v1_uerr = np.abs(calc_vel(w = peaks1+0.064, w0 = 410.4)-v1)
# v1_lerr = np.abs(v1-calc_vel(w = peaks1-0.064, w0 = 410.4))

# # plt.plot(delay1, v1, 'ro')
# # plt.errorbar(x=delay1, y=v1, yerr=[v1_lerr,v1_uerr], elinewidth=0.5, capsize=2, color = 'k')
# # plt.plot(delay1, v1, 'k-')

# v2 = calc_vel(w = peaks2, w0 = 410.3)
# v2_uerr = np.abs(calc_vel(w = peaks2+0.064, w0 = 410.3)-v2)
# v2_lerr = np.abs(v2-calc_vel(w = peaks2-0.064, w0 = 410.3))

# # plt.plot(delay2, v2, 'go')
# # plt.errorbar(x=delay2, y=v2, yerr=[v2_lerr,v2_uerr], elinewidth=1, capsize=2, color = 'b')
# # plt.plot(delay2, v2, 'k-')

# velocity = (v1+v2)/2
# blue_v = []
# red_v = []

# blue_delay = []
# red_delay= []

# for i in range(len(velocity)):
#     if(velocity[i]<=0):
#         blue_delay.append(delay1[i])
#         blue_v.append(velocity[i])
        
#     else:
#         red_delay.append(delay1[i])
#         red_v.append(velocity[i])
        
# blue_v=np.array(blue_v)
# red_v=np.array(red_v)

# plt.plot(blue_delay,blue_v, 'bo')
# plt.plot(red_delay,red_v, 'ro')
# plt.plot(delay1,velocity, 'k-')
# plt.errorbar(x=delay2, y=velocity, yerr=[v2_lerr,v2_uerr], elinewidth=0.5, capsize=2, capthick=0.5, color = 'k')
# plt.title("velocity (v/c) for pos 17")
# plt.grid(lw = 0.5, color = "black")
# plt.xlabel("Delay (ps)")
# plt.ylabel(r"$\beta$ (v/c)")

# # Generate sample data
# x = np.linspace(min(delay1), max(delay1), 100)
# y1 = np.ones(len(x))*(max(v2)+max(v2_uerr))
# y2 = np.ones(len(x))*(min(v2)-min(v2_lerr))

# # Set background colors based on y-values
# plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
# plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
# plt.xlim(-5,15.3)
# plt.show()

plt.plot(delay1,v1, 'ro')
plt.plot(delay1,v1, 'k-')
plt.errorbar(x=delay1,y=v1, yerr=[v1_lerr,v1_uerr],lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(x=delay1,y=v1, yerr=[v1_lerr,v1_uerr],lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(blue_delay1,blue_v1, 'bo')
plt.plot(delay1,v1,'ko-', markersize=3,label = 'scan 1')
plt.fill_between(delay1,v1-v1_lerr,v1+v1_uerr,color="k",alpha=0.3)

plt.plot(delay2,v2, 'rd')
plt.plot(delay2,v2, 'k-')
plt.errorbar(x=delay2,y=v2, yerr=[v2_lerr,v2_uerr],lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(x=delay2,y=v2, yerr=[v2_lerr,v2_uerr],lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(blue_delay2,blue_v2, 'bo')
plt.plot(delay2,v2,'gd-', markersize=3,label = 'scan 1')
plt.fill_between(delay2,v2-v2_lerr,v2+v2_uerr,color="g",alpha=0.3)

# plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3, 'rd')
# plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3, 'k-')
# plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,lw=0, elinewidth=2, capsize=2, color = 'r')
# plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,lw=0, elinewidth=1, capsize=2, color = 'k')
# plt.plot((delay2[0:len(peaks2)])[peaks2-410.3<=0],peaks2[peaks2-410.3<=0]-410.3, 'bd')
# plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'kd', markersize=3)
# plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'gd-', markersize=3,label = 'scan 2')
# plt.plot(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3,'gd', markersize=3)
# plt.fill_between(delay2[0:len(peaks2)-1],peaks2[0:len(peaks2)-1]-410.3-np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,peaks2[0:len(peaks2)-1]-410.3+np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,color="k",alpha=0.3)

# plt.plot(delay2[0:len(peaks2)],peaks2[0:len(peaks2)]-410.3, 'go', label = 'scan 2')
# plt.plot(delay2[0:len(peaks2)],peaks2[0:len(peaks2)]-410.3, 'k-')
# plt.errorbar(x=delay2[0:len(peaks2)-1],y=peaks2[0:len(peaks2)-1]-410.3, yerr=np.ones(len(peaks2[0:len(peaks2)-1]))*0.074,color='k', capsize=1, linewidth=0.5)

plt.title("Doppler shift")
plt.xlabel("Probe delay (ps)")
plt.legend()
plt.ylabel("velocity (cm/s)")
plt.xlim(-5,15.5)
plt.ylim(-2e7,0.5e7)
plt.title("Critical surface velocity\n"+r"Intensity: $1.3\times 10^{18}\ W/cm^2$")
plt.grid(lw = 1, color = "black")

# Generate sample data
x = np.linspace(-5, 15.5, 100)
y1 = np.ones(len(x))*1e7
y2 = -np.ones(len(x))*4e7

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='r', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='b', alpha=0.3)
plt.xticks()
plt.yticks()
plt.show()


############################################################################
############################################################################

v1 = calc_vel(w = peaks1, w0 = 410.4)
v1_uerr = np.abs(calc_vel(w = peaks1+0.064, w0 = 410.4)-v1)
v1_lerr = np.abs(v1-calc_vel(w = peaks1-0.064, w0 = 410.4))

# plt.plot(delay1, v1, 'ro')
# plt.errorbar(x=delay1, y=v1, yerr=[v1_lerr,v1_uerr], elinewidth=0.5, capsize=2, color = 'k')
# plt.plot(delay1, v1, 'k-')

v2 = calc_vel(w = peaks2, w0 = 410.3)
v2_uerr = np.abs(calc_vel(w = peaks2+0.064, w0 = 410.3)-v2)
v2_lerr = np.abs(v2-calc_vel(w = peaks2-0.064, w0 = 410.3))

# plt.plot(delay2, v2, 'go')
# plt.errorbar(x=delay2, y=v2, yerr=[v2_lerr,v2_uerr], elinewidth=1, capsize=2, color = 'b')
# plt.plot(delay2, v2, 'k-')

velocity = (v1+v2)/2
blue_v = []
red_v = []

blue_delay = []
red_delay= []

for i in range(len(velocity)):
    if(velocity[i]<=0):
        blue_delay.append(delay1[i])
        blue_v.append(velocity[i])
        
    else:
        red_delay.append(delay1[i])
        red_v.append(velocity[i])
        
blue_v=np.array(blue_v)
red_v=np.array(red_v)

plt.plot(blue_delay,blue_v, 'bo')
plt.plot(red_delay,red_v, 'ro')
plt.plot(delay1,velocity, 'k-')
plt.errorbar(x=delay2, y=velocity, yerr=[v2_lerr,v2_uerr], elinewidth=0.5, capsize=2, capthick=0.5, color = 'k')
plt.title("velocity (v/c) for pos 17")
plt.grid(lw = 0.5, color = "black")
plt.xlabel("Delay (ps)")
plt.ylabel(r"$\beta$ (v/c)")

# Generate sample data
x = np.linspace(min(delay1), max(delay1), 100)
y1 = np.ones(len(x))*(max(v2)+max(v2_uerr))
y2 = np.ones(len(x))*(min(v2)-min(v2_lerr))

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='blue', alpha=0.3)
plt.fill_between(x, y1, where=(y1 > 0), color='red', alpha=0.3)
plt.xlim(-5,15.3)
plt.show()




##########################################################################
# Simulation
##########################################################################


# Load the dataset

files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\Simulation with Jian\Hydro simulation 3e17 and 1e18 till 30 ps\TIFR_hydro_30ps\TIFR_1D_3_1e18_2\tifr_hdf5_plt_cnt_*")

pos = []

for file in files:
    ds = yt.load(file)
    # Create a data object (like the entire domain)
    ad = ds.all_data()

    index = find_index(np.array(ad['gas', 'El_number_density']),6.97e21)
    e_dens_in_10_e_21 = np.array(ad['gas', 'El_number_density'])/1e21
    
    x = np.array(ad['gas', 'x'])*1e4
    
    pos.append(x[index])

    plt.plot(x,e_dens_in_10_e_21)
    
plt.legend()
plt.axhline(6.97,linestyle="--",color="k",lw=2)
plt.xlim(2,3)
plt.ylim(-6.97/2,6.97*3)
plt.show()
    

pos[0] = 0.00025*1e4
# pos = np.array(pos)-pos[0]
t = np.linspace(0,30,len(pos))

plt.plot(t,pos,"k-")
plt.xlabel("delay (ps)")
plt.ylabel("Simulation position of critical surface (um)")
plt.show()

dt = 1e-13
vel_sim = -1e-4*np.diff(pos)/dt*1.556   # 1/cos(50) = 1.556 as the simulation was done on 50 degree AOI and velocity calculated along that angle
plt.plot(t[1:],vel_sim)
plt.xlabel("delay (ps)")
plt.ylabel("Simulated velocity")
plt.show()




true_delay = t[1:][vel_sim!=0]
true_vel = vel_sim[vel_sim!=0]




x = moving_average(true_delay, 4)*0.95
y = moving_average(true_vel, 4)

y = y[0:find_index(x,18.5)]
x = x[0:find_index(x,18.5)]


# Threshold for spacing
min_spacing = 0.5

# Filtering logic
filtered_x = [x[0]]
filtered_y = [y[0]]
last_x = x[0]

for i in range(1, len(x)):
    if abs(x[i] - last_x) >= min_spacing:
        filtered_x.append(x[i])
        filtered_y.append(y[i])
        last_x = x[i]
        
        
        
        

# delay1 -= 2
# delay2 -= 2
# Creating the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data for the first scan
ax.plot(delay1, v1, 'ro', color="brown",markersize=10,label="scan 1")
ax.plot(delay1, v1, 'k-', lw=1)
# ax.errorbar(x=delay1, y=v1, yerr=[v1_lerr, v1_uerr], lw=0, elinewidth=2, capsize=2, color='r')
# ax.errorbar(x=delay1, y=v1, yerr=[v1_lerr, v1_uerr], lw=0, elinewidth=1, capsize=2, color='k')
ax.plot(blue_delay1, blue_v1, 'bo',markersize=10)
# ax.plot(delay1, v1, 'ko-', markersize=3, label='scan 1', lw=0.5)
ax.fill_between(delay1, v1 - v1_lerr, v1 + v1_uerr, color="k", alpha=0.1)

# Plot data for the second scan
ax.plot(delay2, v2, 'rd',markersize=10,label="scan 2",color="brown")
# ax.plot(delay2, v2, 'k-', lw=0.5)
# ax.errorbar(x=delay2, y=v2, yerr=[v2_lerr, v2_uerr], lw=0, elinewidth=2, capsize=2, color='r')
# ax.errorbar(x=delay2, y=v2, yerr=[v2_lerr, v2_uerr], lw=0, elinewidth=1, capsize=2, color='k')
ax.plot(blue_delay2, blue_v2, 'bd',markersize=10)
ax.plot(delay2, v2, 'gd-', markersize=3, lw=1)
ax.fill_between(delay2, v2 - v2_lerr, v2 + v2_uerr, color="k", alpha=0.1)

ax.plot(
    filtered_x,
    filtered_y,
    marker="s",
    color="k",
    label="Simulation",
    lw=0
)


# Load your image (replace 'dot_image.png' with your actual image file)
image_path = "C:\\Users\\mrsag\\OneDrive\\Desktop\\3d balls.png"  
dot_image = mpimg.imread(image_path)

# Function to create image markers
def image_scatter(x, y, image, ax, zoom=0.2):
    for (x0, y0) in zip(x, y):
        im = OffsetImage(image, zoom=zoom)  # Adjust zoom for size
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
        # ax.plot(x,y,'ko')
        
# Scatter plot using images
image_scatter(filtered_x[:-2], filtered_y[:-2], dot_image, ax, zoom=0.04)

# Styling
ax.set_title(
    "Velocity of 400 nm critical surface, " + r"$n_c=6.97\times 10^{21} cm^{-3}$" + "\n" +
    fr"Intensity: 1.3$\times$" + r"$10^{18}$ W/cm$^2$" + "; (Red: inside the target)",
    fontweight='bold',
    fontsize=20
)
ax.set_xlabel("Probe delay (ps)", fontsize=25, fontweight='bold')
ax.set_ylabel("velocity (cm/s)", fontsize=25, fontweight='bold', color="brown")

# Axis limits
ax.set_xlim(-4, 15.5)
ax.set_ylim(-2e7, 1e7)

# Ticks styling
ax.tick_params(axis='y', labelcolor='brown')
ax.tick_params(axis='both', labelsize=20)
ax.yaxis.set_tick_params(labelcolor="brown")
plt.setp(ax.get_xticklabels(), fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')

# Legend styling
legend = ax.legend(fontsize=20, facecolor="lightgray",loc="upper right")
for text in legend.get_texts():
    text.set_fontstyle('italic')


# Generate sample data for background colors
x = np.linspace(min(delay1) * 1.1, max(delay1) * 1.1, 100)

# Define number of layers for the gradient
n_layers = 200

# Maximum Y to shade (depends on your plot's y-limits)
ymax = ax.get_ylim()[1]
ymin = ax.get_ylim()[0]

# Gradient fill above y=0 (blue shades)
for i in range(1, n_layers + 1):
    level = i / n_layers
    ax.fill_between(
        x,y1=np.ones(len(x))*(i)/n_layers*ymax,y2=np.ones(len(x))*(i+0.9)/n_layers*ymax,
        color='blue',
        alpha = 0.3-((1-level)**0.3)*0.3,
    )

# Gradient fill below y=0 (red shades)
for i in range(1, n_layers + 1):
    level = i / n_layers
    ax.fill_between(
        x,y1=np.ones(len(x))*(i)/n_layers*ymin,y2=np.ones(len(x))*(i+0.9)/n_layers*ymin,
        color='red',
        alpha = 0.3-((1-level)**0.3)*0.3,
        zorder=-1
    )



# Background shading
x = np.linspace(-5, 15.5, 100)
y1 = np.ones(len(x)) * 1e7
y2 = -np.ones(len(x)) * 4e7
# ax.fill_between(x, y2, where=(y2 <= 0), color='r', alpha=0.15)
# ax.fill_between(x, y1, where=(y1 > 0), color='b', alpha=0.15)

# Add the image at a specific data coordinate (e.g., at x=2, y=3)
imbox = OffsetImage(dot_image, zoom=0.04)  # Adjust zoom to scale the image
ab = AnnotationBbox(imbox, (10.6, 0.25e7),frameon=False, zorder=10)  # High z-order ensures it's in front
ax.add_artist(ab)

# Display the plot
plt.show()
