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