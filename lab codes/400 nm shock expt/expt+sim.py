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
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


# Date: 13/01/2025
# Energy on target: 120 mJ
# pulse width (assumed): 55 fs
# focal spot: 6-7 micron
# intensity: 6e18 W/cm^2 

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




delays = np.array(sorted([28.9,29.65,30.4,30.7,31.15,31.9,32.2,32.65,32.95,33.4,33.7,34.9,35.65]))
time_delays = (delays-30.4)*2/c
peaks = []
std = []

for delay in delays:
    files = glob.glob(f"D:\\data Lab\\400 vs 800 doppler experiment\\400 pump 400 probe\\13012025\\values for plot\\{delay}\\*.txt")
    peak = []
    
    for file in files:
        f = open(file)
        r=np.array(np.loadtxt(f,skiprows=17,comments='>'))
        
        wavelength = r[:,0]
        intensity = r[:,1]
        
        
        intensity -= np.mean(intensity[0:200])
        intensity /= max(intensity)
        minw = find_index(wavelength, 400)
        
        maxw = find_index(wavelength, 420)
        
        wavelength = wavelength[minw:maxw]
        intensity = intensity[minw:maxw] 
        
        fit_I,parameters,string = Gf.Gaussfit(wavelength, intensity)
        
        peak.append(np.mean(wavelength[intensity>0.95]))
        
    peaks.append(np.mean(peak))
    std.append(max([np.std(peak)*2,0.03]))
    
    
peaks = np.array(peaks)
std = np.array(std)/2
shifts = peaks-np.mean(peaks[0:3])

blue_delay = time_delays[shifts<0]
blue_shifts = shifts[shifts<0]

red_delay = time_delays[shifts>0]
red_shifts = shifts[shifts>0] 

plt.errorbar(time_delays, shifts, yerr=std,lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(time_delays, shifts, yerr=std,lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(time_delays, shifts,'ko-')
plt.plot(blue_delay,blue_shifts,"bo")
plt.plot(red_delay, red_shifts,"ro")
plt.plot(time_delays, shifts,'ko', markersize=2)
plt.grid(lw=0.3,color='k')
plt.xlabel("delay (ps)")
plt.ylabel("spectral shift (nm)")
plt.fill_between(time_delays,shifts-std,shifts+std,color="k",alpha=0.3)
plt.title("Pu-Pr delay vs probe doppler shift")


# Generate sample data
x = np.linspace(min(time_delays)*1.1, max(time_delays)*1.1, 100)
y1 = np.ones(len(x))*(max(shifts+std))*1.1
y2 = np.ones(len(x))*(min(shifts-std))*1.1

plt.xlim(min(time_delays)*1.1, max(time_delays)*1.05)
plt.ylim((min(shifts-std))*1.1,(max(shifts+std))*1.1)

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='b', alpha=0.2)
plt.fill_between(x, y1, where=(y1 > 0), color='r', alpha=0.2)

plt.show()




def calc_vel(w, w0):
    c_norm = -3e10
    v = (w**2-w0**2)/(w**2+w0**2)*c_norm
    return v

# def calc_vel(w, w0):
#     c_norm = 3e10
#     v = -0.5*(w-w0)/w*c_norm
#     return v



velocity = calc_vel(peaks,np.mean(peaks[0:3]))
velocity_lower_lim = calc_vel(peaks+std,np.mean(peaks[0:3]))
velocity_upper_lim = calc_vel(peaks-std,np.mean(peaks[0:3]))

velocity_lerr = velocity - velocity_lower_lim
velocity_uerr = velocity_upper_lim - velocity

blue_vel = velocity[shifts<0]
red_vel = velocity[shifts>=0]




plt.errorbar(time_delays, velocity, yerr=[velocity_lerr,velocity_uerr],lw=0, elinewidth=2, capsize=2, color = 'r')
plt.errorbar(time_delays, velocity, yerr=[velocity_lerr,velocity_uerr],lw=0, elinewidth=1, capsize=2, color = 'k')
plt.plot(time_delays, velocity,'ko-')
plt.plot(blue_delay,blue_vel,"bo")
plt.plot(red_delay, red_vel,"ro")
plt.plot(time_delays, velocity,'ko', markersize=2)
plt.fill_between(time_delays,velocity_lower_lim,velocity_upper_lim,color="g", alpha=0.3)
plt.grid(lw=0.3,color='k')
plt.xlabel("delay (ps)")
plt.ylabel("velocity (cm/s)")
plt.title(r"delay vs velocity (crit surf); I=$6\times 10^{18}W/cm^2$")


# Generate sample data
x = np.linspace(min(time_delays)*1.1, max(time_delays)*1.1, 100)
y1 = np.ones(len(x))*(max(velocity_upper_lim))*1.1
y2 = np.ones(len(x))*(min(velocity_lower_lim))*1.1

plt.xlim(min(time_delays)*1.1, max(time_delays)*1.05)
plt.ylim((min(velocity_lower_lim))*1.1,(max(velocity_upper_lim))*1.1)

# Set background colors based on y-values
plt.fill_between(x, y2, where=(y2 <= 0), color='r', alpha=0.2)
plt.fill_between(x, y1, where=(y1 > 0), color='b', alpha=0.2)

plt.show()



######################################################################
# Simulation
######################################################################


# Load the dataset

files = glob.glob(r"D:\data Lab\400 vs 800 doppler experiment\Simulation with Jian\Result for hydro simulation 09-04-2025\TIFR_hydro\TIFR_hydro\TIFR_1D_4_6e18\tifr_hdf5_plt_cnt_*")

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

x = moving_average(true_delay, 1)
y = moving_average(true_vel, 1)

y = y[0:find_index(x,30)]
x = x[0:find_index(x,30)]

# Threshold for spacing
min_spacing = 1

# Filtering logic
filtered_x = [x[0]]
filtered_y = [y[0]]
last_x = x[0]

for i in range(1, len(x)):
    if abs(x[i] - last_x) >= min_spacing:
        filtered_x.append(x[i])
        filtered_y.append(y[i])
        last_x = x[i]
        

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Error bars
# ax.errorbar(time_delays, velocity, yerr=[velocity_lerr, velocity_uerr], lw=0, elinewidth=2, capsize=2, color='r')
# ax.errorbar(time_delays, velocity, yerr=[velocity_lerr, velocity_uerr], lw=0, elinewidth=1, capsize=2, color='k')

# Data points and lines
ax.plot(time_delays, velocity, 'k-')
ax.plot(blue_delay, blue_vel, 'bo',markersize=10)  # Assuming blue_delay and blue_vel are predefined
ax.plot(red_delay, red_vel, 'ro',markersize=10,label="Experiment",color="brown")  # Assuming red_delay and red_vel are predefined
# ax.plot(time_delays, velocity, 'ko', markersize=2)

# Fill between areas for lower and upper limits
ax.fill_between(time_delays, velocity_lower_lim, velocity_upper_lim, color="k", alpha=0.3)

# Grid settings
# ax.grid(lw=0.3, color='k')

# Labels and title
ax.set_xlabel("delay (ps)", fontsize=20, fontweight='bold')
ax.set_ylabel("velocity (cm/s)", fontsize=20, fontweight='bold')
ax.set_title(r"delay vs velocity (crit surf); I=$6\times 10^{18}W/cm^2$", fontsize=20, fontweight='bold')


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
image_scatter(filtered_x, filtered_y, dot_image, ax, zoom=0.04)

# Styling
ax.set_title(
    "Velocity of 400 nm critical surface, " + r"$n_c=6.97\times 10^{21} cm^{-3}$" + "\n" +
    fr"Intensity: 6$\times$" + r"$10^{18}$ W/cm$^2$" + "; (Red: inside the target)",
    fontweight='bold',
    fontsize=20
)
ax.set_xlabel("Probe delay (ps)", fontsize=25, fontweight='bold')
ax.set_ylabel("velocity (cm/s)", fontsize=25, fontweight='bold', color="brown")

# Legend styling
legend = ax.legend(fontsize=20, facecolor="lightgray",loc="upper left")
for text in legend.get_texts():
    text.set_fontstyle('italic')
    

ax.tick_params(axis='y', labelcolor='brown')
ax.tick_params(axis='both', labelsize=20)
ax.yaxis.set_tick_params(labelcolor="brown")
plt.setp(ax.get_xticklabels(), fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')

# Generate sample data for background colors
x = np.linspace(min(time_delays) * 1.1, max(time_delays) * 1.1, 100)
y1 = np.ones(len(x)) * (max(filtered_y)) * 1.1
y2 = np.ones(len(x)) * (min(velocity_lower_lim)) * 1.1

# Axis limits
ax.set_xlim(min(time_delays) * 1.1, max(time_delays) * 1.05)
ax.set_ylim((min(velocity_lower_lim)) * 1.1, (max(filtered_y)) * 1.1)



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
        alpha = 0.4-((1-level)**0.3)*0.4,
        zorder=-1
    )


# Set background colors based on y-values
# ax.fill_between(x, y2, where=(y2 <= 0), color='r', alpha=0.2)
# ax.fill_between(x, y1, where=(y1 > 0), color='b', alpha=0.2)

# Add the image at a specific data coordinate (e.g., at x=2, y=3)
imbox = OffsetImage(dot_image, zoom=0.04)  # Adjust zoom to scale the image
ab = AnnotationBbox(imbox, (-7.7, 1.25e7),frameon=False, zorder=10)  # High z-order ensures it's in front
ax.add_artist(ab)


# Show the plot
plt.show()


