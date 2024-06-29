# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:26:32 2024

@author: mrsag
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:45:10 2024

@author: mrsag
"""

#----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ---------------------------------------------------------------------------------------
import numpy as np
from math import pi, sqrt, sin, cos 
import matplotlib.pyplot as plt

l0 = 2. * pi             # laser wavelength [in code units]
t0 = l0                  # optical cycle
Lsim = [100.0*l0]  # length of the simulation
Tsim = 400.0*t0            # duration of the simulation
resx = 128               # nb of cells in one laser wavelength
rest = resx*sqrt(2.)/0.95 # nb of timesteps in one optical cycle 
#Tsim = t0


Lx = Lsim[0]       # Simulation box length
x0 = 10.*l0     # Target thickness (width of step function)    
Xsurface = Lx - x0  # Vacuum region

aL=3.0  # Laser Intensity (a0 = 0.855*1e-9*sqrt(I)*Lambda   (I in W/cm^2,  Lambda in micron)
waist = 5.*l0   # Focal spot size
Tfwhm = 8.*l0   # Laser time FWHM;  8*t0 means 8 optical cycle pulse

n0 = 100.   # Target density, n0 comes from lambda dependence
one_ev = 1/511.0e3   # from formula
T0_eon = 10.0*one_ev    # Temperature of electron plasma at start of simulation
T0_ion = 10.0*one_ev    #     "        "    ion     "     "   "    "     "   
nppc = 16               # no of particles per unit cell

@np.vectorize
def density_profile(x):
    if(x<Xsurface-6.0*l0):
        return 0.
    elif(x>=Xsurface-6.0*l0 and x<Xsurface):
        return n0*np.exp((x-Xsurface)/(1*l0))
    else:
        return n0

# @np.vectorize
# def density_profile(x):
#     if(x<Xsurface):
#         return 0.
#     else:
#         return n0    


# @np.vectorize
# def density_profile(x):
#     if(x>Xsurface):
#         return n0
#     else:
#         return n0*np.exp((x-Xsurface)/(1*l0))    



### fs structure profile
@np.vectorize
def time_profile(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    # w1 = 39.2439786/2.6666666*l0
    w1 = 19.2439786/2.6666666*l0
    #xc1 = (4.80421665 + 600)/2.6666666*l0
    xc1 = 250/2.66666666*l0
    A2 = 4.297148e1
    w2 = 80.0356576/2.6666666*l0
    xc2 = (-83.484182 + 250)/2.6666666*l0
    maxx = 0.4040909459763201
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)

### Single peak - 60fs
@np.vectorize
def time_profile2(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/2.6666666*l0
    xc1 = (4.80421665 + 500)/2.66666666*l0
    maxx = 0.36773593926954434
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2))/maxx)

@np.vectorize
def time_profile3(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0/3.
    xc1 = (4.80421665 + 300)/(8./3.)*l0
    A2 = 4.297148e1
    w2 = 281.0356576/(8./3.)*l0/3.
    xc2 = (-113.484182 + 370)/(8./3.)*l0
    #maxx = 1.
    maxx = 1.2021913830894313
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)

### Single peak - 60fs
@np.vectorize
def time_profile4(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0/3.
    xc1 = (4.80421665 + 300)/(8./3.)*l0
    #maxx = 1.
    maxx = 1.1108121526768813
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2))/maxx)

@np.vectorize
def time_profile5(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0
    xc1 = (4.80421665 + 600)/(8./3.)*l0
    A2 = 4.297148e1
    w2 = 39.2439786/(8./3.)*l0
    xc2 = (-113.484182 + 550)/(8./3.)*l0
    maxx = 0.37079699580774755
    #maxx=1
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)




x = np.linspace(0,Lsim[0],1001)
density = density_profile(x)

plt.plot(x,density)
plt.show()


# t = np.linspace(0, Tsim, 1001)
# pulse = time_profile(t)

# plt.plot(t/t0*2.666,pulse)
# plt.show()

