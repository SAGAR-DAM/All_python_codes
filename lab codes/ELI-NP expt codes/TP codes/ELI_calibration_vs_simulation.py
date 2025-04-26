# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:55:40 2025

@author: mrsag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.integrate as integrate
# from Curve_fitting_with_scipy import polynomial_fit as pft
import glob
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 17
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=500 # highres display

# %%
# Defining constants:
cm = 1e-2
mm = 1e-3
e = 1.6e-19
me = 9.11e-31
keV_to_Joule = 1.6e-16
c_light=3e8  # in SI unit

# %%
B_field = 0.45 # Magnetic Field (B)  (in Tesla)
lB = 5*cm  # Magnetic field length (lB)  (in cm)
DB = (32.3+1.75)*cm # Distance of screen from magnetic field region endpoint (DB)  (in cm)
qp = 1*e # Charge of proton in terms of proton charge (q)
mH = 1836*me # Mass of particle (in terms of electron mass)

# %%
def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index


def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)


def parabolic_curve(x, a, b, c):
    """Returns y values for a parabola y = ax^2 + bx + c."""
    return a * x**2 + b * x + c


# %%
@np.vectorize
def radius_of_curvature(energy,m=mH,q=qp,B=B_field):
    f1=(m*c_light)/(q*B)
    f2=np.sqrt((1+energy/(m*c_light**2))**2-1)
    R = f1*f2
    return R

@np.vectorize
def magnetic_shift(energy,DB=DB,lB=lB,m=mH,q=qp,B=B_field):
    R = radius_of_curvature(energy,m,q,B)
    l1 = R-np.sqrt(R**2-lB**2)
    l2 = lB*DB/np.sqrt(R**2-lB**2)
    L = l1+l2
    return L

@np.vectorize
def angular_dependence(energy,DB=DB,lB=lB,m=mH,q=qp,B=B_field):
    R = radius_of_curvature(energy,m,q,B)
    val = np.sqrt(R**2-lB**2)/R
    return val

# %%
energy_chart = np.linspace(100,200000,10000)*keV_to_Joule
LB_chart = magnetic_shift(energy=energy_chart,DB=DB,lB=lB,m=mH,q=qp,B=B_field)

# Since LB is decreasing, reverse both arrays to make interpolation work
LB_to_E = interp1d(LB_chart[::-1], energy_chart[::-1], kind='cubic', fill_value='extrapolate')

given_LB = 13.256*mm  # energy in keV
E_result = LB_to_E(given_LB)/keV_to_Joule

print(f"For LB = {given_LB/mm} mm, E_proton = {E_result/1000 :.3f} MeV")

chart_file_path = r"D:\data Lab\ELI-NP March 2025\TP dimensions\Core1_045T_50mm_ELI.xlsx"
# Read the Excel file
df = pd.read_excel(chart_file_path)
df = np.array(df)
dist = df[2:,0]
energy_calib = df[2:,1]

LB_chart_mm = LB_chart/mm
energy_chart_MeV = energy_chart/keV_to_Joule/1e3

# Create a DataFrame
df = pd.DataFrame({
    'Proton deflection in magnetic field (mm)': LB_chart_mm,
    'KE (MeV)': energy_chart_MeV
})

# Save to Excel
df.to_excel(f"D:\data Lab\ELI-NP March 2025\Core1_045T_50mm.xlsx", index=False)

# plt.figure(figsize=(2,1))
plt.plot(energy_chart_MeV,LB_chart_mm,"r-",label="From calculation")
plt.plot(energy_calib,dist,"go",markersize=8,label="ELI calibration")
plt.plot(energy_chart/keV_to_Joule/1e3,LB_chart/mm,"ko",markersize=3)
plt.xlabel("Energy (MeV)")
plt.ylabel("Shift due to B field (mm)")
plt.xscale("log")
# plt.yscale("log")
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=1, color='k')
plt.grid(which='minor', linestyle='-', linewidth=0.5, color='k')
plt.title("Proton energy vs magnetic shift")
plt.legend()
plt.show()



