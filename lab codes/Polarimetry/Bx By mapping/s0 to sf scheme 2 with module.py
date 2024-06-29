# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import magfield.magmodule_sagar as mms

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display



chi=0.0
psi=0.0

Bx=39e6
By=-6e6
Bz=5e6

L=float(5.33e-12*6e6)
dz=1e-8

s0_1=[np.cos(2*chi)*np.cos(2*psi)]
s0_2=[np.cos(2*chi)*np.sin(2*psi)]
s0_3=[np.sin(2*chi)]

s0=np.array([s0_1,s0_2,s0_3])

sf = mms.final_stokes(s0,Bx,By,Bz,L,dz)
print("s0: \n",s0)
print("sf: \n",sf)
print("sf-s0: \n",sf-s0)

final_chi=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)/2
print(f"final chi: {final_chi}")

final_psi=np.arctan(sf[1][0]/sf[0][0])/2
print(f"final psi:  {final_psi}")