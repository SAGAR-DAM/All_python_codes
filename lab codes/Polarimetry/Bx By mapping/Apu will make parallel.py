# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:27:49 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import magfield.magmodule_sagar as mms

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display


t=2.66    # in ps # t = 0.33, 0.66, 1.0..... 10.

B_max=100
resolution=201
Bx,By=np.mgrid[-B_max+0.001:B_max+0.001:resolution*1j,-B_max+0.001:B_max+0.001:resolution*1j]

Bz = 80e6    # Bz = 1,2,3,4,...,100 *e6

L = t*1e-12*6e6 #Plasma total length

Bx_By_to_ellip_farad = np.vectorize(mms.Bx_By_to_ellip_farad)

final_chi, final_psi = Bx_By_to_ellip_farad(Bx*10**6,By*10**6, L=L, Bz=Bz)

final_ellip_array = np.tan(final_chi)
final_faraday_rot = np.tan(final_psi) 



ellip_file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/Bx_By_to_ellipticity_t_{t}_Bz_{int(Bz/1e6)}.txt"
np.savetxt(ellip_file_name, final_ellip_array, fmt='%f', delimiter='\t')


farad_file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/Bx_By_to_tan_faraday_rot_t_{t}_Bz_{int(Bz/1e6)}.txt"
np.savetxt(farad_file_name, final_faraday_rot, fmt='%f', delimiter='\t')


chi_file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/Bx_By_to_chi_t_{t}_Bz_{int(Bz/1e6)}.txt"
np.savetxt(chi_file_name, final_chi, fmt='%f', delimiter='\t')


psi_file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/Bx_By_to_psi_t_{t}_Bz_{int(Bz/1e6)}.txt"
np.savetxt(psi_file_name, final_psi, fmt='%f', delimiter='\t')