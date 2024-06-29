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



BT_BA_to_ellip_farad = np.vectorize(mms.BT_BA_to_ellip_farad)

t=0.66    # in ps

resolution=20
B_max=80
BT,BA=np.mgrid[0.001:B_max+0.001:resolution*1j,0.001:B_max+0.001:resolution*1j]


L = t*1e-12*6e6 #Plasma total length


final_chi, final_psi = BT_BA_to_ellip_farad(BT*10**6,BA*10**6, L=L)

final_ellip_array = np.tan(final_chi)
final_faraday_rot = np.tan(final_psi) 



plt.figure()
plt.imshow(final_ellip_array,cmap="jet",origin="upper",extent=[0,B_max,0,B_max])
plt.colorbar()
plt.xlabel(r"$B_{Transverse}$ (MG)")
plt.ylabel(r"$B_{Axial}$ (MG)")
plt.title(r"variation of ellipticity (tan($\chi$))"+"\nfor t="+f"{t} ps")
#plt.savefig(f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.png",bbox_inches="tight")
plt.show()


plt.figure()
plt.imshow(final_faraday_rot,cmap="jet",origin="upper",extent=[0,B_max,0,B_max])
plt.colorbar()
plt.xlabel(r"$B_{Transverse}$ (MG)")
plt.ylabel(r"$B_{Axial}$ (MG)")
plt.title(r"variation of Faraday Rotation (tan($\psi$))"+"\nfor t="+f"{t} ps")
#plt.savefig(f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.png",bbox_inches="tight")
plt.show()


#file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.txt"
#np.savetxt(file_name, final_ellip_array, fmt='%f', delimiter='\t')