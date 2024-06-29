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



t=0.66    # in ps # t = 0.33, 0.66, 1.0..... 10.

B_max=80
resolution=20
Bx,By=np.mgrid[-B_max+0.001:B_max+0.001:resolution*1j,-B_max+0.001:B_max+0.001:resolution*1j]

Bz = 80e6    # Bz = 1,2,3,4,...,100
L = t*1e-12*6e6 #Plasma total length

Bx_By_to_ellip_farad = np.vectorize(mms.Bx_By_to_ellip_farad)

final_chi, final_psi = Bx_By_to_ellip_farad(Bx*10**6,By*10**6, L=L, Bz=Bz)

final_ellip_array = np.tan(final_chi)
final_faraday_rot = np.tan(final_psi) 



plt.figure()
plt.imshow(final_ellip_array,cmap="jet",origin="upper",extent=[-B_max,B_max,-B_max,B_max])
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.title(r"variation of ellipticity (tan($\chi$))"+"\nfor t="+f"{t} ps,  Bz: {int(Bz/1e6)}  MG")
#plt.savefig(f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.png",bbox_inches="tight")
plt.show()


plt.figure()
plt.imshow(final_faraday_rot,cmap="jet",origin="upper",extent=[-B_max,B_max,-B_max,B_max])
plt.colorbar()
plt.xlabel("Bx (MG)")
plt.ylabel("By (MG)")
plt.title(r"variation of Faraday Rotation (tan($\psi$))"+"\nfor t="+f"{t} ps,  Bz: {int(Bz/1e6)}  MG")
#plt.savefig(f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.png",bbox_inches="tight")
plt.show()


#file_name = f"/Volumes/Extreme SSD/Experiments/Polarimetry 18Aug2023/python codes/B_chart/t_{t}_Bz_{int(Bz/1e6)}.txt"
#np.savetxt(file_name, final_ellip_array, fmt='%f', delimiter='\t')