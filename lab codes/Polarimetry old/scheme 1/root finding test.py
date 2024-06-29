"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab


import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (abs(array - value)).argmin()
    val=array.flat[idx]
    index=np.where(array==val)
    return val,index

def dist_in_array(index1,index2):
    dist=0
    for i in range(len(index2)):
        dist+=(index2[i]-index1[i])**2
    return(dist)
    
def closest_index(array,value,old_index):
    closest_val,closest_val_indices=find_nearest_value(array,value)
    #print(f"val: {val}")
    #print(f"index: {index}")
    dist=[]
    for i in range(len(closest_val_indices[0])):
        index2=[]
        for j in range(len(old_index)):
            index2.append(closest_val_indices[j][i])
        #print(f"index2: {index2}")
        dist.append(dist_in_array(old_index,index2))
    
    mindist,mindist_index=find_nearest_value(dist,np.min(dist))
    #print(f"mindist: {mindist}")
    #print(f"mindist_index: {mindist_index}")
    mindist_index_of_closest_val_in_arr=[]
    
    for i in range(len(old_index)):
        #print(mindist_index[0][0])
        mindist_index_of_closest_val_in_arr.append(closest_val_indices[i][mindist_index[0][0]])
    return(closest_val,mindist_index_of_closest_val_in_arr)

def give_nearest_B(final_ellip_array,ellip,old_index):
    ellip_nearest,index=closest_index(final_ellip_array,ellip,old_index)
    
    Bx_nearest=Bx[index[0]][index[1]]
    By_nearest=By[index[0]][index[1]]
    
    return Bx_nearest,By_nearest,index

file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\test1\\B matrix\\Bx.txt"
r = np.loadtxt(file_name, dtype=float, delimiter='\t')
Bx=r[:][:]

file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\test1\\B matrix\\By.txt"
r = np.loadtxt(file_name, dtype=float, delimiter='\t')
By=r[:][:]

import glob
e_files=sorted(glob.glob("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\test1\\*.txt"))
print(e_files)

t=np.linspace(0,5,200)
def testfn(t):
    return(10*t*np.exp(-t**2))

ellip_over_t=testfn(t)
ellip_over_t=ellip_over_t/max(ellip_over_t)
plt.plot(t,ellip_over_t)
plt.show()
                  
calculated_Bx=[]
calculated_By=[]

index=[0,0]
for i in range(len(ellip_over_t)):
    r = np.loadtxt(e_files[-3], dtype=float, delimiter='\t')
    final_ellip_array=r[:][:]
    Bx_now,By_now,index=give_nearest_B(final_ellip_array, ellip_over_t[i], index)
    calculated_Bx.append(Bx_now)
    calculated_By.append(By_now)
    
plt.plot(calculated_Bx,calculated_By,'ro')
plt.plot(calculated_Bx,calculated_By,'k-')
plt.title("Emolution of Bx and By")
plt.xlabel("Bx (in MG)")
plt.ylabel("By (in MG)")
plt.show()


calculated_Bx=np.array(calculated_Bx)
calculated_By=np.array(calculated_By)


plt.plot(t,abs(calculated_Bx),'r-',label="|Bx|")
plt.plot(t,calculated_Bx,'g--',label="Bx",lw=0.5)
plt.xlabel("time (ps)")
plt.ylabel("Bx in MG")
plt.title("evolution of Bx in plasma")
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=1)
plt.show()


plt.plot(t,abs(calculated_By),'r-',label="|By|")
plt.plot(t,calculated_By,'g--',label="By",lw=0.5)
plt.xlabel("time (ps)")
plt.ylabel("By in MG")
plt.title("evolution of By in plasma")
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=1)
plt.show()


plt.plot(t,np.sqrt(abs(calculated_Bx**2+calculated_By**2)),'r-',label="|B|")
plt.xlabel("time (ps)")
plt.ylabel("By in MG")
plt.title("evolution of By in plasma")
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=1)
plt.show()