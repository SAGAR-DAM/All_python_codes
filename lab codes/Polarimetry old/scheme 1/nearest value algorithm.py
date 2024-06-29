"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display

#defining all constants in CGS
e=4.8032e-10                        #electron charge
m=9.1094e-28                        #electron mass
c=2.99792458e10                     #speed of light
wavelength=4e-5                     #used probe wavelength
omega=2*np.pi*c/wavelength          #prob angular frequency
nc=omega**2*m/(4*np.pi*e**2)
e_eular=2.718281

    
def phase_unwrap(arr):
    arr=np.array(arr)
    delta=np.max(arr)-np.min(arr)
    n=int(2*np.pi/delta)
    arr=arr*n
    arr=np.unwrap(arr)
    arr/=n
    return(arr)
    
def final_stokes(Bx,By):
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])

    L=5e-12*6e6 #Plasma total length
    dz=L/100   #one slab length
    
    Bz=10
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    
    #n_steps=int((L-z)/dz)
    n_steps=int(L*np.log(100)/dz)
    z=0
    M=np.matrix(np.identity(3))
    I=np.matrix(np.identity(3))
    A=np.matrix(np.identity(3))
    
    

    for i in range(n_steps):
        
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        omega_c=e*B/(m*c)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1):
            mu1=0
        if np.isnan(mu2):
            mu2=0
            
        prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
        
        O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
        
        O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
        
        O3=prefactor*2*omega*e*Bz/(m*c)
        
        O=np.sqrt(O1**2+O2**2+O3**2)
        
        A[0,0]=0
        A[0,1]=-O3
        A[0,2]=O2
        
        A[1,0]=O3
        A[1,1]=0
        A[1,2]=-O1
        
        A[2,0]=-O2
        A[2,1]=O1
        A[2,2]=0
        
        if(O==0):
            M=(I  +  dz*A  +  1/2*dz**2*A**2)*M
        else:
            M=(I  +  np.sin(O*dz)/O*A  +  1/2*(np.sin(O*dz)/O)**2*A**2)*M
        
        z+=dz
    '''
    for i in range(n_steps):
        
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        omega_c=e*B/(m*c)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1):
            mu1=0
        if np.isnan(mu2):
            mu2=0
            
        prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
        
        O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
        
        O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
        
        O3=prefactor*2*omega*e*Bz/(m*c)
        
        O=np.sqrt(O1**2+O2**2+O3**2)
        
        A[0,0]=0
        A[0,1]=-O3
        A[0,2]=O2
        
        A[1,0]=O3
        A[1,1]=0
        A[1,2]=-O1
        
        A[2,0]=-O2
        A[2,1]=O1
        A[2,2]=0
        
        if(O==0):
            M=(I  +  dz*A  +  1/2*dz**2*A**2)*M
        else:
            M=(I  +  np.sin(O*dz)/O*A  +  1/2*(np.sin(O*dz)/O)**2*A**2)*M
        
        z-=dz
    '''
    sf=np.array(np.dot(M,s0))
    
    try:
        val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)
        if(np.isnan(val)):
            val=np.pi/2
    except:
        None
        
    return np.tan(val/2)

final_stokes=np.vectorize(final_stokes)

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
    
r,phi=np.mgrid[0.00001:50.00001:51j,0:2*np.pi:51j]
Bx=r*np.cos(phi)
By=r*np.sin(phi)
final_ellip_array=np.tan(final_stokes(Bx*10**6,By*10**6))

t=np.linspace(0,5,50)
def testfn(t):
    return(t*np.exp(-t**2))

ellip_over_t=testfn(t)
ellip_over_t=ellip_over_t/max(ellip_over_t)
plt.plot(ellip_over_t)
plt.show()

calculated_Bx=[]
calculated_By=[]

index=[0,0]
for i in range(len(t)):
    Bx_now,By_now,index=give_nearest_B(final_ellip_array, ellip_over_t[i], index)
    calculated_Bx.append(Bx_now)
    calculated_By.append(By_now)
    
plt.plot(calculated_Bx,calculated_By,'ro')
plt.plot(calculated_Bx,calculated_By,'k-')
plt.show()
