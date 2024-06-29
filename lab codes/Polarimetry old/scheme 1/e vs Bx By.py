# -*- coding: utf-8 -*-
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
    global t
    global omega
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])

    L=t*1e-12*6e6 #Plasma total length
    #dz=L/100   #one slab length
    dz=5e-8
    
    Bz=10
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    
    z=0
    #n_steps=int((L-z)/dz)
    n_steps=int(L*np.log(100)/dz)
    
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
    #Bz=Bz*(-1)
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
        
        #O3=prefactor*2*omega*e*Bz/(m*c)
        O3=prefactor*2*omega*e*(-1)*Bz/(m*c)
        
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
        val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
        if(np.isnan(val) or abs(val)>1.56979633):
            val=np.pi/2
    except:
        None
        
    return(val/2)

final_stokes=np.vectorize(final_stokes)

t=0.3
r,phi=np.mgrid[0.00001:101.00001:101j,0:2*np.pi:51j]
Bx=r*np.cos(phi)
By=r*np.sin(phi)
final_ellip_array=np.tan(final_stokes(Bx*10**6,By*10**6))

print(final_ellip_array)

#file_name = "D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\codes\\test1\\e0003_t_0_6.txt"

# Save the matrix to a text file
#np.savetxt(file_name, final_ellip_array, fmt='%f', delimiter='\t')

lensoffset=0
xx = yy = zz = np.arange(-200,200,1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=1)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=1)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=1)
mlab.mesh(Bx,By,50*final_ellip_array,representation='surface')
mlab.axes(extent=[-100, 100, -100, 100, -70, 70], color=(0, 0, 0), nb_labels=5)
#mlab.show()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()