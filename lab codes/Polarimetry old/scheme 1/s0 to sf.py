# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:08:06 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
def final_stokes(s0,Bx,By,Bz,L,dz):
    n_steps=int(L*np.log(100)/dz)
    z=0
    M=np.matrix(np.identity(3))
    I=np.matrix(np.identity(3))
    A=np.matrix(np.identity(3))
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    
    ne_arr=[]
    psi2_arr=[]
    chi2_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    
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
        if np.isnan(mu2) or mu2>1:
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

        sf=np.array(np.dot(M,s0))
        ne_arr.append(ne)
        try:
            res=np.arctan(sf[1][0]/sf[0][0])
            if(np.isnan(res)):
                psi2_arr.append(np.pi/2)
            else:
                psi2_arr.append(res)
        except:
            None
        try:
            res=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)
            if(np.isnan(res)):
                chi2_arr.append(np.pi/2)
            else:
                chi2_arr.append(res)
        except:
            None
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        
        z+=dz
        
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
        if np.isnan(mu2) or mu2>1:
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

        sf=np.array(np.dot(M,s0))
        ne_arr.append(ne)
        try:
            res=np.arctan(sf[1][0]/sf[0][0])
            if(np.isnan(res)):
                psi2_arr.append(np.pi/2)
            else:
                psi2_arr.append(res)
        except:
            None
        try:
            res=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)
            if(np.isnan(res)):
                chi2_arr.append(np.pi/2)
            else:
                chi2_arr.append(res)
        except:
            None
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        
        z-=dz
        
    sf=np.array(np.dot(M,s0))

    
    plt.plot(np.linspace(0,2*L*np.log(100),len(ne_arr))/(L*np.log(100)),np.array(ne_arr)/nc,'r-')
    plt.title("variation of ne \n"+"units in cgs")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel("ne")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    psi2_arr=phase_unwrap(psi2_arr)
    chi2_arr=phase_unwrap(chi2_arr)
    plt.plot(np.linspace(0,2*L*np.log(100),len(psi2_arr))/(L*np.log(100)),psi2_arr/2,'g-',label=r"$\psi$")
    plt.plot(np.linspace(0,2*L*np.log(100),len(chi2_arr))/(L*np.log(100)),chi2_arr/2,'r-',label=r"$\chi$")
    plt.title(r"variation of $\psi$ and $\chi$"+"\nunit in rad")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel("psi")
    plt.legend()
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    plt.plot(np.linspace(0,2*L*np.log(100),len(mu1_arr))/(L*np.log(100)),mu1_arr,'r-',label=r"$\mu_1$")
    plt.plot(np.linspace(0,2*L*np.log(100),len(mu2_arr))/(L*np.log(100)),mu2_arr,'b-',label=r"$\mu_2$")
    plt.legend()
    plt.title(r"variation of $\mu$")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel(r"$\mu$")
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    return(sf)

chi=np.pi/4
psi=0.0

Bx=100e6
By=100e6
Bz=100e6

L=float(5e-12*6e6)
dz=1e-8

s0_1=[np.cos(2*chi)*np.cos(2*psi)]
s0_2=[np.cos(2*chi)*np.sin(2*psi)]
s0_3=[np.sin(2*chi)]

s0=np.array([s0_1,s0_2,s0_3])

sf=final_stokes(s0,Bx,By,Bz,L,dz)
print("s0: \n",s0)
print("sf: \n",sf)
print("sf-s0: \n",sf-s0)

final_chi=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+sf[1][0])**2)/2
print(f"final chi: {final_chi}")

final_psi=np.arctan(sf[1][0]/sf[0][0])/2
print(f"final psi:  {final_psi}")

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()