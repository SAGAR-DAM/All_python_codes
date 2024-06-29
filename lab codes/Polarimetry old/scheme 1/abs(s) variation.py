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
    
def final_stocks(s0,Bx,By,Bz,L,dz):
    #n_steps=int((L-z)/dz)
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
    return (sf[0][0])**2+(sf[1][0])**2+(sf[2][0])**2

#final_stocks=np.vectorize(final_stocks)

sfinal=[]
for i in range(20):
    chi=0.0
    psi=0.0
    
    B=100e6
    Bx=B
    By=B
    Bz=B
    
    L=float(5e-12*6e6)
    dz=1e-8
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]
    
    s0=np.array([s0_1,s0_2,s0_3])
    
    sf=final_stocks(s0,Bx,By,Bz,L,dz)
    sfinal.append(sf)
    print(sf)
    chi+=np.pi/80
    
plt.plot(np.linspace(0,np.pi/4,len(sfinal)),sfinal)
plt.xlabel("chi")
plt.ylabel("|s|")
plt.show()

sfinal=[]
for i in range(21):
    chi=0.0
    psi=0.0
    
    B=100*10**(6*i/10)
    Bx=B
    By=B
    Bz=100
    
    L=float(5e-12*6e6)
    dz=1e-8
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]
    
    s0=np.array([s0_1,s0_2,s0_3])
    
    sf=final_stocks(s0,Bx,By,Bz,L,dz)
    sfinal.append(sf)
    print(sf)
    chi+=np.pi/80
    
plt.plot(2+np.linspace(0,12,len(sfinal)),sfinal)
plt.xlabel("order of magnitude of B")
plt.ylabel("|s|")
plt.show()

print(sf)

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()