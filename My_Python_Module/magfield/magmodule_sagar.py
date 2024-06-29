# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:01:53 2023

@author: sagar
"""

import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy import ndimage
import math
from skimage import io, draw

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display
#from numba import jit


##########################################################################
##########################################################################
##########################################################################
##########################################################################
""" important constants is CGS """


#defining all constants in CGS
e=4.8032e-10                        #electron charge
m=9.1094e-28                        #electron mass
c=2.99792458e10                     #speed of light
wavelength=4e-5                     #used probe wavelength
omega=2*np.pi*c/wavelength          #prob angular frequency
nc=omega**2*m/(4*np.pi*e**2)
e_eular=2.718281

##########################################################################
##########################################################################
##########################################################################
##########################################################################

def get_image_tl_and_br(image, background_factor):
    # Define the vertices of the top-left triangle
    triangle_vertices = np.array([[0, 0], [0, image.shape[1]-1], [image.shape[0]-1, 0]])
    
    # Create a binary mask for the triangle region
    mask = np.zeros_like(image, dtype=bool)
    rr, cc = draw.polygon(triangle_vertices[:, 0], triangle_vertices[:, 1])
    mask[rr, cc] = True
    
    # Apply the mask to the original image
    image_tl = np.zeros_like(image)
    image_tl[mask] = image[mask]
    # Replace NaN values with zero
    image_tl = np.nan_to_num(image_tl, nan=0)
    image_agv = background_factor*np.mean(image[0:400,600:1000])
    image_tl = image_tl*(image_tl>image_agv)
    
    # Define the vertices of the top-left triangle
    triangle_vertices = np.array([[image.shape[0]-1, image.shape[1]-1], [image.shape[0]-1, 0], [0, image.shape[1]-1]])
    
    # Create a binary mask for the triangle region
    mask = np.zeros_like(image, dtype=bool)
    rr, cc = draw.polygon(triangle_vertices[:, 0], triangle_vertices[:, 1])
    mask[rr, cc] = True
    
    # Apply the mask to the original image
    image_br = np.zeros_like(image)
    image_br[mask] = image[mask]
    # Replace NaN values with zero
    image_br = np.nan_to_num(image_br, nan=0)
    image_br = image_br*(image_br>image_agv)
    
    return image_tl, image_br
    

##########################################################################
##########################################################################
##########################################################################
##########################################################################

def get_nonzero_value_average(matrix):
    # Extract the nonzero elements
    nonzero_elements = matrix[matrix != 0.0]

    # Calculate the average of nonzero elements
    average_nonzero = np.mean(nonzero_elements)
    return(average_nonzero)

##########################################################################
##########################################################################
##########################################################################
##########################################################################

def stokes_generator(image_P,
                     image_C,
                     p_filter,
                     d_filter,
                     t_filter,
                     q_filter,
                     d_image_pump_noise,
                     p_image_pump_noise,
                     t_image_pump_noise,
                     q_image_pump_noise,
                     tol):
    
    
    def subtract_pump_noise(matrix, noise):
        noise = np.uint16(noise)
        # Convert the matrix to a NumPy array with the specified data type
        matrix_array = np.array(matrix)
        
        # Replace negative values with zero
        matrix_array[matrix_array > noise] -= noise
        
        return matrix_array.tolist()  # Convert the NumPy array back to a list
    
    def get_stokes(image_P,image_C, background_factor):
        D,P = get_image_tl_and_br(image_P, background_factor)
        
        D=subtract_pump_noise(D, d_image_pump_noise)
        P=subtract_pump_noise(P, p_image_pump_noise)
        
        
        Q,T = get_image_tl_and_br(image_C, background_factor)
        
        
        Q=subtract_pump_noise(Q, q_image_pump_noise)
        T=subtract_pump_noise(T, t_image_pump_noise)
        
        P_int = np.sum(P)*p_filter
        D_int = np.sum(D)*d_filter
        Q_int = np.sum(Q)*q_filter
        T_int = np.sum(T)*t_filter
        
        
        s1 = 2*P_int/D_int-1
        s2 = 2*T_int/D_int-1
        s3 = 1-2*Q_int/D_int
        
        s = np.matrix([[s1],[s2],[s3]])
        mod_s = np.sqrt(np.dot(s.T,s)[0,0])
        
        return P_int, D_int, Q_int, T_int, s, mod_s
    
    def get_mod_s(image_P,image_C, background_factor):
        D,P = get_image_tl_and_br(image_P, background_factor)
        
        D=subtract_pump_noise(D, d_image_pump_noise)
        P=subtract_pump_noise(P, p_image_pump_noise)
        
        
        Q,T = get_image_tl_and_br(image_C, background_factor)

        
        Q=subtract_pump_noise(Q, q_image_pump_noise)
        T=subtract_pump_noise(T, t_image_pump_noise)
    
        
        P_int = np.sum(P)*p_filter
        D_int = np.sum(D)*d_filter
        Q_int = np.sum(Q)*q_filter
        T_int = np.sum(T)*t_filter
        
        
        s1 = 2*P_int/D_int-1
        s2 = 2*T_int/D_int-1
        s3 = 1-2*Q_int/D_int
        
        s = np.matrix([[s1],[s2],[s3]])
        mod_s = np.sqrt(np.dot(s.T,s)[0,0])
        
        return(mod_s-1)
    
    def bisection(f, a, b, image_P, image_C, tol, max_iter=1000):
        # """
        # Bisection method to find the root of the function f(x) within the interval [a, b].
    
        # Parameters:
        # - f: The function for which to find the root.
        # - a: The lower limit of the interval.
        # - b: The upper limit of the interval.
        # - tol: Tolerance for the error. Default is 1e-6.
        # - max_iter: Maximum number of iterations. Default is 1000.
    
        # Returns:
        # - root: Approximate root of the function within the specified tolerance.
        # - iterations: Number of iterations performed.
        # """
    
        if f(image_P, image_C, a) * f(image_P, image_C, b) > 0:
            raise ValueError("The function values at the interval endpoints must have different signs.")
    
        root = None
        iterations = 0
    
        while iterations < max_iter:
            c = (a + b) / 2
            if abs(f(image_P, image_C, a)) <= tol:
                root = a
                break
            elif abs(f(image_P, image_C, b)) <= tol:
                root = b
                break
            elif abs(f(image_P, image_C, c)) <= tol:
                root = c
                break
            elif f(image_P, image_C, c) * f(image_P, image_C, a) < 0:
                b = c
            else:
                a = c
    
            iterations += 1
    
        #root = (a + b) / 2
        return root
    
    def find_interval_with_sign_change(f, lower, upper, image_P, image_C, step=0.1):
        # """
        # Find an interval [a, b] such that f(a) * f(b) < 0.
    
        # Parameters:
        # - f: The function for which to find the interval.
        # - lower: Lower limit of the range.
        # - upper: Upper limit of the range.
        # - step: Step size for iterating over the range. Default is 0.1.
    
        # Returns:
        # - (a, b): Interval [a, b] where f(a) * f(b) < 0.
        # """
    
        a = None
        b = None
    
        x = lower
        while x < upper:
            if f(image_P, image_C, x) * f(image_P, image_C, x + step) < 0:
                a = x
                b = x + step
                break
            x += step
    
        return a, b
    
    def make_s_normalized(image_P, image_C, tol):
        try:
            background_factor = bisection(f = get_mod_s, a=1, b=1.2, image_P=image_P, image_C=image_C, tol=tol)
        except:
            try:
                a, b = find_interval_with_sign_change(f = get_mod_s, lower = 1, upper = 2, image_P= image_P, image_C=image_C)
                background_factor = bisection(f = get_mod_s, a=a, b=b, image_P=image_P, image_C=image_C, tol=tol)
            except:
                background_factor = None
                
        return(background_factor)
    
    
    background_factor = make_s_normalized(image_P, image_C, tol=tol)
    P_int, D_int, Q_int, T_int, s, mod_s = get_stokes(image_P, image_C, background_factor)
    
    
    return P_int, D_int, Q_int, T_int, s, mod_s, background_factor


##########################################################################
##########################################################################
##########################################################################
##########################################################################


def phase_unwrap(arr):
    arr=np.array(arr)
    delta=np.max(arr)-np.min(arr)
    try:
        n=int(2*np.pi/delta)
    except:
        n=1
    arr=arr*n
    arr=np.unwrap(arr)
    arr/=n
    return(arr)


##########################################################################
##########################################################################
##########################################################################
##########################################################################


def final_stokes(s0,Bx,By,Bz,L,dz):
    """

    Parameters
    ----------
    s0 : initial stokes vector
         a column matrix (3,1)
         
    Bx : Magnetic field in X direction
        float
        
    By : same as Bx
        float
        
    Bz : Same as Bx
        float.
        
    L : Length of plasma slab in CGS
        float.
        
    dz : iteration step size
        float

    Returns
    -------
    Final stokes vector sf. same as dimensio s0

    """
    
    """ important constants is CGS """

    ##########################################################################
    #defining all constants in CGS
    e=4.8032e-10                        #electron charge
    m=9.1094e-28                        #electron mass
    c=2.99792458e10                     #speed of light
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)
    e_eular=2.718281
    ##########################################################################
    
    z=0
    #n_steps=int((L-z)/dz)
    n_steps=int(L*np.log(100)/dz)
    M=np.matrix(np.identity(3))
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    
    ne_arr=[]
    psi2_arr=[]
    chi2_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    temp=0
    
    for i in range(n_steps):
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        # omega_c=e*B/(m*c)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
        
        if (mu2>0):
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
            
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            M=np.dot(R1,np.dot(R2,M))
            
            
            s=np.array(np.dot(M,s0))
            s=s.reshape((3,1))
            #print(s)
            try:
                res=np.arctan(s[1][0]/s[0][0])
                if(np.isnan(res)):
                    psi2_arr.append(np.pi/2)
                else:
                    psi2_arr.append(res)
            except:
                None
            try:
                res=np.arctan(s[2][0]/np.sqrt((s[0][0])**2+s[1][0])**2)
                if(np.isnan(res)):
                    chi2_arr.append(np.pi/2)
                else:
                    chi2_arr.append(res)
            except:
                None
            
            
        elif(mu2==0):
            if(temp==0):
                s1=np.dot(M,s0)
                A1 = s1[1]/s1[0]      #tan2psi
                B1 = s1[2]           #sin2chi
                
                s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
                t2eta = s2eta/np.sqrt(1-s2eta**2)
                phi = np.arcsin(B1/s2eta)
                if(np.isnan(phi)):
                    phi = 0
                temp=1
                
            phi = -(mu1*dz*omega/c)+phi
            
            psiN=np.arctan(t2eta*np.cos(phi))            
            #print(psiN)
            chiN=np.arcsin(s2eta*np.sin(phi))
                                                                         
            psi2_arr.append(psiN[0,0])
            chi2_arr.append(chiN[0,0])
            
        z+=dz
        
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        ne_arr.append(ne)
        
    s2_1=[np.cos(chi2_arr[-1])*np.cos(psi2_arr[-1])]
    s2_2=[np.cos(chi2_arr[-1])*np.sin(psi2_arr[-1])]
    s2_3=[np.sin(chi2_arr[-1])]

    s2=np.array([s2_1,s2_2,s2_3])
    s2=s2.reshape((3,1))

    M=np.matrix(np.identity(3))
    temp=0
    
    for i in range(n_steps):
        #ne=(np.exp(z/L)-1)*nc/(e_eular-1)
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        #omega_c=e*B/(m*c)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
        
        if (mu2>0):
            if(temp==0):
                s3_1=np.cos(chi2_arr[-1])*np.cos(psi2_arr[-1])
                s3_2=np.cos(chi2_arr[-1])*np.sin(psi2_arr[-1])
                s3_3=np.sin(chi2_arr[-1])
                
                s3=np.array([s3_1,s3_2,s3_3])
                s3=s3.reshape((3,1))
            
                temp=1
                
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
        
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            M=np.dot(R1,np.dot(R2,M))
            
            s=np.dot(M,s3)
            s=s.reshape((3,1))
            #print(s)
            try:
                res=np.arctan(s[1][0]/s[0][0])
                if(np.isnan(res)):
                    psi2_arr.append(np.pi/2)
                else:
                    psi2_arr.append(res[0,0])

            except:
                None
            try:
                res=np.arctan(s[2][0]/np.sqrt((s[0][0])**2+s[1][0])**2)
                if(np.isnan(res)):
                    chi2_arr.append(np.pi/2)
                else:
                    chi2_arr.append(res[0,0])
            except:
                None

        
        elif(mu2==0):
            try:
                phi = -(mu1*dz*omega/c)+phi
            except:
                s1=np.dot(M,s0)
                A1 = s1[1]/s1[0]      #tan2psi
                B1 = s1[2]           #sin2chi
                
                s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
                t2eta = (s2eta/np.sqrt(1-s2eta**2))
                phi = np.arcsin(B1/s2eta)
                if (np.isnan(phi)):
                    phi=0
                
            psiN=np.arctan(t2eta*np.cos(phi))            
            chiN=np.arcsin(s2eta*np.sin(phi))
            
            psi2_arr.append(psiN[0,0])
            chi2_arr.append(chiN[0,0])
            
        z-=dz
        
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        ne_arr.append(ne)
        
    
        
    sf=np.array(np.dot(M,s3))
    
    plt.plot(np.linspace(0,2*L*np.log(100),len(ne_arr))/(L*np.log(100)),np.array(ne_arr)/nc,'r-')
    plt.title(r"variation of $n_e/n_c$"+"\n"+"units in cgs")
    plt.xlabel("z/(L ln 100)")
    plt.ylabel(r"$n_e/n_c$")
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



##########################################################################
##########################################################################
##########################################################################
##########################################################################


@np.vectorize
def Bx_By_to_ellip_farad(Bx, By, **kwargs):
    
    ##########################################################################
    #defining all constants in CGS
    e=4.8032e-10                        #electron charge
    m=9.1094e-28                        #electron mass
    c=2.99792458e10                     #speed of light
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)
    e_eular=2.718281
    ##########################################################################
    
    L=kwargs.get("L", 1e-12*6e6) #Plasma total length
    Bz=kwargs.get("Bz",10)
    
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])

    dz=1e-8
    
    z=0
    n_steps=int(L*np.log(100)/dz)
    M1=np.matrix(np.identity(3))
    M2=np.matrix(np.identity(3))
    
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    phi = 0
    

    # ne_arr=[]
    # psi2_arr=[]
    # chi2_arr=[]
    # mu1_arr=[]
    # mu2_arr=[]
    
    for i in range(n_steps):
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
            
        if(mu2 != 0):

            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            # O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
            
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            
            R = np.dot(R1,R2)
            
            M1 = np.dot(R,M1)
            M2 = np.dot(M2,R)
            
        elif (mu2==0.0):
            phi += (mu1*dz*omega/c)
            
        z += dz
        
    s1 = np.dot(M1,s0)
    s1 = s1.reshape((3,1))
    
    A1 = s1[1]/s1[0]      #tan2psi
    B1 = s1[2]           #sin2chi
    
    s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
    t2eta = s2eta/np.sqrt(1-s2eta**2)
    
    eta1 = np.arcsin(np.sqrt((A1**2+B1**2)/(1+A1**2)))/2
    phi1 = np.arcsin(B1/s2eta)
    
    if(np.isnan(phi1)):
        phi1=0
    
    eta2 = eta1
    phi2 = phi1-phi
    
    eta3 = eta2
    phi3 = phi2-phi
    
    psi3 = np.arctan(np.tan(2*eta3)*np.cos(phi3))/2
    chi3 = np.arctan(np.sin(2*eta3)*np.sin(phi3))/2
    
    s3_1=[np.cos(2*chi3)*np.cos(2*psi3)]
    s3_2=[np.cos(2*chi3)*np.sin(2*psi3)]
    s3_3=[np.sin(2*chi3)]

    s3=np.array([s3_1,s3_2,s3_3])
    s3=s3.reshape((3,1))
    
    sf = np.dot(M2,s3)
    sf=sf.reshape((3,1))
    
    try:
        chi2_val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
        if(np.isnan(chi2_val) or abs(chi2_val)>1.56979633):
            chi2_val=np.pi/2
    except:
        None
        
    try:
        psi2_val=np.arctan(sf[1][0]/sf[0][0])
        if(np.isnan(psi2_val) or abs(psi2_val)>1.56979633):
            psi2_val=np.pi/2
    except:
        None
        
    chi_val = chi2_val/2
    psi_val = psi2_val/2

    return chi_val, psi_val



##########################################################################
##########################################################################
##########################################################################
##########################################################################



@np.vectorize
def BT_BA_to_ellip_farad(B_Transverse, B_Axial, **kwargs):
    
    if(B_Transverse<0):
        raise ValueError("B_Transverse should be positive ...")
        return None
    
    ##########################################################################
    #defining all constants in CGS
    e=4.8032e-10                        #electron charge
    m=9.1094e-28                        #electron mass
    c=2.99792458e10                     #speed of light
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)
    e_eular=2.718281
    ##########################################################################
    
    L=kwargs.get("L", 1e-12*6e6) #Plasma total length
    
    Bx = B_Transverse/1.41421356237     #Bx = B_Transverse/sqrt(2)   
    By = B_Transverse/1.41421356237
    Bz = B_Axial
    
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])

    dz=1e-8
    
    z=0
    n_steps=int(L*np.log(100)/dz)
    M1=np.matrix(np.identity(3))
    M2=np.matrix(np.identity(3))
    
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    phi = 0
    

    # ne_arr=[]
    # psi2_arr=[]
    # chi2_arr=[]
    # mu1_arr=[]
    # mu2_arr=[]
    
    for i in range(n_steps):
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
            
        if(mu2 != 0):

            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            # O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
            
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            
            R = np.dot(R1,R2)
            
            M1 = np.dot(R,M1)
            M2 = np.dot(M2,R)
            
        elif (mu2==0.0):
            phi += (mu1*dz*omega/c)
            
        z += dz
        
    s1 = np.dot(M1,s0)
    s1 = s1.reshape((3,1))
    
    A1 = s1[1]/s1[0]      #tan2psi
    B1 = s1[2]           #sin2chi
    
    s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
    t2eta = s2eta/np.sqrt(1-s2eta**2)
    
    eta1 = np.arcsin(np.sqrt((A1**2+B1**2)/(1+A1**2)))/2
    phi1 = np.arcsin(B1/s2eta)
    
    if(np.isnan(phi1)):
        phi1=0
    
    eta2 = eta1
    phi2 = phi1-phi
    
    eta3 = eta2
    phi3 = phi2-phi
    
    psi3 = np.arctan(np.tan(2*eta3)*np.cos(phi3))/2
    chi3 = np.arctan(np.sin(2*eta3)*np.sin(phi3))/2
    
    s3_1=[np.cos(2*chi3)*np.cos(2*psi3)]
    s3_2=[np.cos(2*chi3)*np.sin(2*psi3)]
    s3_3=[np.sin(2*chi3)]

    s3=np.array([s3_1,s3_2,s3_3])
    s3=s3.reshape((3,1))
    
    sf = np.dot(M2,s3)
    sf=sf.reshape((3,1))
    
    try:
        chi2_val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
        if(np.isnan(chi2_val) or abs(chi2_val)>1.56979633):
            chi2_val=np.pi/2
    except:
        None
        
    try:
        psi2_val=np.arctan(sf[1][0]/sf[0][0])
        if(np.isnan(psi2_val) or abs(psi2_val)>1.56979633):
            psi2_val=np.pi/2
    except:
        None
        
    chi_val = chi2_val/2
    psi_val = psi2_val/2

    return chi_val, psi_val




####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################



@np.vectorize
def Bx_By_to_ellip_farad_adaptive_step(Bx, By, **kwargs):
    
    ##########################################################################
    #defining all constants in CGS
    e=4.8032e-10                        #electron charge
    m=9.1094e-28                        #electron mass
    c=2.99792458e10                     #speed of light
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)
    e_eular=2.718281
    
    L=kwargs.get("L", 1e-12*6e6) #Plasma total length
    Bz=kwargs.get("Bz",10)
    dz_min=kwargs.get("dz_min", 1e-8)
    ##########################################################################
    
    ##########################################################################
    """ Reequired local functions """
    
    def z_to_n(z):
        n = nc*np.exp(z/L-np.log(100))
        return(n)

    def n_to_z(n):
        z=L*np.log(100*n/nc)
        return(z)

    dn0 = z_to_n(z=L*np.log(100))-z_to_n(z=(L*np.log(100)-dz_min))
    n_arr = np.arange(nc/100, nc, dn0)
    z_arr = n_to_z(n_arr)
    dz_arr= np.diff(z_arr)
    
    ##########################################################################
    
    chi=0.0     #initial chi
    psi=0.0     #initial psi
    
    s0_1=[np.cos(2*chi)*np.cos(2*psi)]
    s0_2=[np.cos(2*chi)*np.sin(2*psi)]
    s0_3=[np.sin(2*chi)]

    s0=np.array([s0_1,s0_2,s0_3])
    
    z=0
    n_steps=len(dz_arr)
    M1=np.matrix(np.identity(3))
    M2=np.matrix(np.identity(3))
    
    R1=np.matrix(np.identity(3))
    R2=np.matrix(np.identity(3))
    
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)
    phi = 0

    
    for i in range(n_steps):
        dz = dz_arr[i]
        
        ne=nc*np.exp(z/L-np.log(100))
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)
        
        F=2*omega/omega_c*(1-omega_p**2/omega**2)*np.cos(theta)/(np.sin(theta))**2
        mu1_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1+np.sqrt(1+F**2)))
        mu2_sq=1-omega_p**2/omega**2*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-omega_p**2/omega**2))*(-1-np.sqrt(1+F**2)))
        
        mu1=np.sqrt(mu1_sq)
        mu2=np.sqrt(mu2_sq)
        
        N=(omega_p/omega)**2
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        if np.isnan(mu1) or mu1>=1:
            mu1=0
        if np.isnan(mu2) or mu2>=1:
            mu2=0
            
        if(mu2 != 0):

            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-omega_p**2/omega**2))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-omega_p**2/omega**2))
            
            O3=prefactor*2*omega*e*Bz/(m*c)
            
            # O=np.sqrt(O1**2+O2**2+O3**2)
            
            w1=O1*dz/2
            w2=O2*dz/2
            w3=O3*dz/2
            
            
            sw1=np.sin(w1)
            sw2=np.sin(w2)
            sw3=np.sin(w3)
            
            cw1=np.cos(w1)
            cw2=np.cos(w2)
            cw3=np.cos(w3)
            
            R1[0,0]=cw3*cw2
            R1[0,1]=cw3*sw2*sw1-sw3*cw1
            R1[0,2]=sw3*sw1+cw3*sw2*cw1
            
            R1[1,0]=sw3*cw2
            R1[1,1]=cw3*cw1+sw3*sw2*sw1
            R1[1,2]=sw3*sw2*cw1-cw3*sw1
            
            R1[2,0]=-sw2
            R1[2,1]=cw2*sw1
            R1[2,2]=cw2*cw1
            
            
            
            R2[0,0]=cw3*cw2
            R2[1,0]=cw3*sw2*sw1+sw3*cw1
            R2[2,0]=sw3*sw1-cw3*sw2*cw1
            
            R2[0,1]=-sw3*cw2
            R2[1,1]=cw3*cw1-sw3*sw2*sw1
            R2[2,1]=sw3*sw2*cw1+cw3*sw1
            
            R2[0,2]=sw2
            R2[1,2]=-cw2*sw1
            R2[2,2]=cw2*cw1
            
            
            R1=np.matrix(R1)
            R2=np.matrix(R2)
            
            R = np.dot(R1,R2)
            
            M1 = np.dot(R,M1)
            M2 = np.dot(M2,R)
            
        elif (mu2==0.0):
            phi += (mu1*dz*omega/c)
            
        z += dz
        
    s1 = np.dot(M1,s0)
    s1 = s1.reshape((3,1))
    
    A1 = s1[1]/s1[0]      #tan2psi
    B1 = s1[2]           #sin2chi
    
    s2eta = np.sqrt((A1**2+B1**2)/(1+A1**2))
    t2eta = s2eta/np.sqrt(1-s2eta**2)
    
    eta1 = np.arcsin(np.sqrt((A1**2+B1**2)/(1+A1**2)))/2
    phi1 = np.arcsin(B1/s2eta)
    
    if(np.isnan(phi1)):
        phi1=0
    
    eta2 = eta1
    phi2 = phi1-phi
    
    eta3 = eta2
    phi3 = phi2-phi
    
    psi3 = np.arctan(np.tan(2*eta3)*np.cos(phi3))/2
    chi3 = np.arctan(np.sin(2*eta3)*np.sin(phi3))/2
    
    s3_1=[np.cos(2*chi3)*np.cos(2*psi3)]
    s3_2=[np.cos(2*chi3)*np.sin(2*psi3)]
    s3_3=[np.sin(2*chi3)]

    s3=np.array([s3_1,s3_2,s3_3])
    s3=s3.reshape((3,1))
    
    sf = np.dot(M2,s3)
    sf=sf.reshape((3,1))
    
    try:
        chi2_val=np.arctan(sf[2][0]/np.sqrt((sf[0][0])**2+(sf[1][0])**2))
        if(np.isnan(chi2_val) or abs(chi2_val)>1.56979633):
            chi2_val=np.pi/2
    except:
        None
        
    try:
        psi2_val=np.arctan(sf[1][0]/sf[0][0])
        if(np.isnan(psi2_val) or abs(psi2_val)>1.56979633):
            psi2_val=np.pi/2
    except:
        None
        
    chi_val = chi2_val/2
    psi_val = psi2_val/2

    return chi_val, psi_val