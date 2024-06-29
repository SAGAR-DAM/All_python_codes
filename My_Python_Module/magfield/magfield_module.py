# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:03:18 2023

@author: Anandam
"""
import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy import ndimage
import math
from skimage import io, draw
#from numba import jit


def zoom_custom(image, zoom_factor):
    row, col = image.shape
    
    zoomed_image = zoom(image, zoom_factor)
    nrow, ncol = zoomed_image.shape

    # Padding or Cropping
    if nrow > row:
        crop_row = nrow - row
        crop_col = ncol - col
        if crop_row%2 == 1:
            crop_row-=1
            crop_col-=1
        final_image = zoomed_image[crop_row//2:(crop_row-crop_row//2)+row,crop_col//2:(crop_col-crop_col//2)+col]
    elif row > nrow:
        pad_row = row - nrow
        pad_col = col - ncol
        final_image = np.pad(zoomed_image,[(pad_row//2,(pad_row-pad_row//2)),(pad_col//2,(pad_col-pad_col//2))],mode='constant')
    else:
        final_image = zoomed_image
    
    return final_image

def maximize_correlation(image1, image2,scan_lim):
    max_corr = 0
    best_shift = (0, 0)
    best_angle = 0
    best_scale = 1
    
    Norm1 = (image1 - np.mean(image1)) / np.std(image1)

    #Iterate over all possible shifts
    for dx in range(-1*scan_lim,scan_lim):
        for dy in range(-1*scan_lim,scan_lim):
            shift_image2 = np.roll(np.roll(image2, dx, axis=1), dy, axis=0)
            Norm2 = (shift_image2 - np.mean(shift_image2)) / np.std(shift_image2)
            correlation= np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
            print(correlation,dx,dy,'\n')
            if correlation > max_corr:
                max_corr = correlation
                best_shift = (dx, dy)
    #max_corr1 = max_corr
    #print(max_corr,best_shift)
    shift_image2 = np.roll(np.roll(image2, best_shift[0], axis=1), best_shift[1], axis=0)
    
    #Iterate over all possible rotations
    for angle in range(-5, 6):
        rot_image2 = rotate(shift_image2, angle, reshape=False)
        for scale in np.linspace(0.9, 1.1, 21):
            scale_image2 = zoom_custom(rot_image2, scale)  
            Norm2 = (scale_image2 - np.mean(scale_image2)) / np.std(scale_image2)
            correlation= np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
            print(correlation,best_shift,angle,scale,'\n')

            if correlation > max_corr:
                max_corr = correlation
                best_angle = angle
                best_scale = scale
                
    #print(max_corr,best_shift, best_angle, best_scale,max_corr1)
    corr_image2 = zoom_custom(rotate(shift_image2, best_angle, reshape=False),best_scale)             
    #return max_corr, best_shift, best_angle, best_scale
    return(corr_image2,max_corr, best_shift, best_angle, best_scale)
    #return shift_image2
    
def corr_shft(image, dx, dy, angle, scale, image2):
    
    corr_image = zoom_custom(rotate(np.roll(np.roll(image, dx, axis=1), dy, axis=0), angle, reshape=False),scale) 
    
    Norm1 = (image2 - np.mean(image2)) / np.std(image2)
    Norm2 = (corr_image - np.mean(corr_image)) / np.std(corr_image)
        
    corr = np.corrcoef(Norm1.flatten(), Norm2.flatten())[0,1]
    print(corr)
    
    return corr_image


def find_bestcut(start, end, image):   # cut two different images with best central line 
    
    l=350
    count=0
    theta_range = np.linspace(0,90,31)
    
    for i in range(start,end,1):
        for theta in theta_range:
            l_start=[round(i+l*np.sin(np.radians(theta))),round(i-l*np.cos(np.radians(theta)))]
            l_end=[round(i-l*np.sin(np.radians(theta))),round(i+l*np.cos(np.radians(theta)))]
            
            count+=1
            mask = np.zeros_like(image)
            cv2.line(mask, l_start, l_end, 30000, 1)
            line_pixels = image[mask == 30000]
            total_counts = np.sum(line_pixels)
            
            if count == 1:
                min_counts = total_counts
                b_s = l_start
                b_e = l_end
            
            if total_counts < min_counts:
                min_counts = total_counts
                b_s = l_start
                b_e = l_end
                
            #print(total_counts)
    #print(min_counts,b_s,b_e)
    print('#')
    return(b_s,b_e)

def edit_image(image,start_pos,end_pos,fac_up,fac_dn,noise,min_count):
    
    slope = round((end_pos[1]-start_pos[1])/(end_pos[0]-start_pos[0]))
    X=image.shape[0]
    Y=image.shape[1]
    
    for i in range(X):
        for j in range(Y):
            #print('\n',image[i][j])
            if image[i][j] < min_count:
                x=0
            else:
                delta = (j-start_pos[1])-(i-start_pos[0])*slope
                if delta <= 0:
                    image[i][j] = image[i][j] - noise
                    image[i][j] = image[i][j] * fac_up
                    image[i][j] = image[i][j] + noise
                elif delta > 0:
                    image[i][j] = image[i][j] - noise
                    image[i][j] = image[i][j] * fac_dn
                    image[i][j] = image[i][j] + noise
            #print(' ',image[i][j])
            
    return(image)


def separate_image(start_pos, end_pos, image):
    
    slope = round((end_pos[1]-start_pos[1])/(end_pos[0]-start_pos[0]))
    X=image.shape[0]
    Y=image.shape[1]
    base_noise=np.mean(image[4*X//5:,:Y//5])
    Top = np.asarray([[base_noise]*Y]*X)
    Down = np.asarray([[base_noise]*Y]*X)
    
    for i in range(X):
        for j in range(Y):
            delta = (j-start_pos[1])-(i-start_pos[0])*slope
            if delta >= 0:
                Down[i,j] = image[i,j]
            else:
                Top[i,j] = image[i,j]
                
    for i in range(X):
        for j in range(Y):
            if Top[i,j] < 1050:
                Top[i,j]=0
            if Down[i,j] < 1050:
                Down[i,j]=0
    cond = 1            
    for i in range(X):
        for j in range(Y):
            if Top[i,j] > 0:
                U = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1    
    for i in range(X):
        for j in range(Y):
            if Top[1023-i,j] > 0:
                D = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1
    for i in range(X):
        for j in range(Y):
            if Top[j,i] > 0:
                L = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1    
    for i in range(X):
        for j in range(Y):
            if Top[j,1023-i] > 0:
                R = i
                cond = 0
                break
        if cond == 0:
            break
        
    Top = np.roll(np.roll(Top,(D-U)//2 ,axis=0),(R-L)//2 ,axis=1)
    
    cond = 1            
    for i in range(X):
        for j in range(Y):
            if Down[i,j] > 0:
                U = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1    
    for i in range(X):
        for j in range(Y):
            if Down[1023-i,j] > 0:
                D = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1
    for i in range(X):
        for j in range(Y):
            if Down[j,i] > 0:
                L = i
                cond = 0
                break
        if cond == 0:
            break
    cond = 1    
    for i in range(X):
        for j in range(Y):
            if Down[j,1023-i] > 0:
                R = i
                cond = 0
                break
        if cond == 0:
            break
                
  
    Down = np.roll(np.roll(Down,(D-U)//2 ,axis=0),(R-L)//2 ,axis=1)
                
    return(Top, Down)

def mean_image(image,noise,min_signal):
    x = image.shape[0]
    y = image.shape[1]
    count = 0
    Sum= 0
        
    for i in range(x):
        for j in range(y):
            if image[i][j] > min_signal:
                Sum+=image[i][j]-noise
                count+=1
    
    mean = Sum//count
            
    return (Sum,mean)       

def modify(image,noise,min_signal):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > min_signal:
                image[i][j] = image[i][j]-noise
            else:
                image[i][j] = 0
    return image
    
    
def stokes_calculation(I,I0,I45,Iq45):
    s1 = 2*I0-I
    s2 = 2*I45-I
    s3 = I-2*Iq45
    
    return s1, s2, s3

def polarimetry(P_image,D_image,T_image,Q_image):
    if P_image.shape == D_image.shape == T_image.shape == Q_image.shape:
        x = P_image.shape[0]
        y = P_image.shape[1]
    
        ellip = np.asarray([[0.]*y]*x)
        farad_rot = np.asarray([[0.]*y]*x)
    
        # for i in range(x):
        #     for j in range(y):
        #         if P_image[i][j] != 0 and D_image[i][j] != 0 and T_image[i][j] != 0 and Q_image[i][j] != 0:
        #             s1 = (2*P_image[i][j]/D_image[i][j])-1.0
        #             s2 = (2*T_image[i][j]/D_image[i][j])-1.0
        #             s3 = 1.0-(2*Q_image[i][j]/D_image[i][j])
                
        #             if s1 <= 1:
        #                 ellip[i][j] = 0.5*np.arctan(s3/(s1**2+s2**2)**0.5)
        #                 farad_rot[i][j] = 0.5*np.arctan(s2/s1)
    
        s1 = (2*np.sum(P_image)/np.sum(D_image))-1.0
        s2 = (2*np.sum(T_image)/np.sum(D_image))-1.0
        s3 = 1.0-(2*np.sum(Q_image)/np.sum(D_image))
        
        print(mean_image(P_image,0.,0.)[0],mean_image(D_image,0.,0.)[0],mean_image(T_image,0.,0.)[0],mean_image(Q_image,0.,0.)[0])
        print(s1,s2,s3)
    
        e = 0.5*np.arctan(s3/(s1**2+s2**2)**0.5)
        f = 0.5*np.arctan(s2/s1)
        
        print('\nellipticity - ',e,'\nfaraday rotation - ',f)
        
        return(ellip, farad_rot, e, f)
    
    else:
        raise ValueError("Images size not matching")
    
def svector(Bx,By,Bz,L,dz):
    I=np.identity(3)    #Identity matrix
    M=np.identity(3)    #Declaring Plasma transition matrix
    A=np.zeros([3,3])   #Declaring A matrix
    O=np.zeros(3)       #Declaring omega vector
    s=np.array([1,0,0]) #Input stokes vector
    

    e=4.803e-10     #in esu
    w=4.712e15      #probe frequency 400nm wavelength
    m=9.1e-28       #electron mass
    c=3e10          #speed of light
    
    n=int(4*L/dz)     #number of slabs

    wc2=(Bx*Bx+By*By+Bz*Bz)*(e/(m*c))**2        #cyclotron freqency
    sin2=(Bx*Bx+By*By)/(Bx*Bx+By*By+Bz*Bz)      #sin^2 and cos of angle  
    cos=Bz/(Bx*Bx+By*By+Bz*Bz)**0.5             #angle b/w B and propagation direction

    for i in range(n):
        z=(i+1)*dz          #only one side integration
        N=np.exp(-z/L)
        F=(1-N)*(2*w*cos)/(sin2*wc2**0.5)
        D=1-((e/(w*m*c))**2)*(((Bx*Bx+By*By)/(1-N))+Bz*Bz)
        mu1=(1-N/(1+(wc2/(w*w))*(sin2/(2*(1-N)))*(-1+(1+F*F)**0.5)))**0.5
        mu2=(1-N/(1+(wc2/(w*w))*(sin2/(2*(1-N)))*(-1-(1+F*F)**0.5)))**0.5
        if np.isnan(mu2):
            mu2=0
        #print(N," ",mu1," ",mu2," ",i)

        O[0]=(N/((mu1+mu2)*c*w*D))*((Bx*Bx-By*By)/(1-N))*(e/(m*c))**2
        O[1]=(N/((mu1+mu2)*c*w*D))*((2*Bx*By)/(1-N))*(e/(m*c))**2
        O[2]=(N/((mu1+mu2)*c*w*D))*(2*w*Bz*(e/(m*c)))
        
        omega=np.dot(O,O)
        omega=omega**0.5

        A[0][1]=-O[2]
        A[0][2]=O[1]
        A[1][0]=O[2]
        A[1][2]=-O[0]
        A[2][0]=-O[1]
        A[2][1]=O[0]
    
        M=np.dot((I+(np.sin(omega*dz)/omega)*A+(2*(np.sin(omega*dz/2)/omega)**2)*np.dot(A,A)),M)
        #print(M,'\n')
        
    s=np.dot(M,s)
    print('N ',N,'M ',M)
    
    ellip = 0.5*np.arctan(s[2]/(s[0]**2+s[1]**2)**0.5)
    farad = 0.5*np.arctan(s[1]/s[0])
    
    return(s, ellip, farad)    

def final_stokes(s0,Bx,By,Bz,t):
    
    #defining all constants in CGS
    
    e=4.8032e-10                        #electron charge in esu (cgs)
    m=9.1094e-28                        #electron mass in g
    c=2.99792458e10                     #speed of light cm/sec^2
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)        #plasma critical frequency     
    z=0                                 #probe entry from vacuum
    cs=6e6                              #plasma self similar expansion velocity
    dz=1e-8                             #plasma slab element length
    L = cs*t 
    S0 = s0                           #plasma slab length
    B=np.sqrt(Bx**2+By**2+Bz**2) 
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)                   #cyclotron freq
           
    n_steps=int((L*np.log(100)/dz))
    M=np.identity(3)
    R1=np.identity(3)
    R2=np.identity(3)
    
    phi = 0.0
    phi0 = 0.0
    psiN=0.0
    chiN=0.0
    sf=S0
    
    ne_arr=[]
    psi_arr=[]
    chi_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    modS=[]
    
    for i in range(2*n_steps):
        
        N=np.exp(-(L*np.log(100)-z)/L)    
        ne=np.exp(-(L*np.log(100)-z)/L)*nc    #exponential density gradient
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)          #plasma frequency
                                          
        F=2*omega/omega_c*(1-N)*np.cos(theta)/(np.sin(theta))**2     
        mu1_sq=1-N*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-N))*(-1+np.sqrt(1+F**2)))          #Ordinary refractive index sq
        mu2_sq=1-N*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-N))*(-1-np.sqrt(1+F**2)))          #Extraordinary refractive index sq
        # if i >2432 and i<2436:
        #     print(mu2_sq,N)
        
        mu1=np.sqrt(mu1_sq)     #ord refractive index
        if np.isnan(mu1) or mu1_sq>=1:
            mu1=0
            
        mu2=np.sqrt(mu2_sq)     #extraord refractive index
        if np.isnan(mu2) or mu2_sq>=1:
            mu2=0       
        
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        modS.append((np.dot(sf,sf))**0.5)
        ne_arr.append(ne/nc)
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        
        if i > n_steps and mu2 == 0:
            M=np.identity(3)
            S0[0] = np.cos(2*chiN)*np.cos(2*psiN)
            S0[1] = np.cos(2*chiN)*np.sin(2*psiN)
            S0[2] = np.sin(2*chiN)
                            
        if mu2 > 0:
                                    
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)                   #prefactor for all components
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-N))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-N))
            
            O3=prefactor*2*omega*e*Bz/(m*c)    
            
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
            
            M=np.dot(R1,np.dot(R2,M))
            sf=np.dot(M,S0)
            try:
                val=0.5*np.arctan(sf[1]/sf[0])
                #print(val)
                if(np.isnan(val)):
                    psi_arr.append(np.pi/4)
                else:
                    psi_arr.append(val)
            except:
                None
            try:
                val=0.5*np.arctan(sf[2]/np.sqrt((sf[0])**2+sf[1])**2)
                if(np.isnan(val)):
                    chi_arr.append(np.pi/4)
                else:
                    chi_arr.append(val)
            except:
                None
            
        else:
            A = sf[1]/sf[0]      #tan2psi
            B = sf[2]           #sin2chi
            
            s2eta = ((A**2+B**2)/(1+A**2))**0.5
            t2eta = s2eta/((1-s2eta**2))**0.5
            phi0=np.arcsin(B/s2eta)
            
            
            phi = (mu1*dz*omega/c)+phi
            
            psiN=0.5*np.arctan(t2eta*np.cos(phi0-phi))            
            #print(psiN)
            chiN=0.5*np.arcsin(s2eta*np.sin(phi0-phi))
                                                                         
            psi_arr.append(psiN)
            chi_arr.append(chiN)                       
       
        if i < n_steps:
            z = z + dz
        else:
            z = z - dz
                                                    
    # print(phi0)
    # print(t2eta)
    # print(phi)
    # print(n_steps)
    # print(z)

    return(sf,ne_arr,mu1_arr,mu2_arr,modS,psi_arr,chi_arr)


def final_stokes_new(s0,Bx,By,Bz,t):
    
    #defining all constants in CGS
    
    e=4.8032e-10                        #electron charge in esu (cgs)
    m=9.1094e-28                        #electron mass in g
    c=2.99792458e10                     #speed of light cm/sec^2
    wavelength=4e-5                     #used probe wavelength
    omega=2*np.pi*c/wavelength          #prob angular frequency
    nc=omega**2*m/(4*np.pi*e**2)        #plasma critical frequency     
    z=0                                 #probe entry from vacuum
    cs=6e6                              #plasma self similar expansion velocity
    dz=1e-8                             #plasma slab element length
    L = cs*t
    S0=s0                            #plasma slab length
    B=np.sqrt(Bx**2+By**2+Bz**2) 
    theta=np.arccos(Bz/B)
    omega_c=e*B/(m*c)                   #cyclotron freq
           
    n_steps=int((L*np.log(100)/dz))
    M=np.identity(3)
    R1=np.identity(3)
    R2=np.identity(3)
    
    phi = 0.0
    phi0 = 0.0
    psiN=0.0
    chiN=0.0
    sf=S0
    
    ne_arr=[]
    psi_arr=[]
    chi_arr=[]
    mu1_arr=[]
    mu2_arr=[]
    modS=[]
    
    for i in range(2*n_steps):
        
        N=np.exp(-(L*np.log(100)-z)/L)    
        ne=np.exp(-(L*np.log(100)-z)/L)*nc    #exponential density gradient
        omega_p=np.sqrt(4*np.pi*ne*e**2/m)          #plasma frequency
                                          
        F=2*omega/omega_c*(1-N)*np.cos(theta)/(np.sin(theta))**2     
        mu1_sq=1-N*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-N))*(-1+np.sqrt(1+F**2)))          #Ordinary refractive index sq
        mu2_sq=1-N*1/(1+omega_c**2/omega**2*(np.sin(theta))**2/(2*(1-N))*(-1-np.sqrt(1+F**2)))          #Extraordinary refractive index sq
        # if i >2432 and i<2436:
        #     print(mu2_sq,N)
        
        mu1=np.sqrt(mu1_sq)     #ord refractive index
        if np.isnan(mu1) or mu1_sq>=1:
            mu1=0
            
        mu2=np.sqrt(mu2_sq)     #extraord refractive index
        if np.isnan(mu2) or mu2_sq>=1:
            mu2=0       
        
        D=1-(e/(m*omega*c))**2*((Bx**2+By**2)/(1-N)+Bz**2)
        
        modS.append((np.dot(sf,sf))**0.5)
        ne_arr.append(ne/nc)
        mu1_arr.append(mu1)
        mu2_arr.append(mu2)
        
        if i > n_steps and mu2 == 0:
            M=np.identity(3)
            S0[0] = np.cos(2*chiN)*np.cos(2*psiN)
            S0[1] = np.cos(2*chiN)*np.sin(2*psiN)
            S0[2] = np.sin(2*chiN)
                            
        if mu2 > 0:
                                    
            prefactor=omega_p**2/((mu1+mu2)*c*omega**3*D)                   #prefactor for all components
            
            O1=prefactor*(e/(m*c))**2*((Bx**2-By**2)/(1-N))
            
            O2=prefactor*(e/(m*c))**2*(2*Bx*By/(1-N))
            
            O3=prefactor*2*omega*e*Bz/(m*c)    
            
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
            
            M=np.dot(R1,np.dot(R2,M))
            sf=np.dot(M,S0)
            try:
                val=0.5*np.arctan(sf[1]/sf[0])
                #print(val)
                if(np.isnan(val)):
                    psi_arr.append(np.pi/4)
                else:
                    psi_arr.append(val)
            except:
                None
            try:
                val=0.5*np.arctan(sf[2]/np.sqrt((sf[0])**2+sf[1])**2)
                if(np.isnan(val)):
                    chi_arr.append(np.pi/4)
                else:
                    chi_arr.append(val)
            except:
                None
            
        else:
            A = sf[1]/sf[0]      #tan2psi
            B = sf[2]           #sin2chi
            
            s2eta = ((A**2+B**2)/(1+A**2))**0.5
            t2eta = s2eta/((1-s2eta**2))**0.5
            phi0=np.arcsin(B/s2eta)
            
            
            phi = (mu1*dz*omega/c)+phi
            
            psiN=0.5*np.arctan(t2eta*np.cos(phi0-phi))            
            #print(psiN)
            chiN=0.5*np.arcsin(s2eta*np.sin(phi0-phi))
                                                                         
            psi_arr.append(psiN)
            chi_arr.append(chiN)                       
       
        if i < n_steps:
            z = z + dz
        else:
            z = z - dz
                                                    
    # print(phi0)
    # print(t2eta)
    # print(phi)
    # print(n_steps)
    # print(z)

    return(psi_arr[-1],chi_arr[-1])

def find_magfield(chi,psi,chi_path,psi_path):
    
    B_trans = 0
    B_axial = 0
    temp1 = 0
    temp2 = 0
    
    Chi_arr = np.loadtxt(chi_path)[:,0]
    Psi_arr = np.loadtxt(psi_path)[0,:]
    
    n = len(Chi_arr)
        
    for i in range(n):
        if chi <= Chi_arr[i] and temp1 == 0:
            B_trans = i
            temp1 = 1
        
        if psi <= Psi_arr[i] and temp2 == 0:
            B_axial = i
            temp2 = 1
    #print(Chi_arr)        
    return(B_trans,B_axial)

def ESM_sig(A,L_ip,L_off,B,H_mag,H_ip,Res,D,d,t,Ftime,red_fac):
    
    # A - raw image array
    # L_ip - length of IP in mm
    # L_off - IP edge offset from magnet edge in mm
    # B - B field value inside ESM in tesla
    # H_mag - Height of collimator to magnet edge in mm
    # H_ip - Height of collimator to IP surface in mm
    # Res - Resolution of scanner in micron
    # D - Distance of IP edge from the target in mm
    # d - dia of collimator in mm
    # t - IP cover Al foil thickness in micron
    # Ftime - fading time in min
    # red_fac - reduction factor to make less dense plot
    
    #constant
    m = 9.10939e-31
    c = 2.99792458e+8
    e = 1.60218e-19
    
    # Sensitivity
    sens = 4000 

    #conversion
    H_mag = H_mag*1e-3
    H_ip = H_ip*1e-3
    Res = Res*1e-6
    D = D*1e-3
    d = 2*1e-3
    t = t*1e-4 #in cgs cm
    
    #Calculations
    Solidangle=2*np.pi*(1-np.sqrt(D**2-(d/2)**2)/D)

    #fadratio = funcfading(Ftime)

    row1, col1 = A.shape

    # Apply smoothing to the sum of the image
    Q = ndimage.gaussian_filter1d(np.sum(A, axis=0), 10)
    Q = ndimage.gaussian_filter1d(Q, 10)

    #plt.plot(Q)
    #plt.show()

    # Find the maximum value of the smoothed image
    #R = np.max(Q)

    # Find the index of the maximum value in the smoothed image
    j = np.argmax(Q)

    # Crop image
    X = A[:, j-int((col1*0.025)/2):j+int((col1*0.025)/2)]

    # Find the size of the cropped image
    row, col = X.shape

    # Calculate the signal
    S = np.sum(X, axis=1) / (col * sens)

    # Calculate the distance
    N = np.arange(1, row+1) * (L_ip/row)

    # Calculate the background
    #BG1X = A[:, j-int(col1*0.1):j-int(col1*0.1)+int(col1*0.025)]
    #BG2X = A[:, j+int(col1*0.066):j+int(col1*0.066)+int(col1*0.025)]

    BG1X = A[:, j-int(col1*0.25):j-int(col1*0.25)+int(col1*0.025)]
    BG2X = A[:, j+int(col1*0.25)-int(col1*0.025):j+int(col1*0.25)]

    BG1 = ndimage.gaussian_filter1d(np.sum(BG1X, axis=1) / (col * sens), 5)
    BG2 = ndimage.gaussian_filter1d(np.sum(BG2X, axis=1) / (col * sens), 5)

    # Plots signal and background
    #plt.figure(1)
    #plt.plot(S)
    #plt.show()
    #plt.plot(BG1)
    #plt.show()
    #plt.plot(BG2)
    #plt.show()

    NetSig = []
    NoiseSig = []
    Energy = []

    for i in range(row):
        Length = N[i]-L_off
        Length = Length*1e-3
            
        energy = (1/(e*1e6))*(m*c**2)*(np.sqrt(((((Length**2)+(H_mag**2))/(2*H_mag))**2)*(((e*B)/(m*c))**2)+1)-1)
        
        if H_mag >= Length:
            Theta = -1*np.arctan(abs((Length**2)-(H_mag**2))/(2*H_mag*Length))
            
        else:
            Theta = 1*np.arctan(abs((Length**2)-(H_mag**2))/(2*H_mag*Length))

        Sensitivity = funcsensitivity(energy)
        Fadingratio = funcfading(Ftime)
        Transmittance = functransemittance(energy, Theta, t)

        Correction=int(((H_ip-H_mag)*np.tan(Theta))/Res)
        if i+Correction <= 0:
            SigPSL = 0
            NoisePSL = 0
        elif i+Correction >= row:
            SigPSL = 0
            NoisePSL = 0
        else:
            SigPSL = S[i+Correction]
            NoisePSL = (BG1[i+Correction]+BG2[i+Correction])/2
            SigPSL = SigPSL - NoisePSL
           
    #Calculate differential number density perunit solidangle and energy       
        Sig = (e*1e6)*(SigPSL/(Res*Solidangle*(m*c**2)))*((m*c)/(e*B))*((H_mag*np.cos(Theta))/(Length*Transmittance*Sensitivity*Fadingratio))*np.sqrt(1+(((2*H_mag)/((Length**2)+(H_mag**2)))**2)*(((m*c)/(e*B))**2))        
        Noise = (e*1e6)*(NoisePSL/(Res*Solidangle*(m*c**2)))*((m*c)/(e*B))*((H_mag*np.cos(Theta))/(Length*Transmittance*Sensitivity*Fadingratio))*np.sqrt(1+(((2*H_mag)/((Length**2)+(H_mag**2)))**2)*(((m*c)/(e*B))**2))
      
        NetSig.append(Sig)
        NoiseSig.append(Noise)
        Energy.append(energy)  
        
    Spectrum = np.column_stack((Energy, NetSig, NoiseSig))

    Energy = np.array(Energy)
    NetSig = np.array(NetSig)
   
    for i in range(row):
        if NetSig[i] < 1e3:
              NetSig[i]=np.nan
              NoiseSig[i]=np.nan

    E=[]
    S=[]
    for i in range(row):
        if red_fac*i < row:
            E.append(Energy[red_fac*i])
            S.append(NetSig[red_fac*i])
    
    return(E,S,Spectrum)
            
#Function to calculaate fading ratios
def funcfading(ftime):
    ffa = 0.156    # 0.155857
    ffb = 0.209    # 0.209367
    fta = 0.558    # 0.557602
    ftb = 11.3     # 11.2649
    ftc = 2000.0   # 1999.96

    fadingratio = ffa * pow(0.5, (ftime / fta)) + ffb * pow(0.5, (ftime / ftb)) + (1 - (ffa + ffb)) * pow(0.5, (ftime / ftc))
    
    return fadingratio

#Function to calculate sensitivity of IP
def funcsensitivity(energy):
    # Function 1
    if energy < 0.05:
        sa = -1.37
        sb = 4.8955e-1
        sc = 9.34307e-3
        sensitivity = sa * energy**2 + sb * energy + sc
    # Function 2
    elif energy >= 0.05 and energy < 0.12:
        sa = 29.6061
        sb = -10.7906
        sc = 1.24418
        sd = -8.48498e-3
        sensitivity = sa * energy**3 + sb * energy**2 + sc * energy + sd
    # Function 3
    elif energy >= 0.12 and energy < 0.137:
        sa = -1.30643e-1
        sb = -2.64656e-2
        sc = 4.16476e-2
        sensitivity = sa * energy**2 + sb * energy + sc
    # Function 4
    elif energy >= 0.137 and energy < 0.46:
        sa = 4.06997e-2
        sb = 3.27383
        sc = 9.59762e-3
        sensitivity = sa * math.exp(-sb * energy) + sc
    # Function 5
    elif energy >= 0.46 and energy < 0.79:
        sa = 4.14759e-2
        sb = 3.07174
        sc = 8.5134e-3
        sensitivity = sa * math.exp(-sb * energy) + sc
    # Function 6
    elif energy >= 0.79 and energy < 1.1:
        sa = 1.40477e-2
        sb = -3.53998e-2
        sc = 3.13743e-2
        sensitivity = sa * energy**2 + sb * energy + sc
    # Function 7
    elif energy >= 1.1 and energy < 1.6:
        sa = 1.49836
        sb = 7.00353
        sc = -2.45725e-4
        sd = 9.05957e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy**2 + sd
    # Function 8
    elif energy >= 1.6 and energy < 2.7:
        sa = 4.4432e-3
        sb = 9.45248e-1
        sc = 7.46777e-3
        sensitivity = sa * math.exp(-sb * energy) + sc
    # Function 9
    elif energy >= 2.7 and energy < 5.7:
        sa = 2.74341e-3
        sb = 7.27153e-1
        sc = -5.07852e-5
        sd = 7.56518e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd
    # Function 10
    elif energy >= 5.7 and energy < 9.8:
        sa = 1.51122e-3
        sb = 4.05074e-1
        sc = -1.98777e-5
        sd = 7.28229e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd
    # Function 11
    elif energy >= 9.8 and energy < 13:
        sa = 1.15864e-3
        sb = 2.9994e-1
        sc = -1.29653e-5
        sd = 7.18185e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd
    # Function 12
    elif energy >= 13 and energy < 37:
        sa = 5.42261e-4
        sb = 1.37381e-1
        sc = -6.97495e-6
        sd = 7.03642e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd
    # Function 13
    else:
        sa = 2.04741e-4
        sb = 5.29949e-2
        sc = -5.73563e-6
        sd = 6.96509e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd
    
    return sensitivity

#Function to calculate transemittance
def functransemittance(energy, theta, althickness):
    # cgs Unit
    AlA = 26.981538
    AlZ = 13.0
    AlRho = 2.69
    ta = 0.537
    tb = 0.9815
    tc = 3.1230
    td = 9.2 * (AlZ**-0.2) + 16 * (AlZ**-2.2)
    te = 0.63 * AlZ / AlA + 0.27
    
    range = ta * energy * (1 - tb / (1 + tc * energy))
    taurange = (althickness * AlRho) / (range * np.cos(theta))  # corrected thickness is used for oblique incidence
    transemittance = (1 + np.exp(-td * te)) / (1 + np.exp(td * (taurange - te)))
    
    return transemittance
        
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
    
def get_nonzero_value_average(matrix):
    # Extract the nonzero elements
    nonzero_elements = matrix[matrix != 0.0]

    # Calculate the average of nonzero elements
    average_nonzero = np.mean(nonzero_elements)
    return(average_nonzero)
    
    
def _hello():
    print("hello")

