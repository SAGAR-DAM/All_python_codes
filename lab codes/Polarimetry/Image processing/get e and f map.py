# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:53:53 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# p_filter = 1/(0.5*0.5*0.5)*100/1.3
# d_filter = 1/(0.5*0.5)*100/50

# q_filter = 100/32
# t_filter = 1/(0.5)*100/1

p_filter = 792.82
d_filter = 415.2
t_filter = 1543.17
q_filter = 666.67

def image_reshape(image, average_box):
    new_image = np.zeros(shape=np.array(image.shape)//average_box)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i,j] = np.mean(image[i*average_box:(i+1)*average_box , j*average_box:(j+1)*average_box])
    
    return(new_image)

def image_centre_with_roll(image):
    #brightest_point=brightest(image)
    bp_index=np.where(image==np.max(image))
    brightest_point=[bp_index[0][0],bp_index[1][0]]
    image=np.roll(np.roll(image,-(brightest_point[0]-image.shape[0]//2),axis=0),-(brightest_point[1]-image.shape[1]//2),axis=1)
    # X=image.shape[0]
    # Y=image.shape[1]
    # image=image[X//5:4*X//5,Y//5:4*Y//5]
    return(image)



p_image = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_par\\par_splitted_image\\P_027_br.tif")[256:768,256:768]
d_image = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_par\\par_splitted_image\\P_026_tl.tif")[256:768,256:768]

q_image = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\C_027_tl.tif")[256:768,256:768]
t_image = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\C_027_br.tif")[256:768,256:768]

binning_box_size=4

p_image = image_reshape(image = p_image, average_box = binning_box_size)
d_image = image_reshape(image = d_image, average_box = binning_box_size)
q_image = image_reshape(image = q_image, average_box = binning_box_size)
t_image = image_reshape(image = t_image, average_box = binning_box_size)

p_image = image_centre_with_roll(p_image)
d_image = image_centre_with_roll(d_image)
q_image = image_centre_with_roll(q_image)
t_image = image_centre_with_roll(t_image)

plt.imshow(p_image*p_filter, cmap="jet")
plt.colorbar()
plt.title("p_image * p_filter")
plt.show()

plt.imshow(d_image*d_filter, cmap="jet")
plt.colorbar()
plt.title("d_image * d_filter")
plt.show()

plt.imshow(q_image*q_filter, cmap="jet")
plt.colorbar()
plt.title("q_image * q_filter")
plt.show()

plt.imshow(t_image*t_filter, cmap="jet")
plt.colorbar()
plt.title("t_image * t_filter")
plt.show()

pu_only_p = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_par\\par_splitted_image\\P_015_br.tif")[256:768,256:768]
pu_only_d = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_par\\par_splitted_image\\P_015_tl.tif")[256:768,256:768]

pu_only_t = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\C_015_br.tif")[256:768,256:768]
pu_only_q = io.imread("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\C_015_tl.tif")[256:768,256:768]


pu_only_p = image_reshape(image = pu_only_p, average_box = binning_box_size)
pu_only_d = image_reshape(image = pu_only_d, average_box = binning_box_size)
pu_only_t = image_reshape(image = pu_only_t, average_box = binning_box_size)
pu_only_q = image_reshape(image = pu_only_q, average_box = binning_box_size)


pu_only_p = image_centre_with_roll(image = pu_only_p)
pu_only_d = image_centre_with_roll(image = pu_only_d)
pu_only_q = image_centre_with_roll(image = pu_only_q)
pu_only_t = image_centre_with_roll(image = pu_only_t)


plt.imshow(pu_only_p*p_filter, cmap="jet")
plt.colorbar()
plt.title("pu_only_p * p_filter")
plt.show()

plt.imshow(pu_only_d*d_filter, cmap="jet")
plt.colorbar()
plt.title("pu_only_d * d_filter")
plt.show()

plt.imshow(pu_only_q*q_filter, cmap="jet")
plt.colorbar()
plt.title("pu_only_q * q_filter")
plt.show()

plt.imshow(pu_only_t*t_filter, cmap="jet")
plt.colorbar()
plt.title("pu_only_t * t_filter")
plt.show()

@np.vectorize
def full_image_stokes_generator(P, D, Q, T, pu_only_p, pu_only_d, pu_only_q, pu_only_t, p_avg, d_avg, q_avg, t_avg):
    if(P>=p_avg and D>=d_avg and Q>= q_avg and T>=t_avg):
        def get_off_background_factor(image, image_avg, bg_factor):
            image = image*(image>=image_avg)
            return image
            
            
        def subtract_pump_noise(image, noise):
            image = (image-noise)*(image>noise)
            return image 
        
        
        def get_stokes(P,D,T,Q, bg_factor):
            
            P = get_off_background_factor(P, p_avg, bg_factor)
            D = get_off_background_factor(D, d_avg, bg_factor)
            Q = get_off_background_factor(Q, q_avg, bg_factor)
            T = get_off_background_factor(T, t_avg, bg_factor)
            
            
            P = subtract_pump_noise(image = P, noise = pu_only_p)
            D = subtract_pump_noise(image = D, noise = pu_only_d)
            Q = subtract_pump_noise(image = Q, noise = pu_only_q)
            T = subtract_pump_noise(image = T, noise = pu_only_t)
            
            P *= p_filter
            D *= d_filter
            Q *= q_filter
            T *= t_filter
            
            
            s1 = 2*P/D-1
            s2 = 2*T/D-1
            s3 = 1-2*Q/D
            
            s = np.matrix([[s1],[s2],[s3]])
            mod_s = np.sqrt(np.dot(s.T,s)[0,0])
            
            return P, D, Q, T, s, mod_s
        
        
        def get_mod_s(P,D,T,Q, bg_factor):
            P = get_off_background_factor(P, bg_factor)
            D = get_off_background_factor(D, bg_factor)
            Q = get_off_background_factor(Q, bg_factor)
            T = get_off_background_factor(T, bg_factor)
            
            
            P = subtract_pump_noise(image = P, noise = pu_only_p)
            D = subtract_pump_noise(image = D, noise = pu_only_d)
            Q = subtract_pump_noise(image = Q, noise = pu_only_q)
            T = subtract_pump_noise(image = T, noise = pu_only_t)
            
            P *= p_filter
            D *= d_filter
            Q *= q_filter
            T *= t_filter
            
            
            s1 = 2*P/D-1
            s2 = 2*T/D-1
            s3 = 1-2*Q/D
            
            s = np.matrix([[s1],[s2],[s3]])
            mod_s = np.sqrt(np.dot(s.T,s)[0,0])
            
            return (mod_s-1)
        
        def bisection(f, a, b, P, D, Q, T, tol, max_iter=1000):
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
        
            if f(P, D, Q, T, a) * f(P, D, Q, T, b) > 0:
                raise ValueError("The function values at the interval endpoints must have different signs.")
        
            root = None
            iterations = 0
        
            while iterations < max_iter:
                c = (a + b) / 2
                if abs(f(P, D, Q, T, a)) <= tol:
                    root = a
                    break
                elif abs(f(P, D, Q, T, b)) <= tol:
                    root = b
                    break
                elif abs(f(P, D, Q, T, c)) <= tol:
                    root = c
                    break
                elif f(P, D, Q, T, c) * f(P, D, Q, T, a) < 0:
                    b = c
                else:
                    a = c
        
                iterations += 1
        
            #root = (a + b) / 2
            return root
        
        def find_interval_with_sign_change(f, lower, upper, P, D, Q, T, step=0.1):
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
                if f(P, D, Q, T, x) * f(P, D, Q, T, x + step) < 0:
                    a = x
                    b = x + step
                    break
                x += step
        
            return a, b
        
        def make_s_normalized(P, D, Q, T, tol):
            try:
                background_factor = bisection(f = get_mod_s, a=1, b=1.2, P=P, D=D, Q=Q, T=T, tol=tol)
            except:
                try:
                    a, b = find_interval_with_sign_change(f = get_mod_s, lower = 1, upper = 2,  P=P, D=D, Q=Q, T=T)
                    background_factor = bisection(f = get_mod_s, a=a, b=b, P=P, D=D, Q=Q, T=T, tol=tol)
                except:
                    background_factor = None
                    
            return(background_factor)
        
        background_factor = make_s_normalized(P=p_image, D=d_image, Q=q_image, T=t_image, tol=0.1)
        s, mod_s = get_stokes(P, D, Q, T, background_factor)