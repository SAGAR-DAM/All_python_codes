# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:37:59 2024

@author: mrsag
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 100  # highres display
# Define the function for which you want to calculate Fourier coefficients
def g(x):
    #return((x**3*(abs(x)<2)+np.log(abs(x)+1)))
    return x

@np.vectorize
def f(x):
    if abs(x)>np.pi:
        if(x>0):
            return(f(x-2*np.pi))
        else:
            return(f(x+2*np.pi))
        
    else:
        return g(x)

# Define the range of integration
a = -np.pi
b = np.pi

# Define the number of terms in the Fourier series
n_terms = 5
slider_max = 30

# Function to calculate the Fourier coefficients for a given function and integer n
def fourier_coefficients(n):
    # # Define the integrands for Fourier coefficients a_n and b_n
    # def integrand_cos(x):
    #     return f(x) * np.cos(n * x)

    # def integrand_sin(x):
    #     return f(x) * np.sin(n * x)

    # # Perform numerical integration using quad
    # a_n, _ = quad(integrand_cos, a, b)
    # b_n, _ = quad(integrand_sin, a, b)

    # # Normalize coefficients
    # a_n *= 1 / np.pi
    # b_n *= 1 / np.pi
    if(n==0):
        a_n = 0
        b_n = 0
    else:
        a_n = 0
        b_n = 2/(n)*(-1)**(n+1)

    return a_n, b_n

# Calculate Fourier coefficients for n_terms
coefficients = [fourier_coefficients(n) for n in range(0, n_terms + 1)]

# Print the Fourier coefficients
for n, (a_n, b_n) in enumerate(coefficients, 0):
    print(f"a_{n}: {a_n*(a_n>1e-5):.5f}, b_{n}: {b_n*(b_n>1e-5):.5f}")
    
    
def fourier_calculator(x,coeff):
    y = coeff[0][0]/2
    for i in range(1,len(coeff)):
        y += coeff[i][0]*np.cos(i*x)+coeff[i][1]*np.sin(i*x)
        
    return(y)

x = np.linspace(3*a,3*b,1000)
y = f(x)
y_predict = fourier_calculator(x, coefficients)



# Initial plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
line1, = ax.plot(x, y,'go-', label="Given function", markersize=1)
line, = ax.plot(x, y_predict, 'r')
ax.set_xlabel('x')
ax.set_ylabel('y')
rounding = 5
title = ax.set_title(f"n_terms: {n_terms}")
#time_text = ax.set_title(f't = {degree}')
#ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
#ax.set_ylim(-1,1)
# Slider setup
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'n_terms', 1, slider_max, valinit=n_terms, valstep=1)

# Update function for the slider
def update(val):
    t = slider.val
    # Calculate Fourier coefficients for n_terms
    coefficients = [fourier_coefficients(n) for n in range(0, t + 1)]
    y_predict = fourier_calculator(x, coefficients)
    line.set_ydata(y_predict)
    title.set_text(f"degree: {t}")
    #ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
    #ax.set_ylim(-1,1)
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()