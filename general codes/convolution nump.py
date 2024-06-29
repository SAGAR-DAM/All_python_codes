# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:27:04 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the functions f(x) and g(x)
def f(x):
    return(1*(abs(x)<5))

def g(x):
    return np.exp(-x**2)

# Define the convolution function
def convolution(f, g, x):
    # Discretize the functions
    delta_x = x[1] - x[0]
    result = np.convolve(f(x), g(x), mode='same') * delta_x
    return result

# Generate x values
x = np.linspace(-15, 15, 1000)

# Calculate the convolution of f(x) and g(x)
conv_result = convolution(f, g, x)

# Plot the functions and the convolution result
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(x, f(x), label='f(x)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, g(x), label='g(x)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, conv_result, label='Convolution of f and g')
plt.legend()

plt.tight_layout()
plt.show()


'''
def fg(x,a):
    return(f(x)*g(x-a))

plt.plot(x,fg(x,4))
plt.show()
'''