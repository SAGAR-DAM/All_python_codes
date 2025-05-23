'''
2d FFT of any function f(x,y)
@author: mrsag
'''

#2d FFT of a two variable function

import numpy as np

# Define the function f(x,y)
def f(x, y):
    return 10*np.exp(-0.1*np.sin(2*x**2+y**2))

# Create a grid of x and y values
x, y = np.linspace(-5, 5, 256), np.linspace(-5, 5, 256)
X, Y = np.meshgrid(x, y)

# Compute the FFT of the function
F = np.fft.fft2(f(X, Y))

# Shift the zero-frequency component to the center of the spectrum
F_shifted = np.fft.fftshift(F)

# Compute the magnitude of the FFT
magnitude = np.log(np.abs(F_shifted))

# Plot the magnitude of the FFT
import matplotlib.pyplot as plt
plt.imshow(f(X,Y),cmap='jet')
plt.title("f(x,y)")
plt.colorbar()
plt.show()

plt.imshow(magnitude, cmap='hot')
plt.title(r"$\tilde f(k_x,k_y)$")
plt.colorbar()
plt.show()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__