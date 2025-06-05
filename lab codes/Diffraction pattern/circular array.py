# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:29:04 2024

@author: mrsag
"""

from diffractsim import MonochromaticField, ApertureFromFunction, mm, cm, nm, um
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 300  # highres display

# Define the wavelength of the light
wavelength = 800 * nm  # 500 nm (green light)

# Define the circular hole grating parameters
hole_radius = 50 * um  # Radius of each circular hole along x-axis
num_holes_x = 10  # Number of holes along x-axis
num_holes_y = 10  # Number of holes along y-axis
grid_period_x = 400 * um  # Spacing between holes along x-axis
grid_period_y = 400 * um  # Spacing between holes along y-axis

# Define the circular hole function
def circular_hole_grating(x, y, wavelength):
    hole_pattern = np.zeros_like(x)
    for i in range(num_holes_x):
        for j in range(num_holes_y):
            hole_center_x = (i - (num_holes_x - 1) / 2) * grid_period_x
            hole_center_y = (j - (num_holes_y - 1) / 2) * grid_period_y
            distance_from_center = np.sqrt((x - hole_center_x) ** 2 + (y - hole_center_y) ** 2)
            hole_pattern += np.where(distance_from_center < hole_radius, 1, 0)
    return hole_pattern

@np.vectorize
def get_phase(E):
    phase = (np.log(E)).imag
    return phase


# Create a monochromatic field
field = MonochromaticField(
    wavelength=wavelength,
    extent_x=5 * mm,
    extent_y=5 * mm,
    Nx=1000,
    Ny=1000
)

# Create the aperture pattern
x = field.xx
y = field.yy
aperture_pattern = circular_hole_grating(x, y, wavelength)

# Plot the aperture pattern
plt.figure()
plt.imshow(aperture_pattern, extent=(x.min()/mm, x.max()/mm, y.min()/mm, y.max()/mm), cmap='gray')
plt.title('Aperture Pattern (Circular Holes)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Transmittance')
plt.show()



# Add the circular hole grating aperture to the field
aperture = ApertureFromFunction(circular_hole_grating)
field.add(aperture)

# Propagate the field to a certain distance
field.propagate(80 * cm)

field.plot_intensity(field.get_intensity())