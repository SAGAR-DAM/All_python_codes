# -*- coding: utf-8 -*-
"""
Created on Thu May  1 21:10:13 2025

@author: mrsag
"""

from potential_simulator.LaplassSolver import Simulationbox2d, polygon_plate, SimulationBox3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 100  # highres display


# Create simulation box and add all geometries
box = SimulationBox3D(resolution_x=200,resolution_y=100,resolution_z=100,    # resolutions of grid along corresponding axis
                      box_x=2, box_y=1, box_z=1,    # Box side lengths 
                      potential_offset = 0)

# box.add_sphere(center=(0.25, 0.25, 0.25), radius=0.1, potential=1)
# box.add_box((0.6, 0.8), (0.6, 0.8), (0.6, 0.8), potential=-1)
# box.add_cylinder(base_center=(0.5, 0.5), radius=0.1, height=0.9, axis='z', potential=0.5)
# box.add_ellipsoid(center=(0.75, 0.25, 0.25), radii=(0.05, 0.1, 0.1), potential=0.8)
box.add_hyperboloid(center=(1,0.2,0.2), coeffs=(0.6, 0.1, 0.1), waist=0.2, axis="x", potential=1)
box.add_hyperboloid(center=(1,0.8,0.8), coeffs=(0.6, 0.1, 0.1), waist=0.2, axis="x", potential=-1)
box.add_hyperboloid(center=(0.5,0.5,0.5), coeffs=(0.1, 0.1, 0.4), waist=0.2, axis="z", potential=-1)
box.add_hyperboloid(center=(1.5,0.5,0.5), coeffs=(0.1, 0.4, 0.1), waist=0.2, axis="y", potential=1)
# box.add_plane(coefficients=(0,0,1,0),thickness=0.1,potential=10)
# box.add_plane(coefficients=(0,0,1,-1),thickness=0.1,potential=-10)
# box.add_sphere(center=(1, 0.5, 0.5), radius=0.3, potential=10)
box.add_cylinder(base_center=(1,0.5,0),radius=0.1,axis="z",potential=1,height=1)
# box.add_cylinder(base_center=(0,0,0),radius=0.1,axis="x",potential=-5,height=2)
# box.add_cylinder(base_center=(0,1,0),radius=0.1,axis="x",potential=5,height=2)
# box.add_cylinder(base_center=(0,0,1),radius=0.1,axis="x",potential=5,height=2)
# box.add_cylinder(base_center=(0,1,1),radius=0.1,axis="x",potential=-5,height=2)
# box.add_hollow_pipe(base_center=(0,0.5,0.5),radius=0.1,axis="x",height=2, thickness=0.05,potential=5)



# Solve potential field
# box.solve(max_iter=100, tol=1e-4, method='gauss-seidel', verbose=True)
# box.plot_potential_density()
# box.plot_electric_field_3d()

box.show_geometry()