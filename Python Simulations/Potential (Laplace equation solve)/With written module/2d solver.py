# -*- coding: utf-8 -*-
"""
Created on Thu May  1 21:10:13 2025

@author: mrsag
"""

from potential_simulator.LaplassSolver import Simulationbox2d, polygon_plate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 500  # highres display

# %%

sim = Simulationbox2d(resolution_x=801, resolution_y=401, box_x=2, box_y=1,
                      # b_y0=b_y0, b_x0 = b_x0, b_y1 = b_y1, 
                      match_boundary=True, potential_given_on_boundary=False)

sim.add_disk_plate(center=(0.5,0.52277),radius=0.2,potential=100)
sim.add_disk_plate(center=(1.5,0.52277),radius=0.2,potential=-100)

sim.add_polygon_plate(poly=polygon_plate(edges=[(0.1,0.21777),(0.1,0.22777),(1.9,0.22777),(1.9,0.21777)],potential=100))
sim.add_polygon_plate(poly=polygon_plate(edges=[(0.1,0.81777),(0.1,0.82777),(1.9,0.82777),(1.9,0.81777)],potential=-100))

sim.show_geoemtry()
sim.solve(max_iterations=2000)
sim.plot_potential(lw=0.4,levels=50)

sim.plot_electric_field(stepx=5,stepy=5,scale=5,remove_singularity=2000)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,colorbar=True,logscale=True)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,remove_singularity_Ex=100,remove_singularity_Ey=100,remove_singularity_mod_E=100,colorbar=True,logscale=False)


# %%


plate1 = polygon_plate(
    edges=[(0.1, 0.3), (0.2, 0.6), (0.4, 0.4), (0.2, 0.4)],
    potential=35
)

# Create alternating zigzag points
x_plate2 = list(np.linspace(0.2, 1.8, 22))  # Use odd number for clean pairing
y_plate2 = 0.8*np.ones(len(x_plate2))
y_plate2[::2] +=0.1
x_plate2.extend(x_plate2[::-1])
y_plate2_bottom = list(y_plate2[::-1]-0.1)
y_plate2 = list(y_plate2)
y_plate2.extend(y_plate2_bottom)

# Top zigzag path
vertices_plate2 = [(x_plate2[i], y_plate2[i]) for i in range(len(x_plate2))]

# Now you can create the polygon plate
plate2 = polygon_plate(edges=vertices_plate2, potential=35)

b_y0 = -5*np.sin(np.linspace(0,31.4159,100))
b_x0 = 1*np.linspace(-5,5,100)**2-0.1*25
b_y1 = -25*np.sin(np.linspace(0,3*3.14159,100))

sim = Simulationbox2d(resolution_x=801, resolution_y=401, box_x=2, box_y=1,
                      b_y0=b_y0, b_x0 = b_x0, b_y1 = b_y1, 
                      match_boundary=True, potential_given_on_boundary=True)

sim.add_polygon_plate(plate1)
sim.add_polygon_plate(plate2)
sim.add_disk_plate(center=(0.5,0.5),radius=0.1,potential=-30)
sim.solve(max_iterations=500)

sim.show_geoemtry()
sim.solve(max_iterations=2000)
sim.plot_potential()

sim.plot_electric_field(stepx=5,stepy=5,scale=5,remove_singularity=2000)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,colorbar=True,logscale=True)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,colorbar=True,logscale=False)

# %%


sim = Simulationbox2d(resolution_x=401, resolution_y=401, box_x=2, box_y=1,
                      # b_y0=b_y0, b_x0 = b_x0, b_y1 = b_y1, 
                      match_boundary=True, potential_given_on_boundary=False)

for i in range(1,7):
    for j in range(1,4):
        potential = 100 if (i+j)%2==0 else -100
        sim.add_disk_plate(center=(0.3*i-0.075,0.3*j-0.11),radius=0.1,potential=potential*1/(i+j))


for i in range(0,8):
    for j in range(0,4):
        potential = -100 if (i+j)%2==0 else 100
        sim.add_disk_plate(center=(0.3*i+0.075,0.3*j+0.05),radius=0.05,potential=potential*1/(11-i-j))

sim.show_geoemtry()
sim.solve(max_iterations=2000)
sim.plot_potential(lw=0.4,levels=50)

sim.plot_electric_field(stepx=5,stepy=5,scale=5,remove_singularity=2000)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,colorbar=True,logscale=True)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,remove_singularity_Ex=100,remove_singularity_Ey=100,remove_singularity_mod_E=100,colorbar=True,logscale=False)

