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


sim = Simulationbox2d(resolution_x=401, resolution_y=201, box_x=2, box_y=1,
                      # b_y0=b_y0, b_x0 = b_x0, b_y1 = b_y1, 
                      match_boundary=True, potential_given_on_boundary=False)

sim.add_disk_plate(center=(0.5,0.5),radius=0.2,potential=100)
sim.add_disk_plate(center=(1.5,0.5),radius=0.2,potential=-100)

sim.add_polygon_plate(poly=polygon_plate(edges=[(0.1,0.21777),(0.1,0.22777),(1.9,0.22777),(1.9,0.21777)],potential=100))
sim.add_polygon_plate(poly=polygon_plate(edges=[(0.1,0.81777),(0.1,0.82777),(1.9,0.82777),(1.9,0.81777)],potential=-100))

sim.solve(max_iterations=500)
# sim.plot_potential()

sim.plot_electric_field(stepx=5,stepy=5,scale=5,remove_singularity=2000)
sim.plot_Ex_Ey_E_separately(plot_Ex=True,plot_Ey=True,plot_mod_E=True,colorbar=True,logscale=True)