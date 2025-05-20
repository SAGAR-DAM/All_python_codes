# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:55:40 2025

@author: mrsag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import scipy.integrate as integrate
# from Curve_fitting_with_scipy import polynomial_fit as pft
import glob
from scipy.interpolate import interp1d
import os
from numba import njit

import matplotlib as mpl

# ==== GLOBAL STYLE SETTINGS ====
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
# mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['figure.dpi'] = 500  # High-res display

# === Black background + yellow labels/ticks ===
# mpl.rcParams['figure.facecolor'] = 'black'
# mpl.rcParams['axes.facecolor'] = 'black'
# mpl.rcParams['axes.edgecolor'] = 'green'
mpl.rcParams['axes.labelcolor'] = 'red'
mpl.rcParams['xtick.color'] = 'red'
mpl.rcParams['ytick.color'] = 'red'
mpl.rcParams['text.color'] = 'red'
mpl.rcParams['axes.titlecolor'] = 'red'

# %% Bifurcation data generation

@njit
def eq_pop(r):
    x = 0.5
    for j in range(np.random.randint(low=1000, high=1200)):
        # x = r * x * (1 - x)
        x = r*np.abs(np.sin(np.pi*x))
    return x

r_min = 0
r_max = 1.2
N = 1000000

r_arr = np.linspace(r_min, r_max, N)
p_eq = []

for i in range(N):
    pop = eq_pop(r_arr[i])
    p_eq.append(pop)

# %% Plotting

fig, ax = plt.subplots()

# Plot transformed x-axis
alpha = 2
x_transformed = np.exp(alpha * r_arr)
ax.scatter(x_transformed, p_eq, s=0.3, color='k', edgecolors='none')

# Set custom ticks based on original r values
tick_pos = np.linspace(np.exp(alpha*r_min),np.exp(alpha*r_max),5)
r_tick_vals = np.log(tick_pos)/alpha  # choose 9 evenly spaced r values
x_tick_locs = np.exp(alpha * r_tick_vals)   # transformed x positions

ax.set_xticks(x_tick_locs)
ax.set_xticklabels([f"{r:.2f}" for r in r_tick_vals])  # label with original r

# Axis labels and title
ax.set_xlabel("r (original scale)")
ax.set_ylabel("Equilibrium Population")
ax.set_title("Logistic Map Bifurcation")

plt.show()
