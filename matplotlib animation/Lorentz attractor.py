# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:09:23 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Lorenz system
def lorenz(X, t, sigma=10, rho=28, beta=8/3):
    x, y, z = X
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

# Function to solve the Lorenz system using Euler's method
def solve_lorenz(x0, y0, z0, tmax, dt):
    t = np.arange(0, tmax, dt)
    X = np.zeros((len(t), 3))
    X[0] = np.array([x0, y0, z0])
    
    for i in range(1, len(t)):
        X[i] = X[i-1] + lorenz(X[i-1], t[i-1]) * dt
        
    return t, X

# Parameters
x0, y0, z0 = 0., 1., 1.05  # Initial conditions
tmax = 200                   # Time to run the simulation
dt = 0.01                   # Time step

# Solve the Lorenz system
t, X = solve_lorenz(x0, y0, z0, tmax, dt)

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim((-30, 30))
ax.set_ylim((-30, 30))
ax.set_zlim((5, 55))

# Plot elements
line, = ax.plot([], [], [], lw=0.5, color='g')

# Initialization function
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Update function for the animation
def update(frame):
    line.set_data(X[:frame, 0], X[:frame, 1])
    line.set_3d_properties(X[:frame, 2])
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=1, blit=True)

plt.show()
