# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:24:01 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define masses of the two bodies
m1 = 5.0  # mass of the central body
m2 = 1.0  # mass of the moving object
mu = (m1 * m2) / (m1 + m2)  # reduced mass

# Define the central force F(r)
def F(r):
    k = 1.0  # strength of the force
    # grav_force = -k/r**2
    # perturbed_grav_force = -k/r**2 - 1*k/r
    # inverse_r_force = -k/r
    # exponential_force = -k*np.exp(-r)
    # log_force = -k/(1+np.log(r))
    # const_force = -k
    # spring_force = -k*r
    strange_force = -k*r**2
    return strange_force

# Define the equations of motion for the reduced mass system
def equations(t, y):
    r, r_dot, theta = y
    r_ddot = F(r) / mu + (h**2 / r**3)  # Radial acceleration
    theta_dot = h / r**2  # Angular velocity from conservation of angular momentum
    return [r_dot, r_ddot, theta_dot]

# Initial conditions
r0 = 2.0         # initial radial distance
r_dot0 = 0.2    # initial radial velocity
theta0 = 0.0     # initial angle
y0 = [r0, r_dot0, theta0]

# Set initial angular momentum
h = 1.0  # angular momentum (constant in central force problems)

# Time span for the simulation
t_max = 400
t_eval = np.linspace(0, t_max, 10000)

# Solve the system using solve_ivp
sol = solve_ivp(equations, [0, t_max], y0, t_eval=t_eval, rtol=1e-8)

# Extract the solution
r = sol.y[0]
theta = sol.y[2]

# Convert from polar to Cartesian coordinates for plotting
x = r * np.cos(theta)
y = r * np.sin(theta)

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
ax.set_aspect('equal')

# Plot the central body as a static point
central_body, = ax.plot(0, 0, 'ro', markersize=10)  # Central body (red dot)
moving_body, = ax.plot([], [], 'bo', markersize=5)  # Moving body (blue dot)
line, = ax.plot([], [], lw=0.35, color='k')  # Path of the moving body

# Initialization function for the animation
def init():
    moving_body.set_data([], [])
    line.set_data([], [])
    return moving_body, line

# Update function for the animation
def update(frame):
    moving_body.set_data(x[frame], y[frame])
    line.set_data(x[:frame], y[:frame])
    return moving_body, line

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=2, blit=True)

plt.show()
