# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:02:58 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Define masses and positions of the stationary bodies
m1 = 5.0  # Mass of the first stationary body
m2 = 4.0  # Mass of the second stationary body
pos1 = np.array([-1.0, 0.0])  # Position of the first body
pos2 = np.array([1.0, 0.0])   # Position of the second body

# Gravitational constant
G = 1.0

# Define the gravitational force function
def gravitational_force(r, m):
    return -G * m / np.linalg.norm(r)**3 * r

# Define the equations of motion for the moving body
def equations(t, y):
    x, y, vx, vy = y
    
    # Position and velocity vectors
    r = np.array([x, y])
    v = np.array([vx, vy])

    # Compute forces from the two stationary bodies
    r1 = r - pos1  # Vector from the first body to the moving body
    r2 = r - pos2  # Vector from the second body to the moving body

    F1 = gravitational_force(r1, m1)  # Force from the first body
    F2 = gravitational_force(r2, m2)  # Force from the second body

    # Total acce2eration
    a = (F1 + F2)

    # Return derivatives: dx/dt, dy/dt, dvx/dt, dvy/dt
    return [v[0], v[1], a[0], a[1]]

# Initial conditions
x0, y0 = [0.0, 2.0]   # Initial position of the moving body
vx0, vy0 = [1.2, 0.7]  # Initial velocity of the moving body
y0 = [x0, y0, vx0, vy0]

# Time span for the simulation
t_max = 50
t_eval = np.linspace(0, t_max, 10000)

# Solve the system using solve_ivp
sol = solve_ivp(equations, [0, t_max], y0, t_eval=t_eval, rtol=1e-8)

# Extract the solution
x = sol.y[0]
y = sol.y[1]

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
ax.set_aspect('equal')

# Plot the stationary bodies
stationary_body1, = ax.plot(pos1[0], pos1[1], 'ro', markersize=2, label="Body 1")
stationary_body2, = ax.plot(pos2[0], pos2[1], 'go', markersize=2, label="Body 2")
moving_body, = ax.plot([], [], 'bo', markersize=1, label="Moving Body")
line, = ax.plot([], [], lw=0.2, color='k')  # Path of the moving body

# Initialization function for the animation
def init():
    moving_body.set_data([], [])
    line.set_data([], [])
    return moving_body, line

# Update function for the animation
def update(frame):
    # Ensure frame is within bounds
    frame = min(frame, len(x) - 1)
    moving_body.set_data([x[frame]], [y[frame]])  # Provide lists or arrays
    line.set_data(x[:frame], y[:frame])
    return moving_body, line

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=2, blit=True)

plt.legend()
plt.show()
