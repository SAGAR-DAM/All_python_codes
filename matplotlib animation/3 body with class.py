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
m2 = 3.0  # Mass of the second stationary body
pos1 = np.array([-1.0, 0.0])  # Position of the first body
pos2 = np.array([1.0, 0.0])   # Position of the second body

# Gravitational constant
G = 1.0

# Define the gravitational force function
def gravitational_force(r, m):
    return -G * m / np.linalg.norm(r)**3 * r

# Define the MovingBody class
class MovingBody:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.trajectory = []

    def append_trajectory(self):
        self.trajectory.append(self.position.copy())

# Define the equations of motion for a single moving body
def equations(t, y):
    x, y, vx, vy = y
    r = np.array([x, y])

    # Compute forces from the two stationary bodies
    r1 = r - pos1  # Vector from the first body to the moving body
    r2 = r - pos2  # Vector from the second body to the moving body

    F1 = gravitational_force(r1, m1)  # Force from the first body
    F2 = gravitational_force(r2, m2)  # Force from the second body

    # Total acceleration
    a = F1 + F2

    # Return derivatives: dx/dt, dy/dt, dvx/dt, dvy/dt
    return [vx, vy, a[0], a[1]]

# Initialize a single moving body
body = MovingBody(x=0.0, y=2.0, vx=0.2, vy=0.7)

# Initial conditions for the single body
y0 = np.hstack((body.position, body.velocity))

# Time span for the simulation
t_max = 50
t_eval = np.linspace(0, t_max, 1000)

# Solve the system using solve_ivp
sol = solve_ivp(equations, [0, t_max], y0, t_eval=t_eval, rtol=1e-8)

# Extract the solution for the body
x = sol.y[0]
y = sol.y[1]

# Set up the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
ax.set_aspect('equal')

# Plot the stationary bodies
ax.plot(pos1[0], pos1[1], 'ro', markersize=10, label="Body 1")
ax.plot(pos2[0], pos2[1], 'go', markersize=10, label="Body 2")

# Create a line and point for the moving body
line, = ax.plot([], [], lw=0.5, color='blue')
moving_body, = ax.plot([], [], 'bo', markersize=5, label="Moving Body")

# Initialization function for the animation
def init():
    line.set_data([], [])
    moving_body.set_data([], [])
    return line, moving_body

# Update function for the animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    moving_body.set_data([x[frame]], [y[frame]])
    return line, moving_body

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=20, blit=True)

plt.legend()
plt.show()
