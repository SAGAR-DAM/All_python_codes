import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Constants
G = 1.0  # Gravitational constant
m2 = 1.0  # Mass of the orbiting body
m1_initial = 5.0  # Initial mass of the central body
m1_rate = 0.05  # Rate of change of the central mass (per unit time)

# Define the time-varying mass of the central body
def m1(t):
    return m1_initial + m1_rate * t

# Define the central force F(r, t)
def F(r,t):
    k = 1.0  # strength of the force
    # grav_force = -k*m1(t)*m2/r**2
    # perturbed_grav_force = -k*m1(t)*m2/r**2 - 2*k*m1(t)*m2/r
    # inverse_r_force = -k*m1(t)*m2/r
    # exponential_force = -k*m1(t)*m2*np.exp(-r)
    # log_force = -k*m1(t)*m2/(1+np.log(r))
    # const_force = -k
    # spring_force = -k*m1(t)*m2*r
    strange_force = -k*m1(t)*m2*r**2
    return strange_force

# Define the equations of motion for the system
def equations(t, y):
    r, r_dot, theta = y
    theta_dot = h / r**2  # Angular velocity from conservation of angular momentum
    r_ddot = F(r, t) / m2 + (h**2 / r**3)  # Radial acceleration
    return [r_dot, r_ddot, theta_dot]

# Initial conditions
r0 = 2.0  # Initial radial distance
r_dot0 = 0.2  # Initial radial velocity
theta0 = 0.0  # Initial angle
y0 = [r0, r_dot0, theta0]

# Angular momentum constant (computed from initial conditions)
h = 4 #r0**2 * (10.0 / r0**2)  # Assume initial angular velocity of 1.0

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
fig.patch.set_facecolor('black')  # Set the figure background to black
ax.set_facecolor('black')        # Set the axis background to black

limx = max(abs(x)) * 1.1
limy = max(abs(y)) * 1.1
ax.set_xlim((-limx, limx))
ax.set_ylim((-limy, limy))
ax.set_aspect('equal')

# Adjust axis labels and tick colors for visibility
ax.tick_params(axis='x', colors='white')  # Set x-axis tick color to white
ax.tick_params(axis='y', colors='white')  # Set y-axis tick color to white
for spine in ax.spines.values():
    spine.set_edgecolor('white')  # Set the border of the plot to white

# Plot the central body as a static point
central_body, = ax.plot(0, 0, 'ro', markersize=10)  # Central body (red dot)
moving_body, = ax.plot([], [], 'yo', markersize=5)  # Moving body (blue dot)
line, = ax.plot([], [], lw=0.35, color='white')     # Path of the moving body (white line)

# Initialization function for the animation
def init():
    moving_body.set_data([], [])
    line.set_data([], [])
    return moving_body, line

# Update function for the animation
def update(frame):
    frame = min(frame, len(x) - 1)  # Ensure frame is within bounds
    moving_body.set_data([x[frame]], [y[frame]])  # Provide lists or arrays
    line.set_data(x[:frame], y[:frame])
    return moving_body, line

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=2, blit=True)
# ani.save("D:\\big files from python\\animations\\mass_change_central_force.gif", writer='pillow', fps=30)
plt.show()
