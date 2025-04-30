import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 500  # highres display

# Constants
cm = 1e-2
mm = 1e-3
qe = -1.6e-19
me = 9.11e-31
keV_to_Joule = 1.6e-16
qp = -1*qe
mH = 1836*me

# Input parameters:
Ex = 12000    # in V/m
Ey = 10000    # in V/m
Ez = 0    # in V/m

Bx = 1  # in Tesla
By = 1  # in Tesla
Bz = 0  # in Tesla

vx0, vy0, vz0 = 0, 0, 1
mod_v = np.sqrt(vx0**2 + vy0**2 + vz0**2)
vx0, vy0, vz0 = np.array([vx0, vy0, vz0]) / mod_v

class particle():
    def __init__(self, vx, vy, vz, q, m):
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.q = q
        self.m = m
        self.v = np.sqrt(vx**2 + vy**2 + vz**2)
        self.Energy = m * self.v**2 / 2
        self.pos = [[], [], []]

    def propagator(self):
        global t
        global dt
        global Ex, Ey, Ez, Bx, By, Bz

        for i in range(1, len(t)):
            Fx = self.q * (Ex + (self.vy * Bz - self.vz * By))
            Fy = self.q * (Ey + (self.vz * Bx - self.vx * Bz))
            Fz = self.q * (Ez + (self.vx * By - self.vy * Bz))

            ax, ay, az = Fx / self.m, Fy / self.m, Fz / self.m

            self.x += self.vx * dt + ax * dt**2 / 2
            self.y += self.vy * dt + ay * dt**2 / 2
            self.z += self.vz * dt + az * dt**2 / 2

            self.vx += ax * dt
            self.vy += ay * dt
            self.vz += az * dt

            self.v = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
            self.E = self.m * self.v**2 / 2

            self.pos[0].append(self.x)
            self.pos[1].append(self.y)
            self.pos[2].append(self.z)

        return np.array(self.pos)

class proton(particle):
    def __init__(self, vx, vy, vz):
        super().__init__(vx, vy, vz, q=qp, m=mH)

# Initial energy and velocity
E = 1 * keV_to_Joule
m = mH
v = np.sqrt(2 * E / m)
vx = vx0 * v
vy = vy0 * v
vz = vz0 * v
p = proton(vx=vx, vy=vy, vz=vz)

t_max = 100 * cm / v
time_resolution = 20000
t = np.linspace(0, t_max, time_resolution)
dt = np.diff(t)[0]
num_steps = len(t)

# Propagate the particle's trajectory
pos = p.propagator()
pos = pos / mm  # Convert position to mm

xs = pos[0]
ys = pos[1]
zs = pos[2]

# Ensure that t has the same number of elements as the position arrays
t = t[:len(xs)]  # Trim the extra time step

# Animation setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set tight limits
ax.set_xlim(np.min(xs), np.max(xs))
ax.set_ylim(np.min(ys), np.max(ys))
ax.set_zlim(np.min(zs), np.max(zs))

# Set equal box aspect
ax.set_box_aspect([1, 1, 1])

# Set axis labels
ax.set_xlabel('X (mm)', fontsize=5)
ax.set_ylabel('Y (mm)', fontsize=5)
ax.set_zlabel('Z (mm)', fontsize=5)

# Make ticks smaller
ax.tick_params(axis='both', which='major', labelsize=5)
ax.tick_params(axis='both', which='minor', labelsize=3)

# Line plot to update the trajectory
line, = ax.plot([], [], [], color='b', lw=0.25)

# Create colorbar for time
sc = ax.scatter(xs, ys, zs, c=t, cmap='magma', s=0.1)
# cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label('Time (s)', fontsize=10)

# Initialize the plot
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Update the plot for each frame
def update(frame):
    line.set_data(xs[:frame], ys[:frame])
    line.set_3d_properties(zs[:frame])
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=len(xs), init_func=init, blit=False, interval=5)

plt.show()
