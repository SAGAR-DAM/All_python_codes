import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# === User Input ===
x1, y1 = 1.5, 0.5
x2, y2 = 0.5, 0.9
thickness = 0.1
north_side = 'right'  # or 'right'

# === Grid Setup ===
nx, ny = 100, 100
x = np.linspace(-1, 3, nx)
y = np.linspace(-1, 3, ny)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]
phi = np.zeros((ny, nx))

# === Compute Direction & Normal Vectors ===
L = np.hypot(x2 - x1, y2 - y1)
ux, uy = (x2 - x1) / L, (y2 - y1) / L        # Along magnet (long edge)
nx_, ny_ = -uy, ux                          # Normal to long edge

# === Define Magnet Rectangle ===
corner1 = (x1 - ux * L/2 + nx_ * thickness/2, y1 - uy * L/2 + ny_ * thickness/2)
corner2 = (x2 + ux * L/2 + nx_ * thickness/2, y2 + uy * L/2 + ny_ * thickness/2)
corner3 = (x2 + ux * L/2 - nx_ * thickness/2, y2 + uy * L/2 - ny_ * thickness/2)
corner4 = (x1 - ux * L/2 - nx_ * thickness/2, y1 - uy * L/2 - ny_ * thickness/2)

# === Build Magnet Mask ===
polygon = Path([corner1, corner2, corner3, corner4])
points = np.vstack((X.flatten(), Y.flatten())).T
mask = polygon.contains_points(points).reshape((ny, nx))

# === Detect Magnetic Surfaces ===
# Project distance from magnet center along normal direction
xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
dxs = X - xc
dys = Y - yc
dot_normal = dxs * nx_ + dys * ny_

# Positive side (north), Negative side (south)
north_mask = (dot_normal > thickness * 0.4) & mask
south_mask = (dot_normal < -thickness * 0.4) & mask

# === Apply Scalar Potential Boundary Conditions ===
phi[north_mask] = 1.0 if north_side == 'left' else -1.0
phi[south_mask] = -1.0 if north_side == 'left' else 1.0

# === Solve Laplace's Equation ===
for _ in range(1000):
    phi_old = phi.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if mask[j, i]:
                continue
            phi[j, i] = 0.25 * (phi_old[j+1, i] + phi_old[j-1, i] +
                                phi_old[j, i+1] + phi_old[j, i-1])

# === Compute Magnetic Field (B = -∇φ) ===
By, Bx = np.gradient(-phi, dy, dx)

# === Plot Results ===
plt.figure(figsize=(10, 5))
plt.contourf(X, Y, phi, levels=50, cmap='coolwarm')
plt.colorbar(label='Magnetic Scalar Potential φ')
plt.streamplot(X, Y, Bx, By, color='k', density=2)

# Plot bar magnet rectangle
x_rect = [corner1[0], corner2[0], corner3[0], corner4[0], corner1[0]]
y_rect = [corner1[1], corner2[1], corner3[1], corner4[1], corner1[1]]
plt.plot(x_rect, y_rect, 'k-', lw=2, label='Bar Magnet')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Magnetic Field of a Slanted Bar Magnet")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
