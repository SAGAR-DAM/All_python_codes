import numpy as np
import matplotlib.pyplot as plt

# Grid size and spatial domain
nx, ny = 100, 100
x = np.linspace(0-1, 2+1, nx)
y = np.linspace(0-1, 2+1, ny)
X, Y = np.meshgrid(x, y)

# Magnetic scalar potential φ (B = -∇φ in free space)
phi = np.zeros((ny, nx))

# Define bar magnet corners (in real space units)
magnet_coords = {
    "x1": 1.0, "x2": 1.1,
    "y1": 0.5, "y2": 1.5
}

# Map physical coordinates to grid indices
ix1 = np.searchsorted(x, magnet_coords["x1"])
ix2 = np.searchsorted(x, magnet_coords["x2"])
iy1 = np.searchsorted(y, magnet_coords["y1"])
iy2 = np.searchsorted(y, magnet_coords["y2"])

# Set fixed scalar potential boundary conditions for the magnet
phi[iy1:iy2, ix1] = 100.0    # North pole (left edge)
phi[iy1:iy2, ix2] = -100.0   # South pole (right edge)

# Iterative Laplace solver (Gauss-Seidel method)

for _ in range(500):
    phi_old = phi.copy()
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if ix1 <= i <= ix2 and iy1 <= j <= iy2:
                continue  # Skip inside the magnet (BC already defined)
            phi[j, i] = 0.25 * (phi_old[j+1, i] + phi_old[j-1, i] +
                                phi_old[j, i+1] + phi_old[j, i-1])

# Calculate magnetic field B = -∇φ
By, Bx = np.gradient(-phi, y, x)
B_mag = np.sqrt(Bx**2 + By**2)

# Plot scalar potential and magnetic field lines
# plt.figure(figsize=(10, 5))
plt.contourf(X, Y, phi, levels=50, cmap='coolwarm')
plt.colorbar(label='Magnetic Scalar Potential φ')
plt.streamplot(X, Y, Bx, By, color='k', density=3)



# Draw bar magnet rectangle
plt.plot([x[ix1], x[ix2], x[ix2], x[ix1], x[ix1]],
         [y[iy1], y[iy1], y[iy2], y[iy2], y[iy1]], 'k-', lw=2, label='Bar Magnet')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Magnetic Field Simulation from Bar Magnet")
plt.axis('equal')
# plt.grid(True)
# plt.legend()
plt.show()
