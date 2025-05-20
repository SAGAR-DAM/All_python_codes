import numpy as np
from skimage import measure
from mayavi import mlab





# Create a disk (used to cap the top and bottom of pillars)
def create_disk(radius=10, z_height=0, resolution=100, color=(0.6, 0.6, 0.6), x0=0, y0=0):
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta) + x0
    y = radius * np.sin(theta) + y0
    z = np.full_like(x, z_height)

    # Add center point
    x = np.append(x, x0)
    y = np.append(y, y0)
    z = np.append(z, z_height)

    # Create triangle fan
    n = len(theta)
    triangles = [[i, (i + 1) % n, n] for i in range(n)]
    triangles = np.array(triangles)

    return mlab.triangular_mesh(x, y, z, triangles, color=color)

# Create a single solid pillar (side + top/bottom disks)
def create_pillar(x0, y0, radius=0.3, height=2.0, resolution=30):
    # Side
    phi, z = np.mgrid[0:2*np.pi:resolution*1j, 0:height:2j]
    x = radius * np.cos(phi) + x0
    y = radius * np.sin(phi) + y0
    mlab.mesh(x, y, z, colormap="copper")

    # Caps
    create_disk(radius, z_height=0, color=(0.7, 0.7, 0.0), x0=x0, y0=y0)
    create_disk(radius, z_height=height, color=(1, 1, 0.0), x0=x0, y0=y0)


# Pillar parameters
pillar_radius = 0.037
pillar_height = 1.0
spacing = 0.2  # Grid spacing
grid_extent = 1.8  # From -grid_extent to +grid_extent in both x and y

# Generate grid and place pillars based on the inequality
x_vals = np.arange(-grid_extent, grid_extent + spacing, spacing)
y_vals = np.arange(-grid_extent, grid_extent + spacing, spacing)

for x in x_vals:
    for y in y_vals:
        # Evaluate the inequality: (y² - 50x²)(x² - 50y²) ≤ 100
        expr = (y**2 - 50*x**2) * (x**2 - 50*y**2)
        if expr <= 100:
            create_pillar(x, y, radius=pillar_radius, height=pillar_height)


# Parameters
x_range = (-2, 2)
y_range = (-2, 2)
thickness = 0.2
resolution = 400

# Create 3D grid (X, Y, Z) for voxel space
x = np.linspace(*x_range, resolution)
y = np.linspace(*y_range, resolution)
z = np.linspace(0, thickness, 4)  # few layers for thickness

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Evaluate inequality in 2D, expand to 3D
ineq_2d = ((Y[:, :, 0]**2 - 50 * X[:, :, 0]**2) * (X[:, :, 0]**2 - 50 * Y[:, :, 0]**2)) <= 100
volume = np.repeat(ineq_2d[:, :, np.newaxis], len(z), axis=2).astype(np.uint8)

# Use marching cubes to create a solid mesh
verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(
    (x_range[1]-x_range[0])/resolution,
    (y_range[1]-y_range[0])/resolution,
    thickness / (len(z)-1)
))

# Shift vertices to real coordinates
verts[:, 0] += x_range[0]
verts[:, 1] += y_range[0]

# Plot solid base
mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, color=(0.6, 0.6, 0.6))

# Define domain
X, Y = np.mgrid[-2:2:201j, -2:2:201j]
mask = ((Y**2 - 50 * X**2) * (X**2 - 50 * Y**2)) <= 100

# Plate thickness
thickness = 0.2

# Z values for top and bottom
Z_top = np.where(mask, thickness, np.nan)
Z_bot = np.where(mask, 0, np.nan)

mlab.surf(X, Y, Z_bot, color=(0.6, 0.6, 0.6))
mlab.surf(X, Y, Z_top, color=(0.5, 0.5, 0.5))
mlab.view(azimuth=45, elevation=75, distance=10)
mlab.show()
