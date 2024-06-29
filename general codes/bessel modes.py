import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

# Define the range of x values
x_values = np.linspace(0, 10, 200)

# Plot the Bessel functions for orders 0 to 20
n = np.linspace(0,20,41)
for i in range(len(n)):
    bessel_values = jn(n[i], x_values)
    plt.plot(x_values, bessel_values, label=f'J_{n[i]}(x)')

# Add labels, title, legend, and grid
plt.xlabel('x')
plt.ylabel('J_n(x)')
plt.title('Bessel Functions of Orders 0 to 20')
#plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jn_zeros, jn

# Define the range of radial (r) and angular (theta) values
r_values = np.linspace(0, 20, 1500)  # Define the range of r from 0 to 1
theta_values = np.linspace(0, 2 * np.pi, 500)  # Define the range of theta from 0 to 2pi

# Create a meshgrid for r and theta
r, theta = np.meshgrid(r_values, theta_values)

# Plot the modes for orders 0 to 20
for i in range(len(n)):
    # Compute the Bessel function values for the radial direction
    radial_values = jn(n[i], r)

    # Initialize the figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(r * np.cos(theta), r * np.sin(theta), radial_values, cmap='jet', rcount=100, ccount=400)

    # Set labels and title
    ax.set_title(f'Mode {n[i]}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Amplitude')

    plt.show()
