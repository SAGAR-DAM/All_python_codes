import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 2000  # Number of velocity points
v_min = 0.6      # Minimum velocity (starting point)
v_max = 0.99999  # Maximum velocity (asymptotic limit)

# Create exponentially spaced values
x = np.linspace(0, 1, num_points)  # Uniform spacing from 0 to 1
v_array = v_max - (v_max - v_min) * np.exp(-4 * x)  # Exponential decay

# Round for readability
# v_array = np.round(v_array, 6)

print(v_array)


plt.hist(v_array,bins=30)
plt.show()