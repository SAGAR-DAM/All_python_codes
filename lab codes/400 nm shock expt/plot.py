import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where the edited files are located
edited_directory = "/home/varun/Documents/MEGA/tifr/activities/06 Doppler Data: Sagar/data/edited/10ps"

# Specify the exact file you want to plot
file_name = "s_01598.txt"
file_path = os.path.join(edited_directory, file_name)

def plot_spectral_data(file_path):
    # Load the data from the file
    data = np.loadtxt(file_path, skiprows = 4)

    # Separate the columns into x and y
    x = data[:, 0]  # First column: wavelength (x)
    y = data[:, 1]  # Second column: intensity (y)

    # Plot the data
    plt.figure()
    plt.plot(x, y, label=os.path.basename(file_path))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title(f"Spectral Data - {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Check if the file exists before plotting
if os.path.exists(file_path):
    plot_spectral_data(file_path)
else:
    print(f"File {file_name} not found in {edited_directory}.")

