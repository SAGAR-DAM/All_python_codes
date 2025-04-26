import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where the edited files are located
directory = "/home/varun/Documents/MEGA/tifr/activities/06 Doppler Data: Sagar/reference/edited/-10ps"

# Create a folder to save the plots if it doesn't exist
output_directory = os.path.join(directory, "plots")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def plot_and_save_spectral_data(file_path):
    # Load the data from the file, skipping the first 4 rows
    data = np.loadtxt(file_path, skiprows=4)

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

    # Save the plot as a PNG file
    output_file = os.path.join(output_directory, os.path.basename(file_path).replace(".txt", ".png"))
    plt.savefig(output_file)
    plt.close()

# Iterate over all the .txt files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        plot_and_save_spectral_data(file_path)

print("Plots saved in the 'plots' subfolder.")

