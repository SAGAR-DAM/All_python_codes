import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

# Directory where the edited files are located
directory = "/home/varun/Documents/MEGA/tifr/activities/06 Doppler Data: Sagar/data/edited/25ps"

# Create a folder to save the plots if it doesn't exist
output_directory = os.path.join(directory, "fits")
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Gaussian function for fitting
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Function to calculate R^2
def calculate_r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Function for baseline correction using polynomial fitting
def baseline_correction(x, y, degree=3):
    # Fit a polynomial to the baseline
    p = Polynomial.fit(x, y, degree)
    baseline = p(x)
    y_corrected = y - baseline
    return y_corrected, baseline

def plot_and_save_spectral_data(file_path):
    # Load the data from the file, skipping the first 4 rows
    data = np.loadtxt(file_path, skiprows=4)

    # Separate the columns into x and y
    x = data[:, 0]  # First column: wavelength (x)
    y = data[:, 1]  # Second column: intensity (y)

    # Apply baseline correction
    y_corrected, baseline = baseline_correction(x, y)

    # Plot the original data with the baseline
    plt.figure()
    plt.plot(x, y, 'k--', linewidth=1, label="Original Data")
    plt.plot(x, baseline, 'b-', linewidth=1, label="Baseline")

    # Plot the baseline-corrected data
    plt.plot(x, y_corrected, 'g-', linewidth=1, label="Corrected Data")

    # Perform Gaussian fit on the corrected data
    mean_est = x[np.argmax(y_corrected)]  # Initial guess for the mean (peak)
    amplitude_est = max(y_corrected)      # Initial guess for the amplitude
    stddev_est = np.std(x)                # Initial guess for the standard deviation

    popt, _ = curve_fit(gaussian, x, y_corrected, p0=[amplitude_est, mean_est, stddev_est])

    # Plot the Gaussian fit (no label)
    y_fit = gaussian(x, *popt)
    plt.plot(x, y_fit, 'r-')

    # Calculate FWHM and R^2
    amplitude, mean, stddev = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * stddev  # FWHM formula for Gaussian
    r_squared = calculate_r_squared(y_corrected, y_fit)

    # Add the peak, FWHM, and R^2 to the plot as a label
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.text(0.05, 0.95, f"Peak: {mean:.2f} nm\nFWHM: {fwhm:.2f} nm\nR²: {r_squared:.4f}", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    # Save the plot
    output_file = os.path.join(output_directory, os.path.basename(file_path).replace(".txt", ".png"))
    plt.savefig(output_file)
    plt.close()

# Iterate over all the .txt files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        plot_and_save_spectral_data(file_path)

print("Plots with Gaussian fits and baseline correction saved in the 'plots' subfolder.")

