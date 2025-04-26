import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Gaussian function definition
def gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        ctr = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr) ** 2) / (2 * wid ** 2))
    return y

# Directory where the edited files are located
edited_directory = "/home/varun/Documents/MEGA/tifr/activities/06 Doppler Data: Sagar/data/edited/0ps"

# Threshold for peak detection
prominence_threshold = 100  # Adjust based on noise level

# Loop through each file in the directory
for file_name in os.listdir(edited_directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(edited_directory, file_name)

        # Load the data (assuming skipping 4 rows as before)
        data = np.loadtxt(file_path, skiprows=4)
        x = data[:, 0]  # Wavelength (nm)
        y = data[:, 1]  # Intensity

        # Detect peaks based on the prominence threshold
        peaks, _ = find_peaks(y, prominence=prominence_threshold)

        if len(peaks) == 0:
            print(f"No peaks found in {file_name}. Skipping.")
            continue

        # Prepare initial guesses for Gaussian fitting
        initial_guess = []
        for peak in peaks:
            initial_guess.append(y[peak])      # Amplitude guess
            initial_guess.append(x[peak])      # Center guess
            initial_guess.append(1.0)          # Width guess (assuming small)

        # Perform the Gaussian fit
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
        except RuntimeError:
            print(f"Fit failed for {file_name}. Skipping.")
            continue

        # Calculate R^2 for the fit
        y_fit = gaussian(x, *popt)
        residuals = y - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plotting
        plt.figure()
        plt.plot(x, y, 'k--', linewidth=0.8)  # Original data as dashed black line
        plt.plot(x, y_fit, 'r-', linewidth=2)  # Gaussian fit as red line

        # Label the plot with peak positions and FWHM
        labels = []
        for i in range(0, len(popt), 3):
            amp = popt[i]
            ctr = popt[i+1]
            wid = popt[i+2]
            fwhm = 2 * np.sqrt(2 * np.log(2)) * wid  # FWHM calculation
            labels.append(f"Peak at {ctr:.2f} nm, FWHM: {fwhm:.2f} nm")

        label_text = ", ".join(labels)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title(f"{file_name} (R² = {r_squared:.4f})")
        plt.legend([label_text], loc="best", frameon=False)
        plt.grid(True)

        # Save the plot
        output_path = os.path.join(edited_directory, f"{file_name}_fit.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Plot saved for {file_name} with R² = {r_squared:.4f}")

