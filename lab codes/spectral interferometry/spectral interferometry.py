import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams["figure.dpi"]=300

def generate_spectral_pulse(frequencies, frequency_peak, width):
    return np.exp(-0.5 * ((frequencies - frequency_peak) / width)**2)

def plot_spectral_interference_with_delay(pulse1, pulse2, frequencies, time_delay):
    spectrum1 = pulse1(frequencies, frequency_peak=10-0.1, width=0.1)
    spectrum2 = pulse2(frequencies, frequency_peak=10+0.1, width=0.1)

    # Introduce time delay
    phase_factor = np.exp(-2j * np.pi * frequencies * time_delay)

    # Combine spectra with time delay
    combined_spectrum = spectrum1 * phase_factor + spectrum2

    # Plot individual spectra
    plt.subplot(3, 1, 1)
    plt.plot(frequencies, np.abs(spectrum1), label='Pulse 1')
    plt.title('Spectrum of Pulse 1')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(frequencies, np.abs(spectrum2), label='Pulse 2')
    plt.title('Spectrum of Pulse 2')
    plt.legend()

    # Plot interference in the spectral domain after time delay
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, np.abs(combined_spectrum), label='FDI')
    plt.title('Spectral Interference with Time Delay')
    plt.xlabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Set the frequency values
frequencies = np.linspace(9.5, 10.5, 1000)

# Call the function to plot spectral interference with time delay
plot_spectral_interference_with_delay(generate_spectral_pulse, generate_spectral_pulse, frequencies, time_delay=50.0)
