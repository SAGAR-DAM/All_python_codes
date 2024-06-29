import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the parabolic function
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def parabolic_fit(x, y):
    # Fit the data to the parabolic function
    params, covariance = curve_fit(parabola, x, y)

    # Extracting the coefficients
    a, b, c = params

    return a, b, c

# Example data
x_data = np.linspace(-10,10,201)
y_data = parabola(x_data,-0.5,-1,9)
y_data+=np.random.uniform(size=len(x_data),low=-4,high=4)

# Get parabolic fit coefficients
a, b, c = parabolic_fit(x_data, y_data)

# Generate fitted curve using the coefficients
fitted_curve = lambda x: a * x**2 + b * x + c

# Plotting the original data and the fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_data, fitted_curve(x_data), color='red', label='Parabolic Fit')
plt.title('Parabolic Fit of Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
