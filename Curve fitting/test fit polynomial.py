# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:33:55 2024

@author: mrsag
"""
import numpy as np
import matplotlib.pyplot as plt
from Curve_fitting_with_scipy import polynomial_fit as pft

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 300  # highres display


# Extract features
age = np.random.uniform(low=-10, high=10, size = 500)

factorial = lambda n: 1 if n == 0 else n * factorial(n - 1)

def make_random_polynomial(x, degree):
    """
    Generate a polynomial with random coefficients that decrease in magnitude as the order increases.
    
    Parameters:
    x (array-like): Input values for which to compute the polynomial.
    degree (int): Degree of the polynomial.
    
    Returns:
    array-like: Values of the polynomial at the input values x.
    """
    # Generate random coefficients decreasing in magnitude
    #coefficients = [np.random.uniform(-1, 1) / (factorial(i) + 1) for i in range(degree + 1)]
    coefficients = [-0.4768900743259392, 0.09598512252467273, -0.26734940092458515, -0.11682443034010721, -0.022274722649203323, -0.006105938345754179, 0.0002278723835546729, 7.68978952426078e-05]
   
    # Print the coefficients for reference
    print("Generated coefficients:\n", coefficients)
    
    # Compute the polynomial values
    polynomial_values = sum(coeff * (x ** i) for i, coeff in enumerate(coefficients))
    
    return polynomial_values
    
# Target variable
#f = 150-50*age-100*np.sin(age)+2*age**2+age**3+100*np.random.uniform(low=-1, high=1, size=len(age))
#f = 100-10*age+20*age**2+3*age**3+100*np.random.uniform(low=-1, high=1, size=len(age))
f = np.sin(age)+0.2*np.random.uniform(low=-1, high=1, size=len(age))

degree = 15

#f = make_random_polynomial(age, degree) 
f += max(abs(f))/20*np.random.uniform(low=-1, high=1, size=len(age))
f_predict, coefficients = pft.polynomial_fit(age, f, degree)

print("\nPredicted coefficients:")
print(coefficients)

sorted_indices = np.argsort(age)
age = age[sorted_indices]
f = f[sorted_indices]
f_predict = f_predict[sorted_indices]

plt.scatter(age,f, label='data')
plt.plot(age,f_predict,'r-', label='fit')
plt.plot(age,f-f_predict,'k-', label='deviation')
plt.legend()
plt.show()

error = np.std(f-f_predict)
print(f"\nError: {error}")

