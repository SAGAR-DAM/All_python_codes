import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 100  # highres display
# Define the function for which you want to calculate Fourier coefficients
def g(x):
    #return((x**3*(abs(x)<2)+np.log(abs(x)+1)))
    return(abs(x)<1)

@np.vectorize
def f(x):
    if abs(x)>np.pi:
        if(x>0):
            return(f(x-2*np.pi))
        else:
            return(f(x+2*np.pi))
        
    else:
        return g(x)

# Define the range of integration
a = -np.pi
b = np.pi

# Define the number of terms in the Fourier series
n_terms = 50

# Function to calculate the Fourier coefficients for a given function and integer n
def fourier_coefficients(n):
    # Define the integrands for Fourier coefficients a_n and b_n
    def integrand_cos(x):
        return f(x) * np.cos(n * x)

    def integrand_sin(x):
        return f(x) * np.sin(n * x)

    # Perform numerical integration using quad
    a_n, _ = quad(integrand_cos, a, b)
    b_n, _ = quad(integrand_sin, a, b)

    # Normalize coefficients
    a_n *= 1 / np.pi
    b_n *= 1 / np.pi

    return a_n, b_n

# Calculate Fourier coefficients for n_terms
coefficients = [fourier_coefficients(n) for n in range(0, n_terms + 1)]

# Print the Fourier coefficients
for n, (a_n, b_n) in enumerate(coefficients, 0):
    print(f"a_{n}: {a_n*(a_n>1e-5):.5f}, b_{n}: {b_n*(b_n>1e-5):.5f}")
    
    
def fourier_calculator(x,coeff):
    y = coeff[0][0]/2
    for i in range(1,len(coeff)):
        y += coeff[i][0]*np.cos(i*x)+coeff[i][1]*np.sin(i*x)
        
    return(y)

x = np.linspace(a,b,1000)
z = fourier_calculator(x, coefficients)

plt.plot(x,f(x),'ro-',label='direct', markersize = 1, lw=0.5)
plt.plot(x,z,'k-', lw=0.5, label='from FT')
plt.legend()
plt.xlabel("x"+f"\nRelative Error: {np.std(f(x)-z)/max(abs(z))}")
plt.ylabel("y")
plt.show()
