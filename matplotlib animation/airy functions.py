import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display
# Define the range of x values
x_values = np.linspace(-20, 2, 500)

# Compute the Airy functions
ai_values, aip_values, bi_values, bip_values = airy(x_values)

# Plot the solutions
#plt.figure(figsize=(10, 6))

plt.plot(x_values, ai_values, 'b-',label='Ai(x)',markersize=1.5)
plt.plot(x_values, bi_values, 'r-',label='Bi(x)',markersize=1.5)
plt.plot(x_values, aip_values, label="Ai'(x)", color='green')
plt.plot(x_values, bip_values, label="Bi'(x)", color='magenta')

plt.title('Airy Functions')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()


for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()