# Imprting necessary libraries
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# Setting our linspace limit:
n = 256 

# Creating X and Y arrays:
x = np.linspace(-3., 3., n) 
y = np.linspace(-3., 3., n) 

# Converting arrays into meshgrid:
X, Y = np.meshgrid(x, y) 

def f(x,y):
    z=1*(X<0.4*np.sin(10*Y)+1.5)
    return(z)
# Computing Z: 2D Scalar Field
Z = f(X,Y)


# Plotting our 2D Scalar Field using pcolormesh()
#plt.figure(figsize=(40,40))
plt.imshow(Z)
plt.axis("off")
plt.colorbar()
plt.show()