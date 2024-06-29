# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:18:56 2023

@author: sagar
"""
import numpy as np
import matplotlib.pyplot as plt
#from numba import njit

# Different layer dimensions
layerdims=[20,20,20,20,1] 

#defining the transfer matrices of different layers
# different layer w and b matrices
w=[]
b=[]

for i in range(len(layerdims)):
    if(i==0):
        n0=2
    else:
        n0=layerdims[i-1]
        
    n1=layerdims[i]

    w1=np.random.uniform(low=-3,high=+3, size=(n1,n0))
    b1=np.random.uniform(low=-1,high=+1, size=n1)
    
    w.append(w1)
    b.append(b1)

# defining the nonlinear operation... Here the sigmoid function.
def sigmoid(x):
    y=1/(1+np.exp(-5*x))
    return(y)



# Going from the input to final output.
# This function takes a 2 d vector (y1,y2) goes througth all the layers and gives final output
def f(y_in):
    global layerdims,w,b
    y=y_in
    for i in range(len(layerdims)):
        transfer=w[i]
        offset=b[i]
        z=np.dot(transfer,y)+offset  # doing the layer input to output linear transform
        z=np.array(z)
        y_out=sigmoid(z)
        y=y_out
        
    return(y)

m=200
y_out=np.zeros([m,m])

for i in range(m):
    for j in range(m):
        y1=2*float(i)*1/m-1
        y2=2*float(j)*1/m-1
        y_in=[y1,y2]
        y_in=np.array(y_in)        
        y_out[i,j]=f(y_in)

plt.imshow(y_out,extent=(-1,1,-1,1))
plt.colorbar()
plt.show()