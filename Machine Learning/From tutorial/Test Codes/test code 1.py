# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:26:43 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

n0=2
n1=1

w=np.random.uniform(low=-1,high=+1, size=(n1,n0))
b=np.random.uniform(low=-1,high=+1, size=n1)

y_in=np.array([0.1,0.2])

#z=np.dot(w,y_in)+b
#y_out=1/(1+np.exp(-z))


def f(y_in):
    global w,b
    z=np.dot(w,y_in)+b
    y=1/(1+np.exp(-z))
    return(y)

m=50
y_out=np.zeros([m,m])

for i in range(m):
    for j in range(m):
        y1=float(i)/m-0.5
        y2=float(j)/m-0.5
        
        y_in=[y1,y2]
        y_in=np.array(y_in)
        
        y_out[i,j]=f(y_in)
        
print(y_out)

plt.imshow(y_out,extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.show()