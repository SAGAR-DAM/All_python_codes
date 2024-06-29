# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:13:40 2023

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x=np.linspace(-7,7,141)
y=np.linspace(-7,7,141)

X,Y=np.meshgrid(x,y)

def f(x,y):
    xlim=100
    ylim=100
    xrep=0.4
    yrep=0.4
    value=0
    for i in range(1,xlim):
        for j in range(1,ylim):
            term1=np.cos(i*x/xrep)*np.cos(j*y/yrep)*((-1)**(i+j))/(i**2*j**2)
            term2=np.cos(i*x/xrep)*np.sin(j*y/yrep)/(1+(i**2*j**2))
            term3=np.sin(i*x/xrep)*np.cos(j*y/yrep)/(1+(i**2*j**2))
            term4=np.sin(i*x/xrep)*np.sin(j*y/yrep)*((-1)**(i+j))/(i**2*j**2)
            value+=term1 + term2 + term3 + term4
    return(2*np.pi**2/3-value)

Z=f(X,Y)

plt.set_cmap('hot')
plt.imshow(Z)
plt.show()
'''
# creating the visualization
fig = plt.figure()
wf = plt.axes(projection ='3d')
wf.plot_wireframe(X, Y, Z, color ='green')
 
# displaying the visualization
plt.show()'''