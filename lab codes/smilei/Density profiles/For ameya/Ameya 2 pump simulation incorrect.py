# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:49:49 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi'] = 500  # highres display

l0=1
res=0.1

X_dim=200
Y_dim=64

x=np.arange(0,(X_dim+res)*l0,res)
y=np.arange(0,(Y_dim+res)*l0,res)

X,Y=np.meshgrid(x,y)

def density(x,y):
    if(x<=50*l0):
        if((25-3)*l0<=x<=(25+3)*l0):
            if(y<=12*l0):
                return(0.999)
            else:
                return(0.999*np.exp(-0.05*(y-12*l0)))
            
        else:
            if(y<=12*l0):
                return(0.999)
            else:
                return(0.999*np.exp(-0.05*(y-12*l0))*np.exp(-0.08*(abs(x-25*l0)-3*l0)))
        
    elif(x>50*l0):
        if((75-6)*l0<=x<=(75+6)*l0):
            if(y<=12*l0):
                return(0.999)
            else:
                return(0.999*np.exp(-0.05*(y-12*l0)))
            
        else:
            if(y<=12*l0):
                return(0.999)
            else:
                return(0.999*np.exp(-0.05*(y-12*l0))*np.exp(-0.08*(abs(x-75*l0)-6*l0)))

    else:
        return(0)
    
density=np.vectorize(density)

dense=density(X,Y)

plt.imshow(dense,cmap="hot",extent=[0,X_dim,0,Y_dim])
plt.title("Density map of electron", fontname="Times New Roman", fontsize=10)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()

