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
Y_dim=100

x=np.arange(0,(X_dim+res)*l0,res)
y=np.arange(0,(Y_dim+res)*l0,res)

X,Y=np.meshgrid(x,y)

def sawtooth(amplitude, period,t):
    t = t % period  # Ensure t is within one period
    return(t*amplitude/period)

def density(x,y):

    if((25-3)*l0<=x<=(25+3)*l0):
        if(y<=12*l0+sawtooth(10*l0,5*l0,x)):
            return(0.999)
        else:
            return(0.999*np.exp(-0.05*(y-12*l0-sawtooth(10*l0,5*l0,x))))
    if((75-6)*l0<=x<=(75+6)*l0):
        if(y<=12*l0+sawtooth(10*l0,5*l0,x)):
            return(0.999)
        else:
            return(0.999*np.exp(-0.05*(y-12*l0-sawtooth(10*l0,5*l0,x))))
        
    else:
        if(y<=12*l0+sawtooth(10*l0,5*l0,x)):
            return(0.999)
        else:
            return(0.999*np.exp(-0.05*(y-12*l0-sawtooth(10*l0,5*l0,x)))*(np.exp(-0.08*(abs(x-25*l0)-3*l0))+np.exp(-0.08*(abs(x-75*l0)-6*l0))))

    
density=np.vectorize(density)

dense=density(X,Y)

plt.imshow(dense,cmap="hot",extent=[0,X_dim,0,Y_dim])
plt.title("Density map of electron", fontname="Times New Roman", fontsize=12, fontweight='bold')
plt.xlabel(r"X   (in $l_0$)",fontname="Times New Roman",fontweight='bold')
plt.ylabel(r"Y   (in $l_0$)",fontname="Times New Roman",fontweight='bold')
plt.colorbar()
plt.show()

