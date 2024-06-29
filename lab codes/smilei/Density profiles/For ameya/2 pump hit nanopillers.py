# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:59:31 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib
import time

matplotlib.rcParams['figure.dpi'] = 500  # highres display

x = np.linspace(-1, 3,401)
y = np.linspace(-1, 3,401)
X, Y = np.meshgrid(x, y)

amplitude = 0.2
period = 0.2
pillar_width = 0.075

def periodic_pillars(t):
    t = t % period  # Ensure t is within one period

    step_function = np.zeros_like(t)
    step_function[t <= pillar_width] = amplitude

    return step_function



def f(x,y):
    if(x<=1):
        if(-0.2<=x<=0.2):
            if(y<(-0.2+periodic_pillars(t=x))):
                return(0.999)
            elif(y>=(-0.2+periodic_pillars(t=x))):
                return(0.999*np.exp(-5*(y-(-0.2+periodic_pillars(t=x)))))
            else:
                return(0)
        else:
            if(y<(-0.2+periodic_pillars(t=x))):
                return(0.999)
            elif(y>=(-0.2+periodic_pillars(t=x))):
                return(0.999*np.exp(-5*(y-(-0.2+periodic_pillars(t=x))))*np.exp(-5*(abs(x)-0.2 )))
            else:
                return(0)
    elif(x>1):
        return(f(x-2,y))
    
f=np.vectorize(f)
    
dense=f(X,Y)
print(dense)

plt.imshow(dense,cmap="hot",extent=[min(x),max(x),min(y),max(y)])
plt.title("Density map of electron", fontname="Times New Roman", fontsize=10)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar()
plt.show()


for __var__ in dir():
    exec('del '+ __var__)
    del __var__
