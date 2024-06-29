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

x = np.linspace(-1, 3,1201)
y = np.linspace(-1, 3,1201)
X, Y = np.meshgrid(x, y)

saw_period = 0.1
saw_amplitude = 2

def sawtooth(amplitude, period,t):
    t = t % period  # Ensure t is within one period
    return(amplitude*t)



def f(x,y):
    if(x<=1):
        if(-0.2<=x<=0.2):
            if(y<(-0.2+sawtooth(saw_amplitude,saw_period,x))):
                return(0.999)
            elif(y>=(-0.2+sawtooth(saw_amplitude,saw_period,x))):
                return(0.999*np.exp(-5*(y-(-0.2+sawtooth(saw_amplitude,saw_period,x)))))
            else:
                return(0)
        else:
            if(y<(-0.2+sawtooth(saw_amplitude,saw_period,x))):
                return(0.999)
            elif(y>=(-0.2+sawtooth(saw_amplitude,saw_period,x))):
                return(0.999*np.exp(-5*(y-(-0.2+sawtooth(saw_amplitude,saw_period,x))))*np.exp(-5*(abs(x)-0.2 )))
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
