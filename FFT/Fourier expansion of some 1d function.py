# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:46:00 2023

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-10,10,2001)
y=np.array(np.zeros(len(x)))
def f(x,terms):
    lim=terms
    value=0
    for i in range(lim):
        an=(-1)**i/(2*i+1)
        bn=np.exp(-i**2)
        
        value += an*np.cos((2*i+1)*x)
    return(value)

y=f(x,3)
plt.plot(x,y,label='terms=3')

y=f(x,10)
plt.plot(x,y,label='terms=10')

y=f(x,50)
plt.plot(x,y,label='terms=50')

y=f(x,200)
plt.plot(x,y,label='terms=200')

y=f(x,5000)
plt.plot(x,y,label='terms=5000')

plt.legend()
plt.show()