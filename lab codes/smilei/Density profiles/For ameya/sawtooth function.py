# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:08:04 2023

@author: sagar
"""
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib
import time

matplotlib.rcParams['figure.dpi'] = 500  # highres display

x=np.linspace(0,10,1001)

def sawtooth(amplitude, period,t):
    t = t % period  # Ensure t is within one period
    return(t*amplitude/period)

y=sawtooth(amplitude=10, period=0.5, t=x)

plt.plot(x,y)
plt.show()