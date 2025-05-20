
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:00:17 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import matplotlib
matplotlib.rcParams['figure.dpi']=100 # highres display


sample=1000
sum_length=100
dist=[]

for i in range(sample):
    x=np.random.randint(low=1,high=+7,size=sum_length)
    dist.append(np.mean(x))

dist=np.array(dist)
dist=(dist-np.mean(dist))/np.std(dist)

plt.hist(dist,bins=50,density=True)
x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', label='Standard Normal $N(0,1)$')
plt.show()
