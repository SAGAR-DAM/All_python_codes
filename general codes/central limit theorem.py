# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:00:17 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display


sample=10000
sum_length=10000
dist=[]

for i in range(sample):
    x=np.random.randint(low=1,high=+7,size=sum_length)
    dist.append(sum(x))

dist=np.array(dist)
dist=(dist-np.mean(dist))/np.sqrt(sum_length)

plt.hist(dist,bins=500)
plt.show()
