# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:04:07 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

def point_avg(arr,n):
    arr1=[]
    for i in range(int(len(arr)/n)):
        x=np.mean(arr[n*i:n*(i+1)])
        arr1.append(x)
    arr1.append(np.mean(arr[(int(len(arr)/n))*n:]))
    
    return(arr1)


n=10
x=np.random.rand(10000)
x=x+1*np.sin(np.arange(len(x))/100)
y=point_avg(x, n)
xrng=np.arange(len(x))
yrng=point_avg(xrng,n)

plt.plot(xrng,x)
plt.plot(yrng,y)
plt.show()