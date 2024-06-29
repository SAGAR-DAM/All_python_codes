# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:40:02 2023

@author: sagar
"""

import numpy as np

def nearest_index(arr,x):
    index=(abs(np.array(arr)-x)).argmin()
    return(index)

x=np.linspace(0,100,201)
index=nearest_index(x,50.3)

print(index)
print(x[index])