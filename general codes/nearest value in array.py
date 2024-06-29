# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:03:55 2023

@author: sagar
"""

import numpy as np
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (abs(array - value)).argmin()
    val=array.flat[idx]
    index=np.where(array==val)
    return val,index

array = np.random.uniform(low=1,high=20,size=(5,6))
print(array)
val,index=find_nearest_value(array,10.2)

print(val)
print(index)
print(index[1][0])

