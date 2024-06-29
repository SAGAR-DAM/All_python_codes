# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:28:26 2023

@author: sagar
"""

import numpy as np
import pandas as pd

def nearest_sq(n):
    x=np.sqrt(n)
    a=int(x)
    if(n<=(a**2+(a+1)**2)/2):
        return(a**2)
    else:
        return(a+1)**2
    
number=np.arange(0,50)
nesq=np.zeros(len(number))
for i in range(len(number)):
    nesq[i]=nearest_sq(number[i])

s=pd.Series(nesq,index=number)
print(s)