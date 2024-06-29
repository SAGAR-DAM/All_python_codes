# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 00:02:59 2023

@author: sagar
"""

from decimal import Decimal
import numpy as np
x=[1.12565645,2.5649464,6.6849465,0.6464653,25.6849465]
x=np.array(x)
print(x)

#x = np.array([ '%.2f' % elem for elem in x ])
x=list(np.array([ '%.2f' % elem for elem in x ]).astype(np.float))
print(x)