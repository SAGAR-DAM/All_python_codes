# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:39:14 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display


f1=np.matrix([[0,0],[0,0.16]])
b1=np.matrix([[0],[0]])

f2=np.matrix([[0.85,0.07],[-0.04,0.85]])
b2=np.matrix([[0],[1.6]])

f3=np.matrix([[0.2,-0.16],[0.4,0.22]])
b3=np.matrix([[0],[1.6]])

f4=np.matrix([[-0.15,0.28],[0.26,0.24]])
b4=np.matrix([[0],[0.44]])

x0=np.matrix([[0],[0]])

x=[0]
y=[0]

for i in range(100000):
    rand=np.random.randint(low=1,high=101)
    if(rand<2):
        x0=np.dot(f1,x0)
    elif(2<=rand<85):
        x0=np.dot(f2,x0)+b2
    elif(85<=rand<92):
        x0=np.dot(f3,x0)+b3
    else:
        x0=np.dot(f4,x0)+b4
        
    x.append(x0[0,0])
    y.append(x0[1,0])
    

plt.figure(figsize=(5,5))
plt.plot(x,y,'ro',markersize=0.1)
plt.title("points: 100000")
#plt.grid(color='green', linestyle='-', linewidth=1)
plt.axis('off')
plt.show()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()