# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:22:31 2023

@author: sagar
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

number=(0.0+1j)
maxtower=100

def power_tower(x,n):
    if n==0:
        return(1)
    else:
        return(x**power_tower(x, n-1))
    
    
reals=[]
imgs=[]

for i in range(0,maxtower):
    z=power_tower(number, i)
    print(z)
    reals.append(z.real)
    imgs.append(z.imag)
    
plt.plot(reals,imgs,'ro')
plt.plot(reals,imgs,'bo')
plt.title("Power tower of x in complex plane:   "+r"z=$x^{x^{x^{...^{x}}}}$;   "+f"x={number}", fontname='Times New Roman')
plt.xlabel("Re(z)",fontname='Times New Roman')
plt.ylabel("Im(z)",fontname='Times New Roman')
plt.show()

plt.plot(reals)
#plt.plot(imgs)
plt.title("Tetration real plot  "+r"y=Re(a$\uparrow\uparrow$x) ;    (x$ϵ$$\mathbb{N}$; a$ϵ$$\mathbb{C}$)",fontname='Times New Roman')
plt.xlabel("x",fontname='Times New Roman')
plt.ylabel(f"Re(  ({number.real}+{number.imag}i)"+r"$\uparrow\uparrow$x )",fontname='Times New Roman')
plt.show()