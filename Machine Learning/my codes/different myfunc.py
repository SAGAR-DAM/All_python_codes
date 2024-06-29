# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:44:20 2023

@author: sagar
"""
import numpy as np
import matplotlib.pyplot as plt

def smilei_face(x0,x1):
    r_face2=x0**2+x1**2
    r_reye2=(x0-0.20)**2+(x1-0.15)**2
    r_leye2=(x0+0.20)**2+(x1-0.15)**2
    r_smile2=x0**2/10+10*(x1+0.200)**2
    #val=(r_face2<0.25)*(r_leye2>0.01)*(r_reye2>0.01)*(r_smile2>0.01)
    val=(r_face2<0.25)*(r_leye2>0.01)*(r_reye2>0.01)*(r_smile2>0.0)*(1-1.1*np.exp(-r_leye2*10))*(1-1.1*np.exp(-r_reye2*10))*np.exp(-15*r_face2)
    return(val)

def oneside(x0,x1):
    val=(x0>0)*abs(x0)**0.2
    return(val)

def twoside(x0,x1):
    val=(x0*x1>0)*abs(x0*x1)**0.2
    return(val)

def sin(x0,x1):
    r2=x0**2+x1**2
    val=np.sin(100*r2)
    return val

xrange=np.linspace(-0.5,0.5,501)

X0,X1=np.meshgrid(xrange,xrange)
plt.imshow(smilei_face(X0,X1),interpolation='nearest',origin='lower')
plt.show()


plt.imshow(oneside(X0,X1),interpolation='nearest',origin='lower')
plt.show()


plt.imshow(twoside(X0,X1),interpolation='nearest',origin='lower')
plt.show()


plt.imshow(sin(X0,X1),interpolation='nearest',origin='lower')
plt.show()
