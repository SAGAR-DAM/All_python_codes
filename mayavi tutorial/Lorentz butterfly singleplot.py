# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:57:48 2023

@author: sagar
"""
from mayavi import mlab
import numpy as np
from scipy.integrate import odeint

def Lorentz(x,y,z, s=10, r=28, b=8/3):
    u=s*(y-x)
    v=r*x-y-x*z
    w=x*y-b*z
    return u,v,w

def Lorentz_ode(state,t):
    x,y,z=state
    return(np.array(Lorentz(x, y, z)))

t=np.linspace(0,200,60000)
r=odeint(Lorentz_ode, (10,5,5), t)

x,y,z=r.T

mlab.clf()
lensoffset=0
xx = yy = zz = np.arange(min(x),max(x),0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=None)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=None)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=None)
mlab.plot3d(x,y,z,t,tube_radius=None)

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
