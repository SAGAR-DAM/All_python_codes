# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:49:36 2023

@author: sagar
"""
import numpy as np
from mayavi import mlab

theta,phi=np.mgrid[0.0001:np.pi:50j,0:2*np.pi:100j]
x=np.sin(theta)*np.cos(phi)
y=np.sin(theta)*np.sin(phi)
z=np.cos(theta)

mlab.clf()

lensoffset=0
xx = yy = zz = np.arange(-5,5,0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01)
mlab.mesh(x,y,z,representation='wireframe')


for __var__ in dir():
    exec('del '+ __var__)
    del __var__