# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:35:07 2023

@author: sagar
"""

from mayavi import mlab

import numpy as np

tau=2

sigma,phi=np.mgrid[-np.pi:np.pi:100j,0:2*np.pi:100j]

x=np.sinh(tau)*np.cos(phi)/(np.cosh(tau)-np.cos(sigma))
y=np.sinh(tau)*np.sin(phi)/(np.cosh(tau)-np.cos(sigma))
z=np.sin(sigma)/(np.cosh(tau)-np.cos(sigma))

mlab.clf()

lensoffset=0
xx = yy = zz = np.arange(-5,5,0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.01)
mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.01)
mlab.mesh(x,y,z,representation='wireframe')

# for __var__ in dir():
#     exec('del '+ __var__)
#     del __var__