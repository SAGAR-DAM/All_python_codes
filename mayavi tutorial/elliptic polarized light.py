# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:17:29 2023

@author: sagar
"""

import numpy as np
from mayavi import mlab

x,y,z=np.mgrid[-2:2:600j,-0:0:1j,-0:0:1j]

u=0*x
v=2*np.cos(20*x)
w=np.sin(20*x)

mlab.clf()
mlab.plot3d(x,v/7,w/7)
mlab.plot3d(x,0*v,0*w,color=(0,0,1))
mlab.quiver3d(x,y,z,u,v,w)


for __var__ in dir():
    exec('del '+ __var__)
    del __var__