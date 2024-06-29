# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:15:35 2023

@author: sagar
"""

import numpy as np
from mayavi import mlab
import types

def f(x,y):
    #z=10*np.exp(-(2*x**2+y**2))
    z=10*np.sin(2*x**2+2*y**2)*np.exp(-(2*x**2+y**2))
    #z=np.sin(10*x)*np.sin(10*y)
    return z

r,phi=np.mgrid[0.00001:3.00001:301j,0:2*np.pi:101j]
x=r*np.cos(phi)
y=r*np.sin(phi)
del r
del phi

z=f(x,y)
print(z.shape)
print(x.ndim)
print(x)
print(x[60][20])

lensoffset=0
xx = yy = zz = np.arange(-5,5,0.1)
xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
#mlab.plot3d(yx,yy+lensoffset,yz,line_width=0.01,tube_radius=0.1)
#mlab.plot3d(zx,zy+lensoffset,zz,line_width=0.01,tube_radius=0.1)
#mlab.plot3d(xx,xy+lensoffset,xz,line_width=0.01,tube_radius=0.1)
mlab.mesh(x,y,z,representation='fancymesh')
mlab.axes(extent=[-5, 5, -5, 5, -5, 5], color=(0, 0, 0), nb_labels=5)
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("f(x,y")

#mlab.show()

__del_vars__ = []
# Print all variable names in the current local scope
print("Deleted Variables:")
for __var__ in dir():
    if not __var__.startswith("_") and not callable(locals()[__var__]) and not isinstance(locals()[__var__], types.ModuleType):
        __del_vars__.append(__var__)
        exec("del "+ __var__)
    del __var__
    
print(__del_vars__)
