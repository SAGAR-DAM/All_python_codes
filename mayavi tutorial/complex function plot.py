# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:14:59 2023

@author: sagar
"""

import numpy as np
from mayavi import mlab

r,theta=np.mgrid[0:5:100j,-4*np.pi:4*np.pi:400j]
x=r*np.cos(theta)
y=r*np.sin(theta)
z=r*np.exp(1j*theta)


f=(np.sqrt(r)*(np.cos(theta/2)+np.sin(theta/2)*1j)).imag
#f=(np.log(1+r)+1j*theta).imag

mlab.mesh(x,y,f,representation="wireframe")