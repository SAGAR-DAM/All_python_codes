# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:52:32 2023

@author: sagar
"""

import plotly.graph_objects as go
import numpy as np
X, Y, Z = np.mgrid[-20:20:60j, -20:20:60j, -20:20:60j]

def f(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arcsin(z/r)
    phi=np.arctan(y/x)
    
    psi=r**2*np.exp(-r/3)*(3*(np.cos(theta))**2-1)
    return(psi**2)
values = f(X,Y,Z)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=-0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    ))
fig.show()