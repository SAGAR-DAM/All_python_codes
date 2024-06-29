import numpy
from mayavi.mlab import *

def Lorentz(x,y,z, s=10, r=14, b=3):
    u=s*(y-x)
    v=r*x-y-x*z
    w=x*y-b*z
    return u,v,w

def test_flow():
    x, y, z = np.mgrid[-40:40:100j, -40:40:100j, -40:40:100j]
    u,v,w=Lorentz(x,y,z)
    mlab.clf()
    obj = flow(x,y,z, u, v, w, seedtype='plane',seed_resolution=20)
    return obj

test_flow()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__