import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=500 # highres display

import time

t_start=time.time()

order=4
iteration=50

root=[]
for i in range(order):
    root.append(np.exp(1j*i*2*np.pi/order))


roots=len(root)
colors=np.linspace(0,1,order)

def f(z):
    val=z**order-1
    return(val)

def df(z):
    val=order*z**(order-1)
    return(val)


x = np.linspace(-1, 1, 2001)
y = np.linspace(-1, 1, 2001)

X, Y = np.meshgrid(x, y)
z = X + Y * 1j

for i in range(iteration):
    z=z-f(z)/df(z)

def colour_the_complex_plane(z, root, colors):
    dist = np.abs(z[:, :, np.newaxis] - root)
    try:
        nearest = np.argmin(dist, axis=2)
    except:
        nearest=0
    output = colors[nearest]
    return output


output = colour_the_complex_plane(z, root, colors)
plt.imshow(output)
plt.axis('off')
plt.show()

print(time.time()-t_start)

for __var__ in dir():
    exec('del '+ __var__)
    del __var__