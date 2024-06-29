from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace, transpose,sin

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

def myFunc(x0,x1):
    r2=x0**2+x1**2
    return(100*exp(-5*r2)*abs((x1+x0)*(x0-x1)*x0*x1)*sin(5*r2)*exp(1/(1*r2+0.1)))


def make_batch():
    global batchsize

    inputs=random.uniform(low=-1,high=+1,size=[batchsize,2])
    targets=zeros([batchsize,1]) # must have right dimensions
    targets[:,0]=myFunc(inputs[:,0],inputs[:,1])
    return(inputs,targets)

batchsize=10
xrange=linspace(-1,1,200)
X0,X1=meshgrid(xrange,xrange)
plt.imshow(myFunc(X0,X1),interpolation='nearest',origin='lower')
#plt.axis("off")
plt.show()

y_in,y_target=make_batch()
print(y_in)
print(y_target)