'''
making FT of BOX function
Name: SAGAR DAM; DNAP
'''

import numpy as np
import matplotlib.pyplot as plt

#initializing variables
a=100
h=0.1

#defining box function
def f(x):
    if(abs(x)<=1):
        z=1
    else:
        z=0
    return z

#taking arrays for FT
x=np.arange(-a,a+h,h)
X=np.ones(len(x))
W=np.zeros(len(x),dtype=complex)
for i in range(len(x)):
    X[i]=f(x[i])
    
N=int(len(x))
freq=np.zeros(N)
kmin=-np.pi*(N-1)/(h*N)

#taking frequency points
for j in range(len(x)):
    freq[j]=kmin+j*2*np.pi/(h*N)

#doing fourier transform
for i in range(len(x)-1):
    for k in range(len(x)-1):
        W[i]=W[i]+(X[k]*np.exp(-complex(0,1)*freq[i]*x[k]))
    W[i]=W[i]/np.sqrt(N/np.pi)

#creating true (analytic) FT of box. i.e sinc
w=np.arange(-10,10,0.1)
sinc=np.zeros(len(w))
box=np.zeros(len(w))
def true(t):
    if(t!=0):
        z=np.sin(t)/t
    else:
        z=1
    return z*np.sqrt(2/np.pi)

for j in range(len(w)):
    sinc[j]=true(w[j])
    box[j]=f(w[j])

#plotting box function, numerical FT of box function and true FT of box (sinc)    
plt.plot(freq,W,label='numerical FT')
plt.plot(w,sinc,'ro',markersize='2',label='analytic FT (true sinc function)')
plt.plot(w,box,'k',label='box function')
plt.gca().set_xlim(-10,10)
plt.title('plotting box function, its numerical FT and the true FT')
plt.legend()
plt.grid()
plt.show()