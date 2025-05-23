# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:57:13 2023

@author: mrsag
"""
"""
This is the fitting function for Lorentzian data. The inputs are two same length array of datatype float.
There are 3 outputs:
1. The array after fitting the data. That can be plotted.
2. The used parameters set as the name parameters.
3. The string to describe the parameters set and the fit function.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from decimal import Decimal

def Lorentzian(x,a,b,x0):
    y=a/(b**2+(x-x0)**2)
    return y

def Lorentzfit(w,I):
    xdata=w
    ydata=I
    
    ymax_val = max(ydata)
    ymax_index=(list(ydata)).index(max(ydata))
    xmax_val=xdata[ymax_index]
    xdata=xdata-xmax_val
    
    ydata = ydata/ymax_val
    
    parameters, covariance = fit(Lorentzian, xdata, ydata,maxfev=100000)
    fit_y = Lorentzian(xdata, *parameters)

    
    xdata=xdata+xmax_val
    parameters[2]+=xmax_val
    parameters[0]*=ymax_val

    fit_y=np.asarray(fit_y)*ymax_val
    
    string1=r"Fit: f(x)=$\frac{a}{b^2+(x-x_0)^2}$; "
    string2=rf"with a={Decimal(str(parameters[0])).quantize(Decimal('1.00'))}, b={Decimal(str(parameters[1])).quantize(Decimal('1.00'))}, $x_0$={Decimal(str(parameters[2])).quantize(Decimal('1.00'))}"
    string=string1+string2
    # print(*parameters)
    return fit_y,parameters,string

# x=np.linspace(0,20,201)      #data along x axis
# y=Lorentzian(x=x, a=100, b=1, x0=10)             #data along y axis
# y += np.random.uniform(-1,1,size=len(y))*max(y)/10

# fit_y,parameters,string=Lorentzfit(x,y)
# plt.plot(x,y,'ko',label='Data')
# plt.plot(x,fit_y,'r-',lw=3,label='Fit')
# plt.title(string)
# plt.legend()
# plt.show()

# print(*parameters)
# print('FWHM= ', 2*parameters[1])
