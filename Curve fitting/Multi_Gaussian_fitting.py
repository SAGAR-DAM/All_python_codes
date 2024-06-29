# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:57:13 2023

@author: mrsag
"""
"""
This is the fitting function for Multi Gaussian data. The inputs are two same length array of datatype float.
There are 3 outputs:
1. The array after fitting the data. That can be plotted.
2. The used parameters set as the name parameters.
3. The string to describe the parameters set and the fit function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from decimal import Decimal
import pandas as pd 

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
# mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

def Gauss1(x,b,x0):
    y=np.exp(-(x-x0)**2/(2*b**2))
    return y

def Gaussfit(w,I):
    xdata=w         #Taking the x axis data
    ydata=I         #Taking the y axis data
    
    ''' 
        here the code fits only the normalized Gaussian
        So, we first normalize the array and later multiply with the amplitude factor to get the main array
    '''
    y_maxval=max(ydata)      #taking the maximum value of the y array
    ymax_index=(list(ydata)).index(y_maxval)   
    
    xmax_val=xdata[ymax_index]  #Shifting the array as a non-shifted Gausian 
    xdata=xdata-xmax_val        #Shifting the array as a non-shifted Gausian
    
    ydata=ydata/y_maxval
    
    parameters, covariance = fit(Gauss1, xdata, ydata,maxfev=100000)
    fit_y = Gauss1(xdata, *parameters)
    
    
    xdata=xdata+xmax_val
    parameters[1]+=xmax_val
    
    fit_y=np.asarray(fit_y)
    fit_y=fit_y*y_maxval       # again multiplying the data to get the actual value
    
    string1=r"Fit: $f(x)=Ae^{-\frac{(x-x_0)^2}{2b^2}}$;"
    string2=rf"with A={Decimal(str(y_maxval)).quantize(Decimal('1.00'))}, b={Decimal(str(parameters[0])).quantize(Decimal('1.00'))}, $x_0$={Decimal(str(parameters[1])).quantize(Decimal('1.00'))}"
    string=string1+string2
    return fit_y,parameters,string


def Multi_Gaussfit(x,y,n):

    fit_y = np.zeros(len(y))
    parameters_data = []
    parameters_name = []
    
    for i in range(n):
        fit_y1,parameters1,string1=Gaussfit(x,y-fit_y)
        
        fit_y += fit_y1
        parameters_data.append(format(max(fit_y1), '.2f'))
        parameters_data.append(format(parameters1[0], '.2f'))
        parameters_data.append(format(parameters1[1], '.2f'))
        
        parameters_name.append(rf"$A_{1+i}$:")
        parameters_name.append(rf"$\sigma_{1+i}$:")
        parameters_name.append(rf"$x_{1+i}$:")
    
    
    parameters=pd.Series(parameters_data,index=parameters_name)
    
    return(fit_y,parameters)    



x=np.linspace(-50,50,1001)      #data along x axis
y=np.zeros(len(x))

#y=50*np.exp(-(x+40)**2/5)+30*np.exp(-(x+10)**2/7)+10*np.exp(-(x-30)**2/50)+40*np.exp(-(x-10)**2/0.5)+80*np.exp(-(x-0)**2/5)            #data along y axis

for i in range(5):
    A = np.random.randint(low = 5, high = 50)
    x0 = np.random.randint(low = -40, high = 40)
    sigma = np.random.randint(low = 1, high = 3)
    y += A*Gauss1(x, sigma, x0)

random_noise=np.random.uniform(low=-2,high=2,size=(len(y)))
y=y+random_noise

gaussfit_order = 4
fit_y,parameters=Multi_Gaussfit(x, y, gaussfit_order)
print(parameters)



plt.figure()
plt.plot(x,y,'ko',label="data")
plt.plot(x,fit_y,'r-',lw=2, label=f"order {gaussfit_order} Gaussfit")
for i in range(len(parameters)//3):
    plt.text(float(parameters[3*i+2]), 1.02*float(parameters[3*i]), f'{1+i}', fontsize=12, ha='center', va='bottom', color="blue")
plt.figtext(0.95,0.2,str(parameters))
plt.legend()
plt.ylim(1.2*min(y),1.15*max(y))
plt.title(f"Gaussian fit of order {gaussfit_order}")
plt.grid(lw=0.5, color="black")
plt.show()