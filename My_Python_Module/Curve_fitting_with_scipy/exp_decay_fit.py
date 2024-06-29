import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit as fit

import matplotlib as mpl


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display

def find_index(array, value):
    # Calculate the absolute differences between each element and the target value
    absolute_diff = np.abs(array - value)
    
    # Find the index of the minimum absolute difference
    index = np.argmin(absolute_diff)
    
    return index

def exp_decay(t,t0,tau,y0):
    y = np.exp(-(t-t0)/tau)+y0
    return(y)


def fit_exp_decay(x,y):
    maxy = max(y)
    y /= maxy
    xmax_index =find_index(y,max(y))
    xmax_val = x[xmax_index]
    
    x -= xmax_val
    
    parameters, covariance = fit(exp_decay, x, y, maxfev=100000)
    fit_y = exp_decay(x,*parameters)
    x += xmax_val
    fit_y *= maxy
    y *= maxy
    return fit_y, parameters


