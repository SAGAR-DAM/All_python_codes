# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:58:37 2023

@author: sagar
"""

from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, linspace, transpose

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# set up all the weights and biases

NumLayers=3 # does not count input-layer (but does count output)
LayerSizes=[2,3,4,1] # input-layer,hidden-1,hidden-2,...,output-layer

# initialize random weights and biases for all layers (except input of course)
Weights=[random.uniform(low=-1,high=+1,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Biases=[random.uniform(low=-1,high=+1,size=LayerSizes[j+1]) for j in range(NumLayers)]

# define the batchsize
batchsize=1

# set up all the helper variables

y_layer=[zeros([batchsize,LayerSizes[j]]) for j in range(NumLayers+1)]
df_layer=[zeros([batchsize,LayerSizes[j+1]]) for j in range(NumLayers)]
dw_layer=[zeros([LayerSizes[j],LayerSizes[j+1]]) for j in range(NumLayers)]
db_layer=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]

print(dw_layer)
print((dw_layer[-1]).shape)

print(y_layer[-2].shape)

