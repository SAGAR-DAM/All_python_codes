# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:55:52 2024

@author: mrsag
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:18:32 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from matplotlib.widgets import Slider
import types

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi'] = 120  # highres display


def polynomial_fit(x,y,degree):
    ''' Will take two 1 d arrays x and y and the required degree of 
    fitting and will return the fitted curve along with the fitting coefficients.
    
    y = c0 + c1*x + c2*x**2 + c3*x**3 + ...
    
    the return will be y_predict curve and coefficient array: [c0,c1,c2,c3,...]'''
    
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]

    model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression(fit_intercept=False))

    # Train the model using the training sets
    model.fit(x, y)

    # Make predictions using the testing set
    y_predict = model.predict(x)
    
    # Access the linear regression model within the pipeline
    linear_reg_model = model.named_steps['linearregression']
    
    return y_predict.flatten(), (linear_reg_model.coef_).flatten()


@np.vectorize
def f(x):
    coeff=(0.01,-0.1,1,-10,100)
    return np.polyval(coeff,x)*np.cos(x)

@np.vectorize
def g(x):
    return np.tan(x)


x = np.linspace(-10,10,1001)
y = f(x)

degree = 5

y_predict, coefficients = polynomial_fit(x, y, degree)

print(coefficients)


# Initial plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
line1, = ax.plot(x, y,'go', label="Given function", markersize=3)
line, = ax.plot(x, y_predict, 'r')
ax.set_xlabel('x')
ax.set_ylabel('y')
rounding = 5
txt = np.array([f"{elem:.4f}" for elem in np.round(coefficients,rounding)])
title = ax.set_title(f"degree: {degree}"+"\n"+f"coefficients: {txt}")
#time_text = ax.set_title(f't = {degree}')
#ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
#ax.set_ylim(-1,1)
# Slider setup
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'degree', 1, 20, valinit=degree, valstep=1)

# Update function for the slider
def update(val):
    t = slider.val
    y_predict, coefficients = polynomial_fit(x, y, t)
    txt = np.array([f"{elem:.4f}" for elem in np.round(coefficients,rounding)])
    line.set_ydata(y_predict)
    title.set_text(f"degree: {degree}"+"\n"+f"coef: {txt}")
    #ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
    #ax.set_ylim(-1,1)
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()



# __del_vars__ = []
# # Print all variable names in the current local scope
# print("Deleted Variables:")
# for __var__ in dir():
#     if not __var__.startswith("_") and not callable(locals()[__var__]) and not isinstance(locals()[__var__], types.ModuleType):
#         __del_vars__.append(__var__)
#         exec("del "+ __var__)
#     del __var__
    
# print(__del_vars__)