# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:58:50 2024

@author: mrsag
"""
import numpy as np

def is_even(f, domain):
    for x in domain:
        if f(x) != f(-x):
            return False
    return True

def is_odd(f, domain):
    for x in domain:
        if f(x) != (-1)*f(-x):
            return False
    return True

# Define your function f(x) here
def f(x):
    return (x**2+np.exp(-x**2))*(abs(x)<1)  # Example function x^2

# Define the domain of the function (e.g., range of x values to test)
domain = np.linspace(-5,5,51)  # Test from -10 to 10

# Test the function
if(is_even(f, domain)):
    print("f(x) is even")
if(is_odd(f, domain)):
    print("f(x) is odd")
if not (is_even(f, domain)) and not (is_odd(f, domain)):
    print("f(x) is neither even nor odd")
