# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:03:04 2024

@author: mrsag
"""

class B:
    def __init__(self, param_b):
        self.param_b = param_b
    
    def f(self,x):
        return(x**2)
class C:
    def __init__(self, param_c):
        self.param_c = param_c

class D:
    def __init__(self, param_d):
        self.param_d = param_d

class A(B, C, D):
    def __init__(self, param_a, param_b, param_c, param_d):
        # Call __init__ of parent classes using super()
        B.__init__(self, param_b)
        C.__init__(self, param_c)
        D.__init__(self, param_d)
        # Initialize parameter specific to A
        self.param_a = param_a
        
    def h(self):
        print(self.param_d)
        
    def g(self,x):
        print(self.param_c)
        self.h()
        return(x**3)
    

# Creating an instance of class A
a = A("param_a_value", "param_b_value", "param_c_value", "param_d_value")

# Accessing parameters of A and its base classes
print(a.param_b)  # Output: param_b_value
print(a.param_c)  # Output: param_c_value
print(a.param_d)  # Output: param_d_value
print(a.param_a)  # Output: param_a_value


print(a.f(2))
print(a.g(2))
