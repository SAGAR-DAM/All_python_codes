# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:28:27 2024

@author: mrsag
"""

def my_decorator(func):
    print("Something is happening before the function is called.")
    def wrapper(*args,**kwargs):
        return func(*args,**kwargs)
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")
    pass

say_hello()

@my_decorator
def f(x,y):
    return x+y

print(f(2,5))