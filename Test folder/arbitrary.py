# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:24:17 2023

@author: sagar
"""

import class_test as bsc

p1=bsc.myclass  # an object like myclass

print(p1.x)
print(p1.f(2,[2,0,-3]))   # p1 has a property f which is a function that takes a value and an array of dim>=2... We are using that funtion and giving enternal output
print(p1.f(5,[1,2,3]))    # we are putting the same arguments as inside the class as externally
print(p1.z)               # we are computing the line 2 from the internal property of the class

y=p1.f(2,[2,0,-3])
print(y)

print("############################")

p2=bsc.myclass1
y=p2.g(2,4)
print(y)