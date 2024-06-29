# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:30:59 2023

@author: mrsag
"""

import numpy as np

class myclass:     #creates a parent class
    x=5             #different class values can be assigned
    y=[1,2,3]
    def f(x,y):         #a class function
        return(x+y[2])
    z=f(x,y)            #inter class function application
    
p1=myclass  # an object like myclass

print(p1.x)
print(p1.f(2,[2,0,-3]))   # p1 has a property f which is a function that takes a value and an array of dim>=2... We are using that funtion and giving enternal output
print(p1.f(5,[1,2,3]))    #we are putting the same arguments as inside the class as externally
print(p1.z)               # we are computing the line 2 from the internal property of the class

#Class __init__ method:

class employee:    # define the class
    def __init__(self,n,s,a,r):    # gives the property of the class 
        self.name=n                # assign the name as // eqiv to employee.name=n
        self.salary=s              # assign the salary as s // eqiv to employee.salary=s
        self.age= a                # assign the age as a //....
        self.role=r                # assign the role as r //....
        
    def details(self):      #some class function on the class object
        return (f"The name of the employee is {self.name} with salary {self.salary} \n aged {self.age} years and his role is {self.role}")
    def array(self):        # another class function on the class object
        x=[str(self.name),str(np.array([self.salary])),str(self.age),str(self.role)]
        return(x)
    
sagar=employee("sagar",100000,24,"student")     # sagar is a class of employee type

print(sagar.details())    # print s the details function in the class as the self class argument   
print(sagar.array())      # same thing 


# Parent and child class:
class Parent:         # the parent class 
  def __init__(self, txt, value, role):     #initializing the class
    self.message = txt                 # different properties of the class
    self.value=value
    self.role=role

  def printmessage(self):        # class function
    print(f"\nmessage: {self.message}  value: {self.value}  role: {self.role}")

class Child(Parent):     # defining some class which is of type parent... i.e has the same properties of parent class
  def __init__(self, txt,value,role,age):     #initializing the child class... age is some extra property of the class that it has earned itself
    super().__init__(txt,value,role)   # inheriting all the properties of the parent to the child class... this is needed other than just defining the child as a parent type class
    self.age=age  #defining the new property of the child class
    
  def salary(self):  # a new function of the chin=ld class type
  	return(f"{self.message}s salary is: {self.value*10}")
  def agefunction(self):   # Some new function of the child class property that is not inherited
  	return(f"Age of {self.message} (a {self.role}) is {self.age}")


x = Child("Bachha",10,"student",24)  # class of child type with some inherited and some self property

x.printmessage()
print(x.salary())
print(x.agefunction())
