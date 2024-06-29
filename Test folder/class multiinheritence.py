# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun 23 00:14:08 2024

# @author: mrsag
# """

# class a:
#     def __init__(self):
#         print("a")
#         pass
#     ax = "ax"
#     pass

# class b:
#     def __init__(self):
#         print("b")
#         pass
#     bx = "bx"
#     pass

# class c(b):
#     def __init__(self):
#         super().__init__()
#         print("c")
#         pass
#     cx = "cx"
#     pass
    

# class d(c,a):
#     def __init__(self):
#         a.__init__(self)
#         print("d")
#         pass
#     dx = "dx"
#     def f(e):
#         print(f"""{e.ax}
#         {e.bx}
#         {e.cx}
#         {e.dx}""")
#     pass

# e = d()

# print(f"""{e.ax}
# {e.bx}
# {e.cx}
# {e.dx}""")

# e.f()











# class a:
#     def __init__(self):
#         print("a")
#     ax = "ax"

# class b:
#     def __init__(self):
#         print("b")
#     bx = "bx"

# class c(a, b):
#     def __init__(self):
#         super().__init__()
#         print("c")
#     cx = "cx"

# class d(c):
#     def __init__(self):
#         super().__init__()
#         print("d")
#     dx = "dx"

# print("Using super()")
# x = d()

# print("\nUsing direct call")
# class c2(a, b):
#     def __init__(self):
#         a.__init__(self)
#         b.__init__(self)
#         print("c2")
#     cx = "cx"

# class d2(c2):
#     def __init__(self):
#         c2.__init__(self)
#         print("d2")
#     dx = "dx"

# y = d2()

# class Temperature:
#     def __init__(self, celsius=0):
#         self._celsius = celsius

#     # Define a getter for 'celsius'
#     @property
#     def celsius(self):
#         print("Getting value...")
#         return self._celsius

#     # Define a setter for 'celsius'
#     @celsius.setter
#     def celsius(self, value):
#         if value < -273.15:
#             raise ValueError("Temperature below -273.15°C is not possible.")
#         print("Setting value...")
#         self._celsius = value

#     # Define a getter for 'fahrenheit'
#     @property
#     def fahrenheit(self):
#         return (self._celsius * 9/5) + 32

#     # Define a setter for 'fahrenheit'
#     @fahrenheit.setter
#     def fahrenheit(self, value):
#         self._celsius = (value - 32) * 5/9

# # Create an instance of Temperature
# temp = Temperature()

# # Using the getter
# print(temp.celsius)  # Output: Getting value... 0

# # Using the setter
# temp.celsius = 25    # Output: Setting value...
# print(temp.celsius)  # Output: Getting value... 25

# # Using the getter for the derived property 'fahrenheit'
# print(temp.fahrenheit)  # Output: 77.0

# # Using the setter for the derived property 'fahrenheit'
# temp.fahrenheit = 100
# print(temp.celsius)  # Output: 37.77777777777778


# class temperature:
#     def __init__(self, celsius=0):
#         self._celsius=celsius

# t=temperature()

# print(t._celsius)
# t._celsius=50
# print(t._celsius)

class globalclass:
    name = "your name"
    @staticmethod
    def f():
        print("abc")
        pass
    
    @classmethod
    def g(cls):
        print(f"{cls.name}")
        pass
    
    @staticmethod
    def h(x):
        return x**2
        
x = globalclass()
x.name = "sagar"
# x.f()
# x.g()

# def f(x)-> None : ...
def __mod__(x:globalclass) -> str:
    return x.h(2)

y=__mod__(x)
print(y)
    

y=__mod__(x)
print(y)



