# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:49:44 2024

@author: mrsag
"""

import cmath
import math

class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
        self.r = None
        self.theta = None
    
    def rect_to_polar(self):
        self.r = abs(complex(self.real, self.imag))
        self.theta = cmath.phase(complex(self.real, self.imag))
    
    def polar_to_rect(self):
        self.real = self.r * math.cos(self.theta)
        self.imag = self.r * math.sin(self.theta)
    
    def get_real(self):
        return self.real
    
    def get_imag(self):
        return self.imag
    
    def get_r(self):
        self.rect_to_polar()
        return self.r
    
    def get_theta(self):
        self.rect_to_polar()
        return self.theta
    
    def get_complex(self):
        if self.imag >= 0:
            print(f"The Complex number is: {self.real} + {self.imag}j")
        else:
            print(f"The Complex number is: {self.real} {self.imag}j")
        pass
    
    def get_complex_polar(self):
        self.rect_to_polar()
        return f"The complex number in polar form: {self.r} exp({self.theta}j)"
    
    @staticmethod
    def add_complex(c1, c2):
        return Complex(c1.real + c2.real, c1.imag + c2.imag)
    
    @staticmethod
    def subtract_complex(c1, c2):
        return Complex(c1.real - c2.real, c1.imag - c2.imag)
    
    @staticmethod
    def mult_complex(c1, c2):
        c1.rect_to_polar()
        c2.rect_to_polar()
        r_mult = c1.r * c2.r
        theta_mult = c1.theta + c2.theta

        real_mult = r_mult * math.cos(theta_mult)
        imag_mult = r_mult * math.sin(theta_mult)
        return Complex(real_mult, imag_mult)
    
    @staticmethod
    def div_complex(c1, c2):
        c1.rect_to_polar()
        c2.rect_to_polar()
        r_div = c1.r / c2.r
        theta_div = c1.theta - c2.theta

        real_div = r_div * math.cos(theta_div)
        imag_div = r_div * math.sin(theta_div)
        return Complex(real_div, imag_div)
    
    @staticmethod
    def complex_exponentiation(c, n):
        c.rect_to_polar()
        r_exp = c.r ** n
        theta_exp = n * c.theta

        real_exp = r_exp * math.cos(theta_exp)
        imag_exp = r_exp * math.sin(theta_exp)
        return Complex(real_exp, imag_exp)
    
    @staticmethod
    def complex_pow(c1, c2):
        c1.rect_to_polar()
        c2.rect_to_polar()
        r_pow = (c1.r ** (c2.r * math.cos(c2.theta))) * math.exp(-c2.r * c1.theta * math.sin(c2.theta))
        theta_pow = c2.r * math.sin(c2.theta) * math.log(c1.r) + c2.r * math.cos(c2.theta) * c1.theta

        real_pow = r_pow * math.cos(theta_pow)
        imag_pow = r_pow * math.sin(theta_pow)
        return Complex(real_pow, imag_pow)
    
    @staticmethod
    def real_pow_complex(c, n):
        return Complex.complex_pow(Complex(n, 0), c)
    
    @staticmethod
    def sin_complex(c):
        real_sin = math.sin(c.real) * math.cosh(c.imag)
        imag_sin = math.cos(c.real) * math.sinh(c.imag)
        return Complex(real_sin, imag_sin)
    
    @staticmethod
    def cos_complex(c):
        real_cos = math.cos(c.real) * math.cosh(c.imag)
        imag_cos = -math.sin(c.real) * math.sinh(c.imag)
        return Complex(real_cos, imag_cos)
    
    @staticmethod
    def tan_complex(c):
        return Complex.div_complex(Complex.sin_complex(c), Complex.cos_complex(c))
    
    @staticmethod
    def csc_complex(c):
        return Complex.div_complex(Complex(1, 0), Complex.sin_complex(c))
    
    @staticmethod
    def sec_complex(c):
        return Complex.div_complex(Complex(1, 0), Complex.cos_complex(c))
    
    @staticmethod
    def cot_complex(c):
        return Complex.div_complex(Complex.cos_complex(c), Complex.sin_complex(c))
    
    @staticmethod
    def log_complex(c):
        c.rect_to_polar()
        return Complex(math.log(c.r), c.theta)
    
    @staticmethod
    def asin_complex(c):
        iota = Complex(0, 1)
        z = Complex.add_complex(Complex.mult_complex(iota, c), Complex.complex_exponentiation(Complex.subtract_complex(Complex(1, 0), Complex.mult_complex(c, c)), 0.5))
        z = Complex.log_complex(z)
        z = Complex.div_complex(z, iota)
        return z
    
    @staticmethod
    def acos_complex(c):
        iota = Complex(0, 1)
        z = Complex.add_complex(c, Complex.complex_exponentiation(Complex.subtract_complex(Complex.mult_complex(c, c), Complex(1, 0)), 0.5))
        z = Complex.log_complex(z)
        z = Complex.div_complex(z, iota)
        return z
    
    @staticmethod
    def atan_complex(c):
        iota = Complex(0, 1)
        z = Complex.div_complex(Complex.subtract_complex(iota, c), Complex.add_complex(iota, c))
        z = Complex.log_complex(z)
        z = Complex.div_complex(z, Complex(0, 2))
        return z
    
    @staticmethod
    def acsc_complex(c):
        return Complex.asin_complex(Complex.div_complex(Complex(1, 0), c))
    
    @staticmethod
    def asec_complex(c):
        return Complex.acos_complex(Complex.div_complex(Complex(1, 0), c))
    
    @staticmethod
    def acot_complex(c):
        return Complex.atan_complex(Complex.div_complex(Complex(1, 0), c))