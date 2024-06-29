# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:26:28 2023

@author: sagar
"""
import pandas as pd

def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime_using_wilson(p):
    if p <= 1:
        return False
    return (factorial(p - 1) + 1) % p == 0

def nth_prime_using_wilson(n):
    if n == 1:
        return 2  # The first prime number is 2.
    
    count = 1  # Start with the first prime (2).
    number = 3  # Start checking from the next odd number.
    
    while True:
        if is_prime_using_wilson(number):
            count += 1
            if count == n:
                return number
        number += 2  # Check only odd numbers for efficiency.

# Example usage:
'''
n = 400
nth_prime = nth_prime_using_wilson(n)
print(f"The {n}-th prime number is {nth_prime}")
'''
primes=[]
for i in range(20):
    val=nth_prime_using_wilson(i+1)
    primes.append(val)

primes={"n'th prime":primes}
primes=pd.DataFrame(primes)
primes.index+=1
print(primes)

for v in dir():
    exec('del '+ v)
    del v