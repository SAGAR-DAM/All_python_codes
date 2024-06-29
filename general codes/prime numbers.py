# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:48:51 2023

@author: sagar
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

t1=time.time()

def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True

def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

n = 1000
primes = generate_primes(n)
primes={"n'th prime":primes}
primes=pd.DataFrame(primes)
primes.index+=1
print(primes)

print("time: ",time.time()-t1)

'''
index=np.arange(len(primes))+2
plt.plot(primes)
plt.plot(1.2*index*np.log(index),'k-')
plt.show()
'''

for v in dir():
    exec('del '+ v)
    del v