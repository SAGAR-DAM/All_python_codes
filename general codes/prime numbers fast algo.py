# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:48:51 2023

@author: sagar
"""
import numpy as np
import pandas as pd
import time


t1=time.time()

def is_prime(num,primes):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(len(primes)):
        if num % primes[i] == 0:
            return False
    return True

def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        try:
            if is_prime(num,primes[0:int(1.2*np.sqrt(primes[-1]/np.log(primes[-1])))]):
                primes.append(num)
        except:
            if is_prime(num,primes):
                primes.append(num)
        num += 1
    return primes

n = 20000
primes = generate_primes(n)
primes={"n'th prime":primes}
primes=pd.DataFrame(primes)
primes.index+=1
print(primes)

print("time: ",time.time()-t1)
