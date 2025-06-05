# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:07:49 2025

@author: mrsag
"""

import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

import matplotlib as mpl
mpl.rcParams['figure.dpi']=500 # highres display

def simulate_coin_tosses(n=10, trials=10000):
    outcomes = []

    for _ in range(trials):
        binary_string = ''.join(str(random.randint(0, 1)) for _ in range(n))
        decimal_value = int(binary_string, 2)
        outcomes.append(decimal_value)

    return outcomes

def plot_histogram(outcomes, n):
    count = Counter(outcomes)
    x = list(range(2**n))
    y = [count.get(i, 0) for i in x]

    # Stats
    total = np.sum(y)
    min_count = np.min(y)
    max_count = np.max(y)
    avg_count = np.mean(y)
    nonzero = sum(1 for val in y if val > 0)

    # Find all events with the least count
    rare_events = [i for i, freq in enumerate(y) if freq == min_count]
    rare_binaries = [format(i, f'0{n}b') for i in rare_events]
    
    # Find all events with the least count
    freq_events = [i for i, freq in enumerate(y) if freq == max_count]
    freq_binaries = [format(i, f'0{n}b') for i in freq_events]

    # Print stats
    print(f"Total outcomes recorded = {total}")
    print(f"Minimum frequency of any outcome = {min_count}")
    print(f"Maximum frequency of any outcome = {max_count}")
    print(f"Average frequency per outcome = {avg_count:.2f}")
    print(f"Number of distinct outcomes seen = {nonzero} out of {2**n}")
    
    print(f"\nRarest events (count = {min_count} ; possibilities = {len(rare_binaries)}):")
    for b in sorted(rare_binaries):
        print(b)
        
    print(f"\nMost freq events (count = {max_count} ; possibilities = {len(freq_binaries)}):")
    for b in sorted(freq_binaries):
        print(b)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(x, y, width=1.0,color="b")
    plt.xlabel("Decimal value of binary coin flips")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {len(outcomes)} coin set (n={n}) toss simulations")
    plt.grid(False)#lw=1,color="k")
    plt.show()
    
    plt.hist(y,bins=len(set(y)),color="r")
    plt.title("Histogram for different counts...")
    plt.show()

# Parameters
n = 10         # Number of coins
trials = 100000 # Number of simulations

# Run simulation and plot
outcomes = simulate_coin_tosses(n, trials)
plot_histogram(outcomes, n)
