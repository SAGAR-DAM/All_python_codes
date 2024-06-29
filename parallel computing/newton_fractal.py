# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 10:01:13 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import time
from multiprocessing import Pool
import multiprocessing

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 500  # highres display
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
# mpl.rcParams['font.style'] = 'italic'

t_start = time.time()


def create_random_root(n):
    root = []
    Re = np.random.uniform(low=-1, high=1, size=n)
    Im = np.random.uniform(low=-1, high=1, size=n)
    for i in range(n):
        root.append(Re[i] + Im[i] * 1j)
    return root


root = create_random_root(5)

iteration = 30
root = np.array(root)
colors = np.linspace(0, 1, len(root))

print("Roots:")
for i in range(len(root)):
    print(f"z{i + 1}:    {root[i]: .3f}")


def f(z):
    val = 1
    for i in range(len(root)):
        val = val * (z - root[i])
    return val


def df(z):
    val = 0
    for i in range(len(root)):
        mult = 1
        for j in range(len(root)):
            if (j != i):
                mult = mult * (z - root[j])
        val += mult
    return val


def iteration_step(z):
    return z - f(z) / df(z)


def parallel_iterations(iteration_step, z, num_iterations):
    for i in range(num_iterations):
        z = iteration_step(z)
    return z


def parallel_coloring(args):
    z, root, colors, x_start, x_end, y_start, y_end = args
    dist = np.abs(z[y_start:y_end, x_start:x_end, :, np.newaxis] - root)
    try:
        nearest = np.argmin(dist, axis=3)
    except:
        nearest = 0
    output = colors[nearest]
    return output


res = 25

x = np.linspace(-max(abs(root.real)) * 1.1, max(abs(root.real)) * 1.1, int(2 * res * (max(abs(root.real)) + 1)))
y = np.linspace(-max(abs(root.imag)) * 1.1, max(abs(root.imag)) * 1.1, int(2 * res * (max(abs(root.imag)) + 1)))

X, Y = np.meshgrid(x, y)
z = X + Y * 1j


# Number of parallel threads (adjust as needed)
num_cores = multiprocessing.cpu_count()

# Split the total iterations into chunks for parallel execution
iterations_per_thread = iteration // num_cores
iterations_list = [iterations_per_thread] * num_cores

# If the total iterations are not divisible evenly by num_cores, distribute the remaining iterations
iterations_list[-1] += iteration % num_cores

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(parallel_iterations, iteration_step, z.copy(), iterations) for iterations in
               iterations_list]
    concurrent.futures.wait(futures)

# Combine the results from parallel iterations
for future in futures:
    z = future.result()

# Define the ranges for each core for multiprocessing
ranges = []
for i in range(num_cores):
    x_start = i * X.shape[1] // num_cores
    x_end = (i + 1) * X.shape[1] // num_cores if i < num_cores - 1 else X.shape[1]
    ranges.append((z, root, colors, x_start, x_end, 0, X.shape[0]))

# Create a pool of workers for multiprocessing
with Pool(num_cores) as p:
    results = p.map(parallel_coloring, ranges)

# Combine the results
output = np.concatenate(results, axis=1)

output = np.flip(output, axis=0)

plt.imshow(output, cmap='jet', extent=[min(x), max(x), min(y), max(y)])
for i in range(len(root)):
    if abs(root[i].real) <= max(x) and abs(root[i].imag) <= max(y):
        plt.scatter(root[i].real, root[i].imag, color='red', marker='o')
        plt.text(root[i].real, root[i].imag, r'z$_{%d}$' % (1 + i), color="white", fontsize=10, ha='left',
                 va='bottom')
plt.title(r"NEWTON'S FRACTAL $\ \ by\ \$\alpha\widetilde g\alpha R$", fontname="Times New Roman", fontsize=10)
plt.grid(color="blue", linewidth=0.5)
plt.xlabel("Re(z)", fontsize=7)
plt.ylabel("Im(z)", fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

print("\n\n" + f"Image shape: {output.shape}")
print("Time taken:   ", time.time() - t_start)
