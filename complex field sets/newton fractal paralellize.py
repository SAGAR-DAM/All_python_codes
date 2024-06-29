import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import matplotlib
import time

matplotlib.rcParams['figure.dpi'] = 500  # highres display

t_start=time.time()

order = 4
iteration = 50

root = [np.exp(1j * i * 2 * np.pi / order) for i in range(order)]

roots = len(root)
colors = np.linspace(0, 1, order)

def f(z):
    val = z**order - 1
    return val

def df(z):
    val = order * z**(order - 1)
    return val

def iteration_step(z):
    return z - f(z) / df(z)

def colour_the_complex_plane(z, root, colors):
    dist = np.abs(z[:, :, np.newaxis] - root)
    try:
        nearest = np.argmin(dist, axis=2)
    except:
        nearest = 0
    output = colors[nearest]
    return output

x = np.linspace(-1, 1, 2001)
y = np.linspace(-1, 1, 2001)

X, Y = np.meshgrid(x, y)
z = X + Y * 1j

# Define a function for parallelizing the iterations
def parallel_iterations(iteration_step, z, num_iterations):
    for i in range(num_iterations):
        z = iteration_step(z)
    return z

# Number of parallel threads (adjust as needed)
num_threads = 4

# Split the total iterations into chunks for parallel execution
iterations_per_thread = iteration // num_threads
iterations_list = [iterations_per_thread] * num_threads

# If the total iterations is not divisible evenly by num_threads, distribute the remaining iterations
iterations_list[-1] += iteration % num_threads

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(parallel_iterations, iteration_step, z.copy(), iterations) for iterations in iterations_list]
    concurrent.futures.wait(futures)

# Combine the results from parallel iterations
for future in futures:
    z = future.result()

output = colour_the_complex_plane(z, root, colors)
plt.imshow(output)
plt.axis('off')
plt.show()

print(time.time()-t_start)

for __var__ in dir():
    exec('del '+ __var__)
    del __var__