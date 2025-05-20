# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:30:14 2025

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt

# Number of points to generate
n_points = 500000
n_lobes = 7         # 🌸 Try 3, 5, 6, 8, 12...
scale = 0.3333         # Scaling factor for each copy (0.3–0.6 works well)

# Generate transformations
transforms = []

# Central transformation (optional: comment out if you want a hole in center)
transforms.append((scale * np.eye(2), np.array([[0.5], [0.5]])))  # Center

# Surrounding lobes (evenly spaced on circle)
for k in range(n_lobes):
    angle = 2 * np.pi * k / n_lobes
    dx = 0.5 * np.cos(angle) * (1 - scale)
    dy = 0.5 * np.sin(angle) * (1 - scale)
    shift = np.array([[0.5 + dx], [0.5 + dy]])
    transforms.append((scale * np.eye(2), shift))

# Equal probabilities
probs = [1 / len(transforms)] * len(transforms)

# Initial point
x = np.array([[0.5], [0.5]])
x_vals = []
y_vals = []

# Iterate
for _ in range(n_points):
    i = np.random.choice(len(transforms), p=probs)
    A, b = transforms[i]
    x = A @ x + b
    x_vals.append(x[0, 0])
    y_vals.append(x[1, 0])

# Plot
plt.figure(figsize=(6, 6), facecolor="k")
plt.scatter(x_vals, y_vals, s=0.3, color='cyan', edgecolors='none')
plt.axis('off')
# plt.title(f"{n_lobes}-Lobed IFS Fractal", color='white')

plt.show()
