# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:41:06 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the number of colors we want in each gradient segment
n_colors = 1000  # Four segments of 1000 colors each will give us 4000 colors

# Generate colors for each segment using numpy linspace
# Segment 1: Blue to Cyan (0, 0, 255) -> (0, 255, 255)
blue_to_cyan = [(0, int(g), 255) for g in np.linspace(0, 255, n_colors)]

# Segment 2: Cyan to Green (0, 255, 255) -> (0, 255, 0)
cyan_to_green = [(0, 255, int(b)) for b in np.linspace(255, 0, n_colors)]

# Segment 3: Green to Yellow (0, 255, 0) -> (255, 255, 0)
green_to_yellow = [(int(r), 255, 0) for r in np.linspace(0, 255, n_colors)]

# Segment 4: Yellow to Red (255, 255, 0) -> (255, 0, 0)
yellow_to_red = [(255, int(g), 0) for g in np.linspace(255, 0, n_colors)]

# Combine all segments to form the full gradient
gradient_list = blue_to_cyan + cyan_to_green + green_to_yellow + yellow_to_red

# Convert gradient_list to a numpy array with shape (n_colors, 1, 3) for display
gradient_array = np.array(gradient_list).reshape(len(gradient_list), 1, 3) / 255.0  # Normalize to 0-1 range for imshow

# Display the gradient as an image
plt.figure(figsize=(10, 2))  # Set figure size
plt.imshow(gradient_array, aspect='auto')
# plt.axis('off')  # Hide the axis
plt.show()
