import numpy as np
from skimage import io, feature

# Load the image
image = io.imread('D:\\data Lab\\Supercontinuum-march 2023\\10th march 2023\\Focal spot measurement\\Raw image\\Spot size_20_0_300.bmp', as_gray=True)

# Find the coordinates of the brightest point using the corner_peaks function
coords = feature.corner_peaks(np.abs(image), min_distance=10)

# Print the coordinates of the brightest point
print("The coordinates of the brightest point are: ", coords[0])
