import numpy as np
from skimage import io, feature
from skimage.color import rgb2gray, rgba2rgb

# Load the image
image = io.imread('D:\\Codes\\Test folder\\images\\Spot size_10_0.bmp', as_gray=True)

# Find the coordinates of the brightest point using the corner_peaks function
if(image.ndim==2):
    coords = feature.corner_peaks(np.abs(image), min_distance=10)
elif(image.ndim==3):
    if(image.shape[2]==3):
        coords = feature.corner_peaks(np.abs(rgb2gray(image)), min_distance=10)
    elif(image.shape[2]==4):
        coords = feature.corner_peaks(np.abs(rgba2rgb(rgb2gray(image))), min_distance=10)

# Print the coordinates of the brightest point
print("The coordinates of the brightest point are: ", coords[0])
