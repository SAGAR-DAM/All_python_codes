# this code gives linecut bewteen any two given pixels

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Load the image
image = io.imread('D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_33.png')

# Convert the image to grayscale
#gray_image = color.rgb2gray(image)

# Define the start and end points of the line cut
start_point = [320,0]
end_point = [320,700]

# Use the numpy function linspace to get the coordinates of the line
x, y = np.linspace(start_point[0], end_point[0], num=700), np.linspace(start_point[1], end_point[1], num=700)

# Get the grayscale values along the line
gray_values = image[x.astype(int),y.astype(int)]    
linecut=[]
for i in range(len(gray_values)):
    linecut_value=np.sqrt((gray_values[i][0])**2+(gray_values[i][1])**2+(gray_values[i][2])**2)
    linecut.append(linecut_value)
    
# Plot the grayscale values along the line
plt.plot(linecut)
plt.show()

print(gray_values)
print(gray_values.shape)

#print(image[320,697])