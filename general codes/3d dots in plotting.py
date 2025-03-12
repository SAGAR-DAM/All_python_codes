import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg


import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


# Load your image (replace 'dot_image.png' with your actual image file)
image_path = "C:\\Users\\mrsag\\OneDrive\\Desktop\\3d balls.png"  
dot_image = mpimg.imread(image_path)

# Generate random data
np.random.seed(42)
x = np.linspace(0,10,20)
y = np.sin(x)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Function to create image markers
def image_scatter(x, y, image, ax, zoom=0.2):
    for (x0, y0) in zip(x, y):
        im = OffsetImage(image, zoom=zoom)  # Adjust zoom for size
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
        ax.plot(x,y,'k-')

# Scatter plot using images
image_scatter(x, y, dot_image, ax, zoom=0.06)

# Adjust plot
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_title("Scatter Plot with Image Dots")
ax.grid(color="k", lw=0.5)

# Show plot
plt.show()
