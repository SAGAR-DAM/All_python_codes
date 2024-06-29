import matplotlib.pyplot as plt
import numpy as np
import mpld3

# Define the grid and function
x, y = np.mgrid[-10:10:501j, -10:10:501j]

def f(x, y):
    r = x**2 + y**2
    val = np.sin(r) * np.sin(5 * x)
    return val

z = f(x, y)

# Create the plot
fig, ax = plt.subplots()
im = ax.imshow(z, extent=(-10, 10, -10, 10), origin='lower', cmap='jet')
plt.colorbar(im, ax=ax)

# Set the title
ax.set_title('Interactive Heatmap with Colorbar')

# Use mpld3 to save the plot as an HTML file
mpld3.save_html(fig, 'D:\\Codes\\Test folder\\interactive_plot.html')

# Optionally, you can also display the plot in a Jupyter Notebook
mpld3.display()
