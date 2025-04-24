import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

# Sample data
x = np.linspace(0.1, 10, 100)
y = np.exp(x)

# Create the plot
plt.figure()
plt.plot(x, y)

# Set y-axis to log scale
plt.yscale('log')

# High-resolution minor ticks for y-axis (log scale)
subs = np.linspace(1.01, 10, 10, endpoint=False)
ax = plt.gca()
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=1000))

# High-resolution minor ticks for x-axis (linear scale)
ax.xaxis.set_minor_locator(MultipleLocator(0.2))  # Adjust this for desired grid density

# Hide minor tick marks and labels
ax.tick_params(axis='x', which='minor', length=0)
ax.tick_params(axis='y', which='minor', length=0)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# Add gridlines
plt.grid(which='major', linestyle='-', linewidth=1, color='black')
plt.grid(which='minor', linestyle='-', linewidth=0.5, color='k')

# Labels
plt.xlabel("x (linear scale)")
plt.ylabel("y (log scale)")
plt.title("Minor Grid Lines: Linear x-axis & Log y-axis")

# Show the plot
plt.show()
