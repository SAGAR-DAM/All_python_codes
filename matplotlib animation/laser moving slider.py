import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

def f(x, t):
    return (3 * np.exp(-0.2 * (x - t)**2) * np.cos(5 * (x - t)))**2 * np.sign(-t)

x = np.linspace(-50, 50, 1001)

inc_angle = np.pi/6
theta = np.pi / 2 - inc_angle

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
t0 = -1

# Define the initial x range and y values based on t0
if t0 <= 0:
    x = np.linspace(-50, 0, 1001)
else:
    x = np.linspace(0, 50, 1001)

y = f(x, t0)
xp = np.cos(theta) * x + np.sin(theta) * y
yp = -np.sin(theta) * x + np.cos(theta) * y

line, = plt.plot(xp, yp, lw=0.5, color="r")
ax.set_aspect('equal', adjustable='box')
axcolor = 'lightgoldenrodyellow'
ax.set_xlim(-50, 50)

# Create slider for t
ax_t = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
slider_t = Slider(ax_t, 't', -50.0, 50.0, valinit=t0)

# Set up colormap normalization from -50 to 50
norm = Normalize(vmin=-50, vmax=50)
colormap = plt.cm.jet  # Use the 'jet' colormap for the desired color transition

def update(val):
    t = slider_t.val
    if t < 0:
        x = np.linspace(-50, 0, 1001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = -np.sin(theta) * x + np.cos(theta) * y
    else:
        x = np.linspace(0, 50, 1001)
        y = f(x, t)
        xp = np.cos(theta) * x + np.sin(theta) * y
        yp = np.sin(theta) * x - np.cos(theta) * y

    # Update line data
    line.set_xdata(xp)
    line.set_ydata(yp)

    # Update line color based on the slider value
    color = colormap(norm(t))  # Get color from colormap
    line.set_color("red")

    fig.canvas.draw_idle()

slider_t.on_changed(update)

plt.show()
