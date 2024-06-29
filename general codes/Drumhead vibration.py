# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:33:55 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jn, jn_zeros
import types

# Example data: Vibration of a drumhead
# x = np.linspace(-1, 1, 50)
# y = np.linspace(-1, 1, 50)
# x, y = np.meshgrid(x, y)
r,theta = np.mgrid[0:1:50j,0:2*np.pi:50j]
x = r*np.cos(theta)
y = r*np.sin(theta)

def drumhead_vibration(t, m, n):
    #r = np.sqrt(x**2 + y**2)
    #z = np.cos(np.pi/2*r) * np.cos(t) * np.exp(-r**2)
    zero = jn_zeros(m, n)[n-1]
    z = jn(m, zero * r) * np.cos(m * theta) * np.cos(zero * t)
    return z

# Initial plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)
t_initial = 0
m = 2
n = 1
z = drumhead_vibration(t_initial,m,n)
surface = ax.plot_surface(x, y, z, cmap='jet')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
time_text = ax.set_title(f't = {t_initial:.2f} units')
ax.set_zlim(-0.5,0.5)

# Slider setup
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'time', 0, 10, valinit=t_initial, valstep=0.1)

# Update function for the slider
def update(val):
    t = slider.val
    ax.clear()
    z = drumhead_vibration(t,m,n)
    ax.plot_surface(x, y, z, cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-0.5,0.5)
    time_text = ax.set_title(f't = {t:.2f} units')
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()


