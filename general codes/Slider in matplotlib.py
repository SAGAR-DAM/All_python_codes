# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:30:33 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Example data
x = np.linspace(0, 10, 1000)
def Bz_function(x,t):
    return (np.sin(10*x-10*t)*np.exp(-(x-t)**2))**2


# Initial plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
t_initial = 0
Bz = Bz_function(x,t_initial)
line, = ax.plot(x, Bz, 'r')
ax.set_xlabel('x (code units)')
ax.set_ylabel('Bz (code units)')
time_text = ax.set_title(f't = {t_initial:.2f} code units')
#ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
ax.set_ylim(-1,1)
# Slider setup
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'time', 0, 10, valinit=t_initial, valstep=0.1)

# Update function for the slider
def update(val):
    t = slider.val
    line.set_ydata(Bz_function(x,t))
    time_text.set_text(f't = {t:.2f} code units')
    #ax.set_ylim(min(Bz_function(x,t)),max(Bz_function(x,t)))
    ax.set_ylim(-1,1)
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

plt.show()
