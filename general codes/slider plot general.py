# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:25:12 2024

@author: mrsag
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def make_slider_plot(f, x, t_range, initial_t, rest, xlabel='x', ylabel='y', title_template='t = {:.2f}'):
    """
    Create a 2D plot with a slider to control parameter t in function f(x, t).

    Parameters:
    - f: function, the function to plot. It should take two arguments (x, t).
    - x: array-like, the x values to use for plotting.
    - t_range: tuple, the range of t values (min_t, max_t).
    - initial_t: float, the initial value of t for the plot.
    - xlabel: string, label for the x-axis.
    - ylabel: string, label for the y-axis.
    - title_template: string, template for the title to display the current value of t.
    """
    # Initial plot setup
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    t_initial = initial_t
    y = f(x, t_initial)
    line, = ax.plot(x, y, 'r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = ax.set_title(title_template.format(t_initial))

    # Slider setup
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'time', t_range[0], t_range[1], valinit=t_initial, valstep=(t_range[1] - t_range[0]) / rest)

    # Update function for the slider
    def update(val):
        t = slider.val
        line.set_ydata(f(x, t))
        title.set_text(title_template.format(t))
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()

# # Example usage
# if __name__ == "__main__":
# Example function
def example_function(x, t):
    return np.sin(x - t)#*np.exp(-0.1*(t-5)**2)

# x values
x = np.linspace(0, 10, 100)

# t range
t_range = (0, 10)

# Initial t value
initial_t = 0

# Call the function
make_slider_plot(example_function, x, t_range, initial_t, rest=100, xlabel='x', ylabel='f(x)', title_template='t = {:.2f}')
