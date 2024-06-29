# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:15:02 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 500  # high-res display

amplitude = 5
pillar_width = 0.3
period = 1

def periodic_pillars(t):
    t = t % period  # Ensure t is within one period

    step_function = np.zeros_like(t)
    step_function[t <= pillar_width] = amplitude

    return step_function

t = np.linspace(-10, 10, 1000)
y = periodic_pillars(t)

plt.plot(t, y)
plt.show()
