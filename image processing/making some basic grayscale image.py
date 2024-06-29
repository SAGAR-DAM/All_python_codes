# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:04:22 2023

@author: sagar
"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io,color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np

image=[[0.0,0.1,0.2,0.3,0.4,0.5],[0.1,0.2,0.6,0.5,0.8,0.5],[0.5,0.2,0.2,0.5,0.0,1.5]]

plt.imshow(image)
plt.axis('off')
plt.show()