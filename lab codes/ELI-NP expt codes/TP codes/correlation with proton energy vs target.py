# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:37:48 2025

@author: mrsag
"""

import numpy as np
from skimage import io, transform,color,img_as_ubyte,img_as_float
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from skimage.draw import line
import math
import pandas as pd

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
#mpl.rcParams['font.weight'] = 'bold'
#mpl.rcParams['font.style'] = 'italic'  # Set this to 'italic'
mpl.rcParams['figure.dpi']=300 # highres display


# Read Excel file
df = pd.read_excel(r"D:\data Lab\ELI-NP March 2025\04_04_2025\target info for proton energy correlation.xlsx")

# df.head()

corr_matrix = df.corr()
print(corr_matrix["proton"].sort_values(ascending=True))  # Will show the correlation value (r) between Y and diffirent x


sorted_indices = np.argsort(df.iloc[:,0])
plt.plot(df.iloc[:,0][sorted_indices],df["proton"][sorted_indices],"bo-",lw=1)
plt.xlabel("substrate thickness (um)")
plt.ylabel("proton energy (MeV)")
plt.show()


sorted_indices = np.argsort(df.iloc[:,1])
plt.plot(df.iloc[:,1][sorted_indices],df["proton"][sorted_indices],"ro-",lw=1)
plt.xlabel("diameter (nm)")
plt.ylabel("proton energy (MeV)")
plt.show()

sorted_indices = np.argsort(df.iloc[:,2])
plt.plot(df.iloc[:,2][sorted_indices],df["proton"][sorted_indices],"go-",lw=1)
plt.xlabel("gap (nm)")
plt.ylabel("proton energy (MeV)")
plt.show()

sorted_indices = np.argsort(df.iloc[:,3])
plt.plot(df.iloc[:,3][sorted_indices],df["proton"][sorted_indices],"mo-",lw=1)
plt.xlabel("nanowire height (um)")
plt.ylabel("proton energy (MeV)")
plt.show()

