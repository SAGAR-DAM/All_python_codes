# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:42:01 2023

@author: sagar
"""

# import the Python Image
# processing Library
from PIL import Image

# Giving The Original image Directory
# Specified
Original_Image = Image.open("D:\\Codes\\complex field sets\\quadro dragon 12 1.png")

# Rotate Image By 180 Degree
#rotated_image1 = Original_Image.rotate(180)

# This is Alternative Syntax To Rotate
# The Image
#rotated_image2 = Original_Image.transpose(Image.ROTATE_90)

# This Will Rotate Image By 60 Degree
rotated_image3 = Original_Image.rotate(45)

#rotated_image1.show()
#rotated_image2.show()
rotated_image3.show()
