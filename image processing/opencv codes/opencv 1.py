# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:12:46 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


image_path="D:\\Codes\\image processing\\opencv codes\\test2.jpg"
bgr_image=plt.imread(image_path,1)

image=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
print(image.shape)


b,g,r=cv2.split(image)

cv2.imshow("Red",r)
cv2.imshow("green",g)
cv2.imshow("blue",b)


# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()


