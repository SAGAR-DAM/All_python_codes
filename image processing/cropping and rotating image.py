'''
author: Sagar Dam
image processing with PIL
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
#from skimage import io    # image processing tool
from PIL import Image

# cropping image
img=Image.open("D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_33.png")

cropped_img=img.crop((0,0,300,300))

cropped_img=np.asarray(cropped_img)
plt.axis('off')
#plt.imshow(cropped_img)
#plt.show()

''' rotating image ''' 
img_rotated=img.rotate(90,expand=True)
plt.axis('off')
plt.imshow(img_rotated)
#plt.show()

