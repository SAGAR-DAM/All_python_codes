'''
Image processing using python
Using pillow

creating the Horizomntal and vertical linecut module
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage import io    # image processing tool
from PIL import Image



my_img_plw=Image.open("D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_33.png")   # load the image 
print(type(my_img_plw)) 
print(my_img_plw.format)

#my_img_plw.show()    # To show the image
my_image=np.asarray(my_img_plw)   #converting the image to a numpy array
plt.imshow(my_image)
plt.show()

# taking the horizontal linecut for intensity of an image
hlinecut_no=320    #for the horizontal line by pixel number
hlinecut=[]
for i in range(my_image.shape[1]):
    y=np.sqrt((my_image[hlinecut_no,i][0])**2+(my_image[hlinecut_no,i][1])**2+(my_image[hlinecut_no,i][2])**2)
    hlinecut.append(y)
    
plt.plot(hlinecut,'r-')
plt.show()


# taking the vertical linecut for intensity of an image
vlinecut_no=400    #for the horizontal line by pixel number
vlinecut=[]
for i in range(my_image.shape[0]):
    y=np.sqrt((my_image[i,vlinecut_no][0])**2+(my_image[i,vlinecut_no][1])**2+(my_image[i,vlinecut_no][2])**2)
    vlinecut.append(y)

plt.plot(np.arange(my_image.shape[0]),vlinecut,'g-')
plt.show()



##########################################
'''  reading image with matplotlib'''
##########################################

my_image1=plt.imread("D:\\data Lab\\frog\\Granouille allignment\\16jan2023\\ids raw\\5_33.png")    # readingthe image with matplotlib.image module
#print(my_image1)
print(my_image1.shape)
my_image1=np.array(my_image1*255,dtype=int)   # Making the image from floating to int RGB format
 
plt.imshow(my_image1)