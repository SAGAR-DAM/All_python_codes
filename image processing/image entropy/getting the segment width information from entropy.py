"""
Created on Mon Jan 30 15:40:05 2023

@author: mrsag
"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import color,io
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

image_path='D:\\Codes\\image processing\\image entropy\\qrcode-big-content-2edited.png'

image=io.imread(image_path,as_gray=True)
#image=color.rgba2rgb(image)
#image=color.rgb2gray(image)    # as image entropy only works with grayscale image
print(image.ndim)              # if image.ndim=2 then only entropy function works

entropy_image=entropy(image, disk(14)) # creates a new image from the given image with different entropy (disorder) regions separated out by different colours
threshold=threshold_otsu(entropy_image)     # gives the best value for the pixels on the boundary of the segments of the entropy image

binary=entropy_image <= threshold   # creates a new image for which the pixel_value(entropy_image)<=threshold to give the proper segmentation 

print(threshold)

plt.imshow(image)
plt.title("Main image")
plt.axis("off")
plt.show()

plt.imshow(entropy_image)
plt.axis('off')
plt.title("Entropy image")
plt.show()


plt.imshow(binary)
plt.title("binary image")
plt.axis('off')
plt.show()

print("Total number of blue pixels: ",np.sum(binary==0))
print("Total number of yellow pixels: ",np.sum(binary==1))



###########################################################
'''  Taking the horizontal linecuts and fit... along vertical axis  '''
###########################################################

X=binary.shape[0]
Y=binary.shape[1]

image2=Image.open(image_path)

def draw_on_image(image,x,y):
    draw = ImageDraw.Draw(image)
    draw.line((x[1],x[0],y[1],y[0]), fill=(255, 0, 0), width=2)
    return(image)

def linecut_function(start_point,end_point,image,image2):
    # Use the numpy function linspace to get the coordinates of the line
    num=round(np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2))
    x, y = np.linspace(start_point[0], end_point[0], num), np.linspace(start_point[1], end_point[1], num)
    image2=draw_on_image(image2, start_point,end_point)
    # Get the grayscale values along the line
    gray_values = image[x.astype(int),y.astype(int)]
    linecut=[]
    for i in range(len(gray_values)):
        linecut_value=gray_values[i]
        linecut.append(linecut_value)
        
    return(np.array(linecut),image2)

startpoint_x=np.arange(0,X,20)
startpoint_y=np.zeros(len(startpoint_x))

endpoint_x=startpoint_x
endpoint_y=np.zeros(len(endpoint_x))
endpoint_y=endpoint_y+(Y-1)

linecut_ones=[]

for i in range(len(startpoint_x)):
    x1=startpoint_x[i]
    y1=startpoint_y[i]
    
    x2=endpoint_x[i]
    y2=endpoint_y[i]
    
    start_point=[x1,y1]
    end_point=[x2,y2]
    linecut,image2=linecut_function(start_point,end_point,binary,image2)
    plt.plot(linecut)
    #plt.title("pixel no: (%d"%x+",%d"%y+")    theta=%d"%(theta_degree[i]))
    plt.xlabel('start: %d'%start_point[0]+',%d'%start_point[1]+'\n end: %d'%end_point[0]+',%d'%end_point[1])
    plt.show()
    
    number_of_ones=len([value for value in linecut if value != 0])
    linecut_ones.append(number_of_ones)
    
image2=np.asarray(image2)    
plt.imshow(image2)
plt.axis('off')
plt.show()



def straightline(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def linefit(x,y):
    x=np.array(x)
    parameters,pcov = curve_fit(straightline, x,y) # your data x, y to fit
    line=straightline(x,*parameters)             
    return(line,*parameters)               

line,*parameters=linefit(startpoint_x,linecut_ones)

plt.plot(startpoint_x,linecut_ones,'ko-')
plt.plot(startpoint_x,line,'r-')
plt.title("Width of the segment changing along the vertical direction")
plt.xlabel("Vertical coordinate of the line \nSlope: %f"%parameters[0] +"    offset: %f"%parameters[1])
plt.ylabel("Width of the at the linecut")
plt.show()