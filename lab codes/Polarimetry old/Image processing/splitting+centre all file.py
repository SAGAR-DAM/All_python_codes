# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:40:57 2023

@author: sagar
"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import color,io,img_as_ubyte,feature
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy.optimize import curve_fit
from skimage.draw import line
import time
import glob

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

t_start=time.time()
#########################################################################
#########################################################################
'''Defining required functions'''
#########################################################################
#########################################################################

def draw_on_image(image,x,y):
    image2=np.copy(image)
    rr, cc = line(x[0],x[1],y[0],y[1])
    line_width = 3
    for i in range(-line_width//2, line_width//2 + 1):
        image2[np.clip(rr + i, 0, image.shape[0] - 1), np.clip(cc + i, 0, image.shape[1] - 1)] = [255,0,0]  # Set the color of the line

    #image2[rr, cc] = [1]  # Set the color of the line (red in this example)
    return(image2)




def draw_on_image_pillow(image,x,y):
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

#########################################################################
#########################################################################
'''Giving input files path'''
#########################################################################
#########################################################################

def image_processing(image_path):
    #image_path="D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\17Aug23_par\\Par\\P_110.tif"
    
    image=io.imread(image_path,as_gray=True)
    #image=color.rgba2rgb(image)
    #image=color.rgb2gray(image)    # as image entropy only works with grayscale image
    #image=resize(image,(1000,1000))
    
    imagergb=color.gray2rgb(img_as_ubyte(image))    # make a copy of the image in RGB form to draw the lines
    
    print(image.ndim)              # if image.ndim=2 then only entropy function works
    print(image.shape)
    
    ###########################################################
    
    plt.imshow(image)
    plt.title("Main image")
    plt.axis("off")
    plt.show()
    
    ###########################################################
    
    # Define the points
    x = (3*image.shape[0]//8,3*image.shape[1]//8)
    y = (5*image.shape[0]//8,5*image.shape[1]//8)
    
    
    image2=draw_on_image(imagergb, x, y)    # draw  on the RGB image to show the defined lines
    image2=image2*[5,5,0]       # make the other parts of the line defined image brighter and show them in required colour 
    binary=imagergb*[0,10,0]
    
    '''
    plt.imshow(image2)
    plt.axis("off")
    plt.show()
    '''
    
    no_of_points=30
    
    points_x=np.linspace(x[0],y[0],no_of_points)
    points_y=np.linspace(x[1],y[1],no_of_points)
    for i in range(len(points_x)):
        points_x[i]=int(points_x[i])
        points_y[i]=int(points_y[i])
        
    #########################################################################
    #########################################################################
    '''  Finding the linecut values along all the required direction through given points   '''
    #########################################################################
    #########################################################################
    
    integral_linecut_matrix=[]
    for i in range(len(points_x)):
        #radius=min([points_x[i],image.shape[0]-points_x[i],points_y[i],image.shape[1]-points_y[i]])-1
        radius=image.shape[0]/np.sqrt(2)/2
        theta_degree=np.linspace(0,90,19)    # angels, for which the linecuts will be drawn
        theta=theta_degree*np.pi/180          # angels in radian
        #linecut_ones=[]
        integral_linecut=[]
        for j in range(len(theta)):
            x1=round(points_x[i]+radius*np.sin(theta[j]))
            y1=round(points_y[i]-radius*np.cos(theta[j]))
            
            x2=round(points_x[i]-radius*np.sin(theta[j]))
            y2=round(points_y[i]+radius*np.cos(theta[j]))
            
            start_point=[x1,y1]
            end_point=[x2,y2]
            
            #binary2=draw_on_image(binary2,start_point,end_point)
            linecut,image2=linecut_function(start_point,end_point,image,image2)
            #plt.plot(linecut)
            #plt.title("pixel no: (%d"%x+",%d"%y+")    theta=%d"%(theta_degree[i]))
            #plt.xlabel('start: %d'%start_point[0]+',%d'%start_point[1]+'\n end: %d'%end_point[0]+',%d'%end_point[1])
            #plt.show()
            #number_of_ones=len([value for value in linecut if value > 0.5])
            
            integral_linecut.append(sum(linecut))
        #linecut_ones=np.array(linecut_ones)
        integral_linecut_matrix.append(integral_linecut)
        
    plt.imshow(image2)
    plt.axis('off')
    plt.title("defined lines")
    plt.show()
    
    #########################################################################
    #########################################################################
    '''   Finding the best line to cut the image   '''
    #########################################################################
    #########################################################################
    
    #print(integral_linecut_matrix)
    integral_linecut_matrix=np.array(integral_linecut_matrix)
    m=np.min(integral_linecut_matrix)
    print(m)
    b = np.where(integral_linecut_matrix==m)
    print(b)
    bpi=b[0][0]
    bai=b[1][0]
    print(bpi,bai)
    print(points_x[bpi],points_y[bpi])
    print(theta_degree[bai])
    #print(theta_degree)
    
    #########################################################################
    #########################################################################
    ''' Showing the line of cutting the image '''
    #########################################################################
    #########################################################################
    
    X=image.shape[0]-1
    Y=image.shape[1]-1
    
    if(points_x[bpi]<=binary.shape[0]//2):
        if(theta[bai]<=np.arctan(points_x[bpi]/(Y-points_y[bpi]))):
            x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
            y1=0
            
            x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
            y2=Y
            
        elif(np.arctan(points_x[bpi]/(Y-points_y[bpi])) < theta[bai] <= np.arctan((X-points_x[bpi])/points_y[bpi])):
            x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
            y1=0
            
            x2=0
            y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
        
        elif(np.arctan((X-points_x[bpi])/points_y[bpi])<theta[bai]<np.pi/2):
            x1=X
            y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
            
            x2=0
            y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
            
        else:
            x1=X
            y1=int(points_y[bpi])
            
            x2=0
            y2=y1
            
    elif(points_x[bpi]>binary.shape[0]//2):
        if(theta[bai]<=np.arctan((X-points_x[bpi])/points_y[bpi])):
            x1=int(points_x[bpi]+points_y[bpi]*np.tan(theta[bai]))
            y1=0
            
            x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
            y2=Y
            
        elif(np.arctan((X-points_x[bpi])/points_y[bpi]) < theta[bai] <= np.arctan(points_x[bpi]/(Y-points_y[bpi]))):
            x1=X
            y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
            
            x2=int(points_x[bpi]-(Y-points_y[bpi])*np.tan(theta[bai]))
            y2=Y
            
        elif(np.arctan(points_x[bpi]/(Y-points_y[bpi])) < theta[bai] < np.pi/2):
            x1=X
            y1=int(points_y[bpi]-((X-points_x[bpi])/np.tan(theta[bai])))
            
            x2=0
            y2=int(points_y[bpi]+points_x[bpi]/np.tan(theta[bai]))
            
        else:
            x1=X
            y1=int(points_y[bpi])
            
            x2=0
            y2=y1
            
    
    start_point=[x1,y1]
    end_point=[x2,y2]
    
    
    linecut,binary2=linecut_function(start_point,end_point,binary,binary)
    
    plt.imshow(binary2)
    plt.axis("off")
    plt.show()
    
    
    #########################################################################
    #########################################################################
    '''   cutting the image along the line of cut and showing them  '''
    #########################################################################
    #########################################################################
    
    
    X=image.shape[0]
    Y=image.shape[1]
    
    base_noise=np.mean(image[3*X//4:,:Y//4])
    imagetl = np.asarray([[base_noise]*Y]*X)
    imagebr = np.asarray([[base_noise]*Y]*X)
    
    for i in range(X):
        for j in range(Y):
            if(i/np.tan(theta[bai])+j>=points_y[bpi]+points_x[bpi]/np.tan(theta[bai])):
                imagebr[i,j]=image[i,j]
                #imagetl[i,j]=base_noise*np.random.rand()
            else:
                imagetl[i,j]=image[i,j]
                #imagebr[i,j]=base_noise*np.random.rand()
                
    
    plt.imshow(imagetl)
    plt.axis('off')
    plt.title("Top left")
    plt.show()
    
    plt.imshow(imagebr)
    plt.axis('off')
    plt.title("Bottom right")
    plt.show()
    
    #########################################################################
    #########################################################################
    '''Making imagws at centre'''
    #########################################################################
    #########################################################################
    
    def brightest(image):
        # Find the coordinates of the brightest point using the corner_peaks function
        if(image.ndim==2):
            coords = feature.corner_peaks(np.abs(image), min_distance=10)
        elif(image.ndim==3):
            if(image.shape[2]==3):
                coords = feature.corner_peaks(np.abs(image), min_distance=10)
            elif(image.shape[2]==4):
                coords = feature.corner_peaks(np.abs(image), min_distance=10)
        return(coords[0])
    
    def image_centre_with_roll(image):
        #brightest_point=brightest(image)
        bp_index=np.where(image==np.max(image))
        brightest_point=[bp_index[0][0],bp_index[1][0]]
        image=np.roll(np.roll(image,-(brightest_point[0]-512),axis=0),-(brightest_point[1]-512),axis=1)
        X=image.shape[0]
        Y=image.shape[1]
        #image=image[X//8:7*X//8,Y//8:7*Y//8]
        return(image)
    
    imagetl_centre=image_centre_with_roll(imagetl)
    imagetl_centre=imagetl_centre.astype(np.uint16)
    plt.imshow(imagetl_centre)
    plt.axis("off")
    plt.show()
    
    imagebr_centre=image_centre_with_roll(imagebr)
    imagebr_centre=imagebr_centre.astype(np.uint16)
    plt.imshow(imagebr_centre)
    plt.axis("off")
    plt.show()
    
    name_tl=image_path[-9:-4]
    name_br=name_tl
    
    name_tl=name_tl+"_tl"
    name_br=name_br+"_br"
    
    io.imsave("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\%s.tif"%name_tl,imagetl_centre)
    io.imsave("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross_splitted_image\\%s.tif"%name_br,imagebr_centre)




files=glob.glob("D:\\data Lab\\Aug 2023 Anandam Polarimetry Magfield\\18Aug23_cross\\cross\\*.tif")
for i in range(len(files)):
    image_processing(files[i])

print("Total time: ",time.time()-t_start)