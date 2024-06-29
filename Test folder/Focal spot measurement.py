# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:53:08 2023

@author: Sagar Dam
"""
# A CODE FOR GENERAL LINECUT THROUGH SOME POINT AND TOWARDS ALL DIRECTIONS.
''' 
A code to find the FWHM of the focal spot from different direction
'''


import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageDraw
from scipy.optimize import curve_fit as fit



import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

def draw_on_image(image,x,y):             #function  to draw a line on the image for two given points
    draw = ImageDraw.Draw(image)
    draw.line((x[1],x[0],y[1],y[0]), fill=(255, 0, 0), width=1)
    return(image)
    
def linecut_function(start_point,end_point,image,image2):     #function to give the linecut along the line between two given points
    # Use the numpy function linspace to get the coordinates of the line
    num=round(np.sqrt((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2))      #number of pixels between these two points
    x, y = np.linspace(start_point[0], end_point[0], num), np.linspace(start_point[1], end_point[1], num)     #pixels along x and y
    image2=draw_on_image(image2, start_point,end_point)   #drawing the line between the given points
    # Get the grayscale values along the line
    gray_values = image[x.astype(int),y.astype(int)]
    linecut=[]
    for i in range(len(gray_values)):
        linecut_value=np.sqrt((gray_values[i][0])**2+(gray_values[i][1])**2+(gray_values[i][2])**2)   # taking the grey values along the linecut line
        linecut.append(linecut_value)
        
    return(np.array(linecut),image2)      # return back the linecut array and the image with lines along the linecuts

def Gauss1(x,b,x0):                
    y=np.exp(-(x-x0)**2/(2*b**2))
    return y

def Gaussfit(w,I):      # function for gaussian fitting a data
    xdata=w             # x data array
    ydata=I             # y data array
    ymax_index=(list(ydata)).index(max(ydata))
    xmax_val=xdata[ymax_index]
    xdata=xdata-xmax_val
    parameters, covariance = fit(Gauss1, xdata, ydata,maxfev=100000)  # fitting the data to get the parameters
    fit_y = Gauss1(xdata, *parameters)    # get the proper gaussian with the fit parameters

    
    xdata=xdata+xmax_val
    fit_y=np.asarray(fit_y)
    
    return fit_y,parameters   # return the fit and the parameters


#plt.imshow(image)


def main():
    files=glob.glob("D:\\data Lab\\Supercontinuum-march 2023\\10th march 2023\\Focal spot measurement\\Raw image\\*.bmp")
    all_fwhm_list=[]
    
    for i in range(len(files)):
        print(files[i])
        image = io.imread(files[i])
        image2 = Image.open(files[i])
        
        if (i==0):
            point=[521,525]
        elif (i==1):
            point=[518,524]
        elif (i==2):
            point=[517,516]
        elif (i==3):
            point=[508,517]
        elif (i==4):
            point=[508,509]
        elif (i==5):
            point=[506,514]
        elif (i==6):
            point=[500,490]
        elif (i==7):
            point=[506,485]
        elif (i==8):
            point=[502,461]
        elif (i==9):
            point=[509,444]
        
        #####################################################################################
        radius=90          # the radius of the circle in which the linecuts are drawn
        x=point[0]         #x coordinate (row of the matrix)
        y=point[1]         #y coordinate (column of the matrix)
    
        X=image.shape[0]-1       #image size in x direction
        Y=image.shape[1]-1       #image size in y direction
    
        boundary_distances=[x,y,X-x,Y-y]    # distances of boundary from the given point
    
        #radius=min(boundary_distances)
    
        #####################################################################################
        theta_degree=np.linspace(0,160,9)    # angels, for which the linecuts will be drawn
        theta=theta_degree*np.pi/180          # angels in radian
        FWHM=[]                 
        ####################################################################################
        
        #plt.imshow(image)
        #plt.show()
        filename_at_caption=(files[i]).replace("D:\\data Lab\\Supercontinuum-march 2023\\10th march 2023\\Focal spot measurement\\Raw image\\","")
        print(filename_at_caption)
        
        
        for j in range(len(theta)):
            x1=round(x+radius*np.sin(theta[j]))
            y1=round(y-radius*np.cos(theta[j]))
            
            x2=round(x-radius*np.sin(theta[j]))
            y2=round(y+radius*np.cos(theta[j]))
            
            start_point=[x1,y1]
            end_point=[x2,y2]
            linecut,image2=linecut_function(start_point,end_point,image,image2)   #getting the linecut along the start and end points
            linecut=linecut/linecut.max()   #normalizing the linecut grey values
            
            fit_linecut,parameters=Gaussfit(np.arange(len(linecut)),linecut)    #Gaussian fit for the linecut
            fwhm=2.355*parameters[0]    # getting the FWHM along the linecut
            FWHM.append(fwhm)
            
            
            plt.plot(linecut,label='linecut')
            plt.plot(fit_linecut,label='Gaussian fit')
            plt.legend()
            plt.title("Normalized Linecut at:\npixel no: (%d"%x+",%d"%y+")    theta=%d"%(theta_degree[j])+"\n of image: %s"%filename_at_caption)
            plt.xlabel('start: (%d'%start_point[0]+',%d'%start_point[1]+')      end: (%d'%end_point[0]+',%d'%end_point[1]+")\n FWHM = %f"%fwhm)
            plt.show()
    
        image2=np.asarray(image2)
        
        plt.figure()
        plt.imshow(image2[x-radius:x+radius,y-radius:y+radius])
        plt.title("%s"%filename_at_caption)
        plt.xlabel("Average FWHM from all directions: %f"%np.mean(FWHM))
        plt.savefig("D:\\data Lab\\Supercontinuum-march 2023\\10th march 2023\\Focal spot measurement\\Draw on images\\%s.jpg"%filename_at_caption)
        plt.show()
        #print("Average FWHM from all directions: ", np.mean(FWHM))
        
        all_fwhm_list.append(np.mean(FWHM))
    
    print(f"FWHM of differnet images: {all_fwhm_list}")
    print(f"Min spot size: {min(all_fwhm_list)}")
        
if __name__=='__main__':
    main()