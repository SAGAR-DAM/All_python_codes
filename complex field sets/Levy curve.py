# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:25:55 2023

@author: sagar
"""



from turtle import *
import numpy as np
#function to create Levy curve
def levy(L,level):
        if(level==0):
            forward(L)
        elif(level==1):
            L=L/np.sqrt(2)
            levy(L,0)
            left(90)
            levy(L,0)
        else:
            L=L/np.sqrt(2)
            levy(L,level-1)
            left(90)
            right(90*(level-1))
            levy(L,level-1)
        
# main function
if __name__ == "__main__":

    # defining the speed of the turtle
    speed(1000)				
    length = 300.0			

    # Pull the pen up – no drawing when moving.
    penup()					
    
    # Move the turtle backward by distance,
    # opposite to the direction the turtle
    # is headed.
    # Do not change the turtle’s heading.
    #backward(length/2.0)		
    left(90)
    forward(170)
    right(90)

    # Pull the pen down – drawing when moving.
    pendown()		

    level=6
    
    levy(length,level)
    hideturtle()
    # To control the closing windows of the turtle
    mainloop()