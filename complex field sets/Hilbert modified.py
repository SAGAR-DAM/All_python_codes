# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:25:55 2023

@author: sagar
"""

from turtle import *

def Hilbert(L,level,prime):
    global l
    global curvelength
    
    if(prime==0):
        if (level==0):
            forward(L/3)
            right(90)
            forward(L/3)
            right(90)
            forward(L/3)
            curvelength+=L
        elif(level%2==1):
            L /= 2
            #l=L/10
            Hilbert(L,level-1,1)
            right(90)
            forward(l)
            Hilbert(L,level-1,0)
            left(90)
            forward(l)
            left(90)
            Hilbert(L,level-1,0)
            forward(l)
            right(90)
            Hilbert(L,level-1,1)
            curvelength+=3*l
            
        elif(level%2==0):
            L /=2
            #l=L/10
            Hilbert(L,level-1,1)
            forward(l)
            right(90)
            Hilbert(L,level-1,0)
            forward(l)
            Hilbert(L,level-1,0)
            right(90)
            forward(l)
            Hilbert(L,level-1,1)
            curvelength+=3*l
            
            
    elif(prime==1):
        if(level==0):
            forward(L/3)
            left(90)
            forward(L/3)
            left(90)
            forward(L/3)
            curvelength+=L
            
        elif(level%2==1):
            L /=2
            #l=L/10
            Hilbert(L,level-1,0)
            left(90)
            forward(l)
            Hilbert(L,level-1,1)
            right(90)
            forward(l)
            right(90)
            Hilbert(L,level-1,1)
            forward(l)
            left(90)
            Hilbert(L,level-1,0)
            curvelength+=3*l
            
        elif(level%2==0):
            L /=2
            #l=L/10
            Hilbert(L,level-1,0)
            forward(l)
            left(90)
            Hilbert(L,level-1,1)
            forward(l)
            Hilbert(L,level-1,1)
            left(90)
            forward(l)
            Hilbert(L,level-1,0)
            curvelength+=3*l






# main function
if __name__ == "__main__":

    # defining the speed of the turtle
    speed(5000)
    length = 600.0
    curvelength=0
    
    # Pull the pen up – no drawing when moving.
    penup()
    
    # Move the turtle backward by distance,
    # opposite to the direction the turtle
    # is headed.
    # Do not change the turtle’s heading.
    backward(100)
    right(90)
    forward(300)
    left(90)
    left(90)
    # Pull the pen down – drawing when moving.
    pendown()
    
    level=4
    l=length/(2**level*3)
    
    Hilbert(length,level,0)
    
    hideturtle()
    print(curvelength)
    # To control the closing windows of the turtle
    mainloop()