# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:25:55 2023

@author: sagar
"""



from turtle import *

#function to create koch snowflake or koch curve
def circle(l):
    for i in range (361):
        forward(l)
        right(1)
# main function
if __name__ == "__main__":

    # defining the speed of the turtle
    speed(1000)				
    length = 1			

    # Pull the pen up – no drawing when moving.
    penup()					
    
    # Move the turtle backward by distance,
    # opposite to the direction the turtle
    # is headed.
    # Do not change the turtle’s heading.
    backward(length/2.0)		
    left(90)
    forward(200)
    right(90)

    # Pull the pen down – drawing when moving.
    pendown()		

    circle(length)
    
    hideturtle()
    # To control the closing windows of the turtle
    mainloop()