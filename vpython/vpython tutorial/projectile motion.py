# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:06:12 2023

@author: sagar
"""

# vpython code to make projectile motion
from vpython import *

# Set up the scene
#scene = canvas(title='Projectile Motion', width=1600, height=800, range=20)
floor = box(pos=vector(40-40,-2,0), size=vector(80,1,20), color=color.green)
ball = sphere(pos=vector(0-40,0,0), radius=1, color=color.red)
#barrier1=box(pos=vector(-1-40,3,0),size=vector(2,10,1),color=color.green)

# Set up the initial conditions
v0 = vector(20, 10, 0)   # Initial velocity
g = vector(0, -9.8, 0)   # Acceleration due to gravity
dt = 0.001                # Time step

# Loop to update the position of the ball
while (ball.pos.x<=(floor.pos.x+floor.size.x/2)):
    rate(500)
    ball.pos = ball.pos + v0*dt
    if (ball.pos.y<=-1):
        v0.y = -v0.y
    elif(ball.pos.y>-1):
        v0=v0+g*dt