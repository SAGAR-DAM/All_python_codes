# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 09:38:31 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from PIL import Image
import pygame

import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display
##########################################################################################
##########################################################################################
'''  All Required parameters and defined objects (Double pendulum) '''
##########################################################################################
##########################################################################################

g=981
t=np.linspace(0,30,30001)


class double_pendulum:
    def __init__(self,m1,m2,L1,L2,theta1,theta2,theta1_dot,theta2_dot):
        self.m1=m1
        self.m2=m2
        self.L1=L1
        self.L2=L2
        self.theta1=theta1
        self.theta2=theta2
        self.theta1_dot=theta1_dot
        self.theta2_dot=theta2_dot
        self.initial_state=(theta1,theta2,theta1_dot,theta2_dot)
        self.propagator()
        
    def equations(self,states,t):
        global g
        t1,t2,o1,o2=states
        t1_dot=o1
        t2_dot=o2
        delta_t=t1-t2
        
        o1_dot=(self.m2*g*np.sin(t2)*np.cos(delta_t)-self.m2*np.sin(delta_t)*(self.L1*o1**2*np.cos(delta_t)+self.L2*o2**2)-(self.m1+self.m2)*g*np.sin(t1))/(self.L1*(self.m1+self.m2*(np.sin(delta_t))**2))
        
        o2_dot=((self.m1+self.m2)*(self.L1*o1**2*np.sin(delta_t)-g*np.sin(t2)+g*np.sin(t1)*np.cos(delta_t))+self.m2*self.L2*o2**2*np.sin(delta_t)*np.cos(delta_t))/(self.L1*(self.m1+self.m2*(np.sin(delta_t))**2))
        
        derivatives=np.array([t1_dot,t2_dot,o1_dot,o2_dot])
        
        return(derivatives)
    
    def propagator(self):
        global t
        sol=odeint(self.equations,self.initial_state,t)
        sol=sol.T
        self.sol=sol
    
##########################################################################################
##########################################################################################
'''  defining the pendulums with initial conditions '''
##########################################################################################
##########################################################################################

p1=double_pendulum(m1=40, m2=10, L1=120, L2=150, theta1=np.pi, theta2=np.pi, theta1_dot=0.01, theta2_dot=0.01)
p2=double_pendulum(m1=40, m2=10, L1=120, L2=150, theta1=np.pi, theta2=np.pi, theta1_dot=0.0101, theta2_dot=0.01)
p3=double_pendulum(m1=10, m2=100, L1=120, L2=150, theta1=np.pi/6, theta2=np.pi/4, theta1_dot=0.0101, theta2_dot=0.01)

p=[p1]#,p2]#,p3]       # add the pendulums you want the animation for
savegif=False      # True if you want to save the gif. False if not.

plt.plot(t,p1.sol[0],'k-',label=r'$\theta_1$')
plt.plot(t,p1.sol[1],'r-',label=r'$\theta_2$')
plt.plot(t,p1.sol[2],'g--',label=r'$\omega_1$')
plt.plot(t,p1.sol[3],'b--',label=r'$\omega_2$')
plt.title("Double pendulum animation",fontname="Times New Roman",fontsize=14)
plt.xlabel("Time",fontname="Times New Roman",fontsize=14)
plt.ylabel("values in code unit",fontname="Times New Roman",fontsize=14)
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.legend()
plt.show()

##########################################################################################
##########################################################################################
'''  Pygame visulization '''
##########################################################################################
##########################################################################################

# Screen settings
width, height = 600, 600
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Double Pendulum Animation")
pivot = (width // 2, 300)

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green =(0,255,0)
blue=(0,0,255)
red1=(120,0,0)
green1=(0,120,0)
blue1=(0,0,120)
yellow=(255,255,0)
magenta=(255,0,255)
cyan=(0,255,255)

colors=[white,red,green,blue,red1,green1,blue1,yellow,magenta,cyan]
len_c=len(colors)



##########################################################################################
##########################################################################################
'''  Pygame running '''
##########################################################################################
##########################################################################################


no_of_pendulum=len(p)
clock = pygame.time.Clock()
frames=[]
j=0
font = pygame.font.Font(None, 36)  # Use the default system font
running = True

''' Main Loop '''

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Clear the screen
    screen.fill(black)
    
    if j<len(t):
        text = font.render("Time:  %.3f"%(t[j]), True, (255, 255, 255))
        screen.blit(text, (20, 20))
        
        for i in range(len(p)):
            m1=p[i].m1
            m2=p[i].m2
            L1=p[i].L1
            L2=p[i].L2
            theta1=p[i].sol[0][j]
            theta2=p[i].sol[1][j]
            
            # Draw the pendulums
            pendulum1_x = pivot[0] + L1* np.sin(theta1)
            pendulum1_y = pivot[1] + L1* np.cos(theta1)
            pendulum2_x = pendulum1_x + L2 * np.sin(theta2)
            pendulum2_y = pendulum1_y + L2 * np.cos(theta2)
        
            pygame.draw.line(screen, colors[i], pivot, (pendulum1_x, pendulum1_y), 1)
            pygame.draw.line(screen, colors[i], (pendulum1_x, pendulum1_y), (pendulum2_x, pendulum2_y), 1)
            pygame.draw.circle(screen, colors[-i-1], (int(pendulum1_x), int(pendulum1_y)), 10)
            pygame.draw.circle(screen, colors[-i-1], (int(pendulum2_x), int(pendulum2_y)), 10)
            
            if (savegif==True):
                pygame_image = pygame.surfarray.array3d(screen)
                pil_image = Image.fromarray(pygame_image)
                pil_image = pil_image.rotate(-90, expand=True)
                pil_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Append the frame to the list
                frames.append(pil_image)
    else:
        running=False
    j+=1
    pygame.display.flip()
    clock.tick(1000)
    
if(savegif==True):
    # Save the list of frames as a GIF
    frames[0].save("D:\\Codes\\matplotlib animation\\double_pendulum_animation.gif", save_all=True, append_images=frames[1:], loop=0, duration=50)


pygame.quit()

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()

