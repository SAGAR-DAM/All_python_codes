from vpython import *
from numpy import sin,cos,pi

# Set up scene and objects
scene = canvas(title='Double Pendulum Simulation')
ceiling = box(pos=vector(0,0,0), size=vector(0.2,0.2,0.2), color=color.white)
bob1 = sphere(pos=vector(0,-1,0), radius=0.1, color=color.red)
rod1 = cylinder(pos=ceiling.pos, axis=bob1.pos-ceiling.pos, radius=0.02, color=color.white)
bob2 = sphere(pos=bob1.pos-rod1.axis, radius=0.1, color=color.blue)
rod2 = cylinder(pos=bob1.pos, axis=bob2.pos-bob1.pos, radius=0.02, color=color.white)

# Set up initial conditions
g = 9.81
theta1 = 120*pi/180
theta2 = 180*pi/180
L1 = 2.75
L2 = 0.75
m1 = 10
m2 = 10.5
omega1 = 0
omega2 = 0

# Set up time step and time variables
dt = 0.01
t = 0

# Run simulation
while t < 20:
    rate(100)
    
    # Calculate angular accelerations
    alpha1 = (-g*(2*m1+m2)*sin(theta1) - m2*g*sin(theta1-2*theta2) - 2*sin(theta1-theta2)*m2*(omega2**2*L2+omega1**2*L1*cos(theta1-theta2))) / (L1*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
    alpha2 = (2*sin(theta1-theta2)*(omega1**2*L1*(m1+m2)+g*(m1+m2)*cos(theta1)+omega2**2*L2*m2*cos(theta1-theta2))) / (L2*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
    
    # Update angular velocities and angles
    omega1 += alpha1*dt
    omega2 += alpha2*dt
    theta1 += omega1*dt
    theta2 += omega2*dt
    
    # Update positions of objects
    bob1.pos = vector(L1*sin(theta1),-L1*cos(theta1),0)
    bob2.pos = bob1.pos + vector(L2*sin(theta2),-L2*cos(theta2),0)
    rod1.axis = bob1.pos - ceiling.pos
    rod2.pos = bob1.pos
    rod2.axis = bob2.pos - bob1.pos
    
    # Update time
    t += dt
