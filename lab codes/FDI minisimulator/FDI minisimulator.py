import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
from decimal import Decimal
import required_functions as rf

import matplotlib
matplotlib.rcParams["figure.dpi"]=300

c=3e5       # in nm/ps

def simulate_pulse(pulse1,pulse2,time_delay, animsave):
    pulse1_info="Pulse 1 \n________\n"+r"$\tilde E_0: $"+f"{pulse1.E0}\n"+r"$\omega_0$: "+f"{pulse1.omega0}\n"+r"$\sigma_{\omega}$: "+f"{pulse1.sigma}\n"+r"$\phi_0$: "+f"{pulse1.phi0}\n"+r"$b_1$: "+f"{pulse1.b1}\n"+r"$b_2$: "+f"{pulse1.b2}\n"+r"$b_3$: "+f"{pulse1.b3}\n"+r"$b_4$: "+f"{pulse1.b4}"
    pulse2_info="Pulse 2 \n________\n"+r"$\tilde E_0: $"+f"{pulse2.E0}\n"+r"$\omega_0$: "+f"{pulse2.omega0}\n"+r"$\sigma_{\omega}$: "+f"{pulse2.sigma}\n"+r"$\phi_0$: "+f"{pulse2.phi0}\n"+r"$b_1$: "+f"{pulse2.b1}\n"+r"$b_2$: "+f"{pulse2.b2}\n"+r"$b_3$: "+f"{pulse2.b3}\n"+r"$b_4$: "+f"{pulse2.b4}"

    fig, ax = plt.subplots()
    minw=4700   #pulse1.omega[0]        # good range: 4680
    maxw=4740   #pulse1.omega[-1]       # good range: 4760

    line, = ax.plot([], [], 'r-', lw = 1.0, label="Superposed Pulse")
    # Add text annotation for time delay
    time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12, color='cyan',fontdict={'family': 'Times New Roman', 'style': 'italic', 'weight': 'bold'})
    pulse1_text = ax.text(0.15, 0.75, '', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='yellow')
    pulse2_text = ax.text(0.95, 0.75, '', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='yellow')

    ax.set_title("Animation of FDI signal of two pulses vs time", color="blue", fontsize=12, fontname="cursive", fontweight='bold')
    ax.set_xlim(minw,maxw)
    ax.set_ylim(-1,4*max([pulse1.E0,pulse2.E0])**2)
    ax.set_xlabel(r"$\omega$  (rad/ps)", fontsize = 5)
    ax.set_ylabel(r"$\tilde I(\omega)$", fontsize = 5)
    ax.tick_params(labelsize=5)  # Adjust the font size as needed
    ax.tick_params(labelsize=5)  # Adjust the font size as needed
    ax.legend(loc='upper left', facecolor="black", labelcolor="white", fontsize=7)
    ax.set_facecolor("black")

    # Animation function
    def update(frame):
        if frame < len(time_delay):
            delay=time_delay[frame]
            print(delay)
            combined_E,combined_I=rf.superposition(pulse1, pulse2, delay)
            line.set_data(pulse1.omega, combined_I)
            # Update the time delay text
            time_text.set_text(f'Time Delay: {delay:.2f} ps')
            pulse1_text.set_text(pulse1_info)
            pulse2_text.set_text(pulse2_info)
        return line, time_text, pulse1_text, pulse2_text

    # Create animation
    # Set the frame rate to 30 frames per second
    frame_rate = 30
    interval = 1000 / frame_rate  # Calculate interval in milliseconds
    animation = FuncAnimation(fig, update, frames=len(time_delay)+1, interval=interval, blit=True)

    if (animsave==True):
        animation.save("D:\\Codes\\lab codes\\FDI minisimulator\\FDI animation vs time.gif", writer='pillow')

    # Show the plot
    plt.show()


    #   Define the required pulses   #

w1=399.5        # in nm
w2=399.5        # in nm

# The pulse central frequensy will be calculated from the central  wavelength.
# for 400 nm light omega0 ~ 4720 rad/ps
# sigma is the std of the Gaussian of the spectral profile. 
# All the units of sigma, b1, b2, ... are in angular frequency (omega) term.
# sigma ~ (1-10) rad/ps;  b1 ~ 0.5  

pulse1=rf.pulse(E0=7, omega0=int(2*np.pi*c/w1), sigma=5, phi0=0, b1=1, b2=0.00, b3=0.00, b4=0.00)
pulse2=rf.pulse(E0=7, omega0=int(2*np.pi*c/w2), sigma=5, phi0=0, b1=pulse1.b1, b2=0.00, b3=0.09, b4=0.00)
time_delay=np.linspace(1,10,1001)
animsave = False


simulate_pulse(pulse1=pulse1, pulse2=pulse2, time_delay=time_delay, animsave=animsave)