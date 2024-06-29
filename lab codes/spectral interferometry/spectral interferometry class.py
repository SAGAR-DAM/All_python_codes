# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:03:55 2023

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams["figure.dpi"]=300

c=3e8

def gaussian(omega,E0,omega0,sigma):
    E=E0*np.exp(-(omega-omega0)**2/(2*sigma**2))
    return(E)

def phase_factor(omega,omega0,phi0,b1,b2,b3,b4):
    delta=omega-omega0
    phi=phi0+b1*delta+b2*delta**2+b3*delta**3+b4*delta**4
    e_i_phi=np.exp(1j*phi)
    return e_i_phi

class pulse:
    def __init__(self,E0,omega0,sigma,phi0,b1,b2,b3,b4):
        self.E0=E0
        self.omega0=omega0
        self.sigma=sigma
        self.phi0=phi0
        self.b1=b1
        self.b2=b2
        self.b3=b3
        self.b4=b4
        pulse.omega=np.linspace(393,407,4001)
        
    def pulse_profile(self):
        E_0=gaussian(pulse.omega, self.E0, self.omega0, self.sigma)
        phase=phase_factor(pulse.omega, self.omega0, self.phi0, self.b1, self.b2, self.b3, self.b4)
        E=E_0*phase
        return(E)
    
    def spectral_intensity(self):
        E=self.pulse_profile()
        I=(abs(E))**2
        return I
    
def superposition(pulse1,pulse2,time_delay):
    phase_lag = np.exp(-2j * np.pi * pulse1.omega * time_delay)
    combined_E=pulse1.pulse_profile()*phase_lag+pulse2.pulse_profile()
    combined_I=(abs(combined_E))**2
    return combined_E,combined_I
    
def FDI(pulse1,pulse2,time_delay):
    
    combined_E,combined_I=superposition(pulse1, pulse2, time_delay)
    
    
    plt.plot(pulse1.omega,pulse1.pulse_profile().real,label="E vs freq",lw=1)
    plt.plot(pulse1.omega,abs(pulse1.pulse_profile()),'k--',label="E field envelope",lw=1)
    plt.plot(pulse1.omega,pulse1.spectral_intensity(),label="spectral intensity",lw=1)
    plt.title("Pulse 1: \n"+r"$\tilde E(\omega)=$"+r"$\tilde E_0e^{-\frac{(\omega-\omega_0)^2}{2\sigma^2}}\times\exp(\phi_0+b_1\Delta\omega+b_2\Delta\omega^2+b_3\Delta\omega^3+b_4\Delta\omega^4)$"+"\n"+r"$\tilde E_0: $"+f"{pulse1.E0}; "+r"$\omega_0$: "+f"{pulse1.omega0}; "+r"$\sigma$: "+f"{pulse1.sigma}; "+r"$\phi_0$: "+f"{pulse1.phi0}; "+r"$b_1$: "+f"{pulse1.b1}; "+r"$b_2$: "+f"{pulse1.b2}; "+r"$b_3$: "+f"{pulse1.b3}; "+r"$b_4$: "+f"{pulse1.b4}")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
    #plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    plt.plot(pulse2.omega,pulse2.pulse_profile().real,label="E vs freq",lw=1)
    plt.plot(pulse2.omega,abs(pulse2.pulse_profile()),'k--',label="E field envelope",lw=1)
    plt.plot(pulse2.omega,pulse2.spectral_intensity(),label="spectral intensity",lw=1)
    plt.title("Pulse 2: \n"+r"$\tilde E(\omega)=$"+r"$\tilde E_0e^{-\frac{(\omega-\omega_0)^2}{2\sigma^2}}\times\exp(\phi_0+b_1\Delta\omega+b_2\Delta\omega^2+b_3\Delta\omega^3+b_4\Delta\omega^4)$"+"\n"+r"$\tilde E_0: $"+f"{pulse2.E0}; "+r"$\omega_0$: "+f"{pulse2.omega0}; "+r"$\sigma$: "+f"{pulse2.sigma}; "+r"$\phi_0$: "+f"{pulse2.phi0}; "+r"$b_1$: "+f"{pulse2.b1}; "+r"$b_2$: "+f"{pulse2.b2}; "+r"$b_3$: "+f"{pulse2.b3}; "+r"$b_4$: "+f"{pulse2.b4}")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
    #plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    plt.plot(pulse2.omega,combined_E.real,label="Superposed E",lw=1)
    plt.plot(pulse1.omega,abs(combined_E),'k--',label="E field envelope",lw=1)
    plt.plot(pulse2.omega,combined_I,label="FDI signal",lw=1)
    plt.title("Spectral domain interference"+f"   (delay: {time_delay})")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
    #plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()
    
    
    plt.plot(pulse1.omega,pulse1.spectral_intensity(),'r--',label="Pulse 1",lw=1)
    plt.plot(pulse2.omega,pulse2.spectral_intensity(),'b--',label="Pulse 2",lw=1)
    plt.plot(pulse2.omega,combined_I,'k-',label="FDI signal",lw=1)
    plt.title("Combined plot of spectrum"+"\n"+f"Delay: {time_delay}")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$I(\omega)$")
    #plt.grid(color='black', linestyle='-', linewidth=1)
    plt.show()



#   Define the required pulses   #

pulse1=pulse(E0=5, omega0=399, sigma=1.5, phi0=0, b1=10, b2=3.2, b3=0.5, b4=0.0)
pulse2=pulse(E0=3, omega0=401, sigma=1, phi0=0, b1=8, b2=-1.5, b3=0.2, b4=1.0)

time_delay=4.9
FDI(pulse1,pulse2,time_delay)

for __var__ in dir():
    exec('del '+ __var__)
    del __var__
    
import sys
sys.exit()