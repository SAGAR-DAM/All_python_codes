import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal

import matplotlib
matplotlib.rcParams["figure.dpi"]=300

c=3e5       # in nm/ps

def straightline(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def linefit(x,y):
    x=np.array(x)
    parameters,pcov = curve_fit(straightline, x,y, maxfev=100000) # your data x, y to fit
    line=straightline(x,*parameters)    


    #string=f"Fit: y=mx+c ; with m={Decimal(str(parameters[0])).quantize(Decimal('1.00'))}, c={Decimal(str(parameters[1])).quantize(Decimal('1.00'))} "         
    return(line,parameters)               

def parabola(x, A, B, C): # this is your 'parabola' y=f(x)
    return A*x**2 + B*x + C

def parabola_fit(x,y):
    # Perform a parabolic fit
    parameters = np.polyfit(x, y, 2)

    # Create a function from the coefficients
    # Create a function from the coefficients
    parabolic_fit = np.poly1d(parameters)
    fit_y=parabolic_fit(x)

    return(fit_y,parameters)

def gaussian(omega,E0,omega0,sigma):
    E=E0*np.exp(-(omega-omega0)**2/(2*sigma**2))
    return(E)

def phase_factor(omega,omega0,phi0,b1,b2,b3,b4):
    delta=omega-omega0
    phi=phi0+b1*delta+b2*delta**2+b3*delta**3+b4*delta**4
    e_i_phi=np.exp(1j*phi)
    return phi,e_i_phi

class PulseMeta(type):
    def __new__(cls, name, bases, attrs):
        # Add class-level attributes
        attrs['omega'] = np.linspace(4600, 4830, 9210)
        attrs['phase'] = np.zeros(len(attrs['omega']))

        # Add pulse_profile and spectral_intensity methods to the class
        attrs['calc_pulse_profile'] = cls.calc_pulse_profile
        attrs['calc_spectral_intensity'] = cls.calc_spectral_intensity
        
        attrs['pulse_profile'] = cls.pulse_profile
        attrs['spectral_intensity'] = cls.spectral_intensity

        return super().__new__(cls, name, bases, attrs)

    @staticmethod
    def calc_pulse_profile(self, E0, omega0, sigma, phi0, b1, b2, b3, b4):
        E_0 = gaussian(self.omega, E0, omega0, sigma)
        phase, e_i_phase = phase_factor(self.omega, omega0, phi0, b1, b2, b3, b4)
        E = E_0 * e_i_phase
        self.phase = phase
        return E

    @staticmethod
    def calc_spectral_intensity(self):
        E = self.calc_pulse_profile(self.E0, self.omega0, self.sigma, self.phi0, self.b1, self.b2, self.b3, self.b4)
        I = (abs(E))**2
        return I
    
    @staticmethod
    def pulse_profile(self):
        return(self.calc_pulse_profile(self.E0, self.omega0, self.sigma, self.phi0, self.b1, self.b2, self.b3, self.b4))

    @staticmethod
    def spectral_intensity(self):
        return(self.calc_spectral_intensity())
    

class pulse(metaclass=PulseMeta):
    def __init__(self, E0, omega0, sigma, phi0, b1, b2, b3, b4):
        self.E0 = E0
        self.omega0 = omega0
        self.sigma = sigma
        self.phi0 = phi0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.d_omega = abs(np.diff(pulse.omega)[0])
        
        self.calc_pulse_profile(E0, omega0, sigma, phi0, b1, b2, b3, b4)
        self.calc_spectral_intensity()
    
def superposition(pulse1,pulse2,time_delay):
    phase_lag = np.exp(1j * pulse1.omega * time_delay)
    combined_E=pulse2.pulse_profile()*phase_lag+pulse1.pulse_profile()
    combined_I=(abs(combined_E))**2
    return combined_E,combined_I
    
# def FDI(pulse1,pulse2,time_delay):
    
#     combined_E,combined_I=superposition(pulse1, pulse2, time_delay)
#     minw=4680   #pulse1.omega[0]        # good range: 4680
#     maxw=4760   #pulse1.omega[-1]       # good range: 4760
    
    
#     fig, ax1 = plt.subplots()
#     ax1.plot(pulse1.omega,pulse1.pulse_profile().real,label="E vs freq",lw=1)
#     ax1.plot(pulse1.omega,abs(pulse1.pulse_profile()),'k--',label="E field envelope",lw=1)
#     ax1.plot(pulse1.omega,pulse1.spectral_intensity(),label="spectral intensity",lw=1)
#     ax1.set_title("Pulse 1: \n"+r"$\tilde E(\omega)=$"+r"$\tilde E_0e^{-\frac{(\omega-\omega_0)^2}{2\sigma^2}}\times\exp[i(\phi_0+b_1\Delta\omega+b_2\Delta\omega^2+b_3\Delta\omega^3+b_4\Delta\omega^4)]$"+"\n"+r"$\tilde E_0: $"+f"{pulse1.E0}; "+r"$\omega_0$: "+f"{pulse1.omega0}; "+r"$\sigma$: "+f"{pulse1.sigma}; "+r"$\phi_0$: "+f"{pulse1.phi0}; "+r"$b_1$: "+f"{pulse1.b1}; "+r"$b_2$: "+f"{pulse1.b2}; "+r"$b_3$: "+f"{pulse1.b3}; "+r"$b_4$: "+f"{pulse1.b4}" , bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#     ax1.legend()
#     ax1.set_xlabel(r"$\omega$ (rad/ps)")
#     ax1.set_ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
#     ax1.set_xlim(minw,maxw)
#     #plt.grid(color='black', linestyle='-', linewidth=1)
    
#     bx1=ax1.twinx()
#     bx1.plot(pulse1.omega,pulse1.phase,'g-.',label="Phase",lw=1)
#     bx1.set_ylabel(r"$\phi(\omega)$ (rad)")
#     bx1.legend(loc="upper left")
#     bx1.set_xlim(minw,maxw)
    
#     plt.show()
    
    
#     fig, ax2 = plt.subplots()
#     ax2.plot(pulse2.omega,pulse2.pulse_profile().real,label="E vs freq",lw=1)
#     ax2.plot(pulse2.omega,abs(pulse2.pulse_profile()),'k--',label="E field envelope",lw=1)
#     ax2.plot(pulse2.omega,pulse2.spectral_intensity(),label="spectral intensity",lw=1)
#     ax2.set_title("Pulse 2: \n"+r"$\tilde E(\omega)=$"+r"$\tilde E_0e^{-\frac{(\omega-\omega_0)^2}{2\sigma^2}}\times\exp[i(\phi_0+b_1\Delta\omega+b_2\Delta\omega^2+b_3\Delta\omega^3+b_4\Delta\omega^4)]$"+"\n"+r"$\tilde E_0: $"+f"{pulse2.E0}; "+r"$\omega_0$: "+f"{pulse2.omega0}; "+r"$\sigma$: "+f"{pulse2.sigma}; "+r"$\phi_0$: "+f"{pulse2.phi0}; "+r"$b_1$: "+f"{pulse2.b1}; "+r"$b_2$: "+f"{pulse2.b2}; "+r"$b_3$: "+f"{pulse2.b3}; "+r"$b_4$: "+f"{pulse2.b4}" , bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#     ax2.legend()
#     ax2.set_xlabel(r"$\omega$ (rad/ps)")
#     ax2.set_ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
#     ax2.set_xlim(minw,maxw)
#     #plt.grid(color='black', linestyle='-', linewidth=1)
    
#     bx2=ax2.twinx()
#     bx2.plot(pulse1.omega,pulse1.phase,'g-.',label="Phase",lw=1)
#     bx2.set_ylabel(r"$\phi(\omega)$ (rad)")
#     bx2.legend(loc="upper left")
#     bx2.set_xlim(minw,maxw)
    
#     plt.show()


#     fig, ax3 = plt.subplots()
#     ax3.plot(pulse2.omega,combined_E.real,label="Superposed E",lw=1)
#     ax3.plot(pulse1.omega,abs(combined_E),'k--',label="E field envelope",lw=1)
#     ax3.plot(pulse2.omega,combined_I,label="FDI signal",lw=1)
#     ax3.set_title("Spectral domain interference"+f"   (delay: {time_delay} ps)" , bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#     ax3.legend()
#     ax3.set_xlabel(r"$\omega$ (rad/ps)")
#     ax3.set_ylabel(r"$Re(E(\omega))\ \ &\ \ I(\omega)$")
#     #plt.grid(color='black', linestyle='-', linewidth=1)
#     ax3.set_xlim(minw,maxw)
#     plt.show()
    
    
#     plt.plot(pulse1.omega,pulse1.spectral_intensity(),'r-',label="Pulse 1",lw=1)
#     plt.plot(pulse2.omega,pulse2.spectral_intensity(),'b-',label="Pulse 2",lw=1)
#     plt.plot(pulse2.omega,combined_I,'k-',label="FDI signal",lw=1)
#     plt.title("Combined plot of spectrum"+"\n"+f"Delay: {time_delay} ps" , bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
#     plt.legend()
#     plt.xlabel(r"$\omega$ (rad/ps)")
#     plt.ylabel(r"$I(\omega)$")
#     plt.xlim(minw,maxw)
#     #plt.grid(color='black', linestyle='-', linewidth=1)
#     plt.show()
    
#     return combined_E,combined_I



# def info_table(pulse1,pulse2,time_delay):
#     pulse1_info=r"$\tilde E_0: $"+f"{pulse1.E0}\n"+r"$\omega_0$: "+f"{pulse1.omega0}\n"+r"$\sigma_{\omega}$: "+f"{pulse1.sigma}\n"+r"$\phi_0$: "+f"{pulse1.phi0}\n"+r"$b_1$: "+f"{pulse1.b1}\n"+r"$b_2$: "+f"{pulse1.b2}\n"+r"$b_3$: "+f"{pulse1.b3}\n"+r"$b_4$: "+f"{pulse1.b4}"
#     pulse2_info=r"$\tilde E_0: $"+f"{pulse2.E0}\n"+r"$\omega_0$: "+f"{pulse2.omega0}\n"+r"$\sigma_{\omega}$: "+f"{pulse2.sigma}\n"+r"$\phi_0$: "+f"{pulse2.phi0}\n"+r"$b_1$: "+f"{pulse2.b1}\n"+r"$b_2$: "+f"{pulse2.b2}\n"+r"$b_3$: "+f"{pulse2.b3}\n"+r"$b_4$: "+f"{pulse2.b4}"
#     pulse_info=r"$\tilde E(\omega)=$"+r"$\tilde E_0e^{-\frac{(\omega-\omega_0)^2}{2\sigma^2}}\times\exp[i(\phi_0+b_1\Delta\omega+b_2\Delta\omega^2+b_3\Delta\omega^3+b_4\Delta\omega^4)]$"

#     plt.figure()
#     plt.plot()
#     plt.figtext(0,1,"Pulse 1"+"\n__________", fontsize=15, color="Blue")
#     plt.figtext(0, 0.5,pulse1_info, fontsize=12, color='red')
#     plt.figtext(0.55,1,"Pulse 2"+"\n__________", fontsize=15, color="Blue")
#     plt.figtext(0.55, 0.5,pulse2_info, fontsize=12, color='red')
#     plt.figtext(0.15,0.7,f"<-------Delay: ------->\n           {time_delay} ps",fontsize=15,color="magenta")
#     plt.figtext(0,0.3,pulse_info,fontsize=12,color="green")
#     plt.axis("off")
#     plt.show()