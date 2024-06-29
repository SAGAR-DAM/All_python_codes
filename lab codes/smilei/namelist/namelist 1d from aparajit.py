# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:45:10 2024

@author: mrsag
"""

#----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
import numpy as np
from math import pi, sqrt, sin, cos 

l0 = 2. * pi             # laser wavelength [in code units]
t0 = l0                  # optical cycle
Lsim = [100.0*l0]  # length of the simulation
Tsim = 400.0*t0            # duration of the simulation
resx = 128               # nb of cells in one laser wavelength
rest = resx*sqrt(2.)/0.95 # nb of timesteps in one optical cycle 
#Tsim = t0


Lx = Lsim[0]       # Simulation box length
x0 = 10.*l0     # Target thickness (width of step function)    
Xsurface = Lx - x0  # Vacuum region

aL=3.0  # Laser Intensity (a0 = 0.855*1e-9*sqrt(I)*Lambda   (I in W/cm^2,  Lambda in micron)
waist = 5.*l0   # Focal spot size
Tfwhm = 8.*l0   # Laser time FWHM;  8*t0 means 8 optical cycle pulse

n0 = 100.   # Target density, n0 comes from lambda dependence
one_ev = 1/511.e3   # from formula
T0_eon = 10.0*one_ev    # Temperature of electron plasma at start of simulation
T0_ion = 10.0*one_ev    #     "        "    ion     "     "   "    "     "   
nppc = 16               # no of particles per unit cell

#def density_profile(x):
#    if(x<Xsurface-4.0*l0):
#        return 0.
#    elif(x>=Xsurface-4.0*l0 and x<Xsurface):
#        return n0*np.exp((x-Xsurface)/(0.25*l0))
#    else:
#        return n0


def density_profile(x):
    if(x<Xsurface):
        return 0.
    else:
        return n0    

### fs structure profile
def time_profile(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/2.6666666*l0
    xc1 = (4.80421665 + 600)/2.6666666*l0
    A2 = 4.297148e1
    w2 = 281.0356576/2.6666666*l0
    xc2 = (-113.484182 + 600)/2.6666666*l0
    maxx = 0.4040909459763201
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)

### Single peak - 60fs
def time_profile2(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/2.6666666*l0
    xc1 = (4.80421665 + 500)/2.66666666*l0
    maxx = 0.36773593926954434
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2))/maxx)


def time_profile3(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0/3.
    xc1 = (4.80421665 + 300)/(8./3.)*l0
    A2 = 4.297148e1
    w2 = 281.0356576/(8./3.)*l0/3.
    xc2 = (-113.484182 + 370)/(8./3.)*l0
    #maxx = 1.
    maxx = 1.2021913830894313
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)

### Single peak - 60fs
def time_profile4(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0/3.
    xc1 = (4.80421665 + 300)/(8./3.)*l0
    #maxx = 1.
    maxx = 1.1108121526768813
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2))/maxx)


def time_profile5(t):
    y0 = -2.53477e-3
    A1 = 4.291049e1
    w1 = 39.2439786/(8./3.)*l0
    xc1 = (4.80421665 + 600)/(8./3.)*l0
    A2 = 4.297148e1
    w2 = 39.2439786/(8./3.)*l0
    xc2 = (-113.484182 + 550)/(8./3.)*l0
    maxx = 0.37079699580774755
    #maxx=1
    return np.sqrt((np.sqrt(2/np.pi)*A1/w1*np.exp(-2*(t-xc1)**2/w1**2)+np.sqrt(2/np.pi)*A2/w2*np.exp(-2*(t-xc2)**2/w2**2))/maxx)

Main(
    geometry = "1Dcartesian",       # for 1d simulation
    
    interpolation_order = 2 ,       # not to change
    
    cell_length = [l0/resx],        # Particle cell length
    grid_length  = Lsim,            # total simulation box length
    
    number_of_patches = [512],      # Rule: box_size*res*patches > 6, multiplication of boxex >=128
    
    timestep = t0/rest,             # 
    simulation_time = Tsim,         # total simulation time
     
    EM_boundary_conditions = [
        ['silver-muller']
    ],                              # Silver Muller: absorptive, Thermalize...
    
    random_seed = smilei_mpi_rank,  # No need to change
    print_every = Tsim*rest/t0/20.  # No need to change
    #reference_angular_frequency_SI = 2.*pi*3e8/0.8e-6
   
)


#LaserPlanar1D(
#    box_side	     = "xmin",
#    a0               = 0.01, # normalized amplitude
#    omega            = 1.,
    #focus            = [Lsim[0]*4./5., Lsim[1]/2.], # coordinates of laser focus
    #waist            = 5.*l0,
    #incidence_angle  = 30.0/180.0*3.14159,
    #polarization_phi = 0, # p-polarization 
#    time_envelope    = tgaussian(fwhm=4000.*t0, center=4990.*t0)
#)

#################################################
################ LASER ##########################
#################################################

Laser(
    box_side       = "xmin",    # Laser incoming direction
    omega          = 1.,        # Laser wavelength
    chirp_profile  = tconstant(),   # 
    #time_envelope  = tgaussian(start=0., duration=4.*Tfwhm, fwhm = Tfwhm, center = 4.*Tfwhm, order=2),
    time_envelope  = time_profile3,
    space_envelope = [ 0  , aL   ], # polarization direction
    # phase          = [ PhiY_profile, PhiZ_profile ],  # ??
    delay_phase    = [ 0., 0. ]     # ??
)


###############################################
############# PARTICLES #######################
###############################################

Species(
    name = "ion",
    position_initialization = "random",
    momentum_initialization = "cold",
    particles_per_cell = nppc,
    mass = 1837.,
    #atomic_number = None,
    #number_density = 100,
    number_density = density_profile,
    charge = 1.,
    #mean_velocity = [0.],
    temperature = [T0_ion],
    boundary_conditions = [["reflective","thermalize"]],
    thermal_boundary_temperature = [T0_eon],
    #thermal_boundary_velocity = [0,0,0],
    #time_frozen = 18.0*t0,
    is_test = False,
    pusher = "boris", 
    time_frozen = 0.0
)

Species(
    name = "eon",
    position_initialization = "ion",
    momentum_initialization = "mj",
    particles_per_cell = nppc, 
    mass = 1.,
    #atomic_number = 13.,
    #number_density = 100,
    number_density = density_profile,
    charge = -1.,
    #mean_velocity = [0.],
    temperature = [T0_eon],
    boundary_conditions = [["reflective","thermalize"]],
    thermal_boundary_temperature = [T0_eon],
    #thermal_boundary_velocity = [0,0,0],
    #time_frozen = 18.0*t0,
    is_test = False,
    pusher = "boris",
    time_frozen = 0.0
)


##################################################
##############  DIAGNOSTICS   ####################
##################################################

sub_grid = 1    # resolution of viewing
Period = rest   # Save after dt = res_t time step

DiagScalar(
    every = Period  # after this step it will save the values
    #every = 10.*rest
)

DiagFields(
    every = Period,     # time step for saving
    fields = ['Ex', 'Ey','Bz'],     # fields we want to save
    subgrid = np.s_[::sub_grid] #to reduce the number of points in the output file
)

# 2-dimensional grid diagnostic
#DiagProbe(
#    every = 10.0*rest,
#    number = [1000, 1000], # number of points in the grid
#    origin = [0., 0.], # coordinates of origin point
#    corners = [
#        [, 0.], # coordinates of first corner of the grid
#        [0. , 64.*l0], # coordinates of second corner of the grid
#    ],
#    fields = [ "Ey","Bz","Rho"]
#)

#DiagProbe(
#	every = 10.*rest,
#	origin = [0.],
#	corners = [[Lsim[0]]],
#	number = [512]
#)


DiagParticleBinning(
	name = "density_map",
	deposited_quantity = "weight",  # weight = number density
	every = Period,
	time_average = 1.,
	species = ["eon"],
	axes = [["x", Lsim[0] - 5.0*x0, Lsim[0], 6400]]
)

DiagParticleBinning(
	name = "phase_space",
	deposited_quantity = "weight",
	every = Period,
	time_average = 1,
	species = ["eon"],
	axes = [["x", Lsim[0] - 3.0*x0 , Lsim[0], 1000],["px", -5, 5,1000]]
)



#DiagParticleBinning(
#    deposited_quantity = "weight",
#    every = 100.*rest,
#    time_average = 1,
#    species = ["electron", "ion"],
#    axes = [ ["x", 0., Lsim[0], 5.0*resx],
#	     ["ekin",    0.0000002,    0.0002,   1000, "logscale"] 
#	   ]
#)


#DiagParticleBinning(
#        name = "energy_dist",
#        deposited_quantity = "weight",
#        every = 4.0*rest,
#        time_average = 1,
#        species = ["electron"],
#        axes = [
#        ["x", 0., 20*l0, 2000],
#        ["y", 0., 64*l0, 2000],
#	["ekin", 0.02, 2., 100, "logscale"]]
#)

# probe diagnostic with 1 point
#DiagProbe(
#    every = ,
#    origin = [0.1*Lsim[0], 0.5*Lsim[1]],
#    fields = []
#)

#DiagTrackParticles(
#	species = "electron",
#	every = rest*100.
#
'''
Collisions(
    species1 = ["electron",  "ion"],
    species2 = ["electron",  "ion"],
    debug_every = 50.*rest,
    coulomb_log = 0.,
    coulomb_log_factor = 1.,
    ionizing = False,
#      nuclear_reaction = [],
)
'''
