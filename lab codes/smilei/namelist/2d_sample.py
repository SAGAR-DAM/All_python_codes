#----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
import numpy as np
from math import pi, sqrt, sin, cos 

l0 = 2. * pi             # laser wavelength [in code units]
t0 = l0                  # optical cycle


Lx = 96.0*l0  # length of the simulation
Ly = 112.0*l0


Tsim = 150.0*t0            # duration of the simulation


resx = 32               # nb of cells in one laser wavelength
resy = resx
dx = l0/resx

dt   = 0.95 * dx/np.sqrt(2.)

rest = resx # nb of timesteps in one optical cycle 
#Tsim = t0


target_thickness = 2.0*l0
target_top_vacuum = 75.0*l0                  #-----------------  VACUUM  --------------#
Ysurface = Ly - target_thickness - target_top_vacuum           #-----------------  SURFACE  -------------#

aL1=1.0                      #---------  normalized laser amplitude a0 = 0.86 sqrt(I/10^18)  ---------------#
aL2=1.0 
theta_deg_1 = 45.0
theta_deg_2 = 70
theta_deg_3 = 90 - theta_deg_2
theta_1  = theta_deg_1*np.pi/180.
theta_2  = theta_deg_2*np.pi/180.
theta_3  = theta_deg_3*np.pi/180.
waist_1 = 3.0*l0              #-----------------  FOCAL SPOT  ----------#
waist_2 = 3.0*l0 
Tfwhm = 12.0*t0                #-----------------  PULSE WIDTH  ---------#

xfoc_1 = Lx/2        # the laser is focalized over the surface
yfoc_1 = Ysurface + target_thickness           # the laser is focalized at the target's center     
 
xfoc_2 =Lx/2
yfoc_2 = Ysurface

n0 = 100.0                   #-----------------  DENSITY  -------------#
one_ev = 1/511.e3
T0_elec = 1.0*one_ev        #-----------------  ELECTRON TEMP  -------#
T0_ion = 1.0*one_ev         #-----------------  ION TEMP  ------------#
nppc = 32



#--------------------------------------------------------------------------------------------
#                                 PRE-PLASMA SCALE LENGTH
#--------------------------------------------------------------------------------------------


#def density_profile(x,y):
#    if(x<Xsurface-4*l0):
#        return 0.
#    elif(x>=Xsurface-4.0*l0 and x<Xsurface):
#        return n0*np.exp((x-Xsurface)/(0.25*l0))
#    elif(x>=Xsurface + target_thickness):
#        return 0.
#    else:
#        return n0
#---------------------------------------------------------------------------------------------
#                                    FLAT DENSITY PROFILE
#---------------------------------------------------------------------------------------------


#def density_profile(x,y):
#    if(x<Xsurface):
#        return 0.
#    else:
#        return n0 



#def density_profile(x,y):
#    if(x<Xsurface):
#        return 0.
#    elif(x>=Xsurface + target_thickness):
#        return 0.
#    else:
#        return n0


def density_profile(x,y):
    if(y<Ysurface-4*l0):
        return 0.
    elif(y>=Ysurface-4.0*l0 and y<Ysurface):
        return n0*np.exp((y-Ysurface)/(0.25*l0))
    elif(y>=Ysurface + target_thickness+4.*l0):
        return 0.
    elif (y>=Ysurface + target_thickness and y < Ysurface + target_thickness+4.*l0):
        return n0*np.exp((-y+(Ysurface + target_thickness))/(0.25*l0))
    else:
        return n0
#---------------------------------------------------------------------------------------------
#                                    MAIN PIC CODE
#---------------------------------------------------------------------------------------------



Main(
    geometry = "2Dcartesian",
    
    interpolation_order = 2 ,
    
    cell_length = [l0/resx,l0/resy],
    grid_length  = [Lx,Ly],
    
    number_of_patches = [512,1], 
    timestep = dt,
    simulation_time = Tsim,
     
    EM_boundary_conditions = [['silver-muller'] ],
    
    random_seed = smilei_mpi_rank,
    print_every = int(Tsim*rest/t0/20.)
    #reference_angular_frequency_SI = 2.*pi*3e8/0.8e-6
   
)




#------------------------------------------------------------------------------------------
#                                       LASER 
#------------------------------------------------------------------------------------------

#Laser(
#    box_side       = "xmin",
#    omega          = 1.,
#    chirp_profile  = tconstant(),
    #time_envelope  = tgaussian(start=0., duration=4.*Tfwhm, fwhm = Tfwhm, center = 4.*Tfwhm, order=2),
#    time_envelope  = time_profile3,
#    space_envelope = [ 0  , aL   ],
    # phase          = [ PhiY_profile, PhiZ_profile ],
#    delay_phase    = [ 0., 0. ]
#)




#LaserPlanar1D(
#    box_side        = "xmin",
#    a0               = 0.01, # normalized amplitude
#    omega            = 1.,
    #focus            = [Lsim[0]*4./5., Lsim[1]/2.], # coordinates of laser focus
    #waist            = 5.*l0,
    #incidence_angle  = 30.0/180.0*3.14159,
    #polarization_phi = 0, # p-polarization 
#    time_envelope    = tgaussian(fwhm=4000.*t0, center=4990.*t0)
#



LaserGaussian2D(
     box_side        = "xmin",
     a0              = aL1,
     omega           = 1.,
     focus           = [xfoc_1,yfoc_1],
     waist           = waist_1,
     polarization_phi = 0,        #0 for P-pol
     incidence_angle = - theta_1,
     #space_time_profile = [ Bx_profile, Bz_profile ],
     time_envelope   = tgaussian(start=0.0*t0, duration=48.0*t0, fwhm=12.0*t0, center=24.0*t0, order=2)   
)


LaserGaussian2D(
     box_side        = "xmin",
     a0              = aL2,
     omega           = 1.,
     focus           = [xfoc_2,yfoc_2],
     waist           = waist_2,
     polarization_phi = 0,        #0 for P-pol
     incidence_angle = theta_3,
     #space_time_profile = [ Bx_profile, Bz_profile ],
     time_envelope   = tgaussian(start=0.*Tfwhm, duration=4.*Tfwhm, fwhm=Tfwhm, center=2.*Tfwhm, order=2)
)

#------------------  Parameters  -------------------------------------------------------
#start =  starting time
#duration =  duration of the profile (default is simulation_time  start)
#fwhm =  gaussian FWHM (default is duration/3.)
#center =  gaussian center time (default is in the middle of duration)
#order =  order of the gaussian


#LaserGaussian2D(
#     box_side        = "xmin",
#     a0              = aL,
#     omega           = 1.,
#     focus           = [xfoc_2,yfoc_2],
#     waist           = waist_2,
#     polarization_phi = 0,        #0 for P-pol
#     incidence_angle = theta_2,
#     #space_time_profile = [ Bx_profile, Bz_profile ],
#     time_envelope   = tgaussian(start=0., duration=4.*Tfwhm, fwhm=Tfwhm, center=2.*Tfwhm, order=2)
#)
#----------------------------------------------------------------------------------------
#                                 PARTICLES 
#----------------------------------------------------------------------------------------
'''
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
    boundary_conditions = [["remove"]],
    thermal_boundary_temperature = [T0_ion],
    #thermal_boundary_velocity = [0,0,0],
    #time_frozen = 18.0*t0,
    is_test = False,
    pusher = "boris", 
    time_frozen = 0.0
)

Species(
    name = "electron",
    position_initialization = "ion",
    momentum_initialization = "mj",
    particles_per_cell = nppc, 
    mass = 1.,
    #atomic_number = 13.,
    #number_density = 100,
    number_density = density_profile,
    charge = -1.,
    #mean_velocity = [0.],
    temperature = [T0_elec],
    boundary_conditions = [["remove"]],
    thermal_boundary_temperature = [T0_elec],
    #thermal_boundary_velocity = [0,0,0],
    #time_frozen = 18.0*t0,
    is_test = False,
    pusher = "boris",
    time_frozen = 0.0
)

'''


#-------------------------------------------------------------------------------------
#                               DIAGNOSTICS   
#-------------------------------------------------------------------------------------

sub_grid = 16
Period = rest

DiagScalar(
    every = Period
    #every = 10.*rest
)

DiagFields(
    every = Period,
    #fields = ['Ex', 'Ey','Ez','Bx', 'By','Bz','Rho_ion','Rho','Rho_electron'],
    fields = ['Ex', 'Ey','Ez','Bx', 'By','Bz'],
    subgrid = np.s_[::sub_grid,::sub_grid] #to reduce the number of points in the output file
)






def angle(particles):
	return np.arctan2( particles.py, particles.px )

def angle1(particles):
        return np.arctan2( (particles.y - yfoc_1), (particles.x - xfoc_1) )

def angle2(particles):
        return np.arctan2( (particles.y - yfoc_2), (particles.x - Xsurface - target_thickness) )

'''
DiagParticleBinning(
	deposited_quantity = "weight",
	every = [0, 2.0*Period],
	time_average = 1,
	species = ["electron"],
	axes = [["y",Ysurface + target_thickness,Ly,1],
		["gamma", 1., 100, 4000],
		[angle, -math.pi, math.pi, 360]]
)


DiagParticleBinning(
        deposited_quantity = "weight",
        every = [0, 2.0*Period],
        time_average = 1,
        species = ["electron"],
        axes = [["x",0,Xsurface,1],
                ["gamma", 1., 100, 4000],
                [angle, -math.pi, math.pi, 360]]
)


DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [0.0, 30.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)


DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [0.0, 30.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness ,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle2, -math.pi, math.pi, 360]]
)


DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface,Ly/2],
        vector = [0.0, 30.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",0,Xsurface,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)

DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface,Ly/2],
        vector = [0.0, 30.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",0,Xsurface,1],
               ["gamma", 1.0, 100, 4000],
               [angle1, -math.pi, math.pi, 360]]
)

#########################################Probe of sheathmaking laser#############################

DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [0.0*l0,Ly/2],
        vectors = [
                  [+30.0*l0,-30.0*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)


DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [0.0*l0,Ly/2],
        vectors = [
                  [30.0*l0,30.0*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)

DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [Lx,Ly/2],
        vectors = [
                  [-30.0*l0,30.0*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)

###########################################Probe of surface electron making laser###########################
'''
'''
DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [Lx,Ly/2],
        vectors = [
                  [-30.0*l0,-30.0 * np.tan(theta_3)*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)


DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [Lx,Ly/2],
        vectors = [
                  [-30.0*l0,30.0 * np.tan(theta_3)*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)

DiagProbe(
        every = 1,
        #flush_every = [T1-int(12.5*rest),T1+int(12.5*rest),1000],
        origin = [0,Ly/2],
        vectors = [
                  [30.0*l0,30.0 * np.tan(theta_3)*l0]
                  ],
        number = [1000],
        fields = ["Bz"]
)




DiagParticleBinning(
        deposited_quantity = "weight",
        every = [0, 2.0*Period],
        time_average = 1,
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness - l0/8,Lx,1],
                ["gamma", 1., 100, 4000],
                [angle2, -math.pi, math.pi, 360]]
)

DiagParticleBinning(
        deposited_quantity = "weight",
        every = [0, 2.0*Period],
        time_average = 1,
        species = ["electron"],
        axes = [["x",0,Xsurface + l0/8,1],
                ["gamma", 1., 100, 4000],
                [angle1, -math.pi, math.pi, 360]]
)



DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [0.0, 96.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness - l0/8,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle2, -math.pi, math.pi, 360]]
)

DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [0.0, 23.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness - l0/8,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle2, -math.pi, math.pi, 360]]
)


DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface,Ly/2],
        vector = [0.0, 25.0*l0],
        direction = "forward",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",0,Xsurface + l0/8,1],
               ["gamma", 1.0, 100, 4000],
               [angle1, -math.pi, math.pi, 360]]
)
'''

def my_filter(particles):
    ekin = (1+particles.px**2.0+particles.py**2.0+particles.pz**2.0)**0.5
    return (ekin>2.0)

#DiagTrackParticles(
#        species = "electron",
#        every = 2 * Period,
#        filter = my_filter,
#        attributes = ["x","px","py","y","Ex", "Ey", "Ez", "Bx", "By", "Bz"]
#)

'''
DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["electron"],
       axes = [["x",0.0,Xsurface + l0/8,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)

DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["ion"],
       axes = [["x",0.0,Xsurface + l0/8,1],
               ["gamma", 1.0, 1.02, 4000],
               [angle, -math.pi, math.pi, 360]]
)


DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["electron"],
       axes = [["x",Xsurface + target_thickness,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)


DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["ion"],
       axes = [["x",Xsurface + target_thickness,Lx,1],
               ["gamma", 1.0, 1.02, 4000],
               [angle, -math.pi, math.pi, 360]]
)




DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [-7.0*l0, 0.*l0],
        direction = "both",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",Xsurface + target_thickness,Lx,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)


DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface + target_thickness,Ly/2],
        vector = [-7.0*l0, 0.*l0],
        direction = "both",
        deposited_quantity = "weight",
        species = ["ion"],
        axes = [["x",Xsurface + target_thickness,Lx,1],
               ["gamma", 1.0, 1.02, 4000],
               [angle, -math.pi, math.pi, 360]]
)

DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface,Ly/2],
        vector = [-9.0*l0, 0.*l0],
        direction = "both",
        deposited_quantity = "weight",
        species = ["electron"],
        axes = [["x",0.0,Xsurface + l0/8,1],
               ["gamma", 1.0, 100, 4000],
               [angle, -math.pi, math.pi, 360]]
)

DiagScreen(
        every =[0,2.0*Period],
        shape = "sphere",
        point = [Xsurface,Ly/2],
        vector = [-9.0*l0, 0.*l0],
        direction = "both",
        deposited_quantity = "weight",
        species = ["ion"],
        axes = [["x",0.0,Xsurface + l0/8,1],
               ["gamma", 1.00, 1.02, 4000],
               [angle, -math.pi, math.pi, 360]]
)


DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["electron"],
       axes = [["x",0.0,Xsurface + l0/8,1],
               ["y", 0.0, Ly, 200],
               ["px", -100.0, 100.0,1000 ]]
)

DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["ion"],
       axes = [["x",0.0,Xsurface + l0/8,1],
               ["y", 0.0, Ly, 200],
               ["px", -0.02, 0.02,1000 ]]
)


DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["electron"],
       axes = [["x",Xsurface+target_thickness,Lx,1],
               ["y", 0.0, Ly, 200],
               ["px", -100.0, 100.0,1000 ]]
)

DiagParticleBinning(
       deposited_quantity = "weight",
       every = [0, 2.0*Period],
       time_average = 1,
       species = ["ion"],
       axes = [["x",Xsurface+target_thickness,Lx,1],
               ["y", 0.0, Ly, 200],
               ["px", -0.02, 0.02,1000 ]]
)
'''
#DiagScreen(
#        every =[0,2.0*Period],
#        shape = "plane",
#        point = [Xsurface + target_thickness,Ly/2],
#        vector = [5.*l0, 0.*l0],
#        direction = "both",
#        deposited_quantity = "weight",
#        species = ["electron"],
#        axes = [["y",0,Ly,400],
#        ["gamma" , 1., 100., 4000]
#        ]
#)







#DiagScreen(
#        every =[0,2.0*Period],
#        shape = "sphere",
#        point = [Xsurface, 37.5*l0],
#        vector = [10.*l0, 0.*l0],
#        direction = "forward",
#        deposited_quantity = "weight",
#        species = ["electron"],
#        axes = [
#        ["gamma" , 1., 100., 4000],
#        [angle1 , -math.pi, math.pi,360]
#        ]
#)


#DiagPerformances (
# every = [0,2.0*Period]
#)
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


#DiagParticleBinning(
#	name = "density_map",
#	deposited_quantity = "weight",
#	every = Period,
#	time_average = 1.,
#	species = ["eon"],
#	axes = [["x", Lsim[0] - 5.0*x0, Lsim[0], 6400]]
#)

#DiagParticleBinning(
#	name = "phase_space",
#	deposited_quantity = "weight",
#	every = Period,
#	time_average = 1,
#	species = ["eon"],
#	axes = [["x", Lsim[0] - 3.0*x0 , Lsim[0], 1000],["px", -5, 5,1000]]
#)



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
#ussian2D

