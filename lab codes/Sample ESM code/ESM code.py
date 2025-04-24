# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:48:11 2023
@author: Anandam
"""
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import math

"""
# Input block
"""
# Sensitivity
sens = 4000 

#Name of the trace file with extension
A = np.array(Image.open("D:\\data Lab\\ELI-NP March 2025\\Sample ESM code\\sample_trace.tiff"))


plt.imshow(A)
plt.show()

#Length of the IP in mm
Lengthip = 105

#Length from IP edge to Magnet edge in mm
Lengthoff = 0

#Magenetic field in Tesla
B=0.1

#Height of collimator to magnet edge in mm
Height = 30

#Height of collimator to IP surfac in mm
HeightIP = 31

#Resolution of scanner in micron
Resolution = 25

#Distance of IP edge from the target in mm
D = 260

#Aperture Dia in mm
d = 2

#Al thickness in micron
t = 12

#Fading time in min
Ftime = 80

#Constants
m = 9.10939e-31
c = 2.99792458e+8
e = 1.60218e-19

#conversion
Height = Height*1e-3
HeightIP = HeightIP*1e-3
Resolution = Resolution*1e-6
D = D*1e-3
d = 2*1e-3
t = t*1e-4 #in cgs cm

#Defining functions
#Function to calculaate fading ratios
def funcfading(ftime):
    ffa = 0.156    # 0.155857
    ffb = 0.209    # 0.209367
    fta = 0.558    # 0.557602
    ftb = 11.3     # 11.2649
    ftc = 2000.0   # 1999.96

    fadingratio = ffa * pow(0.5, (ftime / fta)) + ffb * pow(0.5, (ftime / ftb)) + (1 - (ffa + ffb)) * pow(0.5, (ftime / ftc))
    return fadingratio

#Function to calculate sensitivity of IP
def funcsensitivity(energy):
    # Function 1
    if energy < 0.05:
        sa = -1.37
        sb = 4.8955e-1
        sc = 9.34307e-3
        sensitivity = sa * energy**2 + sb * energy + sc
    
    # Function 2
    elif energy >= 0.05 and energy < 0.12:
        sa = 29.6061
        sb = -10.7906
        sc = 1.24418
        sd = -8.48498e-3
        sensitivity = sa * energy**3 + sb * energy**2 + sc * energy + sd

    # Function 3
    elif energy >= 0.12 and energy < 0.137:
        sa = -1.30643e-1
        sb = -2.64656e-2
        sc = 4.16476e-2
        sensitivity = sa * energy**2 + sb * energy + sc

    # Function 4
    elif energy >= 0.137 and energy < 0.46:
        sa = 4.06997e-2
        sb = 3.27383
        sc = 9.59762e-3
        sensitivity = sa * math.exp(-sb * energy) + sc

    # Function 5
    elif energy >= 0.46 and energy < 0.79:
        sa = 4.14759e-2
        sb = 3.07174
        sc = 8.5134e-3
        sensitivity = sa * math.exp(-sb * energy) + sc

    # Function 6
    elif energy >= 0.79 and energy < 1.1:
        sa = 1.40477e-2
        sb = -3.53998e-2
        sc = 3.13743e-2
        sensitivity = sa * energy**2 + sb * energy + sc

    # Function 7
    elif energy >= 1.1 and energy < 1.6:
        sa = 1.49836
        sb = 7.00353
        sc = -2.45725e-4
        sd = 9.05957e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy**2 + sd

    # Function 8
    elif energy >= 1.6 and energy < 2.7:
        sa = 4.4432e-3
        sb = 9.45248e-1
        sc = 7.46777e-3
        sensitivity = sa * math.exp(-sb * energy) + sc

    # Function 9
    elif energy >= 2.7 and energy < 5.7:
        sa = 2.74341e-3
        sb = 7.27153e-1
        sc = -5.07852e-5
        sd = 7.56518e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd

    # Function 10
    elif energy >= 5.7 and energy < 9.8:
        sa = 1.51122e-3
        sb = 4.05074e-1
        sc = -1.98777e-5
        sd = 7.28229e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd

    # Function 11
    elif energy >= 9.8 and energy < 13:
        sa = 1.15864e-3
        sb = 2.9994e-1
        sc = -1.29653e-5
        sd = 7.18185e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd

    # Function 12
    elif energy >= 13 and energy < 37:
        sa = 5.42261e-4
        sb = 1.37381e-1
        sc = -6.97495e-6
        sd = 7.03642e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd

    # Function 13
    else:
        sa = 2.04741e-4
        sb = 5.29949e-2
        sc = -5.73563e-6
        sd = 6.96509e-3
        sensitivity = sa * math.exp(-sb * energy) + sc * energy + sd

    return sensitivity



#Function to calculate transemittance
def functransemittance(energy, theta, althickness):
    # cgs Unit
    AlA = 26.981538
    AlZ = 13.0
    AlRho = 2.69
    ta = 0.537
    tb = 0.9815
    tc = 3.1230
    td = 9.2 * (AlZ**-0.2) + 16 * (AlZ**-2.2)
    te = 0.63 * AlZ / AlA + 0.27

    Range = ta * energy * (1 - tb / (1 + tc * energy))
    taurange = (althickness * AlRho) / (Range * np.cos(theta))  # corrected thickness is used for oblique incidence
    transemittance = (1 + np.exp(-td * te)) / (1 + np.exp(td * (taurange - te)))

    return transemittance

#Calculations
Solidangle=2*np.pi*(1-np.sqrt(D**2-(d/2)**2)/D)
fadratio = funcfading(Ftime)
row1, col1 = A.shape

# Apply smoothing to the sum of the image
Q = ndimage.gaussian_filter1d(np.sum(A, axis=0), 10)
Q = ndimage.gaussian_filter1d(Q, 10)

#plt.plot(Q)
#plt.show()

# Find the maximum value of the smoothed image
R = np.max(Q)

# Find the index of the maximum value in the smoothed image
j = np.argmax(Q)

# Crop image
X = A[:, j-int((col1*0.025)/2):j+int((col1*0.025)/2)]

# Find the size of the cropped image
row, col = X.shape

# Calculate the signal
S = np.sum(X, axis=1) / (col * sens)

# Calculate the distance
N = np.arange(1, row+1) * (Lengthip/row)

# Calculate the background
#BG1X = A[:, j-int(col1*0.1):j-int(col1*0.1)+int(col1*0.025)]
#BG2X = A[:, j+int(col1*0.066):j+int(col1*0.066)+int(col1*0.025)]
BG1X = A[:, j-int(col1*0.3):j-int(col1*0.15)+int(col1*0.025)]
BG2X = A[:, j+int(col1*0.275):j+int(col1*0.275)+int(col1*0.025)]
BG1 = ndimage.gaussian_filter1d(np.sum(BG1X, axis=1) / (col * sens), 5)
BG2 = ndimage.gaussian_filter1d(np.sum(BG2X, axis=1) / (col * sens), 5)

# Plots signal and background
#plt.figure()
#plt.plot(S)
#plt.show()
#plt.plot(BG1)
#plt.show()
#plt.plot(BG2)
#plt.show()

NetSig = []
NoiseSig = []
Energy = []

for i in range(row):
    Length = N[i]-Lengthoff
    Length = Length*1e-3
    energy = (1/(e*1e6))*(m*c**2)*(np.sqrt(((((Length**2)+(Height**2))/(2*Height))**2)*(((e*B)/(m*c))**2)+1)-1)
    #print(energy)

    if Height >= Length:
        Theta = -1*np.arctan(abs((Length**2)-(Height**2))/(2*Height*Length))
    else:
        Theta = 1*np.arctan(abs((Length**2)-(Height**2))/(2*Height*Length))

    Sensitivity = funcsensitivity(energy)
    Fadingratio = funcfading(Ftime)
    Transmittance = functransemittance(energy, Theta, t)
    Correction=int(((HeightIP-Height)*np.tan(Theta))/Resolution)

    if i+Correction <= 0:
        SigPSL = 0
        NoisePSL = 0
    elif i+Correction >= row:
        SigPSL = 0
        NoisePSL = 0
    else:
        SigPSL = S[i+Correction]
        NoisePSL = (BG1[i+Correction]+BG2[i+Correction])/2
        SigPSL = SigPSL - NoisePSL
    

    # print(Correction)

    # Calculate differential number density perunit solidangle and energy       
    Sig =(e*1e6)*(SigPSL/(Resolution*Solidangle*(m*c**2)))*((m*c)/(e*B))*((Height*np.cos(Theta))/(Length*Transmittance*Sensitivity*Fadingratio))*np.sqrt(1+(((2*Height)/((Length**2)+(Height**2)))**2)*(((m*c)/(e*B))**2))        
    Noise =(e*1e6)*(NoisePSL/(Resolution*Solidangle*(m*c**2)))*((m*c)/(e*B))*((Height*np.cos(Theta))/(Length*Transmittance*Sensitivity*Fadingratio))*np.sqrt(1+(((2*Height)/((Length**2)+(Height**2)))**2)*(((m*c)/(e*B))**2))

    if Sig < 1e7:
        Sig=np.nan
        Noise=np.nan

    NetSig.append(Sig)
    NoiseSig.append(Noise)
    Energy.append(energy)  

Spectrum = np.column_stack((Energy, NetSig, NoiseSig))
# np.savetxt("spectrum.txt", Spectrum, delimiter='\t')

plt.figure()
plt.plot(Energy,NetSig,color='red')  
# plt.plot(Energy,NoiseSig,color='blue')
# plt.yscale('log')
plt.xscale("log")
# plt.xlim(0,1.5)
#plt.ylim(0,4e9)
plt.xlabel('Energy (MeV)')
plt.ylabel('d2N/dOmegadE')
plt.title('Electron Energy Spectrum')
plt.show()
