# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:08:25 2021

@author: Arjan
"""

from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### INPUTS ###
aperture = 1                # m
pixelAngle = 0.1            # sr
quantumEff = 0.9            # -
transmittance = 0.9         # -
straddleFac = 1             # -
solarRadiance = 5.79e10     # W/sr
integrationTime = 10        # s
readNoise = 0.1             # e-
poissonNoiseDark = 0.1      # e-


def trailing_loss(pixelWidth, vTransverse, time, distance):
    factor = pixelWidth / ((vTransverse*time / distance) + pixelWidth)
    deltaV = 2.5 * log10(factor)
    return deltaV


def ecliptic_to_galactic(l_ecl, b_ecl):
    # Equinox 2000, Leinert et al.
    b_NGP = 29.81           # Latitude of North Galactic Pole, deg
    l_NGP = 270.02          # Ascending note of galaxy, deg
    l_c = 6.38              # Direction to galactic core, deg
    
    l_ecl = l_ecl / 180 * pi
    b_ecl = b_ecl / 180 * pi
    b_NGP = b_NGP / 180 * pi
    l_NGP = l_NGP / 180 * pi
    l_c = l_c / 180 * pi
    
    b_gal = 180/pi*asin(sin(b_ecl)*sin(b_NGP) -
                        cos(b_ecl)*cos(b_NGP)*sin(l_ecl - l_NGP))
    try:
        l_gal = 180/pi*(l_c + acos(cos(l_ecl - l_NGP)*cos(b_ecl)/cos(b_gal)))
    except ValueError:
        l_gal = 0

    return l_gal, b_gal
    
    

def phase_mag(t_mag, t_x, t_y, t_z, s_x, s_y, s_z):
    r_x = s_x - t_x     # relative x
    r_y = s_y - t_y     # relative y
    r_z = s_z - t_z     # relative z
    t_abs = sqrt(t_x*t_x + t_y* t_y + t_z*t_z)
    r_abs = sqrt(r_x*r_x + r_y* r_y + r_z*r_z)
    elongation = acos((t_x*r_x + t_y*r_y + t_z*r_z) / (t_abs*r_abs))
    phase = acos((-t_x*r_x - t_y*r_y - t_z*r_z) / (t_abs*r_abs))
    
    if elongation / pi * 180 < 60:
        # Per Stokes at al (2003)
        V = t_mag + 5*log10(t_abs*r_abs) + 5.03 - 10.373*log10(pi - phase)
    else:
        # Per Stokes er al (2003)
        phi_1 = exp(-3.33*(tan(phase/2))**0.63)
        phi_2 = exp(-1.87*(tan(phase/2))**1.22)
        V = t_mag + 5*log10(t_abs*r_abs) - 2.5*log10(0.85*phi_1 + 0.15*phi_2)
    
    return V

zodiacGegenschein = pd.read_csv('zodiacgegenschein.csv', index_col=0, header=0)
starBackground = pd.read_csv('starbackground.csv', index_col=0, header=0)

background = np.zeros((180, 361))
background_ZGS = np.zeros((180, 361))
background_SBG = np.zeros((180, 361))

l_solar = 30

for i in range(0, 361):
    for j in range(0, 181):
        l_ecl_g = i
        if l_ecl_g > 180:
            l_ecl_g = 360 - l_ecl_g
        b_ecl = j - 90
        l_gal, b_gal = ecliptic_to_galactic(l_ecl_g, b_ecl)
        l_ecl = i - l_solar
        if l_ecl < 0:
            l_ecl = 360 + l_ecl
        if l_ecl > 180:
            l_ecl = 360 - l_ecl
        
        # Interpolate all table values for Zodiacal light and Gegenschein
        # Get annoyed by string indices....
        for k in range(len(zodiacGegenschein.index) - 1):
            low = zodiacGegenschein.index[k]
            high = zodiacGegenschein.index[k+1]
            if int(high) <= l_ecl <= int(low):
                l_ecl_low = low
                l_ecl_high = high
                # print(l_ecl, low, high)
            else:
                pass
                # print(l_ecl, low, high, "no")
        
        for k in range(len(zodiacGegenschein.columns) - 1):
            low = zodiacGegenschein.columns[k]
            high = zodiacGegenschein.columns[k+1]
            if int(low) <= b_ecl <= int(high):
                b_ecl_low = low
                b_ecl_high = high
        
        # Do linear interpolation
        
        ZGS_1_1 = zodiacGegenschein.loc[l_ecl_low, b_ecl_low]
        # print(l_ecl, b_ecl, l_ecl_low, b_ecl_low, ZGS_1_1)
        ZGS_1_2 = zodiacGegenschein.loc[l_ecl_low, b_ecl_high]
        ZGS_2_1 = zodiacGegenschein.loc[l_ecl_high, b_ecl_low]
        ZGS_2_2 = zodiacGegenschein.loc[l_ecl_high, b_ecl_high]

        fac_l_1 = abs(int(l_ecl_high)-l_ecl) / abs(int(l_ecl_high)-int(l_ecl_low))
        fac_l_2 = abs(l_ecl-int(l_ecl_low)) / abs(int(l_ecl_high)-int(l_ecl_low))
        fac_b_1 = abs(int(b_ecl_high)-b_ecl) / abs(int(b_ecl_high)-int(b_ecl_low))
        fac_b_2 = abs(b_ecl-int(b_ecl_low)) / abs(int(b_ecl_high)-int(b_ecl_low))       
        
        ZGS = fac_l_1*(fac_b_1*ZGS_1_1 + fac_b_2*ZGS_1_2) +\
            fac_l_2*(fac_b_1*ZGS_1_1 + fac_b_2*ZGS_1_2)
            
            
        # Interpolate all table values for background starlight
        # Get annoyed by string indices....
        for k in range(len(starBackground.index) - 1):
            low = starBackground.index[k]
            high = starBackground.index[k+1]
            if int(low) <= l_gal <= int(high):
                l_gal_low = low
                l_gal_high = high
        
        for k in range(len(starBackground.columns) - 1):
            low = starBackground.columns[k]
            high = starBackground.columns[k+1]
            if int(low) <= b_gal <= int(high):
                b_gal_low = low
                b_gal_high = high

        # Do nearest/linear interpolation
        
        SBG_1_1 = starBackground.loc[l_gal_low, b_gal_low]
        SBG_1_2 = starBackground.loc[l_gal_low, b_gal_high]
        SBG_2_1 = starBackground.loc[l_gal_high, b_gal_low]
        SBG_2_2 = starBackground.loc[l_gal_high, b_gal_high]
        '''
        if int(l_gal_high) - l_gal > l_gal - int(l_gal_low):
            if int(b_gal_high) - b_gal > b_gal - int(b_gal_low):
                SBG = SBG_1_1
            else:
                SBG = SBG_1_2
        else:
            if int(b_gal_high) - b_gal > b_gal - int(b_gal_low):
                SBG = SBG_2_1
            else:
                SBG = SBG_2_2
        
        '''
        fac_l_1 = abs(int(l_gal_high)-l_gal) / abs(int(l_gal_high)-int(l_gal_low))
        fac_l_2 = abs(l_gal-int(l_gal_low)) / abs(int(l_gal_high)-int(l_gal_low))
        fac_b_1 = abs(int(b_gal_high)-b_gal) / abs(int(b_gal_high)-int(b_gal_low))
        fac_b_2 = abs(b_gal-int(b_gal_low)) / abs(int(b_gal_high)-int(b_gal_low))       
        
        SBG = fac_l_1*(fac_b_1*SBG_1_1 + fac_b_2*SBG_1_2) +\
            fac_l_2*(fac_b_1*SBG_1_1 + fac_b_2*SBG_1_2)
        
            
        signalBG = ZGS + SBG
        background[j, (i+180)%360] = signalBG
        background_ZGS[j, (i+180)%360] = ZGS
        background_SBG[j, (i+180)%360] = SBG
        
plt.figure(figsize=(10,10))
plt.imshow(background, cmap='gray', vmin=0, vmax=1000)
plt.colorbar()
plt.figure(figsize=(10,10))
plt.imshow(background_ZGS, cmap='gray', vmin=0, vmax=1000)
plt.colorbar()
plt.figure(figsize=(10,10))
plt.imshow(background_SBG, cmap='gray', vmin=0, vmax=1000)
plt.colorbar()