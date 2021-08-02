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
pixelAngle = 7.62e-11       # sr (corr 5 deg x 5 deg at 100MP)
quantumEff = 0.9            # -
transmittance = 0.99        # -
straddleFac = 1             # - (compensated by trailing loss eq)
solarRadiance = 5.79e10     # W/sr??
solarFlux = 1350            # W/m2
integrationTime = 10        # s
readNoise = 400             # J
darkNoise = 1000            # J
coulomb = 1.6E-19           # J/e-
asteroidRange = 0.3           # AU (BEUNFACTOR)
asteroidSize = 50           # m
l_solar = 0                 # solar latitude

def R_2(a):
    R = np.array([[cos(a), 0, 1*sin(a)],
                  [0, 1, 0],
                  [-1*sin(a), 0, cos(a)]])
    return R

def R_3(a):
    R = np.array([[cos(a), 1*sin(a), 0],
                  [-1*sin(a), cos(a), 0],
                  [0, 0, 1]])
    return R


def trailing_loss(pixelWidth, vTransverse, time, distance):
    factor = pixelWidth / ((vTransverse*time / distance) + pixelWidth)
    deltaV = 2.5 * log10(factor)
    return deltaV


def angleCalc(sinValue, cosValue):
    # Copied from MatLab (https://nl.mathworks.com/matlabcentral/fileexchange/42365-angle-calculator-from-sin-and-cos-values)
    # FLOATING POINT ERRORSSSSSSSSSS
    if sinValue > 0.9999999 and sinValue < 1.00000001:
        sinValue = 1
    if sinValue < -0.9999999 and sinValue > -1.00000001:
        sinValue = -1
    theta=asin(sinValue)
    if cosValue < 0:
        if sinValue > 0:
            theta = pi - theta
        elif sinValue < 0:
            theta = -pi-theta
        else:
            theta = theta+pi
    return theta


def ecliptic_to_galactic(l_ecl, b_ecl):
    # Equinox 2000, Leinert et al.
    b_NGP = 29.81           # Latitude of North Galactic Pole, deg
    l_NGP = 270.02          # Ascending node of galaxy, deg
    l_c = 6.38              # Direction to galactic core, deg
    
    l_ecl = l_ecl / 180 * pi
    b_ecl = b_ecl / 180 * pi
    b_NGP = b_NGP / 180 * pi
    l_NGP = l_NGP / 180 * pi
    l_c = l_c / 180 * pi
    
    b_gal = 180/pi*asin(sin(b_ecl)*sin(b_NGP) -
                        cos(b_ecl)*cos(b_NGP)*sin(l_ecl - l_NGP))
    if abs(b_gal - 0.5*pi) > 1e-10:
        l_sin = (sin(b_ecl)*cos(b_NGP) +
                 cos(b_ecl)*sin(b_NGP)*sin(l_ecl - l_NGP))/cos(b_gal/180*pi)
        l_cos = cos(l_ecl - l_NGP)*cos(b_ecl)/cos(b_gal/180*pi)
        try:
            l_gal = 180/pi*angleCalc(l_sin, l_cos) + l_c
            if l_gal < 0:
                l_gal = 360 + l_gal
        except ValueError:
            print(l_sin, l_cos, b_gal)
            raise ValueError
    else:
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
        try:
            V = t_mag + 5*log10(t_abs*r_abs) + 5.03 - 10.373*log10(pi - phase)
        except ValueError:
            # 0 degrees, backlit by sun
            V = 1000
    else:
        # Per Stokes er al (2003)
        phi_1 = exp(-3.33*(tan(phase/2))**0.63)
        phi_2 = exp(-1.87*(tan(phase/2))**1.22)
        V = t_mag + 5*log10(t_abs*r_abs) - 2.5*log10(0.85*phi_1 + 0.15*phi_2)
    
    return V

zodiacGegenschein = pd.read_csv('zodiacgegenschein.csv', index_col=0, header=0)
starBackground = pd.read_csv('starbackground.csv', index_col=0, header=0)

SNR = np.zeros((181, 361))

for i in range(0, 361):
    for j in range(0, 181):
        # Fetch coordinates
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
            
        # Get asteroid position and magnitude
        vec = np.array([-asteroidRange, 0, 0])
        pos = R_3(b_ecl/180*pi)@R_2(l_ecl/180*pi)@vec
        x_t = pos[0] + 1
        y_t = pos[1]
        z_t = pos[2]
        mag_t = -5*log10(asteroidSize * sqrt(0.15) / 1.329E6)
        
        V_t = phase_mag(mag_t, x_t, y_t, z_t, 1, 0, 0)
        fluxTarget = 100**((V_t+26.74)/-5)*solarFlux
        
        
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
        interpolation = 'linear'
        
        SBG_1_1 = starBackground.loc[l_gal_low, b_gal_low]
        SBG_1_2 = starBackground.loc[l_gal_low, b_gal_high]
        SBG_2_1 = starBackground.loc[l_gal_high, b_gal_low]
        SBG_2_2 = starBackground.loc[l_gal_high, b_gal_high]
        if interpolation == 'nearest':
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
        elif interpolation == 'linear':
            fac_l_1 = abs(int(l_gal_high)-l_gal) / abs(int(l_gal_high)-int(l_gal_low))
            fac_l_2 = abs(l_gal-int(l_gal_low)) / abs(int(l_gal_high)-int(l_gal_low))
            fac_b_1 = abs(int(b_gal_high)-b_gal) / abs(int(b_gal_high)-int(b_gal_low))
            fac_b_2 = abs(b_gal-int(b_gal_low)) / abs(int(b_gal_high)-int(b_gal_low))       
            
            SBG = fac_l_1*(fac_b_1*SBG_1_1 + fac_b_2*SBG_1_2) +\
                fac_l_2*(fac_b_1*SBG_1_1 + fac_b_2*SBG_1_2)
        
            
        signalBG = 6.62E-12 * (ZGS + SBG) * pixelAngle * solarFlux / coulomb
        signalTarget = fluxTarget * aperture * pixelAngle * quantumEff *\
            transmittance*integrationTime / coulomb
        ratio = (signalTarget / pixelAngle * straddleFac) /\
              sqrt(readNoise + darkNoise + signalBG + signalTarget)
        SNR[j, (i+180)%360] = ratio

plt.figure(figsize=(10,5))
plt.title(f"Signal-to-noise ratio for {asteroidSize}m asteroid at {asteroidRange}AU from spacecraft.")
plt.imshow(SNR, cmap='hot', extent=[-180, 180, -90, 90])
plt.xlabel('lambda [deg]')
plt.ylabel('beta [deg]')
plt.colorbar()
plt.contour(SNR, 16, cmap='plasma', extent=[-180, 180, 90, -90])

