# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:17:20 2021

@author: Arjan
"""

import numpy as np
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from transformations import angle_calc, ecliptic_to_galactic, R_2, R_3, sun_scale
import matplotlib.pyplot as plt

def blackbody_hz(wavelength, temp, megajansky=False):
    ''' returns blackbody radiation as function of wavelength and temperature'''
    h = 6.626E-34 # Plack constant
    kB = 1.380E-23 # Boltzmann constant
    c = 299792458 # Light speed
    v = c / wavelength
    B = (2*h*v**3)/(c**2) / (np.exp((h*v)/(kB*temp)) - 1)
    if megajansky:
        B = B/(10**-20)
    return B

def blackbody_mum(wavelength, temp, micrometer=False):
    ''' returns blackbody radiation as function of wavelength and temperature'''
    h = 6.626E-34 # Plack constant
    kB = 1.380E-23 # Boltzmann constant
    c = 299792458 # Light speed
    B = (2*h*c**2)/(wavelength**5) / (exp((h*c)/(wavelength * kB * temp)) - 1)
    if micrometer:
        B = B/(10**6)
    return B


def tir_mag(R, t_alb, wavelength, beaming, t_x, t_y, t_z, s_x, s_y, s_z):
    stefanBoltzmann = 5.67E-8
    solarFlux = 1373 / (t_x**2 + t_y**2 + t_z**2)
    em = 0.9
    T_max = ((1 - t_alb)*solarFlux/(beaming*stefanBoltzmann))**0.25
    
    # Next, calculate phase angle
    r_x = s_x - t_x     # relative x
    r_y = s_y - t_y     # relative y
    r_z = s_z - t_z     # relative z
    t_abs = sqrt(t_x*t_x + t_y* t_y + t_z*t_z)
    r_abs = sqrt(r_x*r_x + r_y* r_y + r_z*r_z)
    phase = acos((-t_x*r_x - t_y*r_y - t_z*r_z) / (t_abs*r_abs))
    
    delta_theta = pi/4
    delta_phi = pi/4
    flux = 0
    for theta in [-3/8*pi+phase, -1/8*pi+phase, 1/8*pi+phase, 3/8*pi+phase]:
        for phi in [-3/8*pi, -1/8*pi, 1/8*pi, 3/8*pi]:
            if theta > 7*pi/16:
                continue # No flux
            T = T_max*cos(theta)**0.25 * cos(phi)**0.25
            flux += delta_theta * delta_phi *\
                blackbody_mum(wavelength, T, 1) * cos(phi) * cos(theta - phase)
    power = flux*em*R**2
    receivedFlux = power / ((r_abs * 149_579_900_000)**2)
    return receivedFlux

### ASTEROID PROPERTIES
t_mag_neatm = 16300/2 #Actually radius
t_alb_neatm = 0.02
beaming_neatm = 1.22
t_mag_stm = 12200/2
t_alb_stm = 0.036
beaming_stm = 0.756
e_x = -1.00E0
e_y = 5.377E-2
e_z = 1.989E-4
t_x = -2.335E0
t_y = 1.239E0
t_z = 5.443E-1

wavelengths = np.arange(1, 41) * 1E-6
fluxes_neatm = [tir_mag(t_mag_neatm, t_alb_neatm, l, beaming_neatm, t_x, t_y, t_z, e_x, e_y, e_z) for l in wavelengths]
fluxes_stm = [tir_mag(t_mag_stm, t_alb_stm, l, beaming_stm, t_x, t_y, t_z, e_x, e_y, e_z) for l in wavelengths]

plt.plot(wavelengths, fluxes_neatm, 'b')
#plt.plot(wavelengths, fluxes_stm, 'b')
plt.yscale('log')
plt.xscale('log')
plt.ylim((1E-15, 1E-14))
plt.xlim((5E-6, 4E-5))
plt.grid(which='minor')