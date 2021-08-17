# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:50:02 2021

@author: Arjan

Calculation of IR Zodiacal light
"""

import numpy as np
import matplotlib.pyplot as plt
from ir_data_convert import f

# Constants general
T_0 = 286
delta = 0.467

def blackbody(wavelength, temp, megajansky=False):
    ''' returns blackbody radiation as function of wavelength and temperature'''
    h = 6.626E-34 # Plack constant
    kB = 1.380E-23 # Boltzmann constant
    c = 299792458 # Light speed
    v = c / wavelength
    B = (2*h*v**3)/(c**2) / (np.exp((h*v)/(kB*temp)) - 1)
    if megajansky:
        B = B/(10**-20)
    return B

def coords_cartersian(R_sc, l_sc, b, l, s):
    '''Returns cartesian coordinates from latitude and longitude'''
    X = R_sc*np.cos(l_sc) + s*np.cos(b)*np.cos(l)
    Y = R_sc*np.sin(l_sc) + s*np.cos(b)*np.sin(l)
    Z = s*np.sin(b)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    return X, Y, Z, R

# Constants smooth cloud
n_0_sc = 1.13E-7
alpha_sc = 1.34
beta_sc = 4.14
gamma_sc = 0.942
mu_sc = 0.189
i_sc = 2.03 / 180 * np.pi
Omega_sc = 77.7 / 180 * np.pi
X_0_sc = 0.0119
Y_0_sc = 0.00548
Z_0_sc = -0.00215
E_49_sc = 0.997
E_12_sc = 0.958

def density_c(X, Y, Z):
    '''returns density of smooth cloud at given position'''
    X_sc = X - X_0_sc
    Y_sc = Y - Y_0_sc
    Z_sc = Z - Z_0_sc
    R_c = np.sqrt(X_sc**2 + Y_sc**2 + Z_sc**2)
    Z_c = X_sc*np.sin(Omega_sc)*np.sin(i_sc) -\
        Y_sc*np.cos(Omega_sc)*np.sin(i_sc) +\
            Z_sc*np.cos(i_sc)
    
    zeta = abs(Z_c / R_c)
    g = zeta**2 / (2*mu_sc) if zeta < mu_sc else zeta - mu_sc/2
    n_sc = n_0_sc * R_c**(-1*alpha_sc) * np.exp(-1 * beta_sc * g * gamma_sc)
    
    return n_sc

# Constants dust bands
n_0_b = [5.59E-9, 1.99E-9, 1.44E-10]
delta_zeta_b = [8.78/180*np.pi, 1.99/180*np.pi, 15/180*np.pi]
nu_b = [0.1, 0.9, 0.05]
p_b = [4, 4, 4]
i_b = [0.56/180*np.pi, 1.2/180*np.pi, 0.8/180*np.pi]
Omega_b = [80/180*np.pi, 30.3/180*np.pi, 80/180*np.pi]
delta_r_b = [1.5, 0.94, 1.5]
E_49_b = 0.359
E_12_b = 1.01

def density_b(X, Y, Z, n):
    '''returns density of nth band at given position'''
    R_b = np.sqrt(X**2 + Y**2 + Z**2)
    Z_b = X*np.sin(Omega_b[n])*np.sin(i_b[n]) -\
        Y*np.cos(Omega_b[n])*np.sin(i_b[n]) +\
            Z*np.cos(i_b[n])
    zeta = abs(Z_b / R_b)
    
    n_b = 3*n_0_b[n]/R_b * np.exp(-1*(zeta/delta_zeta_b[n])**6) *\
        (nu_b[n] + (zeta/delta_zeta_b[n])**p_b[n]) *\
            (1-np.exp(-1*(R_b/delta_r_b[n])**20))
    return n_b

# Circumsolar ring
n_0_r = 1.83E-8
R_r = 1.03
sigma_r_r = 0.025
sigma_z_r = 0.054
i_r = 0.49/180*np.pi
Omega_r = 22.3/180*np.pi
E_49_r = 1.06
E_12_r = 1.06

def density_r(X, Y, Z):
    '''returns density of circumsolar ring at given position'''
    R_sr = np.sqrt(X**2 + Y**2 + Z**2)
    Z_sr = X*np.sin(Omega_r)*np.sin(i_r) -\
        Y*np.cos(Omega_r)*np.sin(i_r) +\
            Z*np.cos(i_r)
            
    n_sr = n_0_r * np.exp(-1*(R_sr - R_r)**2 / (2*sigma_r_r**2) -\
                          abs(Z_sr)/sigma_z_r)

    return n_sr


def flux(lambda_sun, lambda_obs, beta_obs, R_sat, wavelength, verbose=False):
    assert wavelength == 4.9E-6 or wavelength == 12E-6
    # Individual contributions
    F_c = 0
    F_b = 0
    F_r = 0
    stepsize = 0.1
    if wavelength == 4.9E-6:
        E_c = E_49_sc
        E_b = E_49_b
        E_r = E_49_r
    else:
        E_c = E_12_sc
        E_b = E_12_b
        E_r = E_12_r
    s = stepsize/2
    R = 0
    while R < 5.2:
        X = R_sat * np.cos(lambda_sun) +\
            s * np.cos(beta_obs) * np.cos(lambda_obs)
        Y = R_sat * np.sin(lambda_sun) +\
            s * np.cos(beta_obs) * np.sin(lambda_obs)
        Z = s * np.sin(beta_obs)
        R = np.sqrt(X**2 + Y**2 + Z**2)
        T = T_0*R**(-1*delta)
        F_c += density_c(X, Y, Z)*E_c*blackbody(wavelength, T, True)*stepsize
        F_b += (density_b(X, Y, Z, 0) + density_b(X, Y, Z, 1) +\
                density_b(X, Y, Z, 2))*\
            E_b*blackbody(wavelength, T, True)*stepsize
        F_r += density_r(X, Y, Z)*E_r*blackbody(wavelength, T, True)*stepsize
        s += stepsize
    if verbose:
        print(f"Wavelength: {wavelength}m")
        print(f"--------------------------------")
        print(f"Contribution Smooth Cloud: {F_c}")
        print(f"Contribution Dust Bands:   {F_b}")
        print(f"Contribution Solar Ring:   {F_r}")
        print(f"Total: {F_c + F_r + F_b}")
    return F_c + F_r + F_b


if False:
    # Generate pretty pictures
    x_arr = np.linspace(-2, 2, 500)
    y_arr = np.linspace(-1, 1, 500)
    
    density_arr = np.zeros((500, 500))
    for i in range(500):
        for j in range(500):
            # Smooth cloud
            density_arr[i][j] += density_c(x_arr[j], 0, y_arr[i])
            # Dust bands
            density_arr[i][j] += density_b(x_arr[j], 0, y_arr[i], 0)
            density_arr[i][j] += density_b(x_arr[j], 0, y_arr[i], 1)
            density_arr[i][j] += density_b(x_arr[j], 0, y_arr[i], 2)
            # Circumsolar ring
            density_arr[i][j] += density_r(x_arr[j], 0, y_arr[i])
                    
    levels_cloud = [0.2E-7, 0.4E-7, 0.6E-7, 0.8E-7, 1E-7, 
                    1.2E-7, 1.4E-7, 1.6E-7, 1.8E-7, 2E-7]
    
    levels_bands_ring = [item for item in levels_cloud]
            
    plt.contour(x_arr, y_arr, density_arr, levels=levels_bands_ring)
    plt.grid()
    plt.colorbar()
    
if True:
    # Combine
    z_arr = np.zeros((360, 720))
    for y_index in range(360):
        print(y_index)
        for x_index in range(720):
            y = y_index/2 - 90
            x = (x_index/2 - 180)%360
            
            z_arr[y_index][x_index] += f(x, y) +\
                0.5*flux(np.pi,x/180*np.pi,y/180*np.pi, 1, 4.9E-6) +\
                    0.5*flux(np.pi, x/180*np.pi, y/180*np.pi, 1, 12E-6)
    plt.imshow(z_arr, extent=[0, 360, -90, 90], vmin=0, vmax=35, cmap='hot')
    plt.colorbar()
    plt.xlabel('Ecliptic longitude [deg]')
    plt.ylabel('Ecliptic latitude [deg]')
    plt.title('IR background averaged over 4.9 and 12 micron [MJy/sr]')