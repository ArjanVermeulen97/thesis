# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:22:49 2021

@author: Arjan
"""

import numpy as np
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos

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


def angle_calc(sinValue, cosValue):
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
            l_gal = 180/pi*angle_calc(l_sin, l_cos) + l_c
            if l_gal < 0:
                l_gal = 360 + l_gal
        except ValueError:
            print(l_sin, l_cos, b_gal)
            raise ValueError
    else:
        l_gal = 0
    

    return l_gal, b_gal


def coords_cartersian(R_sc, l_sc, b, l, s):
    '''Returns cartesian coordinates from latitude and longitude'''
    X = R_sc*np.cos(l_sc) + s*np.cos(b)*np.cos(l)
    Y = R_sc*np.sin(l_sc) + s*np.cos(b)*np.sin(l)
    Z = s*np.sin(b)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    return X, Y, Z, R


def sun_scale(R):
    return R**(-2.3)