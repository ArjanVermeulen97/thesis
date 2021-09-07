# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:16:00 2021

@author: Arjan

Code written to convert silly DIRBE data format to a format similar to
 the data used for the visual observation. Interpolation is NN because
 there are far more data points given than needed, and it saves a lot of
 processing capacity.
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

fits_info = fits.open('DIRBE_SKYMAP_INFO.FITS')
fits_49 = fits.open('DIRBE_BAND04_ZSMA.FITS')
fits_12 = fits.open('DIRBE_BAND05_ZSMA.FITS')

data_info = fits_info[1].data
data_49 = fits_49[1].data
data_12 = fits_12[1].data
if False:
    # No list comprehension so we can check
    data_comb = []
    for i in range(len(data_info)):
        assert data_info[i][0] == data_49[i][0]
        assert data_info[i][0] == data_12[i][0]
        entry = [data_info[i][0], data_49[i][1], data_12[i][1],
                 data_info[i][3], data_info[i][4]]
        data_comb.append(entry)
        
    data_coords = [[item[3], item[4]] for item in data_comb]
    data_z = [(item[1]+item[2])/2 for item in data_comb]
    
    f = NearestNDInterpolator(data_coords, data_z)

if False:
    # Generate pretty picture
    x_coords = np.arange(180, 360, 0.1)
    y_coords = np.arange(-90, 90, 0.1)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = f(X, Y)
    
    plt.pcolormesh(X, Y, Z, vmin=0, vmax=15, cmap='hot')
    
if False:
    # Generate tabulation similar to other tabulation
    x_coords = np.arange(0, 361, 10)
    y_coords = np.array([-90, -80, -70, -60, -50,
                         -40, -30, -20, -15, -10,
                         -5, -2, 0, 2, 5,
                         10, 15, 19, 30, 40,
                         50, 60, 70, 80, 90])
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = f(X, Y).T
    np.savetxt('ir_starbackground.csv', Z, delimiter=',')
    
    