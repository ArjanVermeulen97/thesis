# -*- coding: utf-8 -*-
"""
Author: Arjan Vermeulen
"""

from math import atan, atan2, sin, cos, pi, sqrt
from random import normalvariate, expovariate, uniform, seed
import matplotlib.pyplot as plt
import numpy as np

seed(13)

# Parameters
semiMajorMean = 2
semiMajorStd = 1
semiMajorMax = 4
semiMajorMin = 0
eccentricityMean = 0.4
eccentricityStd = 0.2
eccentricityMin = 0
eccentricityMax = 1
inclinationMean = 0.05
inclinationMin = 0
inclinationMax = 180
anomalyMin = 0
anomalyMax = 360
raanMin = 0
raanMax = 360
argPeriMin = 0
argPeriMax = 360
magMin = 15
magMax = 25

# Rotation matrices
def R_1(a):
    R = np.array([[1, 0, 0],
                  [0, cos(a), 1*sin(a)],
                  [0, -1*sin(a), cos(a)]])
    return R

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


def gen_asteroid():
    semiMajor = semiMajorMin
    eccentricity = eccentricityMin
    inclination = inclinationMin
    anomaly = anomalyMin
    raan = raanMin
    argPeri = argPeriMin
    
    while not (semiMajor > semiMajorMin and semiMajor < semiMajorMax):
        semiMajor = normalvariate(semiMajorMean, semiMajorStd)
    
    while not (eccentricity > eccentricityMin and eccentricity < eccentricityMax):
        eccentricity = normalvariate(eccentricityMean, eccentricityStd)
    
    while not (inclination > inclinationMin and inclination < inclinationMax):
        inclination = expovariate(inclinationMean)
        
    anomaly = uniform(anomalyMin, anomalyMax)
    raan = uniform(raanMin, raanMax)
    argPeri = uniform(argPeriMin, argPeriMax)
    mag = uniform(magMin, magMax)
    
    return [semiMajor, eccentricity, inclination, anomaly, raan,
            argPeri]

def pos_heliocentric(a, e, i, theta, raan, argPeri):
    i = i/180*pi
    raan = raan/180*pi
    argPeri = argPeri/180*pi
    theta = theta/180*pi
    
    r = a*(1+e**2)/(1+e*cos(theta))
    r_orbit = np.array([[r*cos(theta)],
                        [r*sin(theta)],
                        [0]])
    
    r_helio = R_3(-1*raan) @ R_1(-1*i) @ R_3(-1*argPeri) @ r_orbit
    
    return r_helio

def calc_MOID(xs, ys, zs):
    MOID = 1000
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        z = zs[i]
        a = atan2(y, x)
        x_e = cos(a)
        y_e = sin(a)
        d = sqrt((x-x_e)**2 + (y-y_e)**2 + z**2)
        if d < MOID:
            MOID = d
    return MOID

def print_asteroid(asteroid):
    print(f"semi-major axis: {asteroid[0]}")
    print(f"eccentricity:    {asteroid[1]}")
    print(f"icnlination:     {asteroid[2]}")
    print(f"RA of the AN:    {asteroid[4]}")
    print(f"Arg of Periaps:  {asteroid[5]}")
    print(f"True anomaly:    {asteroid[3]}")
    
    return None
        

# Earth
positions = [pos_heliocentric(1, 0, 0, theta, 0, 0) for theta in range(0, 360)]
xs = [item[0][0] for item in positions]
ys = [item[1][0] for item in positions]
zs = [item[2][0] for item in positions]



## ANIMATION ##
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(0, 0, 0, c='yellow')

# ax.plot(xs, ys, zs, c='green')
# ax.scatter(*pos_heliocentric(1, 0, 0, 0, 0, 0), c='green')

# for i in range(3):
#     color = ['red', 'black', 'blue', 'purple', 'cyan'][i]
#     print(f"Asteroid {i+1}: {color}")
#     print("-----------------------")
#     asteroid = gen_asteroid()
#     print_asteroid(asteroid)
    
#     positions = [pos_heliocentric(asteroid[0],
#                                   asteroid[1],
#                                   asteroid[2],
#                                   theta,
#                                   asteroid[4],
#                                   asteroid[5])
#                  for theta in range(0, 360)]
    
#     xs = [item[0][0] for item in positions]
#     ys = [item[1][0] for item in positions]
#     zs = [item[2][0] for item in positions]
#     print(f"MOID:            {calc_MOID(xs, ys, zs)}")
#     print()
    
#     ax.plot(xs, ys, zs, c=color)
#     ax.scatter(*pos_heliocentric(*asteroid), c=color)


# ax.set_xlim3d(-2, 2)
# ax.set_ylim3d(-2, 2)
# ax.set_zlim3d(-2, 2)

# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)