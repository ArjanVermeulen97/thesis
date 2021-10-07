# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:38:47 2021

@author: Arjan
"""

import numpy as np
import pandas as pd
import swifter
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from orbital import pos_heliocentric, theta_solve, theta_step

df_asteroids = pd.read_csv('granvik_neo.csv')
#df_asteroids = df_asteroids.sample(100_000, random_state=9)
df_asteroids['long_node'] = df_asteroids['long_node']/180*pi
df_asteroids['arg_peri'] = df_asteroids['arg_peri']/180*pi
df_asteroids['anomaly'] = df_asteroids['anomaly']/180*pi
df_asteroids['i'] = df_asteroids['i']/180*pi
df_asteroids['t_impact'] = 0

earth = {'a': 1.0,
         'e': 0.0167,
         'i': 0.0,
         'long_node': -11.26/180*pi,
         'arg_peri': 114.21/180*pi,
         'anomaly': 358.62/180*pi,
         'SOI': 0.067}


def asteroid_positions(day, df_asteroids):
    '''Propagate all orbital elements to certain day'''
    df_asteroids['M'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: theta_step(row['a'], row['anomaly'], day), axis=1)
    df_asteroids['theta'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: theta_solve(row['e'], row['M']), axis=1)
    df_asteroids['x'], df_asteroids['y'], df_asteroids['z'] = zip(*df_asteroids.swifter.progress_bar(False).apply(lambda row: pos_heliocentric(row['a'], 
                                                                                                              row['e'], 
                                                                                                              row['i'], 
                                                                                                              row['theta'], 
                                                                                                              row['long_node'], 
                                                                                                              row['arg_peri']),
                                                                                 axis=1
                                                                                 ))
    return df_asteroids


def earth_position(day, earth):
    '''Propagate earth positions'''
    earth['M'] = theta_step(earth['a'], earth['anomaly'], day)
    earth['theta'] = theta_solve(earth['e'], earth['M'])
    earth['x'], earth['y'], earth['z'] = pos_heliocentric(earth['a'],
                                                    earth['e'],
                                                    earth['i'],
                                                    earth['theta'],
                                                    earth['long_node'],
                                                    earth['arg_peri'])
    return earth


def check_SOI(earth, asteroid):
    r = sqrt((asteroid['x'] - earth['x'])**2 + (asteroid['y'] - earth['y'])**2 + (asteroid['z'] - earth['z'])**2)
    if r < earth['SOI']:
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    for day in np.arange(0, 1500, 1):
        # All days from 1-1-2000 to 31-12-2029
        earth = earth_position(day, earth)
        df_asteroids = asteroid_positions(day, df_asteroids)
        df_asteroids['t_impact'] = df_asteroids.apply(lambda row: day if check_SOI(earth, row) and row['t_impact'] == 0 else row['t_impact'],axis=1)
        
        print(f"{day}: {df_asteroids.loc[df_asteroids['t_impact'] > 0]['t_impact'].count()}")
        df_asteroids.to_csv('granvik_impactors.csv')