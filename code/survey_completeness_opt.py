# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:59:53 2021

@author: Arjan
"""

import numpy as np
import pandas as pd
import swifter
from scipy import optimize
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from observations import observation, detection, detection_prob
from orbital import pos_heliocentric, theta_solve, theta_step
import seaborn as sns
from population_models import init_asteroids_jpl, init_asteroids_granvik
import matplotlib.pyplot as plt
import time
import pickle

def init_asteroids():
    return init_asteroids_jpl(5000)


def init_satellites(n, a, e, s):
    satellites = {i: {'a': a,
                      'e': e,
                      'i': 0,
                      'long_node': 0,
                      'arg_peri': 0,
                      'anomaly': s*n,
                      'payload': 'VIS',
                      'cadence': 2} for i in range(n)}
    return satellites


def callbackF(xk, convergence=1):
    print(f"semi-major: {xk[0]:.4}, eccentricity: {xk[1]:.4}, spread: {xk[2]:.4}, convergence: {convergence:.3%}")
    print("")
    return 0


def asteroid_positions(day, df_asteroids):
    '''Propagate all orbital elements to certain day and reset observations'''
    df_asteroids['M'] = df_asteroids.apply(lambda row: theta_step(row['a'], row['anomaly'], day), axis=1)
    df_asteroids['theta'] = df_asteroids.apply(lambda row: theta_solve(row['e'], row['M']), axis=1)
    df_asteroids['x'], df_asteroids['y'], df_asteroids['z'] = zip(*df_asteroids.apply(lambda row: pos_heliocentric(row['a'], 
                                                                                                              row['e'], 
                                                                                                              row['i'], 
                                                                                                              row['theta'], 
                                                                                                              row['long_node'], 
                                                                                                              row['arg_peri']),
                                                                                 axis=1
                                                                                 ))
    df_asteroids['detected'] = df_asteroids.apply(lambda row: day if row['n_obs'] > 4 and row['detected'] == 0 else row['detected'], axis=1)
    df_asteroids['n_obs'] = df_asteroids.apply(lambda row: 0 if day - row['last_obs'] > 90 else row['n_obs'], axis=1)
    return df_asteroids


def satellite_positions(day, satellites):
    '''Propagate satellite positions'''
    for sat in satellites.values():
        sat['M'] = theta_step(sat['a'], sat['anomaly'], day)
        sat['theta'] = theta_solve(sat['e'], sat['M'])
        sat['x'], sat['y'], sat['z'] = pos_heliocentric(sat['a'],
                                                        sat['e'],
                                                        sat['i'],
                                                        sat['theta'],
                                                        sat['long_node'],
                                                        sat['arg_peri'])
    return satellites


def make_observations(sat, day, df_asteroids):
    if day % sat['cadence']:
        return 0
    df_asteroids['SNR'] = df_asteroids.apply(lambda row: observation(row['H'], 
                                                                     row['albedo'], 
                                                                     row['x'], 
                                                                     row['y'], 
                                                                     row['z'], 
                                                                     sat['x'], 
                                                                     sat['y'], 
                                                                     sat['z'], 
                                                                     sat['payload']
                                                                     ), 
                                             axis=1
                                             )
    df_asteroids['step_obs'] = df_asteroids.apply(lambda row: row['step_obs'] + detection_prob(row['SNR']), 
                                                  axis=1
                                                  )
    return day

def do_survey(x):
    semi_major, eccentricity, spread = x[0], x[1], x[2]
    n_sats = 5
    df_asteroids = init_asteroids()
    #print(f"Satellites:   {n_sats}")
    #print(f"Semi-Major:   {semi_major}")
    #print(f"Eccentricity: {eccentricity}")
    #print(f"Spread:       {spread}")
    df_asteroids['n_obs'] = 0
    df_asteroids['step_obs'] = 0
    df_asteroids['last_obs'] = 0
    df_asteroids['detected'] = 0
    df_asteroids = df_asteroids.fillna(0.15)
    satellites = init_satellites(n_sats, semi_major, eccentricity, spread)
    completeness = []
    for day in range(0, 1825, 2):
        n_detected = df_asteroids[df_asteroids['detected'] > 0]['detected'].count()
        n_undetected = df_asteroids[df_asteroids['detected'] == 0]['detected'].count()
        completeness.append(n_detected / (n_detected + n_undetected))
        df_asteroids = asteroid_positions(day, df_asteroids)
        satellites = satellite_positions(day, satellites)
        for sat in satellites.values():
            make_observations(sat, day, df_asteroids)
        df_asteroids['n_obs'] = df_asteroids.apply(lambda row: row['n_obs'] + row['step_obs'], axis=1)
        df_asteroids['last_obs'] = df_asteroids.apply(lambda row: day if row['step_obs'] > 0 else row['last_obs'], axis=1)
        df_asteroids['step_obs'] = 0
    #print(f"Completeness: {completeness[-1]:.2%}")
    #print()
    return float(1 - completeness[-1])
    
    
if __name__ == '__main__':
    bounds = [(0.1, 2.0), (0, 0.9), (0, 2*pi/5)]
    results = dict()
    starttime = time.time()
    results['DE'] = optimize.differential_evolution(do_survey, bounds, workers=6, disp=True, callback=callbackF, polish=False)
    print(f"DE done in:   {time.time() - starttime:.2}")
    results['DE']['time'] = time.time() - starttime
    starttime = time.time()
    f = open('opt_vis_5_jpl_5000_DE.pickle', "wb+")
    pickle.dump(results, f)
    f.close()

