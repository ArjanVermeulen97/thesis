# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:59:53 2021

@author: Arjan
"""

import pandas as pd
import swifter
from math import sqrt, exp, tan, log10, pi, acos, asin, sin, cos, tanh
from observations import observation, detection, detection_prob
from orbital import pos_heliocentric, theta_solve, theta_step
import seaborn as sns
from population_models import init_asteroids_neopop, init_asteroids_granvik_impactors
import matplotlib.pyplot as plt

def init_asteroids():
    return init_asteroids_granvik_impactors(100, False)


def init_satellites(n, a, e, s, anomaly):
    satellites = {i: {'a': a,
                      'e': e,
                      'i': 0,
                      'long_node': 0,
                      'arg_peri': 0,
                      'anomaly': s*i - s*(0.5*n-0.5) + anomaly,
                      'payload': 'TIR',
                      'cadence': 2} for i in range(n)}
    return satellites


def asteroid_positions(day, df_asteroids):
    '''Propagate all orbital elements to certain day and reset observations'''
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
    df_asteroids['detected'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: 1 if row['n_obs'] > 1 and row['detected'] == 0 else row['detected'], axis=1)
    df_asteroids['detect_day'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: day if row['detected'] == 1 and row['detect_day'] == 0 else row['detect_day'], axis=1)
    df_asteroids['n_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: 0 if day - row['last_obs'] > 90 else row['n_obs'], axis=1)
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


def make_observations(sat, day):
    global df_asteroids
    if day % sat['cadence']:
        return 0
    df_asteroids['SNR'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: observation(row['H'], 
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
    df_asteroids['step_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: min(row['step_obs'] + detection_prob(row['SNR']), 1),
                                                  axis=1
                                                  )
    df_asteroids['step_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: row['step_obs'] if row['t_impact']-365 <= day <= row['t_impact'] else 0,
                                                                              axis = 1
                                                                              )
    df_asteroids['impacted'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: 1 if day > row['t_impact'] and row['detected'] < 0.5 else 0,
                                                                              axis=1
                                                                              )
    print(max(df_asteroids['SNR']))
    return day


def asteroid_status(detect, impact):
    if detect > 0:
        return "detected"
    if impact > 0:
        return "impacted"
    return "unknown"

def do_survey(x):
    semi_major, eccentricity, spread, anomaly = x[0], x[1], x[2], x[3]
    anomaly_earth = (358.62+90)/180*pi
    anomaly = anomaly + anomaly_earth # Give anomaly relative to Earth
    n_sats = 1
    df_asteroids = init_asteroids()
    df_asteroids['diameter'] = df_asteroids.apply(lambda row: 1329000/sqrt(row['albedo'])*10**(-1*row['H'] / 5), axis=1)
    df_asteroids['H'] = df_asteroids['H'] - 10
    satellites = init_satellites(n_sats, semi_major, eccentricity, spread, anomaly)
    
    fig, axes = plt.subplots(1,1, figsize=(8,8))
    bg = plt.imread('background.png')
    
    for day in range(-365, 200, 2):
        detected = df_asteroids.loc[df_asteroids['detected'] == 1]['detected'].count() / df_asteroids['detected'].count()
        impacted = df_asteroids.loc[df_asteroids['impacted'] == 1]['impacted'].count() / df_asteroids['impacted'].count()
        print(f"Day: {day}, detected: {detected:.2%}, impacted: {impacted:.2%}")
        df_asteroids = asteroid_positions(day, df_asteroids)
        satellites = satellite_positions(day, satellites)
        for sat in satellites.values():
            make_observations(sat, day)#, df_asteroids)
        df_asteroids['n_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: row['n_obs'] + row['step_obs'], axis=1)
        df_asteroids['last_obs'] = df_asteroids.swifter.progress_bar(False).apply(lambda row: day if row['step_obs'] > 0 else row['last_obs'], axis=1)
        plt.pause(0.0001)
        if day == 0:
            #pass
            a = input("enter to start...")
        axes.clear()
        axes.set_xlim((-5, 5))
        axes.set_ylim((-5, 5))
        axes.imshow(bg, extent=[-5, 5, -5, 5])
        df_asteroids['status'] = df_asteroids.apply(lambda row: asteroid_status(row['detected'], row['impacted']), axis=1)
        sns.scatterplot(data=df_asteroids, x='x', y='y', hue='impacted', ax=axes, marker='o', legend=False, size='diameter')
        sns.scatterplot(data=pd.DataFrame(satellites).transpose(), x='x', y='y', marker='D', color="red", s=30)
        fig.canvas.draw()
        df_asteroids['step_obs'] = 0
    return df_asteroids
    
    
if __name__ == '__main__':
    df_asteroids = do_survey([1.0, 0.0, 2*pi/5, 0])
    