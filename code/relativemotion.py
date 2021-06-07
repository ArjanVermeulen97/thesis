# -*- coding: utf-8 -*-
"""
Author: Arjan Vermeulen
"""

from math import sin, cos, tan, asin, acos, atan, sqrt, pi, atan2, log
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter
import datetime
from observation import gen_asteroid, print_asteroid, calc_MOID

# matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Arjan\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'

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

def pos_heliocentric(a, e, i, theta, raan, argPeri):
    # i = i/180*pi
    # raan = raan/180*pi
    # argPeri = argPeri/180*pi
    # theta = theta/180*pi
    
    r = a*(1+e**2)/(1+e*cos(theta))
    r_orbit = np.array([[r*cos(theta)],
                        [r*sin(theta)],
                        [0]])
    
    r_helio = R_3(-1*raan) @ R_1(-1*i) @ R_3(-1*argPeri) @ r_orbit
    
    return r_helio

def theta_solve(e, M, error=1.0E-14):
    def e_start(e, M):
        t34 = e**2
        t35 = e*t34
        t33 = cos(M)
        return M + (-0.5*t35 + e + (t34 + 3/2*t33*t35)*t33)*sin(M)

    def eps(e, M, x):
        t1 = cos(x)
        t2 = -1 + e*t1
        t3 = sin(x)
        t4 = e*t3
        t5 = -x + t4 + M
        t6 = t5/(0.5*t5*t4/t2 + t2)
        return t5/((0.5*t3 - 1/6.*t1*t6)*e*t6+t2)
    
    Mnorm = M%(2*pi)
    part = 2*pi if Mnorm > pi else 0
    sign = -1 if Mnorm > pi else 1
    E0 = e_start(e, Mnorm)
    dE = error + 1
    n_iter = 0
    while dE > error:
        E = E0 - eps(e, Mnorm, E0)
        dE = abs(E-E0)
        E0 = E
        n_iter += 1
        if n_iter == 1000:
            raise ValueError("Doesn't converge :(")
            break;
    return part + acos((cos(E) - e)/(1-e*cos(E)))*sign

def find_mag(x_s, y_s, z_s, x_t, y_t, z_t, H_t):
    x_r = x_t - x_s
    y_r = y_t - y_s
    z_r = z_t - z_s
    abs_r = sqrt(x_r**2 + y_r**2 + z_r**2)
    abs_t = sqrt(x_t**2 + y_t**2 + z_t**2)
    cos_a = (x_r*x_t + y_r*y_t + z_r*z_t) / (abs_r * abs_t)
    mag = H_t + 5*log(abs_r * abs_t, 10) - 2.5 * log(sqrt(0.5*(1+cos_a)), 10)
    return mag

def observe(x_s, y_s, z_s, x_t, y_t, z_t, H_t):
    x_r = x_t - x_s
    y_r = y_t - y_s
    z_r = z_t - z_s
    abs_r = sqrt(x_r**2 + y_r**2 + z_r**2)
    abs_t = sqrt(x_t**2 + y_t**2 + z_t**2)
    cos_a = (x_r*x_t + y_r*y_t + z_r*z_t) / (abs_r * abs_t)
    mag = H_t + 5*log(abs_r * abs_t, 10) - 2.5 * log(sqrt(0.5*(1+cos_a)), 10)
    r_a = atan2(y_r, x_r)
    dec = atan(z_r/sqrt(x_r**2 + y_r**2))
    return np.array([r_a, dec, mag])
    

mu_sun = 1.327124E11 / (150_000_000**3) * (86_400**2)   # AU^3 / day^2

# Parameters of the spacecraft
a_s = 1             # Semi-major axis
e_s = 0             # Eccentricity
i_s = 0             # Inclination
raan_s = 0          # Right ascencion of the ascending node
argPeri_s = 0       # Argument of periapsis
T_s = 0             # True anomaly at epoch
n_obs = 7           # number of observations
t_obs = 1           # Days between observations
n_asteroids = 10    # number of asteroids to generate

results = np.array([])

for n in range(n_asteroids):
    random.seed(n)
    asteroid = gen_asteroid()
    print()
    print("------------------------------")
    print(f"Asteroid {n}")
    print("------------------------------")
    print_asteroid(asteroid)
    a_t = asteroid[0]
    e_t = asteroid[1]
    i_t = asteroid[2]/180*pi
    raan_t = asteroid[3]/180*pi
    argPeri_t = asteroid[4]/180*pi
    T_t = asteroid[5]/180*pi
    H_t = random.uniform(15, 25)
    
    # Find full orbit
    positions = [pos_heliocentric(a_t, 
                                  e_t, 
                                  i_t,
                                  theta,
                                  raan_t,
                                  argPeri_t) for theta in range(0, 360)]
    xs = [item[0][0] for item in positions]
    ys = [item[1][0] for item in positions]
    zs = [item[2][0] for item in positions]
    MOID = calc_MOID(xs, ys, zs)

    for t in range(n_obs):
        M_s = sqrt(mu_sun / a_s**3)*(t + T_s)
        M_t = sqrt(mu_sun / a_t**3)*(t + T_t)
        theta_s = theta_solve(e_s, M_s)
        pos_s = pos_heliocentric(a_s, e_s, i_s, theta_s, raan_s, argPeri_s)
        theta_t = theta_solve(e_t, M_t)
        pos_t = pos_heliocentric(a_t, e_t, i_t, theta_t, raan_t, argPeri_t)

        x_s = pos_s[0][0]
        y_s = pos_s[1][0]
        z_s = pos_s[2][0]
    
        x_t = pos_t[0][0]
        y_t = pos_t[1][0]
        z_t = pos_t[2][0]
        
        obs = observe(x_s, y_s, z_s, x_t, y_t, z_t, H_t)
        res = np.array([n, t, MOID, obs], dtype=object)
        results = np.append(results, res)

### Animation stuff ###
# # Parameters of the target
# a_t = 1.2             # Semi-major axis
# e_t = 0.439             # Eccentricity
# i_t = 9.06/180*pi             # Inclination
# raan_t = 112.95/180*pi          # Right ascencion of the ascending node
# argPeri_t = 42.25/180*pi       # Argument of periapsis
# T_t = 154.74/80*pi             # True anomaly at epoch
# H_t = 20            # Absolute magnitude

# x_s_arr = []
# y_s_arr = []
# z_s_arr = []

# x_t_arr = []
# y_t_arr = []
# z_t_arr = []

# r_a_arr = []
# dec_arr = []
# mag_arr = []

# for i in range(0, 6000):
#     t_s = i + T_s
#     t_t = i + T_t
#     M_s = sqrt(mu_sun / a_s**3)*t_s
#     M_t = sqrt(mu_sun / a_t**3)*t_t
    
#     theta_s = theta_solve(e_s, M_s)
#     pos_s = pos_heliocentric(a_s, e_s, i_s, theta_s, raan_s, argPeri_s)
#     theta_t = theta_solve(e_t, M_t)
#     pos_t = pos_heliocentric(a_t, e_t, i_t, theta_t, raan_t, argPeri_t)
    
#     x_s = pos_s[0][0]
#     y_s = pos_s[1][0]
#     z_s = pos_s[2][0]
    
#     x_t = pos_t[0][0]
#     y_t = pos_t[1][0]
#     z_t = pos_t[2][0]
    
#     x_rel = x_t - x_s
#     y_rel = y_t - y_s
#     z_rel = z_t - z_s
    
#     r_a = atan2(y_rel, x_rel)
#     dec = atan(z_rel/sqrt(x_rel**2 + y_rel**2))
#     mag = find_mag(x_s, y_s, z_s, x_t, y_t, z_t, H_t)

#     x_s_arr.append(x_s)
#     y_s_arr.append(y_s)
#     z_s_arr.append(z_s)
    
#     x_t_arr.append(x_t)
#     y_t_arr.append(y_t)
#     z_t_arr.append(z_t)
    
#     r_a_arr.append(r_a)
#     dec_arr.append(dec)
#     mag_arr.append(mag)
    
    
# gs = gridspec.GridSpec(3,2)
# fig = plt.figure(figsize=(16,9))
# ax_orb = plt.subplot(gs[:, 0], projection='3d')
# ax_orb.set_xlim3d(-1.2, 1.2)
# ax_orb.set_ylim3d(-1.2, 1.2)
# ax_orb.set_zlim3d(-1, 1)
# orb_line_s = ax_orb.plot(0,0,0)
# orb_dots_s = ax_orb.plot(0,0,0)
# orb_line_t = ax_orb.plot(0,0,0)
# orb_dots_t = ax_orb.plot(0,0,0)
# orb_rel = ax_orb.plot(0,0,0)

# ax_r_a = plt.subplot(gs[0, 1])
# r_a_plt = ax_r_a.plot(0)
# ax_r_a.set_ylabel('Right Ascension')

# ax_dec = plt.subplot(gs[1, 1])
# dec_plt = ax_dec.plot(0)
# ax_dec.set_ylabel('Declination')

# ax_mag = plt.subplot(gs[2, 1]) 
# mag_plt = ax_mag.plot(0)
# ax_mag.set_ylabel('Apparent Magnitude')

# def animate(i):
#     global orb_line_s, orb_dots_s, orb_line_t, orb_dots_t,\
#         orb_rel, r_a_plt, dec_plt, mag_plt
#     i = i + 365
#     orb_line_s.pop(0).remove()
#     orb_dots_s.pop(0).remove()
#     orb_line_t.pop(0).remove()
#     orb_dots_t.pop(0).remove()
#     orb_rel.pop(0).remove()
#     r_a_plt.pop(0).remove()
#     dec_plt.pop(0).remove()
#     mag_plt.pop(0).remove()

#     orb_line_s = ax_orb.plot(x_s_arr[0:500], y_s_arr[0:500], z_s_arr[0:500], c='green')
#     orb_dots_s = ax_orb.plot(x_s_arr[i], y_s_arr[i], z_s_arr[i], 'go')
#     orb_line_t = ax_orb.plot(x_t_arr[0:500], y_t_arr[0:500], z_t_arr[0:500], c='blue')
#     orb_dots_t = ax_orb.plot(x_t_arr[i], y_t_arr[i], z_t_arr[i], 'bo')
#     orb_rel = ax_orb.plot([x_s_arr[i], x_t_arr[i]],
#                           [y_s_arr[i], y_t_arr[i]],
#                           [z_s_arr[i], z_t_arr[i]],
#                           'k--')
#     ax_orb.view_init(15, (i/10)%360)
#     r_a_plt = ax_r_a.plot(r_a_arr[i-365:i], c='blue')
#     dec_plt = ax_dec.plot(dec_arr[i-365:i], c='blue')
#     mag_plt = ax_mag.plot(mag_arr[i-365:i], c='blue')
    
#     ax_orb.plot(0, 0, 'yo') # Add sun
    
# ani = FuncAnimation(fig, animate, interval=10, frames=3650)
# # print("start")
# # start = datetime.datetime.now()
# # f = r"c://Users/Arjan/Desktop/thesis/thesis/relativemotionblue.mp4"
# # writergif = FFMpegWriter(fps=30)
# # ani.save(f, writer=writergif)
# # print(datetime.datetime.now() - start)