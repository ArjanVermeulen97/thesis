# -*- coding: utf-8 -*-
"""
Author: Arjan Vermeulen
"""

from math import sin, cos, tan, asin, acos, atan, sqrt, pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



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

mu_sun = 1.327124E11 / (150_000_000**3) * (86_400**2)   # AU^3 / day^2

a = 1
e = 0.4
t_arr = []
theta_arr = []
r_arr = []

# for t in range(0, 365):
#     M = sqrt(mu_sun / a**3)*t
#     theta = theta_solve(e, M)
#     r = a*(1-e**2)/(1+e*cos(theta))
    
#     t_arr.append(t)
#     theta_arr.append(theta)
#     r_arr.append(r)

#index = count()
fig = plt.figure(figsize = (15, 15))
ax = plt.subplot(111)

xs = []
ys = []

for i in range(0, 366):
    t = i
    M = sqrt(mu_sun / a**3)*t
    theta = theta_solve(e, M)
    theta_arr.append(theta)
    r = a*(1-e**2)/(1+e*cos(theta))
    x = r*cos(theta)
    y = r*sin(theta)
    xs.append(x)
    ys.append(y)
    
xs.append(xs[0])
ys.append(ys[0])

lines, dots = plt.plot(0), plt.plot(0)

def animate_run(i):
    global lines, dots
    i = i%365
    x = xs
    y = ys
    lines.pop(0).remove()
    dots.pop(0).remove()
    lines = plt.plot(x, y, c='red')
    dots = plt.plot(xs[i], ys[i], 'ro')
    plt.plot(0, 0, 'yo')
    
def animate_orbit(i):
    t = i
    M = sqrt(mu_sun / a**3)*t
    theta = theta_solve(e, M)
    r = a*(1-e**2)/(1+e*cos(theta))
    x = r*cos(theta)
    y = r*sin(theta)
    plt.scatter(x, y, c='red')
    
def animate_init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return fig
    
ani = FuncAnimation(fig, animate_run, init_func=animate_init, interval=30, frames=365)
plt.show()
