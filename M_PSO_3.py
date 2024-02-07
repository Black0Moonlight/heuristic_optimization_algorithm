# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        mpso-3
 @Created:          2024/2/4 15:42
 @Software:         PyCharm

 @Author:           Jiayu ZENG
 @Email:            jiayuzeng123@gmail.com

 @Description:      optimize from mpos-2

 ------------------------------------------------------------------
'''

import copy
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import decimal
from copy import deepcopy
from M_PathPlanning_3 import *

decimal.getcontext().prec = 100  # Sets the precision of decimal arithmetic to 100 digits
# random.seed(0)

plt.ion()
plt.plot()

class Problem:
    # Defines the optimization problem parameters
    def __init__(self):
        self.gmax = 300  # Maximum number of generations
        self.np = 50  # Number of particles
        self.nv = 10  # Number of points (variant) on the path

        self.start = (0, 0)
        self.end = (5, 5)
        self.Ub = [6 for _ in range(self.nv*2)]  # Upper bound for each variable
        self.Lb = [-1 for _ in range(self.nv*2)]  # Lower bound for each variable

        self.nPds = 1 + (self.nv + 1) * 50
        self.f_interp = 'quadratic'

        self.iw = 1.45  # Inertia weight
        self.ic1 = 0.25  # Cognitive coefficient
        self.ic2 = 1.5  # Social coefficient
        # [x y]
        self.v0 = [0 for _ in range(self.nv*2)]  # Initial velocity vector for all particles

        self.paths = [None] * self.gmax  # List with the results from all runs
        self.fs = [0 for _ in range(self.gmax)]


prb = Problem()
# Define start, goal, and limits
limits = [-1, 6, -1, 6]
layout = PathPlanning(prb.start, prb.end, limits)

# Add obstacles (polygon verteces must be given in counter-clockwise order)
layout.add_ellipse(x=2.8, y=1.5, theta=-np.pi/6, a=1, b=0.5, Kv=100)
V = [(1.33, 3.08), (3.08, 3.08), (3.08, 5.42), (2.50, 5.42), (1.33, 4.83)]

layout.add_convex(V, Kv=100)
V = [(3.96, 1.92), (5.71, 1.92), (5.71, 4.83), (5.13, 4.25), (5.13, 2.50), (4.54, 2.50), (4.54, 4.25), (3.96, 4.25)]

layout.add_polygon(V, Kv=200)
layout.add_circle(x=1, y=0.8, r=0.6, Kv=100)


def plot_graph(prb, iteration, f, mode):
    folder_path = './fig/'
    file_name = f'chart_w{prb.iw}_c{prb.ic1}_c{prb.ic2}_{iteration+1}.png'
    full_path = folder_path + file_name

    plt.clf()
    ax = plt.gca()
    # Text position on the plots (lower left corner)
    xt = layout.limits[0] + 0.05 * (layout.limits[1] - layout.limits[0])
    yt = layout.limits[2] + 0.05 * (layout.limits[3] - layout.limits[2])

    plt.scatter(prb.start[0], prb.start[1], color='red', label='Start')  # 显示起点
    plt.scatter(prb.end[0], prb.end[1], color='blue', label='End')  # 显示终点

    title = "start=" + str("{}".format(prb.start)) + ", end=" + str("{}".format(prb.end))
    plt.title('Path Planning '+title)

    title = "run=" + str(iteration) + ", F=" + str("{:.2f}".format(f))\
            + ", np=" + str("{}".format(prb.np)) + ", nv=" + str("{}".format(prb.nv))
    ax.text(xt, yt+0.1, title, fontsize=10)
    title = "w=" + str("{:.2f}".format(prb.iw)) + ", c1=" + str("{:.2f}".format(prb.ic1))\
            + ", c2=" + str("{:.2f}".format(prb.ic2))
    ax.text(xt, yt-0.2, title, fontsize=10)

    if mode == 0:
        # start
        layout.plot_obs(ax)
        plt.pause(1)
        plt.savefig(full_path)
        plt.ioff()
    elif mode == 1:
        layout.plot_obs(ax)
        layout.plot_path(ax)
        plt.pause(0.001)
        if (iteration+1) % 20 == 0:
            plt.savefig(full_path)
        plt.ioff()
    else:
        layout.plot_obs(ax)
        layout.plot_path(ax)
        plt.savefig(full_path)
        plt.pause(1)
        plt.ioff()


def plot_line(prb, y):
    plt.close('all')
    folder_path = './fig/'
    file_name = f'line_w{prb.iw}_c{prb.ic1}_c{prb.ic2}.png'
    full_path = folder_path + file_name

    plt.plot()
    plt.plot(list(range(prb.gmax)), y)
    # 定义坐标轴标题
    plt.xlabel('generations')
    plt.ylabel('Fitness')

    title = "np=" + str("{}".format(prb.np)) + ", nv=" + str("{}".format(prb.nv))
    plt.title('Fitness Graph '+title)
    plt.savefig(full_path)
    plt.pause(1)
    plt.ioff()


def plot_subplots(prb):
    plt.close('all')
    folder_path = './fig/'
    file_name = f'sub_w{prb.iw}_c{prb.ic1}_c{prb.ic2}.png'
    full_path = folder_path + file_name

    fig, axs = plt.subplots(3, 5, figsize=(10, 6))
    axs = axs.flatten()
    axi=0
    for run in range(prb.gmax):
        if (run+1)%20 == 0:

            layout = prb.paths[run]         # Layout to plot
            ax = axs[axi]               # Subplot

            L = layout.sol[1]           # Length
            count = layout.sol[2]       # Number of violated obstacles

            # Text position on the plots (lower left corner)
            xt = layout.limits[0] + 0.05 * (layout.limits[1] - layout.limits[0])
            yt = layout.limits[2] + 0.05 * (layout.limits[3] - layout.limits[2])

            layout.plot_obs(ax)         # Plot obstacles
            layout.plot_path(ax)        # Plot path

            # Plot run, length, and count
            title = "run=" + str(run+1) + ", L=" + str("{:.2f}".format(L))
            ax.text(xt, yt, title, fontsize=10)
            axi += 1

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(full_path)
    plt.show()


class PSO:
    def __init__(self):
        self.x = []  # Position of the particle
        self.v = []  # Velocity of the particle
        self.f = 0  # Fitness value of the particle

        self.pbest = []  # Best position found by this particle
        self.fpbest = 0  # Best fitness value found by this particle

        self.w = 0
        self.c1 = 0
        self.c2 = 0

        self.Ub = []
        self.Lb = []


def functionmaker(x, args=0):
    if args == 0:
        args = [prb.start, prb.end, layout.obs, prb.nPds, prb.f_interp]
    f = path_length(x, args)
    return f


if __name__ == '__main__':
    plot_graph(prb, 0, 0, 0)
    # Initialization
    ind = []  # List to store all particles
    for i in range(prb.np):
        temp = PSO()
        ind.append(temp)
    ibest = 0  # Index of the best particle
    gbest = PSO()  # Creates a global best particle

    for i in range(prb.np):
        # Inherit Ub and Lb for each individual
        ind[i].Ub = copy.deepcopy(prb.Ub)
        ind[i].Lb = copy.deepcopy(prb.Lb)
        ind[i].v = copy.deepcopy(prb.v0)

        xtemp = np.array([random.uniform(ind[i].Lb[j], ind[i].Ub[j]) for j in range(prb.nv*2-4)])

        ind[i].x = copy.deepcopy(xtemp)
        ind[i].f = functionmaker(ind[i].x)
        ind[i].pbest = copy.deepcopy(xtemp)
        ind[i].fpbest = copy.deepcopy(ind[i].f)
        if ind[ibest].f > ind[i].f:
            ibest = i

        ind[i].w = prb.iw
        ind[i].c1 = prb.ic1
        ind[i].c2 = prb.ic2

    gbest = copy.deepcopy(ind[ibest])

    # Optimize
    for generation in range(prb.gmax):
        # Main loop of the PSO algorithm
        for i in range(prb.np):
            # Updates each particle's velocity and position
            for j in range(prb.nv*2-4):
                ww = random.uniform(0, ind[i].w)  # Random weight for inertia
                cc1 = random.uniform(0, ind[i].c1)  # Random weight for cognitive component
                cc2 = random.uniform(0, ind[i].c2)  # Random weight for social component
                vv1 = ind[i].pbest[j] - ind[i].x[j]  # Cognitive velocity component
                vv2 = gbest.x[j] - ind[i].x[j]  # Social velocity component
                ind[i].v[j] = ww * ind[i].v[j] + cc1 * vv1 + cc2 * vv2  # Updates velocity
                ind[i].x[j] = ind[i].x[j] + ind[i].v[j]  # Updates position

            ind[i].f = functionmaker(ind[i].x)
            if ind[i].f < ind[i].fpbest:
                # Updates particle's best known position and fitness
                ind[i].fpbest = ind[i].f
                ind[i].pbest = copy.deepcopy(ind[i].x)
                if gbest.f > ind[i].f:
                    ibest = i
                    gbest.x = copy.deepcopy(ind[i].x)
                    gbest.f = ind[i].f
        # Get the results for the best path (<args> has six items)
        args = [prb.start, prb.end, layout.obs, prb.nv, prb.f_interp, []]
        F = functionmaker(gbest.x, args)
        L, count, Px, Py = args[5]
        layout.sol = (gbest.x, L, count, Px, Py)
        print(str(generation) + ":" + str(gbest.f) + " " + str(gbest.x))

        # Save run
        prb.paths[generation] = deepcopy(layout)
        prb.fs[generation] = deepcopy(gbest.f)
        plot_graph(prb, generation, gbest.f, 1)

    # Result
    plot_graph(prb, prb.gmax, gbest.f, 2)
    plot_line(prb, prb.fs)
    plot_subplots(prb)

