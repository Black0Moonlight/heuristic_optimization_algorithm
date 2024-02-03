import csv
import copy
import math
import statistics
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import decimal
from copy import deepcopy
from PathPlanning import *

decimal.getcontext().prec = 100  # Sets the precision of decimal arithmetic to 100 digits
random.seed(0)

plt.ion()
fig, ax = plt.subplots()

class Problem:
    # Defines the optimization problem parameters
    def __init__(self):
        self.gmax = 500  # Maximum number of generations
        self.np = 50  # Number of particles
        self.nv = 20  # Number of points (variant) on the path

        self.start = (0, 0)
        self.end = (5, 5)
        self.Ub = [6 for _ in range(self.nv*2)]  # Upper bound for each variable
        self.Lb = [-1 for _ in range(self.nv*2)]  # Lower bound for each variable

        self.nPts = 10
        self.f_interp = 'quadratic'
        self.Xinit = build_Xinit(self.start, self.end, self.nPts)

        self.iw = 0.5  # Inertia weight
        self.ic1 = 1.5  # Cognitive coefficient
        self.ic2 = 1.5  # Social coefficient
        # [x y]
        self.v0 = [0 for _ in range(self.nv*2)]  # Initial velocity vector for all particles

        self.paths = [None] * self.gmax  # List with the results from all runs


prb = Problem()
# Define start, goal, and limits
# limits = [-2, 10, -6, 6]
limits = [-1, 6, -1, 6]
layout = PathPlanning(prb.start, prb.end, limits)

# Add obstacles (polygon verteces must be given in counter-clockwise order)
layout.add_ellipse(x=2.8, y=1.5, theta=-np.pi/6, a=1, b=0.5, Kv=100)
# V = [(2, 1), (5, 1), (5, 5), (4, 5), (2, 4)]
V = [(1.33, 3.08), (3.08, 3.08), (3.08, 5.42), (2.50, 5.42), (1.33, 4.83)]

layout.add_convex(V, Kv=100)
#V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0), (7.5, 3),  (6.5, 3)]
V = [(3.96, 1.92), (5.71, 1.92), (5.71, 4.83), (5.13, 4.25), (5.13, 2.50), (4.54, 2.50), (4.54, 4.25), (3.96, 4.25)]

layout.add_polygon(V, Kv=200)
layout.add_circle(x=1, y=0.8, r=0.6, Kv=100)


#fig, ax = plt.subplots()
#layout.plot_obs(ax)         # Plot obstacles
#plt.show()


def plot_graph(start, end, iteration, mode):
    # plt.clf()

    # Text position on the plots (lower left corner)
    xt = layout.limits[0] + 0.05 * (layout.limits[1] - layout.limits[0])
    yt = layout.limits[2] + 0.05 * (layout.limits[3] - layout.limits[2])

    layout.plot_obs(ax)

    plt.scatter(start[0], start[1], color='red', label='Start')  # 显示起点
    plt.scatter(end[0], end[1], color='blue', label='End')  # 显示终点

    plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
    if mode == 0:
        # start
        plt.ioff()
    elif mode == 1:
        # L = layout.sol[1]  # Length
        # count = layout.sol[2]  # Number of violated obstacles
        # layout.plot_path(ax)
        plt.ioff()
    else:
        layout.plot_path(ax)
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
        args = [prb.start, prb.end, layout.obs, prb.nv, prb.f_interp]
    f = path_length(x, args)
    return f


if __name__ == '__main__':
    plot_graph(prb.start, prb.end, 0, 0)
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

        xtemp = np.array([random.uniform(ind[i].Lb[j], 0) for j in range(prb.nv*2-4)])

        # xtemp = [np.array(prb.start)]
        # for _ in range(1, prb.nv - 1):
        #     direction = np.random.rand(2) - 0.5
        #     direction /= np.linalg.norm(direction)
        #     next_point = xtemp[-1] + direction
        #     xtemp.append(next_point)
        # xtemp.append(np.array(prb.end))

        ind[i].x = copy.deepcopy(xtemp)
        ind[i].f = functionmaker(ind[i].x)
        ind[i].pbest = copy.deepcopy(xtemp)
        ind[i].fpbest = copy.deepcopy(ind[i].f)
        if ind[ibest].f > ind[i].f:
            ibest = i

        ind[i].w = 1
        ind[i].c1 = 1.4
        ind[i].c2 = 1.4

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
        layout.sol = (gbest.x, L[0], count, Px, Py)
        print(str(generation) + ":" + str(gbest.f) + " " + str(gbest.x))

        # Save run
        prb.paths[generation] = deepcopy(layout)

        if generation % 20 == 0:
            # Print results
            # L = layout.sol[1]  # Length
            # count = layout.sol[2]  # Number of violated obstacles
            # print("\nrun={0:d}, L={1:.2f}, count={2:d}"
            #       .format(generation + 1, L, count), end='', flush=True)
            plot_graph(prb.start, prb.end, generation, 1)

    # Result
    plot_graph(prb.start, prb.end, prb.gmax, 2)

