# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        mpso
 @Created:          2024/2/4 16:22
 @Software:         PyCharm

 @Author:           Jiayu ZENG
 @Email:            jiayuzeng123@gmail.com

 @Description:      used as the basic compare

 ------------------------------------------------------------------
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import decimal
from copy import deepcopy
from E_PathPlanning import *

decimal.getcontext().prec = 100  # Sets the precision of decimal arithmetic to 100 digits
random.seed(0)

# Define start, goal, and limits
start = (0, 0)
goal = (5, 5)
limits = [-2, 10, -6, 6]
layout = PathPlanning(start, goal, limits)

# Set new start and goal position
layout.set_start(0, -3.5)
layout.set_goal(8, 2)

# Add obstacles (polygon verteces must be given in counter-clockwise order)
layout.add_ellipse(x=5, y=-1, theta=-np.pi/6, a=1.0, b=0.5, Kv=100)
V = [(2, 1), (5, 1), (5, 5), (4, 5), (2, 4)]
layout.add_convex(V, Kv=100)
V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0),
     (7.5, 3),  (6.5, 3)]
layout.add_polygon(V, Kv=200)
layout.add_circle(x=2, y=-2, r=1.2, Kv=100)

nRun = 15
nPts = 10
d = 50                  # ns = 551 (points along the spline)
nPop = 100
epochs = 500
f_interp = 'quadratic'
Xinit = build_Xinit(layout.start, layout.goal, nPts)

# Init other parameters
np.seterr(all='ignore')
ns = 1 + (nPts + 1) * d         # Number of points along the spline
best_L = np.inf                 # Best length (minimum)
best_run = 0                    # Run corresponding to the best length
best_count = 0                  # count corresponding to the best length
paths = [None] * nRun           # List with the results from all runs


def optimize(path, nPts=3, ns=100, nPop=40, epochs=500, K=0, phi=2.05,
             vel_fact=0.5, conf_type='RB', IntVar=None, normalize=False,
             rad=0.1, f_interp='cubic', Xinit=None):
    """
    Optimizes the path.
    """
    # Arguments passed to the function to minimize (<args> has five items)
    Xs = np.ones((nPop, 1)) * path.start[0]  # Start x-position (as array)
    Ys = np.ones((nPop, 1)) * path.start[1]  # Start y-position (as array)
    Xg = np.ones((nPop, 1)) * path.goal[0]  # Goal x-position (as array)
    Yg = np.ones((nPop, 1)) * path.goal[1]  # Goal y-position (as array)
    args = [(Xs, Ys), (Xg, Yg), path.obs, ns, f_interp]

    # Boundaries of the search space
    nVar = 2 * nPts
    UB = np.zeros(nVar)
    LB = np.zeros(nVar)
    LB[:nPts] = path.limits[0]
    UB[:nPts] = path.limits[1]
    LB[nPts:] = path.limits[2]
    UB[nPts:] = path.limits[3]

    # Optimize
    X, info = PSO(path_length, LB, UB, nPop, epochs, K, phi, vel_fact,
                  conf_type, IntVar, normalize, rad, args, Xinit)

    # Get the results for the best path (<args> has six items)
    args = [path.start, path.goal, path.obs, ns, f_interp, []]
    F = path_length(X.reshape(1, nVar), args)
    L, count, Px, Py = args[5]
    path.sol = (X, L[0], count, Px, Py)


# Run cases
print("\nns = ", ns)
for run in range(nRun):
    # Optimize (the other PSO parameters have always their default values)
    optimize(path=layout, nPts=nPts, ns=ns, nPop=nPop, epochs=epochs, f_interp=f_interp, Xinit=Xinit)

    # Save run
    paths[run] = deepcopy(layout)

    # Print results
    L = layout.sol[1]               # Length
    count = layout.sol[2]           # Number of violated obstacles
    print("\nrun={0:d}, L={1:.2f}, count={2:d}"
          .format(run+1, L, count), end='', flush=True)

    # Save if best result (regardless the violations)
    if (L < best_L):
        best_L = L
        best_run = run
        best_count = count

# Result
print("\n\nBest:", end='')
print(" run={0:d}, L={1:.2f}, count={2:d}"
      .format(best_run+1, best_L, best_count))

fig, axs = plt.subplots(3, 5)
axs = axs.flatten()
for run in range(nRun):

    layout = paths[run]         # Layout to plot
    ax = axs[run]               # Subplot

    L = layout.sol[1]           # Length
    count = layout.sol[2]       # Number of violated obstacles

    # Text position on the plots (lower left corner)
    xt = layout.limits[0] + 0.05 * (layout.limits[1] - layout.limits[0])
    yt = layout.limits[2] + 0.05 * (layout.limits[3] - layout.limits[2])

    layout.plot_obs(ax)         # Plot obstacles
    layout.plot_path(ax)        # Plot path

    # Plot run, length, and count
    title = "run=" + str(run+1) + ", L=" + str("{:.2f}".format(L)) + \
            ", count=" + str(count)
    ax.text(xt, yt, title, fontsize=10)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
plt.show()
