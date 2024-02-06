import csv
import copy
import math
import statistics
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import decimal
decimal.getcontext().prec = 100

random.seed(0)


def functionmaker(x, ptype):
    if ptype == 0:  # Function Arakawa
        n = int(len(x) * 0.5)
        f = 0.0
        for i in range(n):
            f += (10 * math.exp(-0.01 * (x[2 * i] - 10) ** 2 - 0.01 * (x[2 * i + 1] - 15) ** 2) * math.sin(
                2 * x[2 * i]))
    if ptype == 1:  # Function Rastrigin
        n = int(len(x))
        pi = math.atan(1) * 4
        f = 10.0 * len(x)
        for i in range(n):
            f += (x[i] ** 2 - 10 * math.cos(2 * pi * x[i]))
    if ptype == 2:
        n = int(len(x) * 0.5)
        f = 0.0
        for i in range(n):
            f += (100 * (x[2 * i + 1] - x[2 * i] ** 2) ** 2 + (x[2 * i] - 1) ** 2)
    return f


class problem():
    def __init__(self):
        self.gmax = 5000
        self.np = 50
        self.nv = 16
        self.type = 2  # Rosenbrock
        self.Ub = [5.12 for i in range(self.nv)]
        self.Lb = [-5.12 for i in range(self.nv)]


class PSO:
    __slots__ = [
        'x',
        'v',
        'f',
        'fpbest',
        'pbest',
        'w',
        'c1',
        'c2',
        'Ub',
        'Lb'
    ]

    def __init__(self):
        self.x = []
        self.v = []
        self.f = 0
        self.fpbest = 0
        self.w = 0
        self.c1 = 0
        self.c2 = 0
        self.Ub = []
        self.Lb = []


if __name__ == '__main__':
    prb = problem()
    ind = []
    for i in range(prb.np):
        temp = PSO()
        ind.append(temp)
    gbest = PSO()
    # Initialization
    v0 = [0 for i in range(prb.nv)]  ####Initial vector
    ibest = 0
    for i in range(prb.np):
        # Inherit Ub and Lb for each individual
        ind[i].Ub = copy.deepcopy(prb.Ub)
        ind[i].Lb = copy.deepcopy(prb.Lb)
        ind[i].v = copy.deepcopy(v0)
        xtemp = [random.uniform(ind[i].Lb[j], -5) for j in range(prb.nv)]
        ind[i].x = copy.deepcopy(xtemp)
        ind[i].f = functionmaker(ind[i].x, prb.type)
        ind[i].pbest = copy.deepcopy(xtemp)
        ind[i].fpbest = copy.deepcopy(ind[i].f)
        if ind[ibest].f > ind[i].f:
            ibest = i
        # #Setting of w,c1,c2 We are going to change them in later
        # 2: 0.4, 0.2 ,1.9 generation:125
        # 4: 1.2, 0.6 ,1.9 generation:1331
        # 6: 0.9, 0.4 ,2 generation:397
        ind[i].w = 1
        ind[i].c1 = 1.4
        ind[i].c2 = 1.4

    # Initial Gbest
    gbest = copy.deepcopy(ind[ibest])
    for generation in range(prb.gmax):
        # Equation for PSO
        # v=w*v+c1*(pbest-x)+c2*(Gbest-x)
        for i in range(prb.np):
            # Updating of position and velocity
            for j in range(prb.nv):
                ww = random.uniform(0, ind[i].w)
                cc1 = random.uniform(0, ind[i].c1)
                cc2 = random.uniform(0, ind[i].c2)
                vv1 = ind[i].pbest[j] - ind[i].x[j]
                vv2 = gbest.x[j] - ind[i].x[j]
                ind[i].v[j] = ww * ind[i].v[j] + cc1 * vv1 + cc2 * vv2
                ind[i].x[j] = ind[i].x[j] + ind[i].v[j]
            ind[i].f = functionmaker(ind[i].x, prb.type)
            if ind[i].f < ind[i].fpbest:
                ind[i].fpbest = ind[i].f
                ind[i].pbest = copy.deepcopy(ind[i].x)
                if gbest.f > ind[i].f:
                    gbest.x = copy.deepcopy(ind[i].x)
                    gbest.f = ind[i].f
        print(str(generation) + ":" + str(gbest.f) + " " + str(gbest.x))
