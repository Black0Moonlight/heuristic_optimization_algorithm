import csv
import copy
import math
import statistics
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import decimal
decimal.getcontext().prec = 100  # Sets the precision of decimal arithmetic to 100 digits

random.seed(0)  # Sets the seed for random number generator to ensure reproducibility


def functionmaker(x, ptype):
    # Defines three different mathematical functions based on the type specified by `ptype`
    f=0
    if ptype == 0:  # Arakawa function
        n = int(len(x) * 0.5)
        f = 0.0
        for i in range(n):
            f += (10 * math.exp(-0.01 * (x[2 * i] - 10) ** 2 - 0.01 * (x[2 * i + 1] - 15) ** 2) * math.sin(
                2 * x[2 * i]))
    if ptype == 1:  # Rastrigin function
        n = int(len(x))
        pi = math.atan(1) * 4
        f = 10.0 * len(x)
        for i in range(n):
            f += (x[i] ** 2 - 10 * math.cos(2 * pi * x[i]))
    if ptype == 2:  # Rosenbrock function
        n = int(len(x) * 0.5)
        f = 0.0
        for i in range(n):
            f += (100 * (x[2 * i + 1] - x[2 * i] ** 2) ** 2 + (x[2 * i] - 1) ** 2)
    if ptype == 3:  # Rosenbrock function
        f = 0

    return f  # Returns the calculated function value


class problem():
    # Defines the optimization problem parameters
    def __init__(self):
        self.gmax = 5000  # Maximum number of generations
        self.np = 50  # Number of particles
        self.nv = 4  # Number of variables
        self.type = 2  # Type of function to optimize (0: Arakawa, 1: Rastrigin, 2: Rosenbrock)
        self.Ub = [5.12 for i in range(self.nv)]  # Upper bound for each variable
        self.Lb = [-5.12 for i in range(self.nv)]  # Lower bound for each variable


class PSO:
    # Defines the structure of a particle in the swarm
    __slots__ = [
        'x',  # Position of the particle
        'v',  # Velocity of the particle
        'f',  # Fitness value of the particle
        'fpbest',  # Best fitness value found by this particle
        'pbest',  # Best position found by this particle
        'w',  # Inertia weight
        'c1',  # Cognitive coefficient
        'c2',  # Social coefficient
        'Ub',  # Upper bounds for the particle's position
        'Lb'   # Lower bounds for the particle's position
    ]

    def __init__(self):
        # Initializes a particle with default values
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
    prb = problem()  # Instantiates the optimization problem
    ind = []  # List to store all particles
    for i in range(prb.np):
        temp = PSO()  # Creates a new particle
        ind.append(temp)  # Adds the particle to the list
    gbest = PSO()  # Creates a global best particle
    # Initialization
    v0 = [0 for i in range(prb.nv)]  # Initial velocity vector for all particles
    ibest = 0  # Index of the best particle
    for i in range(prb.np):
        # Initializes each particle's position, velocity, and fitness
        ind[i].Ub = copy.deepcopy(prb.Ub)  # Copies upper bound from problem
        ind[i].Lb = copy.deepcopy(prb.Lb)  # Copies lower bound from problem
        ind[i].v = copy.deepcopy(v0)  # Sets initial velocity
        xtemp = [random.uniform(ind[i].Lb[j], -5) for j in range(prb.nv)]  # Generates random initial position
        ind[i].x = copy.deepcopy(xtemp)  # Sets initial position
        ind[i].f = functionmaker(ind[i].x, prb.type)  # Evaluates fitness
        ind[i].pbest = copy.deepcopy(xtemp)  # Sets initial best position
        ind[i].fpbest = copy.deepcopy(ind[i].f)  # Sets initial best fitness
        if ind[ibest].f > ind[i].f:
            ibest = i  # Updates the index of the best particle if needed
        # Sets initial values for w, c1, and c2 (to be adjusted later)
        ind[i].w = 1
        ind[i].c1 = 1.4
        ind[i].c2 = 1.4

    # Initializes the global best particle
    gbest = copy.deepcopy(ind[ibest])
    for generation in range(prb.gmax):
        # Main loop of the PSO algorithm
        for i in range(prb.np):
            # Updates each particle's velocity and position
            for j in range(prb.nv):
                ww = random.uniform(0, ind[i].w)  # Random weight for inertia
                cc1 = random.uniform(0, ind[i].c1)  # Random weight for cognitive component
                cc2 = random.uniform(0, ind[i].c2)  # Random weight for social component
                vv1 = ind[i].pbest[j] - ind[i].x[j]  # Cognitive velocity component
                vv2 = gbest.x[j] - ind[i].x[j]  # Social velocity component
                ind[i].v[j] = ww * ind[i].v[j] + cc1 * vv1 + cc2 * vv2  # Updates velocity
                ind[i].x[j] = ind[i].x[j] + ind[i].v[j]  # Updates position
            ind[i].f = functionmaker(ind[i].x, prb.type)  # Evaluates new fitness
            if ind[i].f < ind[i].fpbest:
                # Updates particle's best known position and fitness
                ind[i].fpbest = ind[i].f
                ind[i].pbest = copy.deepcopy(ind[i].x)
                if gbest.f > ind[i].f:
                    # Updates global best if necessary
                    gbest.x = copy.deepcopy(ind[i].x)
                    gbest.f = ind[i].f
        print(str(generation) + ":" + str(gbest.f) + " " + str(gbest.x))  # Prints the generation number and the global best fitness and position
