import csv
import copy
import math
import statistics
import time
import random
import numpy as np
import matplotlib.pyplot as plt


class problem:
    __slots__ = [
        'npop',
        'nvariable',
        'ncross',
        'nchild',
        'bitlist',
        'nfunc',
        'nconstraint',
        'functionselect',
        'mutationrate',
        'mutationrate2',
        'xmin',
        'xmax',
        'ptype',
        'minarea',
        'maxarea',
        'maximumgeneration'
    ]

    def __init__(self):
        self.npop = 100  # num of population
        self.nvariable = 2  #
        self.ncross = 15  #
        self.nchild = self.ncross * 2  # num of chidren
        self.bitlist = [4 for i in range(self.nvariable)]  #
        self.nfunc = 1  #
        self.nconstraint = 0  #
        self.functionselect = 0  #
        self.mutationrate = 0.1  # Mutation rate
        self.xmin = [-5.12 for i in range(self.nvariable)]
        self.xmax = [5.12 for i in range(self.nvariable)]
        self.ptype = 2
        self.minarea = 10
        self.maxarea = 100
        self.maximumgeneration = 1000

    class GA(problem):
        __slots__ = [
            'Gtype',
            'Itype',
            'Ptype',
            'Funcval',
            'Fitness'
        ]

        def __init__(self):
            self.Gtype = [0 for i in range(5)]  #
            self.Itype = [0 for i in range(5)]  #
            self.Ptype = [0 for i in range(5)]  #
            self.Funcval = [0 for i in range(5)]  #
            self.Fitness = [0 for i in range(5)]  #

        def initialization(self, prb):
            for i in range(sum(prb.bitlist)):
                a = random.randint(0, 1)
                self.Gtype.append(a)

        def decodingTointeger(self, prb):
            self.Itype = []
            sum = 0
            for i in range(prb.nvariable):
                a = 0
                for j in range(prb.bitlist[i]):
                    a += self.Gtype[sum + j] * 2 ** j
                self.Itype.append(a)
                sum += prb.bitlist[i]

        def decodingToptype(self, prb):
            for i in range(prb.nvariable):
                a = prb.xmin[i] + (prb.xmax[i] - prb.xmin[i]) / (2 ** prb.bitlist[i] - 1) * self.Itype[i]
                self.Ptype.append(a)

        def crossover(self, ind, prb, aparent, bparent, cchild, dchild):
            crossoverpoint = random.randint(1, 3)
            crosswhere = []
            for i in range(crossoverpoint):
                a = random.randint(1, sum(prb.bitlist) - 1)
                if a in crosswhere:
                    i = -1
                else:
                    crosswhere.append(a)
            icross = 0
            for i in range(sum(prb.bitlist)):
                if i in crosswhere:
                    icross += 1
                if icross % 2 == 0:
                    ind[cchild].Gtype[i] = ind[aparent].Gtype[i]
                    ind[dchild].Gtype[i] = ind[bparent].Gtype[i]
                else:
                    ind[dchild].Gtype[i] = ind[aparent].Gtype[i]
                    ind[cchild].Gtype[i] = ind[bparent].Gtype[i]

        def fitnessfunctionmaker(self, fmin, fmax, prb):
            # fit for minimization
            a = prb.minarea + (prb.maxarea - prb.minarea) * (fmax - self.Funcval) / (fmax - fmin)
            # fit for maximization
            # a = prb.minarea+(prb.maxarea-prb.minarea)*(self.Funcval-fmin)/(fmax-fmin)


def mutation(ind, prb, ichild):
    if random.uniform(0, 1) < prb.mutationrate:
        for j in range(sum(prb.bitlist)):
            if random.uniform(0, 1) < prb.mutationrate2:
                if ind[ichild].Gtype[j] == 0:
                    ind[ichild].Gtype[j] = 1
                else:
                    ind[ichild].Gtype[j] = 0

        Flywheel = choiceOfsurvival(ind, prb)
        Flywheel.sort()
        for i in range(prb.npop):
            j = Flywheel[i]
            if j > i:
                ind[i] = copy.deepcopy(ind[j])


def functionMaker(x, ptype):
    if ptype == 0:  # Function Arakawa
        n = int(len(x) * 0.5)
        f = 0
        for i in range(n):
            f += (10 * math.exp(-0.01 * (x[2 * i] - 10) ** 2 - 0.01 * (x[2 * i + 1] - 15) ** 2) * math.sin(
                2 * x[2 * i]))
    elif ptype == 1:  # Function Rastrigin
        n = int(len(x))
        pi = math.atan(1) * 4
        f = 10 * len(x)
        for i in range(n):
            f += (x[i] ** 2 - 10 * math.cos(2 * pi * x[i]))
    elif ptype == 2:
        n = int(len(x) * 0.5)
        f = 0
        for i in range(n):
            f += (100 * (x[2 * i + 1] - x[2 * i] ** 2) ** 2 + (x[2 * i] - 1) ** 2)
    return f


def makingflywheel(ind, prb, type):
    a = []
    if type == 0:
        pmax = prb.npop
    else:
        pmax = prb.npop + prb.nchild
    for i in range(pmax):
        a.append(ind[i].Fitness)
    return (a)


def choiceOfparents(ind, prb):
    ans = []
    flywheel = makingflywheel(ind, prb, 0)
    print(flywheel)
    while len(ans) < prb.nchild:
        maxwheel = sum(flywheel)
        thress = 0
        key = random.uniform(0, maxwheel - 0.0001)
        j = 0
        thress = flywheel[j]
        while thress < key:
            j += 1
            thress += flywheel[j]
        if not (j in ans):
            ans.append(j)
            flywheel[j] = 0
    return ans


prb = problem()
ind = []

for i in range(prb.npop + prb.nchild):
    temp = GA()
    ind.append(temp)

# best solution
bestind = GA()

fmin = 1.0e33
fmax = -1.0e33
ibest = 0

for i in range(prb.npop):
    ind[i].initialization(prb)
    ind[i].decodingTointeger(prb)
    ind[i].decodingToptype(prb)
    ind[i].Funcval = functionMaker(ind[i].Ptype, prb.ptype)

    if fmin > ind[i].Funcval:
        fmin = ind[i].Funcval
        ibest = i  # case for minimization
    if fmax < ind[i].Funcval:
        fmax = ind[i].Funcval
        # ibest=i  # case for maximization

print(fmin, fmax, ibest)

for i in range(prb.nchild):
    j = i + prb.npop
    ind[j].Gtype = [0 for i in range(sum(prb.bitlist))]

for i in range(prb.npop):
    ind[i].fitnessfunctionmaker(fmin, fmax, prb)

bestind.Funcval = copy.deepcopy(ind[ibest].Funcval)
bestind.Gtype = copy.deepcopy(ind[ibest].Gtype)
bestind.Itype = copy.deepcopy(ind[ibest].Itype)
bestind.Ptype = copy.deepcopy(ind[ibest].Ptype)
bestfunc = bestind.Funcval

print("0:" + str(bestind.Funcval) + " " + str(fmin) + " " + str(fmax) + " " + str(bestind.Itype))

for generation in range(prb.maximumgeneration):
    parentslist = choiceOfparents(ind, prb)

    for ic in range(prb.ncross):
        crossover(ind, prb, parentslist[ic * 2], parentslist[ic * 2 + 1], prb.npop + ic * 2, prb.npop + ic * 2 + 1)
        mutation(ind, prb, prb.npop + ic * 2)
        mutation(ind, prb, pprb.npop + ic * 2 + 1)

    for i in range(prb.nchild):
        ic = i + prb.npop

        ind[ic].decodingTointeger(prb)
        ind[ic].decodingToptype(prb)
        ind[ic].Funcval = functionMaker(ind[ic].Ptype, prb.ptype)

    fmin = -1.0e33
    fmax = 1.0e33

    for i in range(prb.npop + prb.nchild):
        if ind[i].Funcval < fmin:
            fmin = ind[i].Funcval
            ibest = i
        if ind[i].Funcval > fmax:
            fmax = ind[i].Funcval

    for i in range(prb.npop + prb.nchild):
        ind[i].fitnessfunctionmaker(fmin, fmax, prb)
    print(str(generation) + ":" + str(bestind.Funcval) + " " + str(ind[ibest].Funcval) + " " + str(fmax) + " " + str(
        bestind.Itype))

    if ind[ibest].Funcval < bestind.Funcval:
        bestind = copy.deepcopy(ind[ibest])

    Flywheel = choiceOfsurvival(ind, prb)
    Flywheel.sort()

    for i in range(prb.npop):
        j = Flywheel[i]
        if j > i:
            ind[i] = copy.deepcopy(ind[j])
