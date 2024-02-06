import math
import os#当座は必要ありません．出力ファイルの管理をしたくなったら必要になります
import csv#当座は必要ありません．ファイル入力をしたくなったら必要になります．
import copy
from cv2 import exp#Pythonでは配列は基本的にはポインタ管理です．値を引き継がせるにはdeepcopyが必要なんだって
import numpy as np#当座は必要ありません．扱う問題が複雑になった時に必要になります．
import random#絶対に必要です．
import statistics

def functionmaker(x,ptype):
    if ptype==0:
        n=int(len(x)*0.5)
        f=0
        for i in range(n):
            f+=(10*math.exp(-0.01*(x[2*i]-10)**2-0.01*(x[2*i]-15)**2))*math.sin(2*x[2*i])

    if ptype==1:
        n=int(len(x))
        pi=math.atan(1)*4
        f=10*len(x)
        for i in range(n):
            f+=(x[i]**2-10*math.cos(2*pi*x[i]))

    if ptype==2:
        n=int(len(x)*0.5)
        f=0
        for i in range(n):
            f+=(100*(x[2*i+1]-x[2*i]**2)**2+(x[2*i]-1)**2)  # 100*(x[0]-x[0]**2)**2+(x[0]-1)**2

    return f

def makingflywheel(ind,prb,type):
    a=[]
    if type==0:
        pmax=prb.npop
    else:
        pmax=prb.npop+prb.nchild
    for i in range(pmax):
        a.append(ind[i].Fitness)
    return a


def choiceofparents(ind,prb):
    ans=[]
    flywheel=makingflywheel(ind,prb,0)
    while len(ans)<prb.nchild:
        maxwheel=sum(flywheel)
        thress=0
        key=random.uniform(0,maxwheel-0.0001)
        j=0
        thress=flywheel[j]
        while thress<key:
            j+=1
            thress+=flywheel[j]
        if not(j in ans):
            ans.append(j)
            flywheel[j]=0
    return ans

def choiceofsurvival(ind,prb):
    ans = []
    flywheel=makingflywheel(ind,prb,1)
    for i in range(prb.npop):
        index=flywheel.index(max(flywheel))
        ans.append(index)
        flywheel[index]=0
        # print(flywheel[i])
    
    return ans

def crossover(ind,prb,aparent,bparent,cchild,dchild):
    crossoverpoint=random.randint(1,3)
    crosswhere=[]
    for i in range(crossoverpoint):
        a=random.randint(1,sum(prb.bitlist)-1)
        if a in crosswhere:
            i-=1
        else:
            crosswhere.append(a)
    icross=0
    for i in range(sum(prb.bitlist)):
        if i in crosswhere:
            icross+=1
        if icross%2==0:
            ind[cchild].Gtype[i]=ind[aparent].Gtype[i]
            ind[dchild].Gtype[i]=ind[bparent].Gtype[i]
        else:
            ind[dchild].Gtype[i]=ind[aparent].Gtype[i]
            ind[cchild].Gtype[i]=ind[bparent].Gtype[i]

        
def mutation(ind,prb,ichild):
    if random.uniform(0,1)<prb.mutationrate:
        for j in range(sum(prb.bitlist)):
            if random.uniform(0,1)<prb.mutationrate2:
                if ind[ichild].Gtype[j]==0:
                    ind[ichild].Gtype[j]=1
                else:
                    ind[ichild].Gtype[j]=0


class problem:
    __slots__ = [
            "npop",
            "nvariable",
            "ncross",
            "nchild",
            "bitlist",
            "nfunc",
            "nconstraint",
            "functionselect",
            "mutationrate",
            "mutationrate2",
            "xmin",
            "xmax",
            "ptype",
            "minarea",
            "maxarea",
            "maxiumgeneration"
            ]

    def __init__(self):
        self.npop=100
        self.nvariable=2
        self.ncross=15
        self.nchild=self.ncross*2
        self.bitlist=[4 for i in range(self.nvariable)]
        self.nfunc=1
        self.nconstraint=0
        self.functionselect=0
        self.mutationrate=0.1        #0.1
        self.mutationrate2=0.1       #0.1
        self.xmax=[5.12 for i in range(self.nvariable)]
        self.xmin=[-5.12 for i in range(self.nvariable)]
        self.ptype=2   #2
        self.minarea=10
        self.maxarea=100
        self.maxiumgeneration=1000    #1000

        
class GA(problem):
    __slots__ = [
            'Gtype',
            'Itype',
            'Ptype',
            'Funcval',
            'Fitness'
            ]
    def __init__(self):
        self.Gtype=[]
        self.Itype=[]
        self.Ptype=[]
        
    def initialization(self, prb):
        self.Gtype=[]
        for i in range(sum(prb.bitlist)):
            a = random.randint(0,1)
            self.Gtype.append(a)

    def decodingtointeger(self, prb):
        self.Itype=[]
        sum_of_bit = 0
        for i in range(prb.nvariable):
            a = 0
            for j in range(prb.bitlist[i]):
                a += self.Gtype[sum_of_bit + j]*2**j
            self.Itype.append(a)
            sum_of_bit += prb.bitlist[i]


    def decodingtoptype(self,prb):
        self.Ptype=[]
        for i in range(prb.nvariable):
            a = prb.xmin[i]+(prb.xmax[i]-prb.xmin[i])/(2**prb.bitlist[i]-1)*self.Itype[i]
            self.Ptype.append(a)


    def fitnessfunctionmaker(self,fmin,fmax,prb):
        # minimization
        self.Fitness=prb.minarea+(prb.maxarea-prb.minarea)*(fmax-self.Funcval)/(fmax-fmin)
        # self.Fitness=prb.minarea+(prb.maxarea-prb.minarea)*(fmax-self.Funcval)/(fmax-fmin+1e-9)
         
        # maxmazation
        # self.Fitness=1e-9+prb.minarea+(prb.maxarea-prb.minarea)*(self.Funcval-fmin)/(fmax-fmin+1e-9)
        # self.Fitness=prb.minarea+(prb.maxarea-prb.minarea)*(self.Funcval-fmin)/(fmax-fmin)




if __name__ == '__main__':
    prb = problem()
    ind=[]
    for i in range(prb.npop+prb.nchild):
        temp=GA()
        ind.append(temp)

    bestind = GA()
    fmin=1.0e33
    fmax=-1.0e33
    for i in range(prb.npop):
        ind[i].initialization(prb)    
        ind[i].decodingtointeger(prb)
        ind[i].decodingtoptype(prb)
        ind[i].Funcval=functionmaker(ind[i].Ptype,prb.ptype)
        if fmin>ind[i].Funcval:
            fmin=ind[i].Funcval
            ibest=i
        if fmax<ind[i].Funcval:
            fmax=ind[i].Funcval

    for i in range(prb.nchild):
        j=i+prb.npop
        ind[j].Gtype=[0 for i in range(sum(prb.bitlist))]

    for i in range(prb.npop):
        ind[i].fitnessfunctionmaker(fmin,fmax,prb)
        # print(str(ind[i].Gtype)+" | "+str(ind[i].Itype)+" | "+str(ind[i].Ptype)+" | "+str(ind[i].Funcval))
    
    # print("----------------------------")

    bestind.Funcval=copy.deepcopy(ind[ibest].Funcval)
    bestind.Gtype=copy.deepcopy(ind[ibest].Gtype)
    bestind.Itype=copy.deepcopy(ind[ibest].Itype)
    bestind.Ptype=copy.deepcopy(ind[ibest].Ptype)
    bestfunc=bestind.Funcval

    # print("0"+": "+str(bestind.Funcval)+" "+str(fmin)+" "+str(fmax)+" "+str(bestind.Itype))

    for generation in range(prb.maxiumgeneration):
        
        parentslist=choiceofparents(ind,prb)

        for ic in range(prb.ncross):
            crossover(ind,prb,parentslist[ic*2],parentslist[ic*2+1],prb.npop+ic*2,prb.npop+ic*2+1)
            mutation(ind,prb,prb.npop+ic*2)
            mutation(ind,prb,prb.npop+ic*2+1)

        for i in range(prb.nchild):
            ic=i+prb.npop
            ind[ic].decodingtointeger(prb)
            ind[ic].decodingtoptype(prb)
            ind[ic].Funcval=functionmaker(ind[ic].Ptype,prb.ptype)
        

        fmin = 1.0e33
        fmax = -1.0e33


        for i in range(prb.npop+prb.nchild):
            # print(str(ind[i].Gtype)+" | "+str(ind[i].Itype)+" | "+str(ind[i].Ptype)+" | "+str(ind[i].Funcval))
            
            if ind[i].Funcval<fmin:
                fmin=ind[i].Funcval
                ibest=i
            if ind[i].Funcval>fmax:
                fmax=ind[i].Funcval
        
        

        for i in range(prb.npop+prb.nchild):
            ind[i].fitnessfunctionmaker(fmin,fmax,prb)
        
        print(str(generation)+": "+str(bestind.Funcval)+" "+str(ind[ibest].Funcval)+" "+str(fmax)+" "+str(bestind.Itype))
        # print(" 0 "+str(ind[0].Gtype)+" | "+str(ind[0].Itype)+" | "+str(ind[0].Ptype)+" | "+str(ind[0].Funcval))
        
        if ind[ibest].Funcval<bestind.Funcval:
            bestind=copy.deepcopy(ind[ibest])

        Flywheel=choiceofsurvival(ind,prb)
        Flywheel.sort()
        # arr=[]
        for i in range(prb.npop):
            j=Flywheel[i]
            # print(j)
            if j>i:
                ind[i]=copy.deepcopy(ind[j])
# print(" 1 "+str(ind[0].Gtype)+" | "+str(ind[0].Itype)+" | "+str(ind[0].Ptype)+" | "+str(ind[0].Funcval))
        

        

        
            
        
       
      
        











