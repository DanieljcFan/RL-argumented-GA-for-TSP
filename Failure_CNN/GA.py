import numpy as np
import pandas as pd
import datetime

from DPX_ import breedDPX, DPX_d, Diverge
from route import Route,City,readCities
import random

random.seed(2019)


def Gene_Alg(maps, popsize,max_it=50 ,mu_rate=0.01,elite=0.2, greedy_pool = 5):
    t = datetime.datetime.now()
    #g0: current generation
    g0 = [Route(maps,selfopt=True) for _ in range(popsize)]
    d_list = [r.d for r in g0]
    rank = sorted(range(len(g0)), key = lambda i: d_list[i])
    d_opt = [d_list[rank[0]]]
    diver = [Diverge(g0,maps)]
    it = 0
    print(popsize, max_it, it, diver[-1], (datetime.datetime.now() - t).seconds)

    while it < max_it: 
        next_g = [g0[i] for i in rank[:int(popsize*elite)]]
        while len(next_g) < popsize:
            p1,p2 = random.sample(range(popsize),2)
            rc_index = breedDPX(g0[p1], g0[p2],maps=maps)
            rc = Route(maps,rc_index, selfopt =True)
            rc.two_opt()
            if random.random() < mu_rate:
                i,k = random.sample(range(len(maps)),2)
                rc.mutate(i,k)
            next_g.append(rc)
        d_list = [r.d for r in next_g]
        rank = sorted(range(len(g0)), key = lambda i: d_list[i])
        d_opt.append(d_list[rank[0]])
        diver.append(Diverge(next_g,maps))
        g0 = next_g.copy()
        it+=1
        print(popsize, max_it, it, diver[-1], (datetime.datetime.now() - t).seconds)

    return(next_g[rank[0]], d_opt, diver, (datetime.datetime.now() - t).seconds)







