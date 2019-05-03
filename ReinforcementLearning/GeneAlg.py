import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from DPX_ import breedDPX, DPX_d, Diverge
from route import Route,City
import random

random.seed(2019)

dataset = np.load('val_data.npy')
dataset = dataset.transpose((0,2,1))
map_list = []
for i,cts in enumerate(dataset):
    if i >= 30: break
    maps = [City(c[0],c[1],i) for i,c in enumerate(cts)]
    map_list.append(maps)

dm_list = []
for i in range(len(map_list)):
    dm = pd.read_csv("mprecord/epoch9_data%d.csv"%i)
    dm_list.append(dm.values)

def Gene_Alg(maps,popsize,use_dm = False,dm = None,max_it=50 ,mu_rate=0.01,elite=0.2, greedy_pool = 5):
    t = datetime.datetime.now()
    #g0: current generation
    g0 = [Route(maps,selfopt=True) for _ in range(popsize)]
    d_list = [r.d for r in g0]
    rank = sorted(range(len(g0)), key = lambda i: d_list[i])
    d_opt = [d_list[rank[0]]]
    it = 0
    print(p, max_it, it, d_opt[-1], (datetime.datetime.now() - t).seconds)

    while it < max_it: 
        next_g = [g0[i] for i in rank[:int(popsize*elite)]]
        while len(next_g) < popsize:
            p1,p2 = random.sample(range(popsize),2)
            rc_index = breedDPX(g0[p1], g0[p2],maps=maps)
            rc = Route(maps,rc_index, selfopt =True)
            if use_dm:
                rc.two_opt_prob(dm)
            else: rc.two_opt()
            if random.random() < mu_rate:
                i,k = random.sample(range(len(maps)),2)
                rc.mutate(i,k)
            next_g.append(rc)
        d_list = [r.d for r in next_g]
        rank = sorted(range(len(g0)), key = lambda i: d_list[i])
        d_opt.append(d_list[rank[0]])
        g0 = next_g.copy()
        it+=1
        print(p, max_it, it, d_opt[-1], (datetime.datetime.now() - t).seconds)

    return(next_g[rank[0]], d_opt, (datetime.datetime.now() - t).seconds)


p=30
max_it = 30
map_size = 30

record_p = []
for i, maps in enumerate(map_list[:map_size]):
    dm = dm_list[i]
    print("map %d"%(i))
    r_opt,d_trace,t = Gene_Alg(maps,popsize = p, use_dm = True, dm = dm, max_it=max_it)
    record_p.append([i,t]+d_trace)
    plt.figure()
    plt.plot(d_trace)
    plt.savefig("GA_test/dwithp_%d.jpg" %(i))
record_p = pd.DataFrame(record_p, columns = ['map','time']+["step"+str(i) for i in range(max_it+1)] )
record_p.to_csv('GA_test_p.csv', index = False)


record = []
for i, maps in enumerate(map_list[:map_size]):
    r_opt,d_trace,t = Gene_Alg(maps,popsize = p, max_it=max_it)
    record.append([i,t]+d_trace)
    plt.figure()
    plt.plot(d_trace)
    plt.savefig("GA_test/d_%d.jpg" %(i))
record = pd.DataFrame(record, columns = ['map','time']+["step"+str(i) for i in range(max_it+1)] )
record.to_csv('GA_test.csv', index = False)


