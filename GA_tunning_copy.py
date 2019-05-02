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
    dm = pd.read_csv("mprecord/epoch9_batch100_data%d.csv"%i)
    dm_list.append(dm.values)

def Gene_Alg(maps,popsize,use_dm = False,dm = None,max_it=50 ,mu_rate=0.01,elite=0.2, greedy_pool = 5):
    t = datetime.datetime.now()
    #g0: current generation
    g0 = [Route(maps,selfopt=True) for _ in range(popsize)]
    d_list = [r.d for r in g0]
    rank = sorted(range(len(g0)), key = lambda i: d_list[i])
    d_opt = [d_list[rank[0]]]
    diver = [Diverge(g0,maps)]
    it = 0
    print(p, max_it, r, it, diver[-1], (datetime.datetime.now() - t).seconds)

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
        diver.append(Diverge(next_g,maps))
        g0 = next_g.copy()
        it+=1
        print(p, max_it, r, it, diver[-1], (datetime.datetime.now() - t).seconds)

    return(next_g[rank[0]], d_opt, diver, (datetime.datetime.now() - t).seconds)


p_list = [30,50]
it_list = [20,30]
repeat = 5

record_p = []
for p in p_list:
    for max_it in it_list:
        for r in range(repeat):
            r_opt,d_trace,diver_trace,t = Gene_Alg(map_list[0],popsize = p, use_dm = True, dm = dm_list[0], max_it=max_it)
            record_p.append([p,max_it,r_opt.d,t])
            plt.figure()
            plt.plot(d_trace)
            plt.savefig("GA_tunning/d_"+str(p)+"_"+str(max_it)+"_"+str(r)+".jpg" )
            plt.figure()
            plt.plot(diver_trace)
            plt.savefig("GA_tunning/diver_"+str(p)+"_"+str(max_it)+"_"+str(r)+".jpg" )
record_p = pd.DataFrame(record_p, columns = ['pop','it','opt_d','time'])
record_p.to_csv('GA_tunning_p.csv', index = False)

record = []
for p in p_list:
    for max_it in it_list:
        for r in range(repeat):
            r_opt,d_trace,diver_trace,t = Gene_Alg(map_list[0], popsize = p, max_it=max_it)
            record.append([p,max_it,r_opt.d,t])
            plt.figure()
            plt.plot(d_trace)
            plt.savefig("GA_tunning/d_"+str(p)+"_"+str(max_it)+"_"+str(r)+".jpg" )
            plt.figure()
            plt.plot(diver_trace)
            plt.savefig("GA_tunning/diver_"+str(p)+"_"+str(max_it)+"_"+str(r)+".jpg" )
record = pd.DataFrame(record, columns = ['pop','it','opt_d','time'])
record.to_csv('GA_tunning.csv', index = False)


p_list = [10,20,30]
it_list = [5,10,15,20]
repeat = 1

record_p = []
for p in p_list:
    for max_it in it_list:
        for r in range(repeat):
            for i, maps in enumerate(map_list):
                dm = dm_list[i]
                r_opt,d_trace,diver_trace,t = Gene_Alg(maps,popsize = p, use_dm = True, dm = dm, max_it=max_it)
                record_p.append([p,max_it,i,r_opt.d,t])
                plt.figure()
                plt.plot(d_trace)
                plt.savefig("GA_test/dwithp_%d_%d_%d_%d.jpg" %(p,max_it,i,r))
                plt.figure()
                plt.plot(diver_trace)
                plt.savefig("GA_test/diverwithp_%d_%d_%d_%d.jpg" %(p,max_it,i,r))
record_p = pd.DataFrame(record_p, columns = ['pop','it','map','opt_d','time'])
record_p.to_csv('GA_test_p.csv', index = False)



record = []
for p in p_list:
    for max_it in it_list:
        for r in range(repeat):
            for i,maps in enumerate(map_list):
                r_opt,d_trace,diver_trace,t = Gene_Alg(maps, popsize = p, max_it=max_it)
                record.append([p,max_it,i,r_opt.d,t])
                plt.figure()
                plt.plot(d_trace)
                plt.savefig("GA_test/d_%d_%d_%d_%d.jpg" %(p,max_it,i,r))
                plt.figure()
                plt.plot(diver_trace)
                plt.savefig("GA_test/diver_%d_%d_%d_%d.jpg" %(p,max_it,i,r))
record = pd.DataFrame(record, columns = ['pop','it','map','opt_d','time'])
record.to_csv('GA_test.csv', index = False)


