import os
os.chdir('/Users/daniel/Documents/academic/DeepLearning/NN-agumented-GA-for-TSP')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
from route import City, readCities, Route
cts = readCities('gr96_tsp.txt',7)
maps = [City(c[0],c[1],i) for i,c in enumerate(cts)]



def two_opt_swap(r,i,k):
    if i >k:
        i,k = k+1, i-1
    return r[:i]+r[i:k+1][::-1]+r[k+1:]


def two_opt_1(index, k = 5,max_it = 10, improvement_threshold = 1e-3, maps = maps):
    must_list = random.sample(range(len(maps)), k)
    record = []
    for swap_first in must_list:
        new_index = index[swap_first:] + index[:swap_first]
        impr = 1e6
        for swap_last in range(len(index)):
            if (swap_first - swap_last)%len(index) < 3:
                continue
            a,b,c,d = swap_first,swap_first-1, swap_last ,(swap_last+1) % len(index)
            d2_former = (maps[index[a]]).distance(maps[index[b]])+(maps[index[c]]).distance(maps[index[d]])
            d2_new = (maps[index[c]]).distance(maps[index[b]])+(maps[index[d]]).distance(maps[index[a]])
            diff = d2_former - d2_new
            if diff > impr:
                impr = diff
                swap = swap_last
        r_temp = Route(maps,two_opt_swap(index, swap_first, swap))
        r_temp.distance()
        r_temp.two_opt(max_it)
        r_temp.cal_score()
        record.append(new_index+[r_temp.score, r_temp.d, r_temp.steps])
    return record

trainset_ = []
n = 200
t = datetime.datetime.now()
for i in range(n):
    r0 = Route(maps)
    record = two_opt_1(r0.index)
    trainset_ = trainset_+record
    print(i, (datetime.datetime.now() - t).seconds)

trainset_ = pd.DataFrame(trainset_, columns=['']*len(maps)+['score','opt_d','step'])
trainset_
trainset_.to_csv('leave1_10step.csv', index = False)


'''
select route from different stage of local seach
only concerns about the next few step
'''


def local_1(index,maps=maps,k=5):
    must_list = random.sample(range(len(maps)), k)
    record = []
    for swap_first in must_list:
        new_index = index[swap_first:] + index[:swap_first]
        impr = -1e6
        for swap_last in range(len(index)):
            if (swap_first - swap_last)%len(index) < 3:
                continue
            a,b,c,d = swap_first,swap_first-1, swap_last ,(swap_last+1) % len(index)
            d2_former = (maps[index[a]]).distance(maps[index[b]])+(maps[index[c]]).distance(maps[index[d]])
            d2_new = (maps[index[c]]).distance(maps[index[b]])+(maps[index[d]]).distance(maps[index[a]])
            diff = d2_former - d2_new
            if diff > impr:
                impr = diff
                swap = swap_last
        record.append(new_index+[impr])
    return record
    
trainset = []
n = 200
for i in range(n):
    if i % 10 == 0:
        print(i)
    r0 = Route(maps)
    record = r0.two_opt(out_ratio=20)
    for route in record:
        trainset = trainset + local_1(route)

trainset = pd.DataFrame(trainset, columns=[""]*len(maps)+["impr"])
trainset.to_csv('leave1_1step.csv', index = False)


