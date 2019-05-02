import datetime
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy import linalg
random.seed(2019)

class City:
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Route:
    def __init__(self, index=None, pool=5, maps=maps):
        self.score = -1e6
        if index:
            self.new_route(index, maps = maps)
        else:
            self.greedy_tour(pool, maps)

    def greedy_tour(self, pool = 5, maps = maps):
        maps_ = maps.copy()
        self.index = random.sample(range(len(maps)), 1)
        self.route = [maps_.pop(self.index[0])]
        while maps_:
            dist_list = [self.route[-1].distance(ct) for ct in maps_]
            ind = np.argsort(dist_list)[:pool]
            ind_s = random.sample(list(ind), 1)[0]
            next_i= maps_[ind_s].index
            self.index.append(next_i)
            self.route.append(maps_.pop(ind_s))

    def distance(self):
        self.d = 0
        for i,city in enumerate(self.route):
            self.d += city.distance(self.route[i-1])
            
    def new_route(self,index, maps=maps):
        self.route = []
        self.index = index
        for i in index:
            city = maps[i]
            self.route.append(city)
        
    def two_opt(self, improvement_threshold = 1e-3, maps = maps):
        two_opt_swap = lambda r,i,k: (r[:i]+r[k:i-1:-1]+r[k+1:])
        index = self.index
        improvement_factor = 1 # Initialize the improvement factor.
        best_distance = self.d # Calculate the distance of the initial path.
        count = 0
        while improvement_factor > improvement_threshold: # If the index is still improving, keep going!
            distance_to_beat = best_distance # Record the distance at the beginning of the loop.
            for swap_first in range(1,len(index)-2): # From each city except the first and last,
                for swap_last in range(swap_first+1,len(index)): # to each of the maps following,
                    new_index = two_opt_swap(index,swap_first,swap_last) # try reversing the order of these maps
                    a,b,c,d = swap_first,swap_first-1, swap_last ,(swap_last+1) % len(index)
                    d2_former = (maps[index[a]]).distance(maps[index[b]])+(maps[index[c]]).distance(maps[index[d]])
                    d2_new = (maps[index[c]]).distance(maps[index[b]])+(maps[index[d]]).distance(maps[index[a]])
                    diff = d2_former - d2_new
                    new_distance = best_distance - diff
                    if new_distance < best_distance: # If the path distance is an improvement,
                        index = new_index # make this the accepted best index
                        best_distance = new_distance # and update the distance corresponding to this index.
                        count +=1
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the index has improved.
        self.opt_index = index
        self.steps = count
  
    def cal_score(self, maps=maps):
        cost = 0
        lamb = 500/len(self.index)
        for i,_ in enumerate(self.opt_index):
            cost += maps[self.opt_index[i]].distance(maps[self.opt_index[i-1]])
        self.opt_d = cost
        self.score = -cost - self.steps*lamb



def readCities(file, skiplines):
    cities_ = pd.read_csv(file,skiprows=skiplines,header = None,sep=' ')
    cities_ = cities_.drop(cities_.columns[[0]], axis=1)
    cities_ = np.array(cities_)[0:len(cities_)-1,[-2,-1]]
    return cities_

cts = readCities('gr96_tsp.txt',7)
maps = [City(c[0],c[1],i) for i,c in enumerate(cts)]

r0 = Route(maps = maps)
r0.distance()
r0.two_opt()
r0.cal_score()        

plt.scatter(cts[:,0],cts[:,1])
plt.plot(cts[r0.index,0],cts[r0.index,1],color = 'blue')
plt.plot(cts[r0.opt_index,0],cts[r0.opt_index,1],color = 'red')




n = 1000
trainset = pd.DataFrame(columns=['']*len(maps), index = range(n))
score = pd.DataFrame(columns=['score','opt_d','step'], index = range(n))
t = datetime.datetime.now()
for i in range(n):
    r0 = Route(maps = maps)
    r0.distance()
    r0.two_opt()
    r0.cal_score()        
    trainset.iloc[i,:] = r0.index
    score.iloc[i,:] = [r0.score, r0.opt_d, r0.steps]
    print(i, (datetime.datetime.now() - t).seconds)

trainset = trainset.join(score)
trainset.to_csv('trainset_1000.csv')


