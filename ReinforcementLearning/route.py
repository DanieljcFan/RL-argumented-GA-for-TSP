#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:01:41 2019

@author: Daniel
"""
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


def two_opt_swap(r,i,k):
    if i>k:
        i,k = k+1, i-1
    return r[:i]+r[i:k+1][::-1]+r[k+1:]
 

class Route:
    def __init__(self, maps, index=None, pool=5,selfopt=False):
        self.score = -1e6
        self.maps = maps
        if index:
            self.new_route(index)
        else:
            self.greedy_route(pool)
        if selfopt:
            self.two_opt()

    def greedy_route(self, pool = 5):
        maps_ = self.maps.copy()
        self.index = random.sample(range(len(maps_)), 1)
        self.route = [maps_.pop(self.index[0])]
        while maps_:
            dist_list = [ self.maps[self.index[-1]].distance(ct) for ct in maps_]
            ind = np.argsort(dist_list)[:pool]
            ind_s = random.sample(list(ind), 1)[0]
            next_i= maps_[ind_s].index
            self.index.append(next_i)
            self.route.append(maps_.pop(ind_s))
        self.distance()

    def distance(self):
        self.d = 0
        for i in range(len(self.index)):
            self.d += self.maps[self.index[i]].distance(self.maps[self.index[i-1]])
            
    def new_route(self,index):
        self.route = []
        self.index = index
        for i in index:
            city = self.maps[i]
            self.route.append(city)
        self.distance()

    def two_opt_swap(self,r,i,k):
        if i>k:
            i,k = k+1, i-1
        return r[:i]+r[i:k+1][::-1]+r[k+1:]
    
    def two_opt(self, max_it = 1e3,  improvement_threshold = 1e-3):
        maps = self.maps
        index = self.index
        improvement_factor = 1 
        best_distance = self.d 
        count = 0
        while improvement_factor > improvement_threshold: 
            distance_to_beat = best_distance 
            #deleted edge: swap_first-1 to swap_first, swap_last to swap_last+1
            #added edge: swap_first-1 t0 swap_last, swap_first to swap_last+1
            #no need to cover first and last point
            for swap_first in range(1,len(index)-2): 
                for swap_last in range(swap_first+1,len(index)): 
                    if count >= max_it:
                                self.steps = count
                                self.oldindex = self.index.copy()
                                self.new_route(index)
                                return
                    new_index = two_opt_swap(index,swap_first,swap_last) 
                    a,b,c,d = swap_first,swap_first-1, swap_last ,(swap_last+1) % len(index)
                    d2_former = (maps[index[a]]).distance(maps[index[b]])+(maps[index[c]]).distance(maps[index[d]])
                    d2_new = (maps[index[c]]).distance(maps[index[b]])+(maps[index[d]]).distance(maps[index[a]])
                    diff = d2_former - d2_new
                    new_distance = best_distance - diff
                    if new_distance < best_distance: 
                        index = new_index 
                        best_distance = new_distance 
                        count +=1
            improvement_factor = 1 - best_distance/distance_to_beat 
        self.steps = count
        self.oldindex = self.index.copy()
        self.new_route(index)

    def two_opt_prob(self, dm,k=5, max_it=1e3):
        maps = self.maps
        self.oldindex = self.index.copy()
        index = self.index
        improvement_factor = 1 
        best_distance = self.d 
        count = 0
        distance_to_beat = best_distance 
        #deleted edge: swap_first-1 to swap_first, swap_last to swap_last+1
        #added edge: swap_first-1 t0 swap_last, swap_first to swap_last+1
        #no need to cover first and last point
        for swap_first in range(1,len(index)-2):
            dm_t = dm[index,:][:,index]
            dp = dm_t[swap_first,:]
            pool = sorted(range(len(dp)), key = lambda i: -dp[i])[:k]
            swap = None
            impr = 0
            for choice in pool:
                swap_last = choice-1 #becase swap_first connect to swap_last+1 
                if (swap_first - swap_last)%len(index) < 3:
                    continue
                if count >= max_it:
                            self.steps = count
                            self.oldindex = self.index.copy()
                            self.new_route(index)
                            return
                new_index = two_opt_swap(index,swap_first,swap_last) 
                a,b,c,d = swap_first,swap_first-1, swap_last ,(swap_last+1) % len(index)
                d2_former = (maps[index[a]]).distance(maps[index[b]])+(maps[index[c]]).distance(maps[index[d]])
                d2_new = (maps[index[c]]).distance(maps[index[b]])+(maps[index[d]]).distance(maps[index[a]])
                diff = d2_former - d2_new
                if diff > impr:
                    impr = diff
                    swap = swap_last
            if impr > 0:
                index = two_opt_swap(index, swap_first, swap)
                count += 1
        self.steps = count
        self.new_route(index)
       
    def mutate(self,swap_first,swap_last):
        self.route = two_opt_swap(self.route, swap_first, swap_last)
        self.index = two_opt_swap(self.index, swap_first, swap_last)
        self.distance()
        self.two_opt()














