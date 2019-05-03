#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 22:01:41 2019

@author: grapeeffect
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from route import Route,City
from numpy.linalg import norm
from collections import OrderedDict
from  more_itertools import unique_everseen

class edge:   #list of cities, length can vary from 1 to n
    def __init__(self, sub_route):
        self.indexes = [c.index for c in sub_route]
        self.cts = sub_route
        self.ends = [sub_route[0],sub_route[-1]]
        #self.end_inds = [tour[-1].index,tour[0].index]
        self.mid = sub_route[1:-1]
   
    def distance(self, edge):
        Dis1 = self.ends[1].distance(edge.ends[0])
        Dis2 = self.ends[1].distance(edge.ends[1])
        distance = min(Dis1,Dis2)
        if Dis1<Dis2:
            return distance,[self.ends[1],edge.ends[0]]
        else:  
            return distance,[self.ends[1],edge.ends[1]]

    def __eq__(self,edge):
        return self.indexes == edge.indexes
    
    def ifConnect(self,edge): #return if connect, and the connected city
        if self.ends[0]==edge.ends[0]:
            return True, self.ends[0]
        if self.ends[1]==edge.ends[1]:
            return True, self.ends[1]
        if self.ends[1]==edge.ends[0]:
            return True, self.ends[1]
        if self.ends[0]==edge.ends[1]:
            return True, self.ends[0]
        return False,None
    
    def __repr__(self):
        return str(self.indexes)
    
def reverse(edge_):
    reversed_ = edge_.cts.copy()
    reversed_.reverse()
    return edge(reversed_)

def nextEdge(edge_,edge_pool): #input an edge object and a list of edge objects
    if len(edge_pool)>0:
        dist = [edge_.distance(ed)[0] for ed in edge_pool]
        ind = np.argmin(dist)
        min_dist_e = edge_.distance(edge_pool[ind])[1]
        return edge(min_dist_e),edge_pool[ind]

def combine_edge(e1,e2): # make sure e1.ends[1] == e2.ends[0]
    mids = e1.mid+[e1.ends[1]]+e2.mid
    l = [e1.ends[0]]+mids+[e2.ends[1]]
    return edge(l)

def combine_edges_in_pool(listOfEdges):
    #new_edge_list = []
    pool = listOfEdges.copy()
    ed_new = None
    while len(pool)>1:
        ed = pool[0]
        new_pool = []
        for ed2 in pool[1:]:
            if ed.ifConnect(ed2)[0]==True:
                con_city = ed.ifConnect(ed2)[1]
                if con_city == ed.ends[0]:
                    ed = reverse(ed)
                if con_city == ed2.ends[1]:
                    ed2 = reverse(ed2)
                ed_new = combine_edge(ed,ed2)
                ed = ed_new
            else:
                new_pool.append(ed2)
                #print('ed2:')
                #print(ed2.indexes)
        if ed_new:
            new_pool.append(ed_new)
            #print('ed_new:')
            #print(ed_new.indexes)
            pool = new_pool
            ed_new = None
        else:
            break
    return pool
        


def breedDPX(parent1,parent2,maps):
    parent1 = parent1.route.copy()
    parent2 = parent2.route.copy()
   
    edges1 = [edge([parent1[i],parent1[i+1]]) for i in range(len(parent1)-1)]
    edges1.append(edge([parent1[0],parent1[-1]]))
    edges2 = [edge([parent2[i],parent2[i+1]]) for i in range(len(parent2)-1)]
    edges2.append(edge([parent2[0],parent2[-1]]))
    common_edges = []
    destroyed_edges = []
    for ed in edges1:
        for ed2 in edges2:
            if ed==ed2:
                common_edges.append(ed)
    destroyed_edges =[ed for ed in edges1 if ed not in common_edges]
    destroyed_edges+=[ed for ed in edges2 if ed not in common_edges]        
    common_edges = combine_edges_in_pool(common_edges)
    
    if len(common_edges) == 0:
        r_t = Route(maps)
        return r_t.index
        #return
    ctsInEdge = sum([ed.cts for ed in common_edges], [])
    edge_pop = common_edges.copy()
    startEdge = common_edges[0]
    currentEdge = startEdge
    edgeToConnect =edge_pop.copy()
    edgeToConnect.remove(startEdge)
    edgeToConnect =edgeToConnect+[edge([ct]) for ct in maps if ct not in ctsInEdge]
    #print('edge_pop:')
    #print(edge_pop)
    #print('currentEdge:')
    #print(currentEdge)
    #print('edgeToConnect:')
    #print(edgeToConnect)
    while len(edgeToConnect)>0:
        nextE = nextEdge(currentEdge,edgeToConnect)[0]
        #print('next_edge:')
        #print(nextE)
        removed = nextEdge(currentEdge,edgeToConnect)[1]
        edge_pop.append(nextE)
        edge_pop = combine_edges_in_pool(edge_pop)
        edgeToConnect.remove(removed)
        currentEdge = edge_pop[0]
        if currentEdge in edgeToConnect:
            edgeToConnect.remove(currentEdge)
        #print('edge_pop:')
        #print(edge_pop)
        #print('currentEdge:')
        #print(currentEdge)
        #print('edgeToConnect:')
        #print(edgeToConnect)
    while len(edge_pop)>2:
        edge_pop.append(nextEdge(edge_pop[0],edge_pop[1:])[0])
        edge_pop = combine_edges_in_pool(edge_pop)    
    while len(edge_pop)==2:
        edge_pop.append(nextEdge(edge_pop[0],[edge_pop[1]])[0])
        edge_pop = combine_edges_in_pool(edge_pop) 
    
    #return Route(maps = maps, index = list(unique_everseen(edge_pop[0].indexes)),selfopt=True)
    return list(unique_everseen(edge_pop[0].indexes))
   

def DPX_d(parent1,parent2,maps):
    parent1 = parent1.route.copy()
    parent2 = parent2.route.copy()
   
    edges1 = [edge([parent1[i],parent1[i+1]]) for i in range(len(parent1)-1)]
    edges1.append(edge([parent1[0],parent1[-1]]))
    edges2 = [edge([parent2[i],parent2[i+1]]) for i in range(len(parent2)-1)]
    edges2.append(edge([parent2[0],parent2[-1]]))
    common_edges = []
    destroyed_edges = []
    for ed in edges1:
        for ed2 in edges2:
            if ed==ed2:
                common_edges.append(ed)
    destroyed_edges =[ed for ed in edges1 if ed not in common_edges]
    destroyed_edges+=[ed for ed in edges2 if ed not in common_edges]        
    common_edges = combine_edges_in_pool(common_edges)

    d = 0
    for e in common_edges:
        d+=len(e.indexes)-1

    return len(maps)-d


def Diverge(generation,maps):
    diver = 0
    for i in generation:
        for j in generation:
            diver+= DPX_d(i,j,maps)
    return diver/(len(generation)**2)