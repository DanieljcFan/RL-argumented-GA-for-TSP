### route.py

class City to represent a point with corresponding x,y axis; class Route to represent a route to cities as ordered list

- class City: 

Attribution: 

	- x,y: *numeric* coordinates of the city point; 

	- index: *int* index of the city in natural order

Methods:

	- distance(*City*): return the distance of current city and given city  

- class Route:

Attributes: 

	- maps: *list* of *City class* in natural order

	- route: *list* of *City class* in route order

	- index: *list* of *int* index in route order

	- oldindex: *list* of *int* previous index in route order before local search

	- d: *numeric* cost of route, sum of distance between connected cities

	- pool_list: *List* of *List* selecting pool of each step when generating route. only exist when self.Greedy_route(show=True)

Methods:

	- Greedy_route(pool=5,show=False): generate a half greedy-half random new route, as completely random routes take more time to improve, and complete greedy routes lack diversity. pool is the size of selecting pool from which the next edge is picked randomly. If show, the pool of each step would be recorded in attribute self.pool_list.

	- new_route(index): generate a fixed route following the order of index 

	- distence(): calculate the distance of current route

	- two_opt(): local search by two-optimal 

	- two_opt_prob(): local search based on decision matrix

	- two_opt_swap(r,i,k): perform a two-swap to *list* r at position (i,k)

	- mutate(): perform a two-swap at random position


## GeneAlg.py

operations on Route, to realize genetic algorithm.

- function Gene_Alg(maps, popsize, use_dm=False, dm = None, max_it=50,mu_rate=0.01,elite=0.2,greedy_pool=5):

Parameters:

	- maps: *list* of City, order irrelevant

	- popsize: population of each generation

	- use_dm: whether use decision matrix to prove local search. If True, a n*n matrix should be provided where n is the length of maps

	- dm: n*n decision matrix, only required when use_dm = True 

	- max_it: max iteration of algorithm, or the number of generation

	- mu_rate: the rate of randomly mutate, to keep diversity and avoid pre-convergence.

	- elite: the ratio of elite passed to next gerenation

	- greedy_pool: size of pool when initialize route. See class Route

Return:

	- *Route* of optimal solution

	- distance of optimal solution

	- time consumed (seconds)

