# NN-agumented-GA-for-TSP
A project to find solution of a TSP problem, in cooperate with Grape Qiao, Han Wang, and Ziyan Lin

TSP is a famous NP-hard problem, and Genetic Algorithm combined with Lin-Kernighan heuristic has been shown to be one of the best approximate solutions. Still, here is a space for imporvement. In this project Neural Network is applied to improve the efficiency of local search and thus improve the total algorithm. 

The whole process is as below:
 - Generate n route as initial population
 	half random-half greedy  generation is applied, as completely random routes take more time to improve, and complete greedy routes lack diversity.
 	In practice, one of first-k nearest neighbors is selected randomly to develop a connection
 - Local search: Due to the complexity of programming a reduced version of Lin-Kernighan heuristic, namely two-optimal swap is implimented here. 
 - Breeding:
 - Update population:
 - Mutate:
 repeats until converge

## Function documentaion

### route.py
 - class City: 
 	Attribution: 
 		- x,y: *numeric* coordinates of the city point; 
 		- index: *int* index of the city in natural order
 	Methods:
 		- distance(*City*) return the distance of current city and given city  

 - class Route:
 	Attributes: 
 		 - maps: *list* of *City class* in natural order
 		 - route *list* of *City class* in route order
 		 - index *list* of *int* index in route order
 		 - d *numeric* cost of route, sum of distance between connected cities
 		 - score: *numeric* score to evaluate the route, cost + penalty for two-opt steps
 	Methods:
 		 - Greedy_route() generate a half greedy-half random new route, for detail
 		 - new_route() generate a fixed route by index 
 		 - distence() calculate the distance of current route
 		 - two_opt() local search by two-optimal 
 		 - two_opt_swap() perform a two-swap at given position
 		 - cal_score() 
 		 - mutate() perform a two-swap at random position

