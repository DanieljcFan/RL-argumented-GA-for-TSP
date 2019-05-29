# RL-agumented-GeneticAlgorithm-for-TSP
A project to find solution of a TSP(Travelling Salesman Problem), in cooperate with Grape Qiao, Han Wang, and Ziyan Lin. [paper here](./553.667-FinalReport-JFan,HWang,XQiu,ZLin.pdf)

A classical method to solve TSP is [Genetic Algorithm(for global search) combined with Lin-Kernighan heuristic(for local search)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.5310), while an emerging method is [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940). They are completely different approaches, but in this work two reduced version of each method is combined to achieve a satisfying result with limited computing resourse.

The key point of this RL-agumented-GA is the way to find out a proper 2-opt imporvement of current route to perform GA-LKH. While previously people try points based on order of distance, what we do here is following the order of probability(of good reward in the end), which is gained from a RL model. It turns out that for TSP with small size(around 50 points), our method is better than the traditional in terms of efficiency, and almost the same in terms of target value(the total cost of route).


The whole process is as below:
 - Generate n route as initial population
 	half random-half greedy  generation is applied, as completely random routes take more time to improve, and complete greedy routes lack diversity.
 	In practice, one of first-k nearest neighbors is selected randomly to develop a connection
 - Local search: Due to the complexity of programming a reduced version of Lin-Kernighan heuristic, namely two-optimal swap is implimented here. 
 - Breeding:
 - Update population:
 - Mutate:
 repeats until converge

[Function documentaion](./function.md)
