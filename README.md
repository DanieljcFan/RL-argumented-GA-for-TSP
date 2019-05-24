# RL-agumented-GeneticAlgorithm-for-TSP
A project to find solution of a TSP problem, in cooperate with Grape Qiao, Han Wang, and Ziyan Lin. [paper heter](./553.667-FinalReport-JFan,HWang,XQiu,ZLin.pdf)

TSP is a famous NP-hard problem, and Genetic Algorithm combined with Lin-Kernighan heuristic has been shown to be one of the best approximate solutions. Still, here is a space for imporvement. In this project Reinforcement Learning is applied to improve the efficiency of local search and thus improve the total algorithm. 

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
