from Tool.priorityQueue import PriorityQueue as PQ
from Model.IC import runIC, runIC_getReward


# def Greedy(G, k, p):
#     """
#     Input: G -- networkx Graph object
#     k -- number of initial nodes needed
#     p -- propagation probability
#     Output: S -- initial set of k nodes to propagate
#     """
#     R = 1  # number of times to run Random Cascade
#     S = []  # set of selected nodes
#     # add node to S if achieves maximum propagation for current chosen + this node
#     for i in range(k):
#         s = PQ()  # priority queue
#         for v in G.nodes():
#             if v not in S:
#                 s.add_task(v, 0)  # initialize spread value
#                 for j in range(R):  # run R times Random Cascade
#                     [priority, count, task] = s.entry_finder[v]
#                     # TODO why do we add task with reward/R
#                     s.add_task(v, priority - runIC(G, S + [v], p)[0] / R)  # add normalized spread value
#         task, priority = s.pop_item()
#         S.append(task)
#     return S
import time
def Greedy(graph, seed_size, probability, iterations, reward_evaluate_func):
    print("mc_iterations", iterations)
    S = []
    for i in range(seed_size):
        influence = dict()  # influence for nodes not in S
        start_time = time.time()
        for v in graph.nodes():
            if v not in S:
                influence[v] = reward_evaluate_func(graph, S + [v], probability, iterations)  # G, S, P not G, P, S
        u, val = max(iter(influence.items()), key=lambda k_v: k_v[1])
        u = int(u)
        S.append(u)
        end_time = time.time()
        print("select one seed time", end_time - start_time)
    return S


class Greedy_class:
    def __init__(self, reward_evaluate_func, mc_iterations=1000):
        self.reward_evaluate_func = reward_evaluate_func
        self.mc_iterations = mc_iterations

    def greedy_func(self, graph, seed_size, probability):

        return Greedy(graph, seed_size, probability, self.mc_iterations, self.reward_evaluate_func)
