# from Model.DC import runDC_getReward
import time
import os
from Model.DC import runDC_getReward_with_multi_process

def Greedy(graph, probability, seed_size, iterations, reward_evaluate_func):

    S = []
    for i in range(seed_size):
        influence = dict()  # influence for nodes not in S

        for v in graph.nodes():
            if v not in S:
                influence[v] = reward_evaluate_func(graph, probability, S + [v], iterations)
        u, val = max(iter(influence.items()), key=lambda k_v: k_v[1])
        u = int(u)
        S.append(u)
    return S

class Greedy_class:
    def __init__(self, reward_evaluate_func, mc_iterations=1000):
        self.reward_evaluate_func = reward_evaluate_func
        self.mc_iterations = mc_iterations

    def greedy_func(self, graph, probability, seed_size):
        return Greedy(graph, probability, seed_size, self.mc_iterations, self.reward_evaluate_func)

