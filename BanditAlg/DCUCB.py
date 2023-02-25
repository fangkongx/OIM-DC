import random
import numpy as np
from Model.DC import runDC



class edge_base:
    def __init__(self, number_of_observations, total_reward, empirical_mean, radius):
        self.observations = number_of_observations
        self.mean = empirical_mean
        self.total_reward = total_reward
        self.p_max = 1
        self.radius = radius

    def updateEdge(self, myReward):
        self.total_reward += myReward
        self.observations += 1
        self.mean = self.total_reward / float(self.observations)

    def get_upper_estimation(self, all_rounds):
        if self.observations == 0:
            return 1
        else:
            upper_p = self.mean + self.radius * np.sqrt(
                3 * np.log(all_rounds) / (2.0 * self.observations))
            if upper_p > self.p_max:
                upper_p = self.p_max
            return upper_p


class DC_UCB_alg:
    def __init__(self, graph, indegree, probabilities, seed_size, oracle, radius=0.1):
        self.G = graph
        self.real_P = probabilities
        self.radius = radius
        self.seed_size = seed_size
        self.oracle = oracle
        self.all_rounds = 0
        self.currentP = {}
        self.indegree = indegree
        self.edge_bases = {}

        for n in self.G.nodes():
            try:
                for i in range(self.indegree[n]):
                    self.edge_bases[(n, i)] = edge_base(0, 0, 0, self.radius)
                    self.currentP[(n, i)] = self.edge_bases[(n, i)].get_upper_estimation(self.all_rounds)
            except:
                continue

    def select_seed(self):
        S = self.oracle(self.G, self.currentP, self.seed_size)
       

        return S

    def simulate(self, S):
        reward, live_edges, live_nodes = runDC(self.G, self.real_P, S)
      
        return live_edges, reward

    def runReal(self, S):
        reward, live_edges, live_nodes = runReal_DC(self.G, S)
       
        return live_edges, reward

    def update(self, observed_probabilities):
        
        for key, val in zip(observed_probabilities.keys(), observed_probabilities.values()):
            node = key[0]
            index = key[1]
            reward = val
            self.edge_bases[(node, index)].updateEdge(reward)

      
        self.all_rounds += 1

      
        for n in self.G.nodes():
            if self.indegree[n] > 0:
                self.currentP[(n, 0)] = self.edge_bases[(n, 0)].get_upper_estimation(self.all_rounds)
                for i in range(self.indegree[n] - 1):
                    self.currentP[(n, i + 1)] = min(self.edge_bases[(n, i + 1)].get_upper_estimation(self.all_rounds),
                                                    self.currentP[(n, i)])

    
