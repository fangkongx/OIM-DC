import numpy as np
from Model.DC import runDC



class node_base:
    def __init__(self, number_of_observations, total_reward, empirical_mean, alpha):
        self.observations = number_of_observations
        self.mean = empirical_mean
        self.total_reward = total_reward
        self.p_max = 1
        self.alpha = alpha

    def updateNode(self, myReward):
        self.total_reward += myReward
        self.observations += 1
        self.mean = self.total_reward / float(self.observations)

    def get_upper_estimation(self, all_rounds):
        if self.observations == 0:
            return 1
        else:
            upper_p = self.mean + self.alpha * np.sqrt(
                3 * np.log(all_rounds) / (2.0 * self.observations))
            if upper_p > self.p_max:
                upper_p = self.p_max
            return upper_p


class CMAB_average_alg:
    def __init__(self, graph, indegree, probabilities, seed_size, oracle, alpha):
        self.G = graph
        self.real_P = probabilities
        self.seed_size = seed_size
        self.oracle = oracle
        self.alpha = alpha
        self.all_rounds = 0
        self.currentP = {}
        self.indegree = indegree
        self.node_bases = {}

        for n in self.G.nodes():
            self.node_bases[n] = node_base(0, 0, 0, alpha)
            self.currentP[n] = self.node_bases[n].get_upper_estimation(self.all_rounds)

    def select_seed(self):
       
        self.all_rounds += 1

        
        for n in self.G.nodes():
            self.currentP[n] = self.node_bases[n].get_upper_estimation(self.all_rounds)

        S = self.oracle(self.currentP, self.seed_size)
        

        return S

    def simulate(self, S):
        reward, live_edges, live_nodes = runDC(self.G, self.real_P, S)
        return live_edges, reward

    def runReal(self, S):
        reward, live_edges, live_nodes = runReal_DC(self.G, S)
        
        return live_edges, reward

    def update(self, S, reward):
      
        for node in S:
            reward = float(reward) / (self.seed_size * (len(self.G.nodes)))
            self.node_bases[node].updateNode(reward)
          
