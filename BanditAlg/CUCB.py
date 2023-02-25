import numpy as np
import networkx as nx


class ArmBaseStruct(object):
    def __init__(self, armID):
        self.armID = armID
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward = 0.0
        self.p_max = 1

    def updateParameters(self, reward):
        self.totalReward += reward
        self.numPlayed += 1
        self.averageReward = self.totalReward / float(self.numPlayed)


class UCB1Struct(ArmBaseStruct):
    def getProb(self, allNumPlayed):
        if self.numPlayed == 0:
            return self.p_max
        else:
            p = self.totalReward / float(self.numPlayed) + np.sqrt(3 * np.log(allNumPlayed) / (2.0 * self.numPlayed))
            if p > self.p_max:
                p = self.p_max
                
            return p



class UCB1Algorithm:
    def __init__(self, G, P, seed_size, oracle, feedback='edge'):
        self.G = G
        self.trueP = P
        self.seed_size = seed_size
        self.oracle = oracle
        self.feedback = feedback
        self.arms = {}
        self.currentP = {}
        for (u, v) in self.G.edges():
            self.arms[(u, v)] = UCB1Struct((u, v))
            self.currentP[(u, v)] = 0
        self.TotalPlayCounter = 0

    def decide(self):
        self.TotalPlayCounter += 1
        for (u, v) in self.G.edges():
            self.currentP[(u, v)] = self.arms[(u, v)].getProb(self.TotalPlayCounter)
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S

    def updateParameters(self, S, live_nodes, live_edges, iter_, update=True):
        for u in live_nodes:  
            for (u, v) in self.G.edges(u):  
                if (u, v) in live_edges:
                    self.arms[(u, v)].updateParameters(reward=live_edges[(u, v)])
                else:
                    self.arms[(u, v)].updateParameters(reward=0)


    def getP(self):
        return self.currentP
