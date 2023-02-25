from math import log, inf, sqrt
from random import choice
import itertools


def flatten(T):
    if not isinstance(T, tuple):
        return (T,)
    elif len(T) == 0:
        return ()
    else:
        return flatten(T[0]) + flatten(T[1:])


class UCBStruct(object):
    def __init__(self, S):
        self.S = S  # 每次play的是seedsize大小的一个set，记为S
        self.totalReward = 0.0
        self.numPlayed = 0
        self.averageReward = 0.0
        self.upperBound = inf
        self.p_max = 1

    def updateParameters(self, reward, delta):
        self.totalReward += reward
        self.numPlayed += 1
        self.averageReward = self.totalReward / float(self.numPlayed)
        self.upperBound = self.averageReward + sqrt(2 * log(1 / delta) / float(self.numPlayed))
        print('arm:', self.S, 'with upper bound:', self.upperBound)

    def getUpperBound(self):
        return self.upperBound


class UCBAlgorithm:
    def __init__(self, G, P, seed_size, feedback='edge', delta=1/2000):
        self.G = G
        self.length = float(len(G.nodes))
        self.trueP = P
        self.seed_size = seed_size
        self.feedback = feedback
        self.arms = {}
        # 排列组合所有的nodes组合将他们分别创建一个UCB结构
        for nodes in itertools.combinations(list(self.G.nodes()), seed_size):
            self.arms[nodes] = UCBStruct(nodes)
        self.TotalPlayCounter = 0
        self.delta = delta

    def decide(self):
        self.TotalPlayCounter += 1
        max_upper_bound = 0
        opt_set = []
        for key, value in self.arms.items():
            current_upper_bound = value.getUpperBound()
            if current_upper_bound > max_upper_bound:
                opt_set.clear()
                opt_set.append(key)
                max_upper_bound = current_upper_bound
            elif current_upper_bound == max_upper_bound:
                opt_set.append(key)
        return list(choice(opt_set))

    def updateParameters(self, S, live_nodes, live_edges, iter_):
        reward = len(live_nodes)
        self.arms[tuple(S)].updateParameters(reward=reward, delta=self.delta)
