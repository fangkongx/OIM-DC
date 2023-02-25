from random import choice, random, sample
import numpy as np
import networkx as nx
from BanditAlg.greedy import ArmBaseStruct
import time

class LinUCBUserStruct:
    def __init__(self, featureDimension, lambda_, userID, RankoneInverse=False):
        self.userID = userID
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.zeros(self.d) 

        self.RankoneInverse = RankoneInverse

        self.pta_max = 1

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.b += articlePicked_FeatureVector * click
        if self.RankoneInverse:
            temp = np.dot(self.AInv, articlePicked_FeatureVector)
            self.AInv = self.AInv - (np.outer(temp, temp)) / (
                        1.0 + np.dot(np.transpose(articlePicked_FeatureVector), temp))
        else:
            self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)  # not necessary
            self.AInv = np.linalg.inv(self.A)

        self.UserTheta = np.dot(self.AInv, self.b)

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def getProb(self, alpha, article_FeatureVector):
        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        if pta > self.pta_max:
            pta = self.pta_max
        return pta


class IMLinUCBAlgorithm:
    def __init__(self, G, P, parameter, seed_size, dim, oracle, alpha, lambda_, FeatureDic, FeatureScaling,
                 feedback='edge'):
        self.G = G
        self.oracle = oracle
        self.seed_size = seed_size
        self.dimension = dim
        self.alpha = alpha
        self.lambda_ = lambda_
        self.FeatureDic = FeatureDic
        self.FeatureScaling = FeatureScaling
        self.currentP = {}

        self.users = {}  
       
        start_time = time.time()

        for u in self.G.nodes():  # for each nodes set a IMLinUCB structure
            self.users[u] = LinUCBUserStruct(self.dimension, self.lambda_, u)
            for v in self.G[u]:
                self.currentP[(u, v)] = 1  # random()
       

    def decide(self):
        for u in self.G.nodes():
            for (u, v) in self.G.edges(u):
                featureVector = np.dot(self.FeatureScaling, np.array(self.FeatureDic[(u, v)]))
                self.currentP[(u, v)] = self.users[u].getProb(self.alpha, featureVector)
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S

    def updateParameters(self, S, live_nodes, live_edges, _iter):
        count = 0
        start_time = time.time()
       
        for u in live_nodes:  
            for (u, v) in self.G.edges(u): 
                featureVector = np.dot(self.FeatureScaling, np.array(self.FeatureDic[(u, v)]))
                if (u, v) in live_edges:  
                    reward = live_edges[(u, v)]
                else:
                    reward = 0
                self.users[u].updateParameters(featureVector, reward) 
                count += 1
        print(time.time() - start_time)

    def getCoTheta(self, userID):
        return self.users[userID].UserTheta

    def getP(self):
        return self.currentP


