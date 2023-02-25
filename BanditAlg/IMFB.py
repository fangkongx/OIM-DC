import numpy as np
import random


class MFUserStruct:
    def __init__(self, featureDimension, lambda_, userID):
        self.userID = userID
        self.dim = featureDimension
        self.A = lambda_ * np.identity(n=self.dim)
        self.C = lambda_ * np.identity(n=self.dim)
        # self.b = np.array([random.random() for i in range(self.dim)])  # should be 0 vector
        # self.d = np.array([random.random() for i in range(self.dim)])  # should be 0 vector
        self.b = np.zeros(self.dim)
        self.d = np.zeros(self.dim)
        self.AInv = np.linalg.inv(self.A)
        self.CInv = np.linalg.inv(self.C)
        self.theta_out = np.dot(self.AInv, self.b)
        self.theta_in = np.dot(self.CInv, self.d)
        self.pta_max = 1

    def updateOut(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += articlePicked_FeatureVector * click
        self.AInv = np.linalg.inv(self.A)
        self.theta_out = np.dot(self.AInv, self.b)

    def updateIn(self, articlePicked_FeatureVector, click):
        self.C += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.d += articlePicked_FeatureVector * click
        self.CInv = np.linalg.inv(self.C)
        self.theta_in = np.dot(self.CInv, self.d)


class IMFBAlgorithm:
    def __init__(self, G, P, parameter, seed_size, dim, oracle, feedback='edge'):
        self.G = G
        # self.trueP = P
        # self.parameter = parameter
        self.oracle = oracle
        self.seed_size = seed_size
        self.q = 0.25
        self.dimension = dim
        print("self.dimension", self.dimension)

        self.it = 0

        self.feedback = feedback
        # self.list_loss = []
        self.currentP = {}
        self.users = {}  # Nodes
        lambda_ = 1
        for u in self.G.nodes():
            self.users[u] = MFUserStruct(self.dimension, lambda_, u)
            for v in self.G[u]:
                self.currentP[(u, v)] = 1 # random.random()  # TODO init as 1

    def decide(self):
        print(f"iter: {self.it}")
        for u in self.G.nodes():
            for (u, v) in self.G.edges(u):
                self.currentP[(u, v)] = self.getP(self.users[u], self.users[v], self.it)
        S = self.oracle(self.G, self.seed_size, self.currentP)
        return S

    def updateParameters(self, S, live_nodes, live_edges, it):
        self.it = it
        count = 0
        # loss_p = 0
        # loss_out = 0
        # loss_in = 0
        for u in live_nodes:
            for (u, v) in self.G.edges(u):
                if (u, v) in live_edges:
                    reward = live_edges[(u, v)]
                else:
                    reward = 0
                self.users[u].updateOut(self.users[v].theta_in, reward)
                self.users[v].updateIn(self.users[u].theta_out, reward)
                # self.currentP[(u, v)] = self.getP(self.users[u], self.users[v], it)

                # estimateP = np.dot(self.users[u].theta_out, self.users[v].theta_in)
                # trueP = self.trueP[u][v]['weight']
                # loss_p += np.abs(estimateP - trueP)
                # loss_out += np.linalg.norm(self.users[u].theta_out - self.parameter[u][1], ord=2)
                # loss_in += np.linalg.norm(self.users[v].theta_in - self.parameter[v][0], ord=2)
                count += 1
        # self.list_loss.append([loss_p / count, loss_out / count, loss_in / count])

    def getP(self, u, v, it):
        alpha_1 = 0.1
        alpha_2 = 0.1
        CB = alpha_1 * np.dot(np.dot(v.theta_in, u.AInv), v.theta_in) + alpha_2 * np.dot(np.dot(u.theta_out, v.CInv),
                                                                                         u.theta_out)
        prob = np.dot(u.theta_out, v.theta_in) + CB + 2 * np.power(self.q, it)
        if prob > 1:
            prob = 1
        if prob < 0:
            prob = 0
        return prob

    # def getLoss(self):
    #     return np.asarray(self.list_loss)
