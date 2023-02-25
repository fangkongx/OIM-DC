# Independent cascade model for influence propagation
from copy import deepcopy
from random import random
import networkx as nx
import numpy as np

def runIC(G, S, P):
    """ Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    T = deepcopy(S)  # copy already selected nodes
    E = {}

    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node
            # if v is already active, T[i] still need to stimulate
            if random() <= P[(T[i], v)]:
                if v not in T:  # if it wasn't selected yet
                    T.append(v)
                E[(T[i], v)] = 1
            else:
                E[(T[i], v)] = 0
        i += 1
    reward = len(T)

    return reward, T, E


def runIC_getReward(G, S, P, iterations):
    rewards = []
    for j in range(iterations):
        reward, T, E = runIC(G, S, P)
        rewards.append(reward)
    return np.mean(rewards)

def runIC_DILinUCB(G, P, S):
    """ Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    T = deepcopy(S)  # copy already selected nodes
    E = {}
    Active = nx.Graph()  # store all the active edge, find path then
    for v in G.nodes():
        E[v] = [0] * len(G.nodes())
    # 将其转为一个向量
    i = 0
    while i < len(T):
        for v in G[T[i]]:
            # for DI, if v is already active, T[i] not need to stimulate
            if v not in T:
                # T[i] try to stimulate v
                if random() <= P[(T[i], v)]:
                    # stimulate success
                    Active.add_edge(T[i], v)
                    T.append(v)
        i += 1
    reward = len(T)

    for u in S:
        for (idx, v) in enumerate(G.nodes()):
            try:
                if nx.has_path(Active, u, v):  # has_edge
                    E[u][idx] = 1
            except:
                E[u][idx] = 0

    return reward, E, T
