import numpy as np
import time
import os


def get_out_degree(graph, nodes_num):
    influence_array = np.zeros((nodes_num, 2))
    for i, (node, out_degree) in enumerate(graph.out_degree):
        influence_array[i][0] = node
        influence_array[i][1] = out_degree
    return influence_array


def degree_count(graph, probability, seed_size, iterations=None, influence_times=None):
    nodes_num = len(graph.nodes())
    S = []
    influence_array = get_out_degree(graph, nodes_num)
    for i in range(seed_size):
        # select the seed with highest ddv
        index = np.argmax(influence_array[:, 1])
        seed_val = influence_array[index]
        u = int(seed_val[0])
        influence_array[index, 1] = -1
        S.append(u)
    assert len(S) == len(set(S)), "Error, one node selected more than once"
    return S