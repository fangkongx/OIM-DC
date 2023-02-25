# 使用 degree discount 方法 获取K个种子节点


"""
1. 计算出每个节点的 degree

2. 选取K个最大ddv最大的节点
"""

from Model.DC import runDC_getReward
import time
import os
import numpy as np
import copy
folder_save_running_time = "running_time/"


def spread_multi_times(graph, probability, nodes_to_visit, nodes_visited_raw, influence_times):
    # 之前激活的节点
    nodes_visited = copy.deepcopy(nodes_visited_raw)
    nodes_visited.extend(nodes_to_visit)
    parents = {}
    T_node = {}
    probs_nodes = {}
    for node in nodes_to_visit:
        parents[node] = copy.deepcopy(nodes_visited_raw)  # 不再更新种子节点的概率
        T_node[node] = 0
        probs_nodes[node] = 1
    for node in graph.nodes():
        T_node[node] = 0
    nodes_to_visit_this_time = nodes_to_visit
    nodes_to_visit_next_time = []
    for i in range(influence_times):
        for node in nodes_to_visit_this_time:
            neighbor = graph[node]
            probs_this_node = probs_nodes[node]
            for neighbor_one in neighbor:
                if (node not in parents or neighbor_one not in parents[node]):
                    # add probability
                    if neighbor_one not in probs_nodes:
                        probs_nodes[neighbor_one] = 0
                    probs_nodes[neighbor_one] = probs_nodes[neighbor_one] + (1-probs_nodes[neighbor_one]) * probs_this_node * probability[(neighbor_one, T_node[neighbor_one])]
                    T_node[neighbor_one] += 1

                    if neighbor_one not in parents:
                        parents[neighbor_one] = []
                    # parents[neighbor_one].extend(parents[node])
                    parents[neighbor_one].append(node)

                    if neighbor_one not in nodes_visited:
                        nodes_to_visit_next_time.append(neighbor_one)
                        nodes_visited.append(neighbor_one)

        nodes_to_visit_this_time = copy.deepcopy(nodes_to_visit_next_time)
        nodes_to_visit_next_time = []

    influence_acc = 0
    for prob_node in probs_nodes:
        influence_acc += probs_nodes[prob_node]
    return influence_acc

def spread_multi_times_record_parents(graph, probability, nodes_to_visit, nodes_visited_raw, influence_times):
    # 之前激活的节点
    nodes_visited = copy.deepcopy(nodes_visited_raw)
    nodes_visited.extend(nodes_to_visit)
    parents = {}
    T_node = {}
    probs_nodes = {}
    for node in nodes_to_visit:
        parents[node] = copy.deepcopy(nodes_visited_raw)  # 不再更新种子节点的概率
        T_node[node] = 0
        probs_nodes[node] = 1
    for node in graph.nodes():
        T_node[node] = 0
    nodes_to_visit_this_time = nodes_to_visit
    nodes_to_visit_next_time = []
    for i in range(influence_times):
        for node in nodes_to_visit_this_time:
            neighbor = graph[node]
            probs_this_node = probs_nodes[node]
            for neighbor_one in neighbor:
                if (node not in parents or neighbor_one not in parents[node]):
                    # add probability
                    if neighbor_one not in probs_nodes:
                        probs_nodes[neighbor_one] = 0
                    probs_nodes[neighbor_one] = probs_nodes[neighbor_one] + (1-probs_nodes[neighbor_one]) * probs_this_node * probability[(neighbor_one, T_node[neighbor_one])]
                    T_node[neighbor_one] += 1

                    if neighbor_one not in parents:
                        parents[neighbor_one] = []
                    parents[neighbor_one].extend(parents[node])
                    parents[neighbor_one].append(node)

                    if neighbor_one not in nodes_visited:
                        nodes_to_visit_next_time.append(neighbor_one)
                        nodes_visited.append(neighbor_one)

        nodes_to_visit_this_time = copy.deepcopy(nodes_to_visit_next_time)
        nodes_to_visit_next_time = []

    influence_acc = 0
    for prob_node in probs_nodes:
        influence_acc += probs_nodes[prob_node]
    return influence_acc


def get_multi_level_influence_spread_multi_step(graph, nodes_num, probability, nodes_visited_raw, influence_times):
    # 给每个节点存储一下path
    influence_array = np.zeros((nodes_num, 2))
    for i, seed in enumerate(graph.nodes()):
        # print(f"initing the {i}th node")
        nodes_to_visit = [seed]
        influence_acc = spread_multi_times(graph, probability, nodes_to_visit, nodes_visited_raw, influence_times)
        influence_array[i][0] = seed
        influence_array[i][1] = influence_acc
    return influence_array

def degree_dis_multi_step(graph, probability, seed_size, influence_times=3):
    nodes_num = len(graph.nodes())
    S = []
    # TODO calculate ddv for each node v
    # influence_array = get_1_level_influence(graph, nodes_num, probability)
    print("before init cal_multi_step")
    influence_array = get_multi_level_influence_spread_multi_step(graph, nodes_num, probability, nodes_visited_raw=[], influence_times=influence_times)
    print("after init cal_multi_step")
    T_node = {}
    for v in graph.nodes():
        T_node[v] = 0
    # u_list = [9905111, 9710046, 110055, 210157, 101126]
    for i in range(seed_size):
        # select the seed with highest ddv
        """
        u, val = max(iter(influence.items()), key=lambda k_v: k_v[1])
        influence[u] = -1
        """
        index = np.argmax(influence_array[:, 1])
        seed_val = influence_array[index]
        u = int(seed_val[0])
        influence_array[index, 1] = -1
        S.append(u)
        # update the ddv of the u's neighbor
        for v in graph[u]:
            index_v = np.argwhere(influence_array[:, 0] == v)
            index_v = index_v[0][0]
            if influence_array[index, 1] > -0.5:
                # did not visited
                # TODO 需要将之前传过来的节点进行排除，但是速度估计会慢一些
                influence_acc = spread_multi_times(graph, probability, nodes_to_visit=[v], nodes_visited_raw=S, influence_times=influence_times)
                influence_array[index_v, 1] = probability[(v, T_node[v])] * influence_acc
                T_node[v] += 1
    # print("128 influence times: ", influence_times)
    assert len(S) == len(set(S)), "Error, one node selected more than once"
    return S

def new_time_activate(graph, probability, visited_nodes, nodes_to_visit_this_time, probs_nodes_to_visit_this_time):
    # activated node
    # 当前时刻之前已经被激活的节点（这部分不能被重新激活）；当前时刻将被激活的节点；当前时刻节点将被激活的概率；
    # 激活一下下一时刻将会被激活的节点
    # back 更新一下这一时刻已经被激活的节点，下一时刻将会被激活的节点，下一时刻将会被激活的节点的概率，
    # v []; node_to_visit_this_time [u]; probs_nodes_to_visit_this_time {u: 1};
    T_node = {}
    nodes_to_visit_next_time = []
    probs_nodes_to_visit_next_time = {}
    for node in nodes_to_visit_this_time:
        neighbors = graph[node]
        prob_node = probs_nodes_to_visit_this_time[node]  # dict
        for next_node in neighbors:
            # 不能激活 之前和本轮激活的节点；# 不考虑一个时刻 一个节点被重复激活的情况。
            if (next_node not in visited_nodes) and (next_node not in nodes_to_visit_this_time) and (next_node not in nodes_to_visit_next_time):
                nodes_to_visit_next_time.append(next_node)
                T_node[next_node] = 0
                new_prob_node = prob_node * probability[(next_node, T_node[next_node])]
                probs_nodes_to_visit_next_time[next_node] = new_prob_node
                T_node[next_node] += 1
            elif (next_node not in visited_nodes) and (next_node not in nodes_to_visit_this_time) and (next_node in nodes_to_visit_next_time):
                # print("T_node[next_node]", T_node[next_node])
                # print("next_node", next_node)
                # print("probability[(next_node, T_node[next_node])]", probability[(next_node, T_node[next_node])])
                new_prob_node = prob_node * probability[(next_node, T_node[next_node])]  # 如果是一个很低的概率来激活，仍然访问到了，让这里的T下降了，其实其prob会减小的。
                # print("new_prob_node", new_prob_node)
                # print("probs_nodes_to_visit_next_time[next_node]", probs_nodes_to_visit_next_time[next_node])
                probs_nodes_to_visit_next_time[next_node] = min(1, probs_nodes_to_visit_next_time[next_node] + (1-probs_nodes_to_visit_next_time[next_node])*new_prob_node)
                # print("probs_nodes_to_visit_next_time[next_node]", probs_nodes_to_visit_next_time[next_node])
                T_node[next_node] += 1
                # print("T_node[next_node]", T_node[next_node])
                # input()

    visited_nodes.extend(nodes_to_visit_this_time)
    return visited_nodes, nodes_to_visit_next_time, probs_nodes_to_visit_next_time

def get_1_level_influence(graph, nodes_num, probability):
    influence_array = np.zeros((nodes_num, 2))

    for i, seed in enumerate(graph.nodes()):
        visited = []
        nodes_to_visit = [seed]
        probs_nodes_to_visit = {seed: 1}
        visited_nodes, nodes_to_visit_next_time, probs_nodes_to_visit_next_time = new_time_activate(graph, probability, visited_nodes=visited, nodes_to_visit_this_time=nodes_to_visit, probs_nodes_to_visit_this_time=probs_nodes_to_visit)
        influence_acc = 0
        for node in nodes_to_visit_next_time:
            influence_acc += probs_nodes_to_visit_next_time[node]
        influence_array[i][0] = seed
        influence_array[i][1] = influence_acc
    return influence_array

def get_2_level_influence(graph, nodes_num, probability):
    influence_array = np.zeros((nodes_num, 2))

    for i, seed in enumerate(graph.nodes()):
        visited = []
        nodes_to_visit = [seed]
        probs_nodes_to_visit = {seed: 1}

        visited_nodes, nodes_to_visit_next_time, probs_nodes_to_visit_next_time = new_time_activate(graph, probability, visited_nodes=visited, nodes_to_visit_this_time=nodes_to_visit, probs_nodes_to_visit_this_time=probs_nodes_to_visit)
        influence_acc = 0
        for node in nodes_to_visit_next_time:
            influence_acc += probs_nodes_to_visit_next_time[node]
        visited_nodes2, nodes_to_visit_next_time2, probs_nodes_to_visit_next_time2 = new_time_activate(graph, probability,
                                                                                                    visited_nodes=visited_nodes,
                                                                                                    nodes_to_visit_this_time=nodes_to_visit_next_time,
                                                                                                    probs_nodes_to_visit_this_time=probs_nodes_to_visit_next_time)
        for node in nodes_to_visit_next_time:
            influence_acc += probs_nodes_to_visit_next_time2[node]
        influence_array[i][0] = seed
        influence_array[i][1] = influence_acc
    return influence_array

def get_multi_level_influence_one_seed(graph, nodes_num, probability, influence_times, seed, visited):
    # visited = []
    nodes_to_visit = [seed]
    probs_nodes_to_visit = {seed: 1}
    influence_acc = len(nodes_to_visit)

    for j in range(influence_times):
        visited, nodes_to_visit, probs_nodes_to_visit \
            = new_time_activate(graph, probability, visited_nodes=visited,
                                nodes_to_visit_this_time=nodes_to_visit,
                                probs_nodes_to_visit_this_time=probs_nodes_to_visit)
        for node in nodes_to_visit:
            influence_acc += probs_nodes_to_visit[node]
    return influence_acc


def get_multi_level_influence(graph, nodes_num, probability, influence_times):
    influence_array = np.zeros((nodes_num, 2))
    for i, seed in enumerate(graph.nodes()):
        # 初始化的时候不受其他种子影响 visited = []
        influence_acc = get_multi_level_influence_one_seed(graph, nodes_num, probability, influence_times, seed, visited=[])
        influence_array[i][0] = seed
        influence_array[i][1] = influence_acc
    return influence_array




def degree_discount(graph, probability, seed_size, iterations=None, influence_times=3):

    nodes_num = len(graph.nodes())
    S = []
    # TODO calculate ddv for each node v
    # influence_array = get_1_level_influence(graph, nodes_num, probability)
    print("before influence_array")
    influence_array = get_multi_level_influence(graph, nodes_num, probability, influence_times=influence_times)
    T_node = {}

    for v in graph.nodes():
        T_node[v] = 0
    # u_list = [9905111, 9710046, 110055, 210157, 101126]
    for i in range(seed_size):
        print("seed_size", seed_size)
        # select the seed with highest ddv
        """
        u, val = max(iter(influence.items()), key=lambda k_v: k_v[1])
        influence[u] = -1
        """
        index = np.argmax(influence_array[:, 1])
        seed_val = influence_array[index]
        u = int(seed_val[0])
        influence_array[index, 1] = -1
        S.append(u)

        # update the ddv of the u's neighbor
        for v in graph[u]:
            index_v = np.argwhere(influence_array[:, 0] == v)
            index_v = index_v[0][0]
            if influence_array[index, 1] > -0.5:
                # did not visited
                influence_acc = get_multi_level_influence_one_seed(graph, nodes_num, probability, influence_times=influence_times, seed=v, visited=S)
                influence_array[index_v, 1] = probability[(v, T_node[v])] * influence_acc
                T_node[v] += 1
    print("293 influence times: ", influence_times)
    assert len(S) == len(set(S)), "Error, one node selected more than once"
    return S
