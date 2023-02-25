from copy import deepcopy
# from random import random
import random
import networkx as nx
import numpy as np
import time
import tqdm
import multiprocessing as mp

random_0_to_1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def runDC_old(G, P, S):
    runDC_mine(G, P, S)

    """
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    T_node = {}

    for v in G.nodes():
        T_node[v] = 0

    T = deepcopy(S)  # copy already selected nodes
    E = {}

    i = 0
    while i < len(T):
        for v in G[T[i]]:
            if v not in T:  # if it wasn't selected yet
                # T[i] attempts to activate node v
                weight = P[(v, T_node[v])]
                if random.random() <= weight:
                    # if random.choice(random_0_to_1) <= weight:
                    T.append(v)
                    E[(v, T_node[v])] = 1
                else:
                    E[(v, T_node[v])] = 0
                T_node[v] += 1
        i += 1
    reward = len(T)
    return reward, E, T


def runDC(G, P, S):
    """
    special: record the  E(v, activate num) nums
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    #################
    graph = G
    seed_set = S
    real_P = P
    E = {}

    # Each node is associated with a number of attempts to activate
    T_node = {}
    for v in graph.nodes():
        T_node[v] = 0

    node_before = []
    last_time_activated_node = deepcopy(seed_set)
    i = 0
    while len(last_time_activated_node) > 0:  # 这个限制了 传播轮数<T
        i += 1
        new_activated_node = []
        for last_time_activated_node_each in last_time_activated_node:
            neighbor = graph[last_time_activated_node_each]
            for v in neighbor:  # 对于 所有节点
                if v not in last_time_activated_node and v not in node_before and v not in new_activated_node:
                    # 尝试将其激活
                    if random.random() <= real_P[(v, T_node[v])]:  # 已经被激活的次数
                        # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:
                        new_activated_node.append(v)  # 一个新的T
                        E[(v, T_node[v])] = 1
                    else:
                        E[(v, T_node[v])] = 0
                    T_node[v] += 1
        node_before.extend(last_time_activated_node)
        last_time_activated_node = new_activated_node
    reward = len(node_before)
    T = node_before

    return reward, E, T


def runDC_DILinUCB(G, P, S):
    """
    special: record the path existed or not
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    """
    T_node = {}  # record count of node try to active node v
    T = deepcopy(S)  # copy already selected nodes
    E = {}
    Active = nx.Graph()

    for v in G.nodes():
        T_node[v] = 0
        E[v] = [0] * len(G.nodes())

    i = 0
    while i < len(T):
        for v in G[T[i]]:
            if v not in T:
                # T[i] attempts to activate node v
                weight = P[(v, T_node[v])]
                if random.random() <= weight:
                    # if random.choice(random_0_to_1) <= weight:
                    Active.add_edge(T[i], v)
                    T.append(v)
                T_node[v] += 1
        i += 1
    reward = len(T)
    # reward LinUCB has_path then
    for u in S:
        for (idx, v) in enumerate(G.nodes()):
            try:
                if nx.has_path(Active, u, v):
                    E[u][idx] = 1
            except:
                E[u][idx] = 0

    return reward, E, T


# TODO 1
def runDC_getReward(graph, real_P, seed_set, iterations):
    rewards = []
    # pbar = tqdm.tqdm(range(iterations))
    # for j in pbar:
    #     pbar.set_description("run DC reward %d" % j)
    for j in range(iterations):
        rewards.append(0)

        # Each node is associated with a number of attempts to activate
        T_node = {}
        start = time.time()

        for v in graph.nodes():
            T_node[v] = 0
        end = time.time()
        # print("time init", end - start)
        """
        random.seed(224)
        T = deepcopy(seed_set)
        i = 0
        while i < len(T):  # 这个限制了 传播轮数<T
            # T[i] attempts to activate node v
            neighbor = graph[T[i]]

            for v in neighbor:  # 对于 所有节点
                if v not in T:
                    # active success
                    if random.random() <= real_P[(v, T_node[v])]:
                    # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:

                        T.append(v)  # 一个新的T
                    T_node[v] += 1
            i += 1
        print("T", len(T))
        print("node_before", len(node_before))
        print("T == node_before", T == node_before)
        """

        for v in graph.nodes():
            T_node[v] = 0

        node_before = []
        last_time_activated_node = deepcopy(seed_set)
        i = 0
        while len(last_time_activated_node) > 0:  # 这个限制了 传播轮数<T

            i += 1
            new_activated_node = []
            for last_time_activated_node_each in last_time_activated_node:
                neighbor = graph[last_time_activated_node_each]
                for v in neighbor:  # 对于 所有节点
                    if v not in last_time_activated_node and v not in node_before and v not in new_activated_node:
                        # 尝试将其激活
                        if random.random() <= real_P[(v, T_node[v])]:  # 已经被激活的次数
                            # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:
                            new_activated_node.append(v)  # 一个新的T

                        T_node[v] += 1
            node_before.extend(last_time_activated_node)
            last_time_activated_node = new_activated_node
        rewards[j] = len(node_before)  # 传播到的点数太多了
        # print("rewards", rewards[j])

    return np.mean(rewards)


def one_simulate(graph, real_P, seed_set):
    T_node = {}
    for v in graph.nodes():
        T_node[v] = 0
    node_before = []
    last_time_activated_node = deepcopy(seed_set)
    i = 0
    while len(last_time_activated_node) > 0:
        i += 1
        new_activated_node = []
        for last_time_activated_node_each in last_time_activated_node:
            neighbor = graph[last_time_activated_node_each]
            for v in neighbor:
                if v not in last_time_activated_node and v not in node_before and v not in new_activated_node:
                    if random.random() <= real_P[(v, T_node[v])]:
                        new_activated_node.append(v)

                    T_node[v] += 1
        node_before.extend(last_time_activated_node)
        last_time_activated_node = new_activated_node
    return len(node_before)


# TODO 2
def runDC_getReward_with_multi_process(graph, real_P, seed_set, iterations, num_cores=25):
    # print("seed_set", seed_set)
    # num_cores = 25  # int(mp.cpu_count())
    # print("num_cores", num_cores)
    with mp.Pool(num_cores) as pool:  # 基本不花费时间
        # rewards = [pool.apply_async(one_simulate, args=(graph, real_P, seed_set)) for i in range(iterations)]  # 基本不花费时间
        # rewards = [p.get() for p in rewards]
        start_time = time.time()
        a = [(graph, real_P, seed_set) for i in range(iterations)]
        end_time = time.time()
        # print("copy time", end_time - start_time)
        start_time = time.time()
        rewards = pool.starmap(one_simulate, a)
        end_time = time.time()
        # print("reward 同步时间", end_time - start_time)
        # print(rewards)
    # print("reward", len(rewards))
    return np.mean(rewards)


# for i in pool.imap_unordered(f, range(10)):
#     print(i)

def runDC_getReward_with_restricted_step(graph, real_P, seed_set, iterations, restricted_step):
    rewards = []
    for j in range(iterations):
        rewards.append(0)

        # Each node is associated with a number of attempts to activate
        T_node = {}
        start = time.time()

        for v in graph.nodes():
            T_node[v] = 0
        end = time.time()
        # print("time init", end - start)

        node_before = []
        last_time_activated_node = deepcopy(seed_set)
        i = 0
        while len(last_time_activated_node) > 0 and i < restricted_step:  # 这个限制了 传播轮数<T

            i += 1
            new_activated_node = []
            for last_time_activated_node_each in last_time_activated_node:
                neighbor = graph[last_time_activated_node_each]
                for v in neighbor:  # 对于 所有节点
                    if v not in last_time_activated_node and v not in node_before and v not in new_activated_node:
                        # 尝试将其激活
                        if random.random() <= real_P[(v, T_node[v])]:  # 已经被激活的次数
                            # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:
                            new_activated_node.append(v)  # 一个新的T

                        T_node[v] += 1
            node_before.extend(last_time_activated_node)  # 这个reward 如果受限有问题
            last_time_activated_node = new_activated_node
        node_before.extend(last_time_activated_node)
        rewards[j] = len(node_before)  # 传播到的点数太多了
        # print("rewards", rewards[j])

    return np.mean(rewards)


def runDC_getReward_static_greedy(graph, real_P, seed_set, iterations, random_num_static_all):
    rewards = []
    for j in range(iterations):
        rewards.append(0)

        # Each node is associated with a number of attempts to activate
        T_node = {}
        start = time.time()

        for v in graph.nodes():
            T_node[v] = 0
        end = time.time()
        # print("time init", end - start)
        """
        random.seed(224)
        T = deepcopy(seed_set)
        i = 0
        while i < len(T):  # 这个限制了 传播轮数<T
            # T[i] attempts to activate node v
            neighbor = graph[T[i]]

            for v in neighbor:  # 对于 所有节点
                if v not in T:
                    # active success
                    if random.random() <= real_P[(v, T_node[v])]:
                    # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:

                        T.append(v)  # 一个新的T
                    T_node[v] += 1
            i += 1
        print("T", len(T))
        print("node_before", len(node_before))
        print("T == node_before", T == node_before)
        """

        for v in graph.nodes():
            T_node[v] = 0

        node_before = []
        last_time_activated_node = deepcopy(seed_set)
        i = 0
        while len(last_time_activated_node) > 0:  # 这个限制了 传播轮数<T

            i += 1
            new_activated_node = []
            for last_time_activated_node_each in last_time_activated_node:
                neighbor = graph[last_time_activated_node_each]
                for v in neighbor:  # 对于 所有节点
                    if v not in last_time_activated_node and v not in node_before and v not in new_activated_node:
                        # 尝试将其激活
                        if random_num_static_all[j][(last_time_activated_node_each, v)] <= real_P[
                            (v, T_node[v])]:  # 已经被激活的次数
                            # if random.choice(random_0_to_1) <= real_P[(v, T_node[v])]:
                            new_activated_node.append(v)  # 一个新的T

                        T_node[v] += 1
            node_before.extend(last_time_activated_node)
            last_time_activated_node = new_activated_node
        rewards[j] = len(node_before)  # 传播到的点数太多了
        # print("rewards", rewards[j])

    return np.mean(rewards)
