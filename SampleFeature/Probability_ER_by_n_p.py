import pickle
import random
import numpy as np
edgeDic = {}

# --------------------------- DC Interval P ----------------------------------- #

dataset = 'SameP/NetHEPT'
dataset = 'Generate'
# graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G", "n4798_e80428_graph.G", "n7487_e126984_graph.G", "n17366_e283866_graph.G", "n21695_e327738_graph.G"]


import glob
node_num_list = [20, 40, 50]
p_list = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
graph_name_list = []
import os
print("afdfads")
print(os.getcwd())
os.chdir(f"../datasets/{dataset}/")

for node_num in node_num_list:
    for p in p_list:
        graph_name = glob.glob(f"ER_n{node_num}_p{p}*.G")
        print(graph_name)
        assert len(graph_name) == 1
        graph_name = graph_name[0]
        graph_name_list.append(graph_name)

os.chdir("../../SampleFeature")
print(graph_name_list)
# low_high_pairs = [(0.01, 0.05)]
# low_high_pairs = [(0.1, 0.5)]
low_high_pairs = [(0.01, 0.05), (0.03, 0.07), (0.05, 0.09)]
low_high_pairs = [(0.01, 0.1)]

for graph_name in graph_name_list:
    for (low, high) in low_high_pairs:
        G = pickle.load(open('../datasets/' + dataset + f'/{graph_name}', 'rb'))
        for (u, in_degree) in G.in_degree:
            prob_pool = -np.sort(-np.random.uniform(low, high, in_degree))
            for index in range(in_degree):
                edgeDic[(u, index)] = prob_pool[index]
        # print('prob dic:', edgeDic)
        pickle.dump(edgeDic, open('../datasets/' + dataset + '/' + str(low) + '-' + str(high) + f'-{graph_name[:-2]}-Probability.dic', "wb"))
        print('../datasets/' + dataset + '/' + str(low) + '-' + str(high) + f'-{graph_name[:-2]}-Probability.dic')
