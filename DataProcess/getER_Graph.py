import networkx as nx
import matplotlib.pyplot as plt
import time
import pickle
start = time.time()

node_num = 1600  # 200 400 800 1600
p = 0.05

node_num_list = [10, 20, 40, 50, 80, 100]
p_list = [0.4, 0.2, 0.1, 0.05]
p_list = [0.6, 0.8]
for node_num in node_num_list:
    for p in p_list:
        ER = nx.random_graphs.erdos_renyi_graph(node_num, p, directed=True) 
      
        edge_num = len(ER.edges())
        print(len(ER.nodes()), len(ER.edges()))
        import os
        os.makedirs("../datasets/Generate/", exist_ok=True)
        print(f'../datasets/Generate/ER_n{node_num}_p{p}_e{edge_num}.G')
        pickle.dump(ER, open(f'../datasets/Generate/ER_n{node_num}_p{p}_e{edge_num}.G', "wb"))
        print('Built Small graph G', time.time() - start, 's')
