import pickle
import networkx as nx
import random
import matplotlib.pyplot as plt

file_address = './raw/flickrEdges.txt'
save_dir = '../datasets/Flickr/'

file_address = './raw/Cit-HepTh.txt'
save_dir = '../datasets/SameP/NetHEPT/'
key_node_num = 170
degree = {}
node_list = []

# count the degree
with open(file_address) as f:
    count = 0
    for line in f:
        if count > 4:  # skip the first four line
            # data = line.split(' ')  # Flickr
            data = line.split('\t')  # NetHEPT
            u = int(data[0])
            v = int(data[1])
            try:
                degree[v] += 1
            except:
                degree[v] = 1
            try:
                degree[u] += 1
            except:
                degree[u] = 1
        count += 1


for key in degree:
    node_list.append(key)


small_node_list = [node_list[i] for i in sorted(random.sample(range(len(node_list)), key_node_num))]

G = nx.Graph()
with open(file_address) as f:
    count = 0
    for line in f:
        
        if count > 4:
            # data = line.split(' ')  # Flickr
            data = line.split('\t')  # NetHEPT
            u = int(data[0])
            v = int(data[1])

            if v in small_node_list or u in small_node_list:
                G.add_edge(u, v)

        count += 1

print("G size : ", len(G.nodes()), len(G.edges()))
component = max(nx.connected_components(G), key=len)  # 选出最大联通分量
print("component", len(component))
Gc = G.subgraph(component).copy()
nodes = Gc.nodes()
edge = Gc.edges()
G = nx.DiGraph()
indegree = {}

with open(file_address) as f:
    count = 0
    for line in f:
        if count > 4:
            # data = line.split(' ')  # Flickr
            data = line.split('\t')  # NetHEPT
            u = int(data[0])
            v = int(data[1])

            if u in nodes and v in nodes:
                G.add_edge(u, v)
                try:
                    indegree[v] += 1
                except:
                    indegree[v] = 1
        count += 1

# nx.draw(G)
# plt.show()
print("G size : ", len(G.nodes()), len(G.edges()))
import os
path_graph_to_save = save_dir + f'n{len(G.nodes())}_e{len(G.edges())}_graph.G'
if os.path.exists(path_graph_to_save):
    print("graph in same size has been created.")
    exit(1)
pickle.dump(G, open(path_graph_to_save, "wb"))




