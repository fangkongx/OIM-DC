import pickle

file_address = '../datasets/Small/'
graph = pickle.load(open(file_address + 'graph.G', 'rb'), encoding='latin1')

indegree = {}
for n in graph.nodes():
    indegree[n] = 0

for (u, v) in graph.edges():
    indegree[v] += 1

pickle.dump(indegree, open(file_address + 'indegree.dic', "wb"))
