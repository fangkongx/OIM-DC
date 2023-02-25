import pickle
import random
import numpy as np

featureDic = {}
"""
# dataset = 'NetHEPT'
dataset = 'Generate'
graph_name = 'ER_n20_p0.2_e88.G'
# graph_name = "ER_n50_p0.2_e508.G"
# graph_name = "ER_n100_p0.2_e1988.G"
# graph_name = "ER_n150_p0.2_e4463.G"
# graph_name = "ER_n200_p0.2_e7927.G"
graph_name = "n786_e10166_graph.G"
graph_name = "n786_e10166_graph.G"
graph_name_list = ["ER_n20_p0.05_e12.G", "ER_n20_p0.2_e70.G"]
dataset = "Fixed_p/Generate"
# graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G"]
# dataset = "Fixed_p/NetHEPT"


graph_name_list = ['ER_n20_p0.4_e156.G', 'ER_n40_p0.4_e621.G', 'ER_n50_p0.4_e976.G',
                   'ER_n20_p0.2_e70.G', 'ER_n40_p0.2_e290.G', 'ER_n50_p0.2_e505.G',
                   'ER_n20_p0.1_e35.G', 'ER_n40_p0.1_e139.G', 'ER_n50_p0.1_e240.G',
                   'ER_n20_p0.05_e12.G', 'ER_n40_p0.05_e91.G', 'ER_n50_p0.05_e113.G']
dataset = "Fixed_p/Generate"



graph_name_list = ['graph.G']
dataset = "original/facebook"

"""
# graph_name_list = ['ER_n20_p0.05_e12.G', 'ER_n20_p0.1_e35.G', 'ER_n20_p0.2_e70.G', 'ER_n20_p0.4_e156.G',]
# dataset = "Generate"

graph_name_list = ["graph.G"]
dataset = 'SameP/Flickr'

dimension = 5

for graph_name in graph_name_list:
    G = pickle.load(open(f'../datasets/{dataset}/{graph_name}', 'rb'))
    i = 0
    for (u, v) in G.edges():
        featureVector = np.array([np.random.normal(-1, 1, 1)[0] for i in range(dimension)])
        l2_norm = np.linalg.norm(featureVector, ord=2)
        featureVector = featureVector / l2_norm
        featureDic[u, v] = [*featureVector]
        i += 1
        # print(i)
    # print('fv dic:', featureDic)
    print(f'../datasets/{dataset}/{graph_name[:-2]}-edgeFeatures-dimension{dimension}.dic')
    pickle.dump(featureDic, open(f'../datasets/{dataset}/{graph_name[:-2]}-edgeFeatures-dimension{dimension}.dic', "wb"))


