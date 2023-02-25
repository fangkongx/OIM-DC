import pickle
import random
import numpy as np
edgeDic = {}

# --------------------------- DC Interval P ----------------------------------- #
# dataset = 'original/NetHEPT'
# dataset = 'SameP/NetHEPT'
# graph_name = "n41_e408_graph.G"
# graph_name = "n85_e403_graph.G"
# graph_name = "n112_e957_graph.G"
# graph_name = "n198_e2270_graph.G"
# graph_name = "n268_e2756_graph.G"
# dataset = "Generate"
# graph_name = "ER_n20_p0.2_e88.G"
# graph_name = "ER_n50_p0.2_e508.G"
# graph_name = "ER_n100_p0.2_e1988.G"
# graph_name = "ER_n150_p0.2_e4463.G"
# graph_name = "ER_n200_p0.2_e7927.G"
# dataset = "original/facebook/"
# graph_name = "graph.G"
# dataset = 'SameP/NetHEPT'
# dataset = 'Generate'
# # graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G", "n4798_e80428_graph.G", "n7487_e126984_graph.G", "n17366_e283866_graph.G", "n21695_e327738_graph.G"]
# graph_name_list = ["n2181_e32213_graph.G", "n3418_e54843_graph.G"]
# graph_name_list = ["ER_n20_p0.2_e70.G"]
# graph_name_list = ["ER_n100_p0.05_e480.G", "ER_n200_p0.05_e1993.G", "ER_n400_p0.05_e7912.G", "ER_n800_p0.05_e31999.G", "ER_n1600_p0.05_e127348.G"]
# low_high_pairs = [(0.01, 0.05), (0.1, 0.5)]



# node_num_list = [10, 20, 40, 50, 80, 100]
# p_list = [0.4, 0.2, 0.1, 0.05]
# graph_name_list = [""]
# low_high_pairs = [(0.01, 0.05)]
# for graph_name in graph_name_list:
#     for (low, high) in low_high_pairs:
#         G = pickle.load(open('../datasets/' + dataset + f'/{graph_name}', 'rb'))
#         for (u, in_degree) in G.in_degree:
#             prob_pool = -np.sort(-np.random.uniform(low, high, in_degree))
#             for index in range(in_degree):
#                 edgeDic[(u, index)] = prob_pool[index]
#         # print('prob dic:', edgeDic)
#         pickle.dump(edgeDic, open('../datasets/' + dataset + '/' + str(low) + '-' + str(high) + f'-{graph_name[:-2]}-Probability.dic', "wb"))
#         print('../datasets/' + dataset + '/' + str(low) + '-' + str(high) + f'-{graph_name[:-2]}-Probability.dic')
# --------------------------- IC/DC Fix P  ----------------------------------- #
# graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G"]
# dataset = 'Fixed_p/NetHEPT'
# prob_list = [0.2, 0.5, 0.8]
#
#
#
#
# graph_name_list = ['ER_n20_p0.4_e156.G', 'ER_n40_p0.4_e621.G', 'ER_n50_p0.4_e976.G',
#                    'ER_n20_p0.2_e70.G', 'ER_n40_p0.2_e290.G', 'ER_n50_p0.2_e505.G',
#                    'ER_n20_p0.1_e35.G', 'ER_n40_p0.1_e139.G', 'ER_n50_p0.1_e240.G',
#                    'ER_n20_p0.05_e12.G', 'ER_n40_p0.05_e91.G', 'ER_n50_p0.05_e113.G']
# dataset = 'Fixed_p/Generate'
# prob_list = [0.2, 0.5, 0.8]

# graph_name_list = ["ER_n20_p0.05_e12.G", "ER_n20_p0.2_e70.G"]
# dataset = 'Fixed_p/Generate'
# prob_list = [0.03]

graph_name_list = ["graph.G"]
dataset = 'SameP/Flickr'
prob_list = [0.2, 0.5, 0.8]

# graph_name_list = ['ER_n20_p0.05_e12.G', 'ER_n20_p0.1_e35.G', 'ER_n20_p0.2_e70.G', 'ER_n20_p0.4_e156.G',]
# dataset = "Generate"
# prob_list = [0.02, 0.05, 0.08, 0.1]


# graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G"]
# dataset = 'SameP/NetHEPT'
# prob_list = [0.02, 0.05, 0.08, 0.1]
#
# graph_name_list = ["graph.G"]
# dataset = 'original/facebook'
# prob_list = [0.02, 0.05, 0.08, 0.1]

for prob in prob_list:
    for graph_name in graph_name_list:
        print('../datasets/' + dataset + '/' + str(prob) + f'-{graph_name[:-2]}-IC-Probability.dic')
        print('../datasets/' + dataset + '/' + str(prob) + f'-{graph_name[:-2]}-Probability.dic')

        G = pickle.load(open('../datasets/' + dataset + f'/{graph_name}', 'rb'))
        edgeDic = {}
        for (u, v) in G.edges():
            edgeDic[(u, v)] = prob


        # print('prob dic:', edgeDic)
        pickle.dump(edgeDic, open('../datasets/' + dataset + '/' + str(prob) + f'-{graph_name[:-2]}-IC-Probability.dic', "wb"))

        edgeDic = {}
        for (u, in_degree) in G.in_degree:
            for index in range(in_degree):
                edgeDic[(u, index)] = prob
        # print('prob dic:', edgeDic)
        pickle.dump(edgeDic, open('../datasets/' + dataset + '/' + str(prob) + f'-{graph_name[:-2]}-Probability.dic', "wb"))
        # input()
