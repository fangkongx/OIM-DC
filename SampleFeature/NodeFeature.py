import pickle

# save_dir = '../datasets/Generate/'
# graph_name = "ER_n100_p0.2_e1988.G"
# graph_name = "ER_n150_p0.2_e4463.G"
# graph_name = "ER_n200_p0.2_e7927.G"
# graph_name = "ER_n20_p0.2_e88.G"
# graph_name = "ER_n50_p0.2_e508.G"
# save_dir = '../datasets/SameP/NetHEPT/'
# graph_name_list = ["n41_e408_graph.G", "n85_e403_graph.G", "n112_e957_graph.G", "n198_e2270_graph.G", "n268_e2756_graph.G"]

# save_dir = "../datasets/original/NetHEPT/"
# save_dir = "../datasets/original/facebook/"
# graph_name_list = ["graph.G"]
"""
save_dir = '../datasets/SameP/NetHEPT/'
save_dir = '../datasets/Generate/'
# graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G", "n4798_e80428_graph.G", "n7487_e126984_graph.G", "n17366_e283866_graph.G", "n21695_e327738_graph.G"]
graph_name_list = ["n2181_e32213_graph.G", "n3418_e54843_graph.G"]
graph_name_list = ["ER_n20_p0.2_e70.G"]
graph_name_list = ["ER_n100_p0.05_e480.G", "ER_n200_p0.05_e1993.G", "ER_n400_p0.05_e7912.G", "ER_n800_p0.05_e31999.G", "ER_n1600_p0.05_e127348.G"]
graph_name_list = ['ER_n10_p0.4_e38.G', 'ER_n20_p0.4_e156.G', 'ER_n40_p0.4_e621.G', 'ER_n50_p0.4_e976.G', 'ER_n80_p0.4_e2470.G', 'ER_n100_p0.4_e4029.G', 'ER_n10_p0.2_e17.G', 'ER_n20_p0.2_e70.G', 'ER_n40_p0.2_e290.G', 'ER_n50_p0.2_e505.G', 'ER_n80_p0.2_e1277.G', 'ER_n100_p0.2_e2008.G', 'ER_n10_p0.1_e2.G', 'ER_n20_p0.1_e35.G', 'ER_n40_p0.1_e139.G', 'ER_n50_p0.1_e240.G', 'ER_n80_p0.1_e629.G', 'ER_n100_p0.1_e996.G', 'ER_n10_p0.05_e3.G', 'ER_n20_p0.05_e12.G', 'ER_n40_p0.05_e91.G', 'ER_n50_p0.05_e113.G', 'ER_n80_p0.05_e346.G', 'ER_n100_p0.05_e480.G']
save_dir = '../datasets/Fixed_p/NetHEPT/'
graph_name_list = ["n786_e10166_graph.G", "n1310_e17026_graph.G"]
save_dir = '../datasets/Fixed_p/Generate/'
graph_name_list = ["ER_n20_p0.05_e12.G", "ER_n20_p0.2_e70.G"]

graph_name_list = ['ER_n20_p0.4_e156.G', 'ER_n40_p0.4_e621.G', 'ER_n50_p0.4_e976.G',
                   'ER_n20_p0.2_e70.G', 'ER_n40_p0.2_e290.G', 'ER_n50_p0.2_e505.G',
                   'ER_n20_p0.1_e35.G', 'ER_n40_p0.1_e139.G', 'ER_n50_p0.1_e240.G',
                   'ER_n20_p0.05_e12.G', 'ER_n40_p0.05_e91.G', 'ER_n50_p0.05_e113.G']

save_dir = '../datasets/Generate/'

graph_name_list = [
    'ER_n20_p0.8_e311.G', 'ER_n20_p0.6_e235.G', 'ER_n20_p0.4_e156.G', 'ER_n20_p0.2_e70.G', 'ER_n20_p0.1_e35.G', 'ER_n20_p0.05_e12.G',
    'ER_n40_p0.8_e1250.G', 'ER_n40_p0.6_e907.G', 'ER_n40_p0.4_e621.G', 'ER_n40_p0.2_e290.G', 'ER_n40_p0.1_e139.G', 'ER_n40_p0.05_e91.G',
    'ER_n50_p0.8_e1930.G', 'ER_n50_p0.6_e1501.G', 'ER_n50_p0.4_e976.G', 'ER_n50_p0.2_e505.G', 'ER_n50_p0.1_e240.G', 'ER_n50_p0.05_e113.G',
]



save_dir = '../datasets/Generate/'

graph_name_list = ['ER_n20_p0.05_e12.G', 'ER_n20_p0.1_e35.G', 'ER_n20_p0.2_e70.G', 'ER_n20_p0.4_e156.G',]
"""
save_dir = "../datasets/SameP/NetHEPT/"
graph_name_list = ["graph.G"]
for graph_name in graph_name_list:
    nodeDic = {}
    edgeDic = {}
    degree = []
    G = pickle.load(open(save_dir + graph_name, 'rb'))
    n = len(G.nodes())
    for index, u in enumerate(G.nodes()):
        fv = [0 for i in range(n)]
        fv[index] = 1
        nodeDic[u] = [fv, fv]
        # print("fv", fv)
    print(save_dir + graph_name[:-2] + '-nodeFeatures.dic')
    pickle.dump(nodeDic, open(save_dir + graph_name[:-2] + '-nodeFeatures.dic', "wb"))
    # print(nodeDic)

    # for node_key in nodeDic:
    #     print("node", node_key, "feature:", nodeDic[node_key])
