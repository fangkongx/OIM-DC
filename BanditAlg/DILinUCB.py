from random import random
import numpy as np
import networkx as nx
import time
import gc
import os


class DILinUCBAlgorithm:
    def __init__(self, G, parameter, seed_size, oracle, alpha, feedback='edge'):
     

        start_time = time.time()

        self.G = G
        self.param = parameter  
        self.node_list = list(G.nodes())
        self.seed_size = seed_size
        self.dimension = len(G.nodes())  # d
        self.feedback = feedback
        self.users = []  # Nodes LinUCB
        self.P = np.ones((len(self.node_list), self.dimension)).astype(np.float16)  # P estimated

        n = len(self.node_list)
        lambda_1 = 1e-4
        self.b = np.zeros((self.dimension, n))  # TODO
        self.inv_cov = (1 / lambda_1) * np.eye(n)
        self.freq = np.zeros((n, 1)) * 1e-3
        self.alpha = np.zeros((self.dimension, n))
        self.t = 1
        self.t1 = 1
        self.param_norm = np.sum(np.power(self.param, 2), axis=0).reshape((1, n))

        end_time = time.time()        
        gc.collect()
        start_time = time.time()
        end_time = time.time()
       
        self.algorithmParameters = {}
        self.algorithmParameters["self.b"] = self.b
        self.algorithmParameters["self.inv_cov"] = self.inv_cov
        self.algorithmParameters["self.freq"] = self.freq
        self.algorithmParameters["self.alpha"] = self.alpha
        self.algorithmParameters["self.t"] = self.t
        self.algorithmParameters["self.t1"] = self.t1

    def decide(self):
        if self.t != 1:
            self.alpha = self.b.dot(self.inv_cov.T)
        temp = np.power(self.freq, -1)
        m = np.tile(temp, [1, len(self.node_list)])
        c = np.sqrt(1.5 * np.log(self.t + 1)) * (m)
        i1 = self.alpha.T.dot(self.param)

        i2 = c.T * np.sqrt((np.diag(self.inv_cov).reshape((len(self.node_list), 1))).dot(self.param_norm))
        

        self.P = np.clip(self.alpha.T.dot(self.param) + c.T * np.sqrt(
            (np.diag(self.inv_cov).reshape((len(self.node_list), 1))).dot(self.param_norm)), a_min=0, a_max=1)
      
        self.t1 += 1

        n = len(self.node_list)
        MG = np.zeros((n, 2))
        MG[:, 0] = np.arange(n)  
        influence_UCB = self.P
       
        MG[:, 1] = np.sum(influence_UCB, axis=1)  

        S = []
        args = []
        prev_spread = 0

        for k in range(self.seed_size):
            if k == 0:
                MG = MG[MG[:, 1].argsort()]
            else:
                for i in range(0, n - k):
                    select_node = int(MG[-1, 0])  
                    MG[-1, 1] = np.sum(np.maximum(influence_UCB[select_node, :], temp)) - prev_spread
                    if MG[-1, 1] >= MG[-2, 1]:
                        prev_spread = prev_spread + MG[-1, 1]
                        break
                    else:
                        val = MG[-1, 1]  
                        idx = np.searchsorted(MG[:-1, 1], val)
                        MG_new = np.zeros(MG.shape)
                        MG_new[:idx, :] = MG[:idx, :]
                        MG_new[idx, :] = MG[-1, :]  
                        MG_new[idx + 1:, :] = MG[idx:-1, :]  
                        MG = MG_new
            args.append(int(MG[-1, 0]))
            S.append(self.node_list[int(MG[-1, 0])])
            
            if k == 0:
                temp = influence_UCB[np.array(args), :]  # 1 x 4039
            else:
                temp = np.amax(influence_UCB[np.array(args), :], axis=0)  #
                
            MG[-1, 1] = -1
            MG_new = np.zeros(MG.shape)
            MG_new[0, :] = MG[-1, :]
            MG_new[1:, :] = MG[:-1, :]
            MG = MG_new

        assert len(S) == self.seed_size, f"number of seed is not k, len(S)={len(S)}, k={k}"
        assert len(S) == len(set(S)), "select repetitive seeds"
        print("after seed selection")
        return S

    def updateParameters(self, S, live_nodes, live_edges, _iter):
        for u in S:
            y = np.array(live_edges[u]).reshape(len(self.node_list), 1)
            add_b = np.matmul(self.param, y)  # param target feature
            u_idx = self.node_list.index(u)
            self.b[:, u_idx] = self.b[:, u_idx] + add_b.flatten()
            self.freq[u_idx] += 1
            temp = self.inv_cov[u_idx, u_idx]
            self.inv_cov = self.inv_cov - (self.inv_cov[:, u_idx:u_idx + 1].dot(self.inv_cov[u_idx:u_idx + 1, :])) / (
                        1 + temp)
        self.t += 1
        assert self.t == self.t1, "计数出错"

       
        self.algorithmParameters["self.b"] = self.b
        self.algorithmParameters["self.inv_cov"] = self.inv_cov
        self.algorithmParameters["self.freq"] = self.freq
        self.algorithmParameters["self.alpha"] = self.alpha
        self.algorithmParameters["self.t"] = self.t
        self.algorithmParameters["self.t1"] = self.t1

       

    def saveParameters(self, mainParametersToSave, SimulationParameters, parameter_file_name):
        from Tool.utilFunc import pickle_save, pickle_load
        print("before save")
        print("save_file_name", parameter_file_name)
        os.system(f'cp {parameter_file_name}.pkl {parameter_file_name}_old.pkl')
        os.system(f'cp {parameter_file_name}.csv {parameter_file_name}_old.csv')

        pickle_save((mainParametersToSave, SimulationParameters, self.algorithmParameters), parameter_file_name)
        print((mainParametersToSave, SimulationParameters, self.algorithmParameters))
        print("after save")

        if SimulationParameters["iterations_done"] > 10:
            exit(1)
