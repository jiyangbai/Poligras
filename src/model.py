import glob
import os
import torch
import metis
import random
import time
import copy
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm, trange


class Poligras(torch.nn.Module):

    def __init__(self, args):

        super(Poligras, self).__init__()
        self.args = args

        self.interLayer_first = torch.nn.Linear(self.args.feat_dim, self.args.filters1)
        self.fully_connected_second = torch.nn.Linear(self.args.filters1, self.args.filters2)
        self.dropout = torch.nn.Dropout(p=self.args.dropout)

        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):

        temp_feat = torch.nn.functional.relu(self.interLayer_first(x))
        temp_feat =  self.fully_connected_second(temp_feat)
        temp_feat = torch.mm(temp_feat, torch.t(temp_feat))

        temp_feat = self.dropout(temp_feat)
        mask_temp_feat = torch.FloatTensor(np.diag([float('-inf')] * temp_feat.size()[0]))
        temp_feat = temp_feat + mask_temp_feat
        temp_feat = torch.nn.functional.softmax(temp_feat.view(1, -1), dim=1).view(temp_feat.size()[0], -1)
        assert(temp_feat.size()[0] == temp_feat.size()[1])

        return temp_feat



class PoligrasRunner(object):

    def __init__(self, args):

        self.args = args

        g_file = open('./dataset/' + self.args.dataset + '/' + self.args.dataset + '_nx_edgeWeighted', 'rb')
        loaded_graph = pickle.load(g_file)
        g_file.close()
        self.init_graph = loaded_graph['G']


        g_file = open('./dataset/' + self.args.dataset + '/' + self.args.dataset + '_staticFeat_1000', 'rb')
        loaded_data = pickle.load(g_file)
        g_file.close()
        self.node_feat = loaded_data['feat']
        self.args.feat_dim = self.node_feat.size()[1]
        # print('feat size: ', self.args.feat_dim)
        self.model = Poligras(self.args)

        init_superNodes_dict = {}
        self.node_belonging = {}
        for node in self.init_graph.nodes():
            init_superNodes_dict[node] = [node]
            self.node_belonging[node] = node


        ij = 0
        self.init_nd_idx = {}
        for nd in self.init_graph.nodes():
            self.init_nd_idx[nd] = ij
            ij += 1

        ## compute the initial group partitioning(index)
        self.num_partitions = self.init_graph.number_of_nodes()//200

        h_function = list(range(self.init_graph.number_of_nodes()))
        random.shuffle(h_function)


        F_A_dict = {}
        for A in init_superNodes_dict:
            F_A = self.init_graph.number_of_nodes()
            for v in init_superNodes_dict[A]:
                f_v = self.init_graph.number_of_nodes()
                for u in list(self.init_graph[v]) + [v]:
                    if(h_function[self.init_nd_idx[int(u)]] < f_v):
                        f_v = h_function[self.init_nd_idx[int(u)]]

                if(f_v < F_A):
                    F_A = f_v

            F_A_dict[A] = F_A
        F_A_list = sorted(F_A_dict.items(), key=lambda item:item[1])

        init_groupIndex = []
        for i in range(self.num_partitions):
            curr_idx = []
            for j in F_A_list[int(i*len(F_A_list)/self.num_partitions): int((i+1)*len(F_A_list)/self.num_partitions)]:
                curr_idx.append(j[0])
            
            init_groupIndex.append(np.array(curr_idx))


        # print('index size: ', len(init_groupIndex))

        f = open('./graph_store/{}/_{}_.best_temp'.format(self.args.dataset, 0), 'wb')
        pickle.dump({'g':self.init_graph, 'group_index':init_groupIndex, 'superNodes_dict':init_superNodes_dict}, f)
        f.close()
 
 
    def select_action(self, curr_feat):

        curr_probs = self.model(curr_feat)

        curr_action = curr_probs.argmax()
        curr_action_row, curr_action_col = curr_action.item() // curr_probs.size()[0], curr_action.item() % curr_probs.size()[0]

        if(curr_action_row == curr_action_col):
            curr_action_row, curr_action_col = random.sample(range(curr_probs.size()[0]), 2)
        self.model.saved_log_probs.append(torch.log(curr_probs[curr_action_row][curr_action_col]))#

        return curr_action_row, curr_action_col


    def update_graph(self, n1, n2, curr_graph):

        curr_reward, graph_modify_dict = 0, {'weight':{}, 'if_true':{}, 'add_edge':{}}## "curr_reward" records the sr of merging n1 & n2; "graph_modify_dict" temporarily stores the modifications of graph when merging two (super)nodes, which will be truly conducted if curr_reward > 0;
        nei_n1, nei_n2 = set(self.curr_graph[n1]), set(self.curr_graph[n2])

        ## compute curr_reward with the help of indicative graph
        for sd in nei_n1 & nei_n2 - set([n1]) - set([n2]):
            if(self.curr_graph[n1][sd]['if_true']):
                if(self.curr_graph[n2][sd]['if_true']):
                    curr_reward += 1
                else:
                    if((self.curr_graph[n1][sd]['weight']+self.curr_graph[n2][sd]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*len(self.superNodes_dict[sd])/2)):
                        curr_reward += (2*self.curr_graph[n2][sd]['weight'] - len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]))
                    else:
                        curr_reward += (1+ len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n1][sd]['weight'])
                        graph_modify_dict['if_true'][(n1,sd)] = False
            else:
                if(curr_graph[n2][sd]['if_true']):
                    if((self.curr_graph[n1][sd]['weight']+self.curr_graph[n2][sd]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*len(self.superNodes_dict[sd])/2)):
                        curr_reward += (2*self.curr_graph[n1][sd]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]))
                        graph_modify_dict['if_true'][(n1,sd)] = True
                    else:
                        curr_reward += (1+ len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n2][sd]['weight'])

            graph_modify_dict['weight'][(n1,sd)] = self.curr_graph[n1][sd]['weight'] + self.curr_graph[n2][sd]['weight']
        

        for sd in nei_n1 - nei_n2 - set([n1]) - set([n2]):
            if(self.curr_graph[n1][sd]['if_true']):
                if(self.curr_graph[n1][sd]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[sd]))/2)):
                    curr_reward += -len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd])
                else:
                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n1][sd]['weight'])
                    graph_modify_dict['if_true'][(n1,sd)] = False


        for sd in nei_n2 - nei_n1 - set([n1]) - set([n2]):
            if(self.curr_graph[n2][sd]['if_true']):
                if(curr_graph[n2][sd]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[sd]))/2)):
                    curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[sd])
                    graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':True}
                else:
                    curr_reward += (1 + len(self.superNodes_dict[n2])*len(self.superNodes_dict[sd]) - 2*self.curr_graph[n2][sd]['weight'])
                    graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':False}
            else:
                graph_modify_dict['add_edge'][(n1, sd)] = {'toAddWei':self.curr_graph[n2][sd]['weight'], 'ifTrue':False}


        if(n1 in nei_n2):
            if(self.curr_graph[n1][n2]['if_true']):
                if(n1 in nei_n1):
                    if(self.curr_graph[n1][n1]['if_true']):
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                curr_reward += 2
                            else:
                                if((self.curr_graph[n2][n2]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n1][n1]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (1 + 2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight']) 
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:# n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (1 - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                                
                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                    else:# self.curr_graph[n1][n1]['if_true'] = False
                        if(n2 in nei_n2): 
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                    curr_reward += (1 + 2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                            else:
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                    curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)

                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:# n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                graph_modify_dict['if_true'][(n1,n1)] = True
                                curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                else:# n1 not in self loop
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (1 - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                        else:# curr_graph[n2][n2]['if_true'] = False
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                    
                    else:# n2 not in self loop
                        if(self.curr_graph[n1][n2]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                            curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':True}

                        else:
                            curr_reward += (1 + len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]) - 2*self.curr_graph[n1][n2]['weight'])
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':False}

            else:# self.curr_graph[n1][n2]['if_true'] = False
                if(n1 in nei_n1):
                    if(self.curr_graph[n1][n1]['if_true']):
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (1 + 2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            else:
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                    curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                    graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        
                        else:# n2 not in self loop
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                    else:# self.curr_graph[n1][n1]['if_true'] = False
                        if(n2 in nei_n2):
                            if(self.curr_graph[n2][n2]['if_true']):
                                if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                    curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                    curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                    graph_modify_dict['if_true'][(n1,n1)] = True
                                else:
                                    curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])

                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']

                        else:# n2 not in self loop
                            graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n1][n2]['weight']

                else:# n1 not in self loop
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                                curr_reward += (2*self.curr_graph[n1][n2]['weight'] - len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2]))
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':True}

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                        else:
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'] + self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                            
                    else:
                        graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n1][n2]['weight'], 'ifTrue':False}

        else: # n1 n2 not connected
            if(n1 in nei_n1):
                if(self.curr_graph[n1][n1]['if_true']):
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += 1
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*self.curr_graph[n2][n2]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                        else:
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                                curr_reward += (2*self.curr_graph[n2][n2]['weight'] - len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2)
                            
                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                                graph_modify_dict['if_true'][(n1,n1)] = False
                        
                        graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']


                    else:# n2 not in self loop 
                        if(self.curr_graph[n1][n1]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                            curr_reward += -len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2
                        else:
                            curr_reward += (1 + len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2 - 2*self.curr_graph[n1][n1]['weight'])
                            graph_modify_dict['if_true'][(n1,n1)] = False

                else:# self.curr_graph[n1][n1]['if_true'] = False
                    if(n2 in nei_n2):
                        if(self.curr_graph[n2][n2]['if_true']):
                            if((self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']) > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                                curr_reward += (2*self.curr_graph[n1][n1]['weight'] - len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2)
                                curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                                # curr_graph[n1][n1]['if_true'] = True
                                graph_modify_dict['if_true'][(n1,n1)] = True

                            else:
                                curr_reward += (1 + len(self.superNodes_dict[n2])*(len(self.superNodes_dict[n2])-1)/2 - 2*curr_graph[n2][n2]['weight'])

                        graph_modify_dict['weight'][(n1,n1)] = self.curr_graph[n1][n1]['weight'] + self.curr_graph[n2][n2]['weight']


            else:# n1 not in self loop
                if(n2 in nei_n2):
                    if(self.curr_graph[n2][n2]['if_true']):
                        if(self.curr_graph[n2][n2]['weight'] > ((len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2]))*(len(self.superNodes_dict[n1])+len(self.superNodes_dict[n2])-1)/4)):
                            curr_reward += -len(self.superNodes_dict[n1])*(len(self.superNodes_dict[n1])-1)/2
                            curr_reward += -len(self.superNodes_dict[n1])*len(self.superNodes_dict[n2])
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':True}
                        else:
                            graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':False}

                    else:
                        graph_modify_dict['add_edge'][(n1, n1)] = {'toAddWei':self.curr_graph[n2][n2]['weight'], 'ifTrue':False}
                    


        self.model.rewards.append(curr_reward)
        if(curr_reward > 0):
            ## modify current graph (indicative graph)
            for pair in graph_modify_dict['weight']:
                self.curr_graph[pair[0]][pair[1]]['weight'] = graph_modify_dict['weight'][pair]
            for pair in graph_modify_dict['if_true']:
                self.curr_graph[pair[0]][pair[1]]['if_true'] = graph_modify_dict['if_true'][pair]
            for pair in graph_modify_dict['add_edge']:
                self.curr_graph.add_edge(pair[0], pair[1], weight=graph_modify_dict['add_edge'][pair]['toAddWei'], if_true=graph_modify_dict['add_edge'][pair]['ifTrue'])

            self.curr_graph.remove_node(n2)
            ## update supernode features
            self.curr_feat[n1] += self.curr_feat[n2]
            for init_n in self.superNodes_dict[n2]:
                self.node_belonging[init_n] = n1
            self.superNodes_dict[n1] += self.superNodes_dict[n2]
            self.superNodes_dict.pop(n2)

        return curr_reward

#---------------------------------------------------------------------------------------------------------------------------------
    def fit(self):
        print("\n-------Model running---------.\n")

        # total_rewards = 0 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.max_reward_by_inner_iter = 0## max_reward_by_inner_iter is to help judge and execute the regrouping
        self.model.train()
        init_time = time.time()
        for count in range(self.args.counts):
            best, bad_counter = -1000000, 0

            curr_time = time.time()
            while(True):
                # start_time = time.time()
                g_file = open('./graph_store/{}/_{}_.best_temp'.format(self.args.dataset, count), 'rb')
                loaded_compre = pickle.load(g_file)
                g_file.close()

                self.curr_graph = loaded_compre['g']
                self.group_index =  loaded_compre['group_index']
                self.superNodes_dict = loaded_compre['superNodes_dict']
                self.curr_feat = copy.deepcopy(self.node_feat)

                count_reward, batch_id = 0, 0
                traverse_time = 0
                for idx in range(len(self.group_index)):
                    if(len(self.group_index[idx]) < 3):
                        continue
                    curr_row, curr_col = self.select_action(self.curr_feat[[self.init_nd_idx[i] for i in self.group_index[idx]]])

                    curr_reward = self.update_graph(self.group_index[idx][curr_row], self.group_index[idx][curr_col], self.curr_graph) 

                    if(curr_reward > 0):
                        count_reward += curr_reward
                        self.group_index[idx] = np.delete(self.group_index[idx], curr_col)


                policy_loss=0
                # len_loss = len(self.model.saved_log_probs)
                returns = torch.FloatTensor(self.model.rewards)
                returns = (returns - max(returns.mean(), 0)) / (returns.std())# + eps)

                for log_prob, R in zip(self.model.saved_log_probs, returns):
                    policy_loss += - log_prob * R

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                print('Count {}; Positive Count Reward: {};\n'.format(count, count_reward))

                del self.model.rewards[:]
                del self.model.saved_log_probs[:]


                if(count < 5):
                    ratio = 0.001
                else:
                    ratio = 0.01
                if(count_reward > (1 + ratio)*best):
                    best, bad_counter = count_reward, 0

                    self.best_graph, self.best_currFeat, self.best_init_groupIndex = self.curr_graph, self.curr_feat, self.group_index
                    self.best_init_superNodes_dict = self.superNodes_dict
                else:
                    bad_counter += 1

                if(bad_counter == self.args.bad_counter):
                    # total_rewards += best
                    break

            
            if(best > self.max_reward_by_inner_iter):
                self.max_reward_by_inner_iter = best
            if(best < (self.max_reward_by_inner_iter/3)):
                ## regrouping (group partitioning)
                assert(self.best_graph.number_of_nodes() == len(self.best_init_superNodes_dict))

                self.num_partitions = self.best_graph.number_of_nodes()//200

                h_function = list(range(self.init_graph.number_of_nodes()))
                random.shuffle(h_function)

                F_A_dict = {}
                for A in self.best_init_superNodes_dict:
                    F_A = self.init_graph.number_of_nodes()
                    for v in self.best_init_superNodes_dict[A]:
                        f_v = self.init_graph.number_of_nodes()
                        for u in list(self.init_graph[v]) + [v]:
                            if(h_function[self.init_nd_idx[int(u)]] < f_v):
                                f_v = h_function[self.init_nd_idx[int(u)]]

                        if(f_v < F_A):
                            F_A = f_v

                    F_A_dict[A] = F_A
                F_A_list = sorted(F_A_dict.items(), key=lambda item:item[1])

                self.best_init_groupIndex = []
                for i in range(self.num_partitions):
                    curr_idx = []
                    for j in F_A_list[int(i*len(F_A_list)/self.num_partitions): int((i+1)*len(F_A_list)/self.num_partitions)]:
                        curr_idx.append(j[0])
                    
                    self.best_init_groupIndex.append(np.array(curr_idx))


            self.node_feat = self.best_currFeat
            f = open('./graph_store/{}/_{}_.best_temp'.format(self.args.dataset, count+1), 'wb')
            pickle.dump({'g':self.best_graph, 'group_index':self.best_init_groupIndex, 'superNodes_dict':self.best_init_superNodes_dict}, f)
            f.close()
            print('------\n')
                


        files = glob.glob('./graph_store/{}/_*_.best_temp'.format(self.args.dataset))
        for fil in files:
            os.remove(fil)

        # print("\n-------Running finished, total reward is {}---------.\n".format(total_rewards))


#---------------------------------------------------------------------------------------------------------------------------------
    def encode(self):
        print("\n-------Model encoding---------.\n")

        self.super_edge = []# {}
        self.correctionSet_plus, self.correctionSet_minus = [], []# {}, {}

        done_pair, i_dx = {}, 0
        self.superNodes_dict = self.best_init_superNodes_dict
        for A in self.superNodes_dict:
            iterative_superNode = []
            # print('{}th supernode'.format(i_dx))
            for init_n in self.superNodes_dict[A]:
                for nei_n in self.init_graph[init_n]:
                    iterative_superNode.append(self.node_belonging[nei_n])

            for B in set(iterative_superNode):
                if(A == B):
                    continue
                if((A, B) in done_pair):
                    continue
                else:
                    done_pair[(A,B)] = 0
                    done_pair[(B,A)] = 0
            

                E_AB = [] # 0
                Pi_E_AB = []
                for n1 in self.superNodes_dict[A]:
                    for n2 in self.superNodes_dict[B]:
                        if((n1, n2) in self.init_graph.edges()):
                            E_AB.append((n1, n2))
                        else:
                            Pi_E_AB.append((n1, n2))

                if(len(E_AB) <= (len(self.superNodes_dict[A])*len(self.superNodes_dict[B])/2)):
                    self.correctionSet_plus += E_AB
                else:
                    self.super_edge.append((A, B))# += 1#
                    self.correctionSet_minus += Pi_E_AB# (Pi_AB - E_AB)


            E_AA = []
            Pi_E_AA = []
            for n1 in self.superNodes_dict[A]:
                for n2 in self.superNodes_dict[A]:
                    if(n1<n2):
                        if((n1, n2) in self.init_graph.edges()):
                            E_AA.append((n1, n2))# += 1
                        else:
                            Pi_E_AA.append((n1, n2))  

            if(len(E_AA) <= (len(self.superNodes_dict[A])*(len(self.superNodes_dict[A])-1)/4)):
                self.correctionSet_plus += E_AA
            else:
                self.super_edge.append((A, A))
                self.correctionSet_minus += Pi_E_AA#(Pi_AA - E_AA) #list(Pi_AA - E_AA)

            i_dx += 1


        print('==============================\n')

        print('#super edge: ', len(self.super_edge))
        print('#correset_plus: ', len(self.correctionSet_plus))
        print('#correset_minus: ', len(self.correctionSet_minus))
        print("\n-------SuperNode encoding ended, total reward is {}---------.\n".format(self.init_graph.number_of_edges() - len(self.super_edge) - len(self.correctionSet_plus) - len(self.correctionSet_minus)))

