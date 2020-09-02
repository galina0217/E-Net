from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
#import cPickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import pdb
import argparse

from scipy.sparse import triu

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='E-net', help='gnn model to use')
cmd_opt.add_argument('-data', default='cora', help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=200, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='32-32-32-1', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=0.6, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=128, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.00001, help='init learning_rate')
cmd_opt.add_argument('-decay_learning_rate', type=bool, default=False, help='whether to decay learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=True, help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')
cmd_opt.add_argument('-sumsort', type=bool, default=True, help='whether to replace sortpooling with sort')
cmd_opt.add_argument('-noise-matrix', type=bool, default=False, help='whether to use noise matrix')
cmd_opt.add_argument('-attention-mode', type=str, default='', help='attention mode')
cmd_opt.add_argument('-noise-hidden-dim', type=int, default=10, help='noise hidden dimension')
cmd_opt.add_argument('-total-num-nodes', type=int, default=2708, help='total-num-nodes')
cmd_opt.add_argument('-reg-smooth', type=bool, default=False, help='whether to use smoothing regularization')
cmd_opt.add_argument('-softmax', type=bool, default=False, help='whether to add softmax on ppr_mat')
cmd_opt.add_argument('-smooth-coef', type=float, default=1, help='smooth reg coef')
cmd_opt.add_argument('-nodefeat-lp', type=bool, default=True, help='add node features when conducting link prediction')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

class GNNGraph(object):
    def __init__(self, g, g_identity, label, node_tags=None, node_features=None, edge_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        # self.query_pair = query_pair
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.edge_features = edge_features
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        if len(g_identity.edges()) != 0:
            x, y = zip(*g_identity.edges())
            self.num_noises = len(x)
            self.noise_pairs = np.ndarray(shape=(self.num_noises, 2), dtype=np.int32)
            self.noise_pairs[:, 0] = x
            self.noise_pairs[:, 1] = y
            self.noise_pairs = self.noise_pairs.flatten()
        else:
            self.num_noises = 0
            self.noise_pairs = np.array([])


