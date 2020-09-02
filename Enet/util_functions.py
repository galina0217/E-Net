
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math
import cPickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../DenoisingGCN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import matplotlib.pyplot as plt
import time
import random
import node2vec
import multiprocessing as mp

def sample_train_val_test(edges, edges_clean, net, split_ratio=[8,1,1], missing_ratio=0.1, neg_pos_ratio=5,
                          train_pos=None, test_pos=None, max_train_num=None):
    edges = list(edges)
    edges_clean = list(edges_clean)
    # mimic missing links
    existing_link_num = int(len(edges) * missing_ratio)
    # sample positive links if not specified
    pos, neg = [], []
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(edges)), existing_link_num)
        edges_filter = [edges[i] for i in perm]
        for e in edges_filter:
            if e in edges_clean:
                pos.append(e)
            else:
                neg.append(e)
    pos_num = len(pos)

    train_pos_num = int(math.ceil(pos_num * split_ratio[0] / np.sum(split_ratio)))
    n = net.shape[0]
    # sample negative links
    while len(neg) < (train_pos_num + neg_pos_ratio*(pos_num-train_pos_num)):
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg.append((i, j))
        else:
            continue
    neg_num = len(neg)
    random.shuffle(pos)
    random.shuffle(neg)
    train_val_split = int(math.ceil(pos_num * split_ratio[0] / np.sum(split_ratio)))
    val_test_split = int(math.ceil(pos_num * np.sum(split_ratio[:-1])/np.sum(split_ratio)))

    train_pos = ([p[0] for p in pos[:train_val_split]], [p[1] for p in pos[:train_val_split]])
    val_pos = ([p[0] for p in pos[train_val_split:val_test_split]], [p[1] for p in pos[train_val_split:val_test_split]])
    test_pos = ([p[0] for p in pos[val_test_split:]], [p[1] for p in pos[val_test_split:]])

    val_pos_num = len(val_pos[0])
    train_neg = ([p[0] for p in neg[:train_pos_num]], [p[1] for p in neg[:train_pos_num]])
    val_neg = ([p[0] for p in neg[train_pos_num:train_pos_num+neg_pos_ratio*val_pos_num]],
               [p[1] for p in neg[train_pos_num:train_pos_num+neg_pos_ratio*val_pos_num]])
    test_neg = ([p[0] for p in neg[train_pos_num+neg_pos_ratio*val_pos_num:]],
                [p[1] for p in neg[train_pos_num+neg_pos_ratio*val_pos_num:]])
    train_val_test = {'train': [train_pos, train_neg], 'val': [val_pos, val_neg], 'test': [test_pos, test_neg]}

    return train_val_test, pos

# data: {'train':[train_pos,train_neg], 'val':[val_pos, val_neg], 'test':[test_pos, test_neg]}
def links2subgraphs(A, Identity, data, h=1, max_nodes_per_hop=None, node_information=None,
                    edge_information=None, lazy_subgraph=False,
                    multi_subgraph=1, num_node_to_walks=5, num_walk=5, mp=False):
    # extract enclosing subgraphs
    max_n_label = {'value': 0}

    if mp == False:
        def helper(A, Identity, links, g_label):
            g_list = []
            for i, j in tqdm(zip(links[0], links[1])):
                if lazy_subgraph:
                    # if lazy_subgraph and mode=='train':
                    for _ in range(multi_subgraph):
                        g, g_identity, n_labels, n_features, e_features = subgraph_extraction_labeling_lazy(
                            (i, j), A, Identity, alpha=0.8, num_walk=num_walk, step_each_pos_node_to_walk=num_node_to_walks,
                            node_information=node_information, edge_information=edge_information)
                        max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                        g_list.append(GNNGraph(g, g_identity, g_label, n_labels, n_features, e_features))
                else:
                    g, g_identity, n_labels, n_features, e_features = subgraph_extraction_labeling(
                        (i, j), A, Identity, h, max_nodes_per_hop, node_information, edge_information)
                    max_n_label['value'] = max(max(n_labels), max_n_label['value'])
                    g_list.append(GNNGraph(g, g_identity, g_label, n_labels, n_features, e_features))
            return g_list
    else:
        def helper(A, Identity, links, g_label):
            # the new parallel extraction code
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            if lazy_subgraph:
                results = pool.map_async(parallel_worker_lazy, [((i, j), A, Identity, 0.8, num_walk, num_node_to_walks,
                                                                 node_information, edge_information)
                                                                for i, j in zip(links[0], links[1])])
            else:
                results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information, edge_information)
                                                           for i, j in zip(links[0], links[1])])
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, n_labels, n_features, e_features)
                      for g, n_labels, n_features, e_features in results]
            max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _, _ in results]), max_n_label['value'])
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    ret = {}
    for key, value in data.items():
        graphs = helper(A, Identity, value[0], 1) + helper(A, Identity, value[1], 0)
        ret[key] = graphs

    print('max_n_label:', max_n_label)
    return ret, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def parallel_worker_lazy(x):
    return subgraph_extraction_labeling_lazy(*x)

def subgraph_extraction_labeling(ind, A, Identity, h=1,
                                 max_nodes_per_hop=None, node_information=None,
                                 edge_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A, max_nodes_per_hop)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # if edge_information is not None:
    #     edge_features = edge_information[nodes]
    # else:
    #     edge_features = None
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    if edge_information is not None:
        edge_features_dict = {(min(x, y), max(x, y)):
                             edge_information[(min(nodes[x], nodes[y]), max(nodes[x], nodes[y]))]
                         for (x, y) in g.edges()}
        keys = sorted(edge_features_dict)
        edge_features = []
        for edge in keys:
            edge_features.append(edge_features_dict[edge])
        edge_features = np.array(edge_features)
    else:
        edge_features = None

    g_identity = nx.from_scipy_sparse_matrix(Identity[nodes, :][:, nodes])
    return g, g_identity, labels.tolist(), features, edge_features

def subgraph_extraction_labeling_lazy(ind, A, Identity, alpha=0.8, num_walk=10,
                                      step_each_pos_node_to_walk=10,
                                      node_information=None, edge_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    # t0 = time.time()
    nodes = set([ind[0], ind[1]])
    for node_start in nodes:
        node_start = [node_start]
        for n_w in range(num_walk):
            node = node_start
            for iteration in range(step_each_pos_node_to_walk):
                if random.uniform(0, 1) < alpha:
                    neighs = neighbors_lazy(node, A)
                    if len(neighs) == 0:
                        break
                    fringe = random.sample(neighs, 1)
                    nodes = nodes.union(fringe)
                    node = fringe
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    # nodes_dist = [1, 1] + [0 for i in range(len(nodes))]
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    if edge_information is not None:
        edge_features_dict = {(min(x, y), max(x, y)):
                             edge_information[(min(nodes[x], nodes[y]), max(nodes[x], nodes[y]))]
                         for (x, y) in g.edges()}
        keys = sorted(edge_features_dict)
        edge_features = []
        for edge in keys:
            edge_features.append(edge_features_dict[edge])
        edge_features = np.array(edge_features)
    else:
        edge_features = None

    g_identity = nx.from_scipy_sparse_matrix(Identity[nodes, :][:, nodes])
    return g, g_identity, labels.tolist(), features, edge_features


def neighbors(fringe, A, max_nodes_per_hop):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        # if max_nodes_per_hop is not None:
        #     if max_nodes_per_hop < len(nei):
        #         nei = random.sample(nei, max_nodes_per_hop)
        res = res.union(nei)
    return res

def neighbors_lazy(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+range(2, K), :][:, [0]+range(2, K)]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divide(d, 2), np.mod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

    
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

