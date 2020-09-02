from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
import scipy.sparse as sp
import time

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib_noise import GNNLIB
from pytorch_util import *

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class Enet(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, total_num_nodes, total_num_tag, latent_dim=[32, 32, 32, 1], k=30,
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU', alpha=0.1,
                 sumsort=False, noise_matrix=True, reg_smooth=False,
                 smooth_coef=1, trainable_noise=False, use_sig=False,
                 use_soft=False, noise_bias=False, noise_init=False):
        print('Initializing Enet')

        super(Enet, self).__init__()
        self.noise_matrix = noise_matrix
        self.sumsort = sumsort
        self.alpha = alpha
        self.reg_smooth = reg_smooth
        self.smooth_coef = smooth_coef
        self.total_num_tag = total_num_tag
        self.trainable_noise = trainable_noise
        self.use_sig = use_sig
        self.use_soft = use_soft
        self.noise_bias = noise_bias

        latent_dim = latent_dim if type(latent_dim)==list else [latent_dim]
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.total_num_nodes = total_num_nodes
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        if self.sumsort:
            self.dense_dim = self.total_latent_dim
        else:
            dense_dim = int((k - 2) / 2 + 1)
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        self.noise_activation = eval('nn.{}()'.format('Sigmoid'))

        weights_init(self)
        if noise_init:
            self.W = nn.Parameter(torch.FloatTensor([[1./6], [1./6], [1./6],
                                                     [1./6], [1./6], [1./6]]))
            if self.noise_bias:
                self.b = nn.Parameter(torch.zeros(1, 1))
        else:
            self.W = nn.Parameter(torch.rand(6, 1))
            if self.noise_bias:
                self.b = nn.Parameter(torch.rand(1, 1))

    def adj_to_bias(self, adj, sizes, nhood=1):
        adj = adj.to_dense()
        mt = torch.eye(sizes)  # self-loops
        for _ in range(nhood):
            mt = torch.matmul(mt, (adj + torch.eye(sizes)))
        for i in range(sizes):
            for j in range(sizes):
                if mt[i][j] > 0.0:
                    mt[i][j] = 1.0
        return -1e9 * (1.0 - mt)

    def forward(self, graph_list, node_feat, edge_feat, M_noise=None):

        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        n2n_sp, e2n_sp, subg_sp, noise_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        ''' construct edge features (symmetry) '''
        edges = []
        edge_indices = n2n_sp._indices().transpose(1, 0)
        for e in edge_indices:
            if e[0] < e[1]:
                edges.append(e)
        edges = torch.stack(edges, 0).cuda()

        new_indices = torch.tensor(edges).transpose(1, 0)
        weight_matrix = []
        for i in range(edge_feat.shape[-1]):
            tmp = torch.sparse.LongTensor(new_indices.cuda(), edge_feat[:,i], n2n_sp.shape)
            tmp = tmp.transpose(1, 0) + tmp  # to symm
            weight_matrix.append(tmp)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            noise_sp = noise_sp.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        noise_sp = Variable(noise_sp)

        h, reg_smooth, record = self.sortpooling_embedding(node_feat, edge_feat, weight_matrix, n2n_sp, e2n_sp,
                                                                   subg_sp, graph_sizes, noise_sp, edges)

        return h, reg_smooth, record

    def sortpooling_embedding(self, node_feat, edge_feat, weight_matrix, n2n_sp, e2n_sp, subg_sp, graph_sizes,
                              noise_sp=None, edges=None):
        ''' if exists edge feature, concatenate to node feature vector '''
        if self.noise_matrix:
            ''' noise matrix '''
            if self.trainable_noise:
                if self.use_soft:
                    W_softmax = F.softmax(self.W, dim=0).squeeze()
                else:
                    W_softmax = self.W.squeeze()
                weighted_score = []
                for i in range(6):
                    tmp = gnn_spmat(weight_matrix[i], W_softmax, i)
                    weighted_score.append(tmp)
                noise_matrix = weighted_score[0] + weighted_score[1] + weighted_score[2] + \
                               weighted_score[3] + weighted_score[4] + weighted_score[5]
                noise_matrix.retain_grad()
                if self.noise_bias:
                    noise_matrix = gnn_spaddscalar(noise_matrix, self.b)
                if self.use_sig:
                    noise_matrix = gnn_spsigmoid(noise_matrix)
            else:
                noise_matrix = weight_matrix[0] + weight_matrix[1] + weight_matrix[2] + \
                               weight_matrix[3] + weight_matrix[4] + weight_matrix[5]
                noise_matrix = noise_matrix / 6
                noise_matrix.retain_grad()
                if self.use_sig:
                    noise_matrix = gnn_spsigmoid(noise_matrix)

            ppr_mat = gnn_spadd(gnn_spminus(n2n_sp, noise_sp), gnn_spmul(noise_matrix, noise_sp))

            # mask noise candidate
            if self.reg_smooth:
                A = ppr_mat
                D = torch.sparse.FloatTensor(torch.LongTensor([[i for i in range(ppr_mat.shape[0])],
                                                               [i for i in range(ppr_mat.shape[0])]]).cuda(),
                                             torch.sparse.sum(A, [1]).to_dense(),
                                             A.shape)
                D.retain_grad()
                L = gnn_spminus(D, A)
                L_quad = torch.mm(gnn_spmm_nograd(L.transpose(1, 0), node_feat).t(), node_feat)
                L_quad.retain_grad()
                reg_smooth = self.smooth_coef * torch.trace(L_quad)
                reg_smooth.retain_grad()
            else:
                reg_smooth = torch.tensor(0)

        else:
            reg_smooth = torch.tensor(0)


        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            if self.noise_matrix:
                if lv == 0:
                    n2npool = gnn_spmm_nograd(ppr_mat, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
                else:
                    n2npool = gnn_spmm(ppr_mat, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            else:
                n2npool = gnn_spmm(ppr_mat, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = F.normalize(node_linear, p=2, dim=1)
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        if not self.sumsort:
            ''' sortpooling layer '''
            sort_channel = cur_message_layer[:, -1]
            batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
            if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

            batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
            accum_count = 0
            for i in range(subg_sp.size()[0]):
                to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
                k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
                _, topk_indices = to_sort.topk(k)
                topk_indices += accum_count
                sortpooling_graph = cur_message_layer.index_select(0, torch.sort(topk_indices)[0])
                if k < self.k:
                    to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                    if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                        to_pad = to_pad.cuda()

                    to_pad = Variable(to_pad)
                    sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
                batch_sortpooling_graphs[i] = sortpooling_graph
                accum_count += graph_sizes[i]
        else:
            ''' sortsuming layer '''
            batch_sortpooling_graphs = torch.zeros(len(graph_sizes), 1, self.total_latent_dim)
            if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

            batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
            accum_count = 0
            for i in range(subg_sp.size()[0]):
                topk_indices = torch.arange(accum_count, accum_count + graph_sizes[i])
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    topk_indices = topk_indices.cuda()
                sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
                batch_sortpooling_graphs[i] = torch.sum(sortpooling_graph, 0)
                accum_count += graph_sizes[i]

        ''' traditional 1d convlution and dense layers '''
        if not self.sumsort:
            to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
            conv1d_res = self.conv1d_params1(to_conv1d)
            conv1d_res = self.conv1d_activation(conv1d_res)
            conv1d_res = self.maxpool1d(conv1d_res)
            conv1d_res = self.conv1d_params2(conv1d_res)
            conv1d_res = self.conv1d_activation(conv1d_res)

            to_dense = conv1d_res.view(len(graph_sizes), -1)
        else:
            to_dense = batch_sortpooling_graphs.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        record = []
        return self.conv1d_activation(reluact_fp), reg_smooth, record
