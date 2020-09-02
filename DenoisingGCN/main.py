import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import pdb
from Enet_embedding import Enet
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args

class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()

        self.regression = regression
        model = Enet
        self.gnn = model(latent_dim=cmd_args.latent_dim,
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                        num_edge_feats=cmd_args.edge_feat_dim,
                        total_num_nodes=cmd_args.total_num_nodes,
                        total_num_tag=cmd_args.feat_dim,
                        k=cmd_args.sortpooling_k,
                        conv1d_activation=cmd_args.conv1d_activation,
                        alpha=0.1,
                        sumsort=cmd_args.sumsort,
                        noise_matrix=cmd_args.noise_matrix,
                        reg_smooth=cmd_args.reg_smooth,
                        smooth_coef=cmd_args.smooth_coef,
                        trainable_noise=cmd_args.trainable_noise,
                        use_sig=cmd_args.use_sig, use_soft=cmd_args.use_soft,
                        noise_bias=cmd_args.noise_bias,
                        noise_init=cmd_args.noise_init)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = self.gnn.dense_dim
        if cmd_args.nodefeat_lp:
            out_dim += (2*cmd_args.attr_dim)
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden,
                                 num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden,
                                     with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):

        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if batch_graph[0].edge_features is not None:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        if cmd_args.nodefeat_lp:
            node_lp_feat_flag = True
            concat_node_lp_feat = []
        else:
            node_lp_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
                if node_lp_feat_flag == True:
                    concat_node_lp_feat.append(tmp[:2].flatten())
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if node_lp_feat_flag == True:
            node_lp_feat = torch.stack(concat_node_lp_feat)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()
            if node_lp_feat_flag == True:
                node_lp_feat = node_lp_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels, node_lp_feat
        else:
            return node_feat, labels, node_lp_feat

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 3:
            node_feat, edge_feat, labels, node_lp_feat = feature_label
            edge_feat = None
        elif len(feature_label) == 4:
            node_feat, edge_feat, labels, node_lp_feat = feature_label
        embed, reg_smooth, record = self.gnn(batch_graph, node_feat, edge_feat)
        if cmd_args.nodefeat_lp:
            embed = torch.cat([node_lp_feat, embed], 1)
        return self.mlp(embed, labels), reg_smooth, record

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
        

def loop_dataset(g_list, classifier, sample_idxes,
                 optimizer=None, bsize=cmd_args.batch_size, test=False, score=False):
    total_loss = []
    if test == False:
        total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    else:
        total_iters = 1
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        if test == False:
            selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        else:
            selected_idx = sample_idxes

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss, mse_nd = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            performance, reg_smooth, record = classifier(batch_graph)
            logits, loss, acc, p, r, f1 = performance[0], performance[1], performance[2], \
                                          performance[3], performance[4], performance[5]
            preds = torch.argmax(logits, 1).data.cpu().numpy()
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification


        if optimizer is not None:
            optimizer.zero_grad()

            if cmd_args.noise_matrix:
                loss_flag = [cmd_args.loss_missing, cmd_args.reg_smooth]
                loss_candidate = [loss, reg_smooth]
                losses = sum([loss_candidate[i] for i,f in enumerate(loss_flag) if f])
                losses.backward()
            else:
                print("singleloss")
                loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        reg_smooth = reg_smooth.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f reg_smooth: %0.5f'
                                 % (loss, mae, reg_smooth))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f p: %0.5f r: %0.5f f1: %0.5f reg_smooth: %0.5f'
                                 % (loss, acc, p, r, f1, reg_smooth))
            total_loss.append(np.array([loss, acc, p, r, f1, reg_smooth]) * len(selected_idx))

        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    if not classifier.regression:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    if score:
        return all_scores, all_targets
    else:
        return avg_loss, preds
