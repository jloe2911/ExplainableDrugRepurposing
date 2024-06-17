import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import HeteroGraphConv, SAGEConv, GraphConv, GATConv
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils import *

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
        
class GNN(torch.nn.Module):
    def __init__(self, gnn_variant, etypes):
        super().__init__()
        if gnn_variant == 'GraphSAGE':
            self.conv1 = HeteroGraphConv({etype: SAGEConv(100, 100, 'mean') for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: SAGEConv(100, 100, 'mean') for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        elif gnn_variant == 'GCN':
            self.conv1 = HeteroGraphConv({etype: GraphConv(100, 100) for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: GraphConv(100, 100) for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        elif gnn_variant == 'GAT':
            self.conv1 = HeteroGraphConv({etype: GATConv(100, 100, 1) for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: GATConv(100, 100, 1) for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        else:
            print('ERROR - No model has been chosen')
        
    def forward(self, pos_g, neg_g, node_features, etype, edge_weight=None):
        if edge_weight is None:
            h = self.conv1(pos_g, node_features)
            h = self.conv2(pos_g, h)
        else:
            h = self.conv1(pos_g, node_features, mod_kwargs={etype:{'edge_weight': edge_weight[etype]} for etype in pos_g.etypes})
            h = self.conv2(pos_g, h, mod_kwargs={etype:{'edge_weight': edge_weight[etype]} for etype in pos_g.etypes})
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype), h
    
class Model():
    def __init__(self, gnn_variant, etypes, etype2pred, g_train, g_test, node_features):
        self.model = GNN(gnn_variant, etypes)
        self.etype2pred = etype2pred
        self.g_train = g_train
        self.g_test = g_test
        self.node_features = node_features
        
    def _train(self, epochs=500):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        
        for epoch in range(epochs+1): 
            self.model.train()
            optimizer.zero_grad()
            
            g_neg_train = construct_negative_graph(self.g_train, self.etype2pred)
            pos_score, neg_score, h = self.model(self.g_train, g_neg_train, self.node_features, self.etype2pred)

            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')
    
    def _eval(self):
        self.model.eval()
        true_heads = self.g_test.edges(etype=self.etype2pred, form='uv')[0]
        true_tails = self.g_test.edges(etype=self.etype2pred, form='uv')[1]
        n = len(true_heads)
        g_neg_test = construct_negative_graph(self.g_test, self.etype2pred)
        pos_score, neg_score, h = self.model(self.g_test, g_neg_test, self.node_features, self.etype2pred)
        
        #hits@5, hits@10
        top5 = 0
        top10 = 0

        for true_head, true_tail in zip(true_heads, true_tails):
            rank = get_rank(true_head, true_tail, h)
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1
        
        #precision, recall, f1-score
        scores = torch.cat([pos_score, neg_score]).view(-1).detach().numpy()
        scores = np.where(scores >= 0.6, 1, 0)
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
        precision = precision_score(labels, scores)
        recall = recall_score(labels, scores)
        f1 = f1_score(labels, scores)
        
        return top5/n, top10/n, precision, recall, f1
    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            # Use the training graph to get the embeddings
            _, _, h = self.model(self.g_train, self.g_train, self.node_features, self.etype2pred)
        return h
    
def split_train_test(g, etype, split=0.8):
    '''Helper function to create train and test graphs'''
    eids = np.arange(g.number_of_edges(etype))
    eids = np.random.permutation(eids)
    eids = torch.tensor(eids, dtype=torch.int64)

    train_size = int(len(eids) * split)
    test_size = g.number_of_edges(etype) - train_size

    train_indices = {etype: eids[test_size:]}
    test_indices = {etype: eids[:test_size]}

    g_train = dgl.remove_edges(g, eids=test_indices[etype], etype=etype)
    g_test = dgl.remove_edges(g, eids=train_indices[etype], etype=etype)
    return g_train, g_test
    
def construct_negative_graph(g, etype, k=5):
    '''Helper function to create negative edges for link prediction'''
    utype, _, vtype = etype
    src, dst = g.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, g.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph({etype: (neg_src, neg_dst)}, num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes})

def compute_loss(pos_score, neg_score):
    '''Helper funtion to compute the loss'''
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()
