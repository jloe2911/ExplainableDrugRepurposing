import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn import HeteroGraphConv, SAGEConv, GraphConv, GATConv
from sklearn.metrics import precision_score, recall_score, f1_score

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
            self.conv1 = HeteroGraphConv({etype: SAGEConv(10, 10, 'mean') for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: SAGEConv(10, 10, 'mean') for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        elif gnn_variant == 'GCN':
            self.conv1 = HeteroGraphConv({etype: GraphConv(10, 10) for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: GraphConv(10, 10) for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        elif gnn_variant == 'GAT':
            self.conv1 = HeteroGraphConv({etype: GATConv(10, 10, 1) for etype in etypes})
            self.conv2 = HeteroGraphConv({etype: GATConv(10, 10, 1) for etype in etypes})
            self.pred = HeteroDotProductPredictor()
        else:
            print('ERROR - No model has been chosen')
        
    def forward(self, pos_g, neg_g, node_features, etype, edge_weight=None):
        if edge_weight is None:
            h = self.conv1(pos_g, node_features)
            h = self.conv2(pos_g, h)
        else:
            h = self.conv1(pos_g, node_features, mod_kwargs={etype:{'edge_weight': edge_weight[etype]} for etype in g.etypes})
            h = self.conv2(pos_g, h, mod_kwargs={etype:{'edge_weight': edge_weight[etype]} for etype in g.etypes})
        return self.pred(pos_g, h, etype), self.pred(neg_g, h, etype)
    
class Model():
    def __init__(self, gnn_variant, etypes, etype2pred, g_train, g_test, node_features):
        self.model = GNN(gnn_variant, etypes)
        self.etype2pred = etype2pred
        self.g_train = g_train
        self.g_test = g_test
        self.node_features = node_features
        
    def _train(self, epochs=300):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        
        for epoch in range(epochs+1): 
            self.model.train()
            optimizer.zero_grad()
            
            g_neg_train = construct_negative_graph(self.g_train, self.etype2pred)
            pos_score, neg_score = self.model(self.g_train, g_neg_train, self.node_features, self.etype2pred)

            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')
    
    def _eval(self):
        self.model.eval()
        g_neg_test = construct_negative_graph(self.g_test, self.etype2pred)
        pos_score, neg_score = self.model(self.g_test, g_neg_test, self.node_features, self.etype2pred)
        scores = torch.cat([pos_score, neg_score]).view(-1).detach().numpy()
        scores = normalize_arr(scores, scores.min(), scores.max())
        scores = np.where(scores >= 0.5, 1, 0)
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
        
        precision = precision_score(labels, scores)
        recall = recall_score(labels, scores)
        f1score = f1_score(labels, scores)
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1score:.4f}')

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

def normalize_arr(arr, x_min, x_max):
    '''Helper function to normalize an array'''
    new_arr = []
    for i in arr:
        new_arr.append((i-x_min)/(x_max-x_min))
    return np.array(new_arr)