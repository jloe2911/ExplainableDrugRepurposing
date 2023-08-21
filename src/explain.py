import pandas as pd
import numpy as np
import torch
import dgl
from captum.attr import Saliency, IntegratedGradients
from functools import partial
from src.utils import *
from src.gnn import *

model = torch.load(f'Output/GNNModels/GCN')
model = model.model

def get_single_instance_graph(g, etype, idx):
    u = g.edges(etype=etype, form='uv')[0][idx]
    v = g.edges(etype=etype, form='uv')[1][idx]
    
    indices2remove = list(np.arange(len(g.edges(etype=etype, form='eid').detach().numpy())))
    indices2remove.pop(idx)
    
    g_single_instance = dgl.remove_edges(g, etype=etype, eids=indices2remove)
    g_neg_single_instance = construct_negative_graph(g_single_instance, etype, 5) 
    return g_single_instance, g_neg_single_instance, u, v

def model_forward(x, imp_ntype, pos_g, neg_g, input_features, etype, idx):
    '''Forward function that returns a single tensor'''
    input_features = input_features.copy()
    input_features[imp_ntype] = x
    return model(pos_g, neg_g, input_features, etype)[0][idx].reshape([1, 1])

def model_forward_edge_weight(x, imp_etype, pos_g, neg_g, input_features, etype, edge_weight, idx):
    '''Forward function with edge weights that returns a single tensor'''
    edge_weight = edge_weight.copy()
    edge_weight[imp_etype] = x
    return model(pos_g, neg_g, input_features, etype, edge_weight)[0][idx].reshape([1, 1])

def get_attr_node(imp_ntype, etype, input_features, pos_g, neg_g, idx, mode):
    '''Returns most important nodes'''
    x_ntype = input_features[imp_ntype]
    
    if mode == 'ig':
        ig = IntegratedGradients(partial(model_forward, imp_ntype=imp_ntype, pos_g=pos_g, neg_g=neg_g, input_features=input_features, 
                                         etype=etype, idx=idx))
        attr_node = ig.attribute(x_ntype, target=0, internal_batch_size=1, n_steps=50)
    
    elif mode == 'sal':
        sal = Saliency(partial(model_forward, imp_ntype=imp_ntype, pos_g=pos_g, neg_g=neg_g, input_features=input_features, 
                               etype=etype, idx=idx))
        attr_node = sal.attribute(x_ntype, target=0)
    
    return scale(attr_node)


def get_imp_node_dic(etype, g_single_instance, g_neg_single_instance, mode):  
    '''Gets dictionary having most important nodes'''
    imp_node_dic = {}
    
    for ntype in g_single_instance.ntypes:
        try:
            attr_node = get_attr_node(ntype, etype, g_single_instance.ndata['h'], g_single_instance, 
                                      g_neg_single_instance, 0, mode)
            
            imp_node_list = []
            for i in range(len(attr_node)):
                if attr_node[i] > 0.5:
                    imp_node_list.append(i)
            if len(imp_node_list) > 0:
                imp_node_dic[ntype] = imp_node_list
                
        except Exception:
            pass
        
    return imp_node_dic

def get_explain_y_hat(g_single_instance, imp_node_dic, etype, u, v):
    '''Explain_y_hat - input: graph without unimportant nodes/edges'''
    g_explain = g_single_instance

    for ntype in g_single_instance.ntypes:
        if ntype in list(imp_node_dic.keys()):
            # keep influential nodes
            nids = list(set(g_single_instance.nodes(ntype).detach().numpy()) - set(imp_node_dic[ntype]))
            g_explain = dgl.remove_nodes(g_explain, nids=nids, ntype=ntype)
        else:
            nids = list(set(g_single_instance.nodes(ntype).detach().numpy()))
            g_explain = dgl.remove_nodes(g_explain, nids=nids, ntype=ntype)

    g_explain = dgl.add_edges(g_explain, u, v, etype=etype)
    g_neg_explain = construct_negative_graph(g_explain, etype, 5)
    return model(g_explain, g_neg_explain, g_explain.ndata['h'], etype)

def get_complement_y_hat(g_single_instance, imp_node_dic, etype, u, v):   
    '''Complement_y_hat - input: graph without unimportant nodes/edges'''
    g_complement = g_single_instance

    for ntype in list(imp_node_dic.keys()):
        # remove influential nodes
        g_complement = dgl.remove_nodes(g_complement, nids=imp_node_dic[ntype], ntype=ntype)

    g_complement = dgl.add_edges(g_complement, u, v, etype=etype)
    g_neg_complement = construct_negative_graph(g_complement, etype, 5)
    return model(g_complement, g_neg_complement, g_complement.ndata['h'], etype)