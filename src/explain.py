import pandas as pd
import numpy as np
import operator
import json
import random
import torch
import dgl
from captum.attr import Saliency, IntegratedGradients
from functools import partial
from src.utils import *
from src.gnn import *

model = torch.load(f'Output/GNNModels/GraphSAGE')
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

def get_imp_node_dic(etype, g_single_instance, g_neg_single_instance, mode, k):  
    '''Gets dictionary having most important nodes'''
    imp_node_dic = {}
    
    for ntype in g_single_instance.ntypes:
        try:
            attr_node = get_attr_node(ntype, etype, g_single_instance.ndata['h'], g_single_instance, 
                                      g_neg_single_instance, 0, mode)
            
            imp_node_list = []
            scores = []
            for i in range(len(attr_node)):
                if attr_node[i] > 0:
                    imp_node_list.append(i)
                    scores.append(attr_node[i].item())
            if len(imp_node_list) > 0:
                imp_node_dic[ntype] = imp_node_list
                imp_node_dic[ntype+'_scores'] = scores
                
        except Exception:
            pass
        
    # only keep top k important nodes
    new_imp_node_dic = {}
    for ntype in g_single_instance.ntypes:
        score_dict = {ntype: score for ntype, score in zip(imp_node_dic[ntype], imp_node_dic[ntype+'_scores'])} 
        sorted_dict = dict(sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())
        sorted_values = list(sorted_dict.values())
        new_imp_node_dic[ntype] = sorted_keys[:k]
        new_imp_node_dic[ntype+'_score'] = sorted_values[:k]
            
    return new_imp_node_dic

def get_imp_node_dicts(g, etype, mode, keys, k):
    '''Gets all the important node dictionaries'''

    for i in range(len(g.edges(etype=etype, form='eid'))):
        
        print(mode + ': ' + str(i))
        
        g_single_instance, g_neg_single_instance, u, v = get_single_instance_graph(g, etype, i)
        imp_node_dic = get_imp_node_dic(etype, g_single_instance, g_neg_single_instance, mode, k)
        
        with open(f"Output/Explainability/Alzheimer/imp_node_dict_{i}_{mode}.json", "w") as file:
            json.dump(imp_node_dic, file)

def get_attr_edge(imp_etype, etype, input_features, pos_g, neg_g, edge_weight, idx, mode):
    '''Returns most important edges'''
    x_etype = edge_weight[imp_etype]
    
    if mode == 'ig':
        ig = IntegratedGradients(partial(model_forward_edge_weight, imp_etype=imp_etype, pos_g=pos_g, neg_g=neg_g, input_features=input_features, 
                                         etype=etype, edge_weight=edge_weight, idx=idx))
        attr_edge = ig.attribute(x_etype, target=0, internal_batch_size=1, n_steps=50)
    
    elif mode == 'sal':
        sal = Saliency(partial(model_forward_edge_weight, imp_etype=imp_etype, pos_g=pos_g, neg_g=neg_g, input_features=input_features, 
                               etype=etype, edge_weight=edge_weight, idx=idx))
        attr_edge = sal.attribute(x_etype, target=0)
    
    return attr_edge

def get_imp_edge_dic(etype, g_single_instance, g_neg_single_instance, edge_weight, mode):   
    '''Gets dictionary having most important edges'''
    imp_edge_dic = {}
    
    for imp_etype in g_single_instance.canonical_etypes:
        try:
            attr_edge = get_attr_edge(imp_etype[1], etype, g_single_instance.ndata['h'], g_single_instance, 
                                      g_neg_single_instance, edge_weight, 0, mode)
            
            attr_edge = attr_edge.clamp(min=0, max=1)
            imp_edge_list = []
            scores = []
            for i in range(len(attr_edge)):
                if attr_edge[i] > 0:
                    imp_edge_list.append(i)
                    scores.append(attr_edge[i].item())
            if len(imp_edge_list) > 0:
                imp_edge_dic[etype[1]] = imp_edge_list
                imp_edge_dic[etype[1]+'_scores'] = scores
                
        except Exception:
            pass
        
    return imp_edge_dic

def get_imp_edge_dicts(g, etype, mode, keys, edge_weight):
    '''Gets all the important edge dictionaries'''

    for i in range(len(g.edges(etype=etype, form='eid'))):
        
        print(mode + ': ' + str(i))
        
        g_single_instance, g_neg_single_instance, u, v = get_single_instance_graph(g, etype, i)
        imp_edge_dic = get_imp_edge_dic(etype, g_single_instance, g_neg_single_instance, edge_weight, mode)
        
        with open(f"Output/Explainability/Alzheimer/imp_edge_dict_{i}_{mode}.json", "w") as file:
            json.dump(imp_edge_dic, file)

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
    '''Complement_y_hat - input: graph without important nodes/edges'''
    g_complement = g_single_instance

    for ntype in list(imp_node_dic.keys()):
        # remove influential nodes
        g_complement = dgl.remove_nodes(g_complement, nids=imp_node_dic[ntype], ntype=ntype)

    g_complement = dgl.add_edges(g_complement, u, v, etype=etype)
    g_neg_complement = construct_negative_graph(g_complement, etype, 5)
    return model(g_complement, g_neg_complement, g_complement.ndata['h'], etype)

def get_all_predictions(g, etype, mode, keys):
    '''Gets all the predictions having as input either g, g_explain or g_complement'''
    y_ = []
    y_explain = []
    y_complement = []
    
    for i in range(len(g.edges(etype=etype, form='eid'))):
        
        print(mode + ': ' + str(i))
    
        g_single_instance, g_neg_single_instance, u, v = get_single_instance_graph(g, etype, i)

        with open(f'Output/Explainability/Alzheimer/imp_node_dict_{i}_{mode}.json') as file:
            imp_node_dic = json.load(file)
        imp_node_dic = {k:v for k,v in imp_node_dic.items() if k in keys}

        pos_score, _, embed = model(g_single_instance, g_neg_single_instance, g_single_instance.ndata['h'], etype)
        #rank = get_rank(u, v, embed)
        y = pos_score[0].reshape([1, 1])
        y_.append(y)

        pos_score, _, embed_explain = get_explain_y_hat(g_single_instance, imp_node_dic, etype, u, v,)
        #rank = get_rank(u, v, embed_explain)
        y = pos_score[0].reshape([1, 1])
        y_explain.append(y)

        pos_score, _, embed_complement = get_complement_y_hat(g_single_instance, imp_node_dic, etype, u, v,)
        #rank = get_rank(u, v, embed_complement)
        y = pos_score[0].reshape([1, 1])
        y_complement.append(y)

    return y_, y_explain, y_complement