import pandas as pd
import numpy as np
import torch

def insert_entry(entry, ntype, dic):
    if ntype not in dic:
        dic[ntype] = {}
    node_id = len(dic[ntype])
    if entry not in dic[ntype]:
         dic[ntype][entry] = node_id
    return dic

def get_node_dict(df):
    '''Creates a dict of node-types -> each dictionary further consists of a dictionary mapping a node to an ID'''
    node_dict = {}
    for triple in df.values.tolist():
        src = str(triple[0])
        src_type = src.split('::')[0]
        dest = str(triple[2])
        dest_type = dest.split('::')[0]
        insert_entry(src, src_type, node_dict)
        insert_entry(dest, dest_type, node_dict)
    return node_dict

def get_edge_dict(df, node_dict):
    '''Creates a dict of edge-types -> the key is the edge-type and the value is a list of (src ID, dest ID) tuples'''
    edge_dict = {}
    for triple in df.values.tolist():
        src = str(triple[0])
        src_type = src.split('::')[0]
        dest = str(triple[2])
        dest_type = dest.split('::')[0]
        
        try:
            src_id = node_dict[src_type][src]
            dest_id = node_dict[dest_type][dest]
        
        except:
            continue
        
        pair = (src_id, dest_id)
        etype = (src_type, triple[1], dest_type)
        if etype in edge_dict:
            edge_dict[etype] += [pair]
        else:
            edge_dict[etype] = [pair]
    return edge_dict

def add_node_features(g, n_node_features=100):
    '''Adds random node features for message passing'''
    node_features = {}
    for ntype in g.ntypes:
        g.nodes[ntype].data['h'] = torch.randn(g.num_nodes(ntype), n_node_features).requires_grad_(True)
        node_features[ntype] = g.nodes[ntype].data['h']
    return g, node_features

def get_edge_weight(g):    
    edge_weight = {}
    for edgetype in g.etypes:
        edge_weight[edgetype] = torch.ones(g.num_edges(edgetype)).requires_grad_(True)
    return edge_weight

def scale(x):
    '''Scales attributes to [0,1]'''
    x = x.abs().sum(dim=1)
    x /= x.max()
    return x

def normalize_arr(arr, x_min, x_max):
    '''Normalizes an array'''
    new_arr = []
    for i in arr:
        new_arr.append((i-x_min)/(x_max-x_min))
    return np.array(new_arr)