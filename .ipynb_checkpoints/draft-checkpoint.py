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
            for i in range(len(attr_edge)):
                if attr_edge[i] > 0:
                    imp_edge_list.append(i)
            if len(imp_edge_list) > 0:
                imp_edge_dic[etype[1]] = imp_edge_list
                
        except Exception:
            pass
        
    return imp_edge_dic