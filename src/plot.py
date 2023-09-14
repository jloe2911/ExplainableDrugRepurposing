import pandas as pd

def get_drug_disease_names(compound_id, disease_id):
    #drugbank
    df_db_voc = pd.read_csv('Input/DRKG/DB_vocabulary.csv')

    mapping_db = {}
    for i, row in df_db_voc.iterrows():
        mapping_db[row['DrugBank ID']] = row['Common name']
    
    #mesh
    df_mesh_voc = pd.read_csv('Input/DRKG/MESH_vocabulary.tsv', sep='\t')

    mapping_mesh = {}
    for i, row in df_mesh_voc.iterrows():
        mapping_mesh[row['d']] = row['name']
        
    #names
    compound_name = mapping_db[compound_id.split('::')[1]]
    disease_name = mapping_mesh['mesh:'+disease_id.split(':')[3]]
    
    return compound_name, disease_name