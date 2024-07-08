#from collections import Counter 

from scRobust import *
import scanpy as sc
import pandas as pd
import numpy as np


cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda_condition else "cpu")

scRobust = scRobust(device)

adata_dict = './data/Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad'

scRobust.read_adata(adata_dict)

gene_vocab, tokenizer = scRobust.set_vocab()

d = 64; attn_heads = 8; hidden = d*attn_heads; n_layers = 1; n_ge = 250;
scRobust.set_encoder(hidden = 64*8, n_layers = 1, attn_heads= 8)
scRobust.set_pretraining_model(hidden = 64*8, att_dropout = 0.3)

save_path = './weights/'

#scRobust.train_SSL(epoch = 1000, lr = 0.00005, batch_size = 128, n_ge = n_ge, save_path = save_path)

weight_path = './weights/Segerstolpe_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_128_encoder.pt'

scRobust.load_encoder_weight(weight_path)

cell_embeddings = scRobust.get_cell_embeddings(n_ge = 400, batch_size = 64)
scRobust_adata = scRobust.get_cell_adata(cell_embeddings, umap = False, tsne = True, leiden = True, 
                                         n_comps = 50, n_neighbors=10, n_pcs=50)

scRobust_adata.obs['label'] = scRobust.adata.obs['label']
sc.pl.tsne(scRobust_adata, color='label')
sc.pl.tsne(scRobust_adata, color=['leiden'])    


clinical_labels = pd.read_csv('./data/Segerstolpe_HumanPancreas_clinical.csv')
sample_ids = clinical_labels['Characteristics [individual]']
sample_ids.index = sample_ids.index.astype(str)

HbA1c_arr = []
count = 0
for i in clinical_labels['Characteristics [clinical information]']:
    try:
        HbA1c_arr.append(float((i[6:9])))
    except:
        HbA1c_arr.append(5.5)
    count +=1
    
HbA1c_arr = np.array(HbA1c_arr)

labels = scRobust_adata.obs['label']
target_cell_type = ['alpha','beta','gamma','delta']

cell_types  = labels[(HbA1c_arr>0) * (labels.isin(target_cell_type))]
HbA1c  = HbA1c_arr[(HbA1c_arr>0) * (labels.isin(target_cell_type))]
s_ids  = sample_ids[(HbA1c_arr>0) *(labels.isin(target_cell_type))]

adata_df = pd.DataFrame(data = cell_embeddings[labels.isin(target_cell_type)],
                               index = labels.index[labels.isin(target_cell_type)])

sc_adata = sc.AnnData(adata_df)   

sc_adata.obs['HbA1c'] = HbA1c 
sc_adata.obs['cell_types'] = cell_types.tolist()
sc_adata.obs['sample_ids'] = s_ids.tolist()

sc.pp.pca(sc_adata, n_comps=50)
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=50)

sc.tl.tsne(sc_adata)
sc.tl.leiden(sc_adata)

sc.pl.tsne(sc_adata, color='HbA1c')    
sc.pl.tsne(sc_adata, color='cell_types')    
sc.pl.tsne(sc_adata, color='sample_ids')    


## Downstream task
indices = ~scRobust.adata.obs['label'].isin(['co-expression','unclassified endocrine']).values
scRobust.adata = scRobust.adata[indices]

n_classes = len(scRobust.adata.obs['label'].cat.categories)
scRobust.set_downstream_model(hidden = 64*8, n_clssses = n_classes, att_dropout = 0.3)

total_val_auc, total_val_loss, total_val_f1, total_val_acc, \
        total_test_auc, total_test_loss, total_test_f1, total_test_acc = scRobust.train_DS(epoch = 20, lr = 5e-5, batch_size = 64, n_ge = 800)
