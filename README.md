# scRobust
The robust self-supervised learning strategy to tackle the inherent sparsity of single-cell RNA-seq data

# Datasets
https://zenodo.org/records/10602754

# Weights
https://zenodo.org/records/10608134
https://zenodo.org/records/12741301

# Tutorial
with Python >= 2.1.2, sklearn, scipy, scanpy

```sh
from scRobust import *

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda_condition else "cpu")
scRo = scRobust(device)
adata_path = './data/Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad'
scRo.read_adata(adata_path)
gene_vocab, tokenizer = scRo.set_vocab()

scRo.set_encoder(hidden = 64*8, n_layers = 1, attn_heads= 8)

## pre_train scRobust
scRo.set_pretraining_model(hidden = 64*8, att_dropout = 0.3)
scRo.train_SSL(epoch = 1000, lr = 0.00005, batch_size = 128, n_ge = 250, save_path = './weights/')

## load weight
weight_path = './weights/Segerstolpe_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_128_encoder.pt'
scRo.load_encoder_weight(weight_path)

## get cell embeddings
cell_embeddings = scRo.get_cell_embeddings(n_ge = 400, batch_size = 64)
scRobust_adata = scRo.get_cell_adata(cell_embeddings, umap = False, tsne = True, leiden = True, n_comps = 50, n_neighbors=10, n_pcs=50)

scRobust_adata.obs['label'] = scRo.adata.obs['label']
sc.pl.tsne(scRobust_adata, color='label')
sc.pl.tsne(scRobust_adata, color=['leiden'])

## Downstream task
indices = ~scRo.adata.obs['label'].isin(['co-expression','unclassified endocrine']).values
scRo.adata = scRo.adata[indices]

n_classes = len(scRo.adata.obs['label'].cat.categories)
scRo.set_downstream_model(hidden = 64*8, n_clssses = n_classes, att_dropout = 0.3)

val_auc, val_loss, val_f1, val_acc, \
        test_auc, test_loss, test_f1, test_acc = scRo.train_DS(epoch = 20, lr = 5e-5, batch_size = 64, n_ge = 800)

```
<img src="https://github.com/DMCB-GIST/scRobust/assets/31497898/00649a67-6005-45b3-8245-6a63c5c37504" alt="cell_embeddings_cell_types" width="600"/>

Large scRobust with 8 layers and large vocaburary with 42,160 tokens

```sh

from scRobust import *
import scanpy as sc
import pandas as pd
import numpy as np

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:3" if cuda_condition else "cpu")

scRo = scRobust(device)

vocab_path = './vocab/whole_human_vocab.csv'
adata_path = './data/Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad'

scRo.load_vocab(vocab_path)

## For the larger version, normalize_total -> log1p step is needed. 
## Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad has already been pre-processed.
scRo.read_adata(adata_path,normalize_total = False, log1p = False) 

## Pre-processing for being sure. Therefore, it's not necessary.
scRo.adata.X = np.expm1(scRo.adata.X)
sc.pp.normalize_total(scRo.adata, target_sum=1e4)
sc.pp.log1p(scRo.adata)

d = 64; attn_heads = 8; hidden = d*attn_heads; n_layers = 0; n_ge = 400;
scRo.set_encoder(hidden = 64*8, n_layers = n_layers, attn_heads= 8)
scRo.set_pretraining_model(hidden = 64*8, att_dropout = 0.3)

save_path = './weights/test_w_weight_ly8_bt_128'
weight_path = './weights/Whole_Human_BERT_Hid_512_Att_8_nGenes_200_ly_8_bt_256.pt'
scRo.load_model_weight(weight_path)

## can do fine-tuning
train_cl_loss, train_ge_loss, test_cl_loss, test_ge_loss = scRo.train_SSL(epoch = 100, lr = 0.0001, batch_size = 128, 
                                                                              n_ge = n_ge, save_path = save_path, simple = True)

## get cell embeddings
cell_embeddings = scRo.get_cell_embeddings(n_ge = 1000, batch_size = 128, use_HUGs = False, use_HVGs = True, simple = True)
scRobust_adata = scRo.get_cell_adata(cell_embeddings, umap = False, tsne = True, leiden = True, 
                                         n_comps = 50, n_neighbors=10, n_pcs=50)

```
# Overview
![Figure1-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/2ec9e5cc-177a-454f-8ce2-6dbdf89b83cb)

# Results
![Main_results-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/a736e655-ca70-4d75-b35a-ad43e27efcaa)
