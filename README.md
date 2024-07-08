# scRobust
The robust self-supervised learning strategy to tackle the inherent sparsity of single-cell RNA-seq data

# Datasets
https://zenodo.org/records/10602754

# Weights
https://zenodo.org/records/10608134

## Tutorial
```sh
from scRobust import *

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda_condition else "cpu")
scRobust = scRobust(device)
adata_dict = './data/Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad'
scRobust.read_adata(adata_dict)
gene_vocab, tokenizer = scRobust.set_vocab()

scRobust.set_model(hidden = 64*8, n_layers = 1, attn_heads= 8)
## pre_train scRobust
scRobust.train_SSL(epoch = 1000, lr = 0.00005, batch_size = 128, n_ge = 250, save_path = './weights/')
## load weight
weight_path = './weights/Segerstolpe_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_128_encoder.pt'
scRobust.load_encoder_weight(weight_path)

## get cell embeddings
cell_embeddings = scRobust.get_cell_embeddings(n_ge = 400, batch_size = 64)
scRobust_adata = scRobust.get_cell_adata(cell_embeddings, umap = False, tsne = True, leiden = True, n_comps = 50, n_neighbors=10, n_pcs=50)

scRobust_adata.obs['label'] = scRobust.adata.obs['label']
sc.pl.tsne(scRobust_adata, color='label')
sc.pl.tsne(scRobust_adata, color=['leiden'])
```
![cell_embeddings_cell_types](https://github.com/DMCB-GIST/scRobust/assets/31497898/00649a67-6005-45b3-8245-6a63c5c37504)

# Overview
![Figure1-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/2ec9e5cc-177a-454f-8ce2-6dbdf89b83cb)

# Results
![Main_results-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/a736e655-ca70-4d75-b35a-ad43e27efcaa)
