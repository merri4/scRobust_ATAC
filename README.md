# scRobust
The robust self-supervised learning strategy to tackle the inherent sparsity of single-cell RNA-seq data

# Datasets
https://zenodo.org/records/10602754

# Weights
https://zenodo.org/records/10608134

## Tutorial
```sh
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

```sh

# Overview
![Figure1-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/2ec9e5cc-177a-454f-8ce2-6dbdf89b83cb)

# Results
![Main_results-8](https://github.com/DMCB-GIST/scRobust/assets/31497898/a736e655-ca70-4d75-b35a-ad43e27efcaa)
