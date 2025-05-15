import torch
from scRobust import *
import scanpy as sc
import pandas as pd
import numpy as np
import argparse

# https://huggingface.co/datasets/UCSC-VLAA/MiniAtlas 데이터셋

def parse_arguments() :
    parser = argparse.ArgumentParser(description='Argparse')
    
    parser.add_argument('--data_path', type=str, default="./miniatlas/kidney_atlas_rna.h5ad")
    parser.add_argument('--save_path', type=str, default="./weights/")
    parser.add_argument('--model_path', type=str, default='./weights/Segerstolpe_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_128_encoder.pt')
    

    parser.add_argument('--seed', type=int, default=821)
    parser.add_argument('--mode', type=str, default="train")
    
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_gene', type=int, default=250)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int, default=1)


    parser.add_argument('--test_data_path', type=str, default="./data/seq_class.test.csv")
    parser.add_argument('--eval_step', type=int, default=20)
    parser.add_argument('--save_step', type=int, default=20)

    args = parser.parse_args()

    return args

if __name__ == "__main__" :

    args = parse_arguments()

    ### =======================================================================
    ### Device Setting 
    ### =======================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)


    ### =======================================================================
    ### Model
    ### =======================================================================    
    scRobust = scRobust(DEVICE)


    ### =======================================================================
    ### Data 
    ### =======================================================================
    scRobust.read_adata(args.data_path)
    gene_vocab, tokenizer = scRobust.set_vocab()
    print(f"Vocab size : {tokenizer.vocab_size}")

    # RAM이 부족해! 
    # 1.2G인 BMMC를 돌릴 때 75%를 사용.
    
    #  11G full_atlas_atac.h5ad
    # 3.6G full_atlas_rna.h5ad
    
    # 6.6G bmmc_atlas_atac.h5ad
    # 1.2G bmmc_atlas_rna.h5ad
    
    # 3.1G kidney_atlas_atac.h5ad
    # 909M kidney_atlas_rna.h5ad
    
    # 4.3G pbmc_atlas_atac.h5ad
    # 1.9G pbmc_atlas_rna.h5ad
    


    ### =======================================================================
    ### param setting
    ### =======================================================================    

    scRobust.set_encoder(hidden = args.embedding_dim*args.num_heads, n_layers = args.num_blocks, attn_heads= args.num_heads)
    scRobust.set_pretraining_model(hidden = args.embedding_dim * args.num_heads, att_dropout = 0.3)

    ### Train
    if args.mode == "train" :
        scRobust.train_SSL(epoch = args.epoch, lr = args.lr, batch_size = args.batch_size, n_ge = args.num_gene, save_path = args.save_path)

    ### Inference
    else :

        scRobust.load_encoder_weight(args.model_path)

        cell_embeddings = scRobust.get_cell_embeddings(n_ge = 400, batch_size = 64)
        scRobust_adata = scRobust.get_cell_adata(cell_embeddings, umap = False, tsne = True, leiden = True, n_comps = 50, n_neighbors=10, n_pcs=50)

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
        scRobust.set_downstream_model(hidden = args.embedding_dim*args.num_heads, n_clssses = n_classes, att_dropout = 0.3)

        total_val_auc, total_val_loss, total_val_f1, total_val_acc, total_test_auc, total_test_loss, total_test_f1, total_test_acc = scRobust.train_DS(epoch = 20, lr = 5e-5, batch_size = 64, n_ge = 800)
