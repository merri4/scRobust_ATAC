import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
#import copy
from utils import *
from models import *
from random import choices
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

class scRobust():
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.adata = None
        self.data_df = None#copy.deepcopy(data_df)
        self.gene_vocab = None
        self.tokenizer = None
        self.vocab_size = None
        self.embedding = None
        self.encoder = None
        self.model = None
    
    def set_vocab(self):
        self.gene_vocab = pd.DataFrame(data = ['PAD', 'SEP', 'UNKWON', 'CLS', 'MASK']+
                                               self.data_df.columns.tolist())
        self.gene_vocab = self.gene_vocab.reset_index()
        self.gene_vocab.columns = ['ID', 'SYMBOL']
        self.tokenizer = Tokenizer2(self.gene_vocab)
        self.vocab_size = self.gene_vocab.shape[0]; 
        
        return self.gene_vocab, self.tokenizer
        
    def transform_data_df(self):
        ex_genes = np.array(self.tokenizer.convert_symb_to_id(self.data_df.columns))
        
        self.data_df.index = list(range(0,len(self.data_df.index)))
        self.data_df.columns = ex_genes
        
        self.data_df = self.data_df[ex_genes[ex_genes!=2]]
        self.data_df[1] = 1.0
        
        return self.data_df
    
    def set_model(self, hidden, n_layers, attn_heads, att_dropout = 0.3):
        self.embedding = Gene_Embedding(vocab_size=self.vocab_size, embed_size=hidden)
        self.encoder = GeneBERT(embedding = self.embedding, hidden=hidden, n_layers= n_layers, attn_heads=attn_heads)
        
        self.model = CL_GE_BERT(y_dim = hidden, dropout_ratio = att_dropout,
                           device = self.device, encoder = self.encoder).to(self.device)
        
        return self.model
    
        
    def read_adata(self, adata_path = '', set_self = True, normalize_total = True, log1p = True, min_genes = 1, min_cells = 1):
        adata = sc.read_h5ad(adata_path)
        
        if normalize_total: sc.pp.normalize_total(adata, target_sum=1e4)
        if log1p: sc.pp.log1p(adata)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        if set_self: 
            self.set_adata(adata); self.set_df(adata); 
        
        return adata
        
    def set_adata(self, adata):
        self.adata = adata
        
    def set_df(self, adata):
        X = csr_matrix(adata.X)
        self.data_df = pd.DataFrame(data = X.toarray(), columns = adata.var.index, index = adata.obs.index)
        self.data_df = self.data_df[self.data_df.columns.sort_values()]
        
        return self.data_df
        
    def df_2_adata(self, df, normalize_total = True, log1p = True, min_genes = 1, min_cells = 1):
        adata = sc.AnnData(df)
        
        if normalize_total: sc.pp.normalize_total(adata, target_sum=1e4)
        if log1p: sc.pp.log1p(adata)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        return adata
    
    def sorted_by_highly_unique_genes(self, data_df):
        zero_count = (np.sum(data_df==0)/data_df.shape[0])
        high_zero_genes = zero_count.sort_values(ascending=False).index
        data_df = data_df[high_zero_genes]

        return data_df
    
    
    def train_SSL(self, epoch = 1000, lr = 0.00005, batch_size = 128, n_ge = 250, save_path = './'):
        
        self.data_df = self.set_df(self.adata)
        self.transform_data_df()
        
        all_genes = self.data_df.columns
        index_range = np.array(range(len(all_genes)))
        
        all_cells = self.data_df.index
        
        ce_fun = torch.nn.CrossEntropyLoss(); mse_fun = torch.nn.MSELoss();
    
        train_cells, test_cells = train_test_split(all_cells,test_size = 0.1)
        train_df = self.data_df.loc[train_cells]; test_df = self.data_df.loc[test_cells]; 
    
        train_ds = TensorDataset(torch.tensor(train_df.values))
        test_ds = TensorDataset(torch.tensor(test_df.values))
                    
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
        train_cl_loss = []; test_cl_loss = []; train_ge_loss = []; test_ge_loss = []; 
            
        best_loss = 100.
        stop_count = 0
        
        for ep in range(epoch):
                
            step =0; sum_cl_loss = 0.0; sum_ge_loss = 0.0;
            self.model.train();
            for ge_values, in train_dataloader:
              
                n_samples = len(ge_values)
                
                optimizer.zero_grad(); step +=1;
                
                ge_values = ge_values.repeat(2,1).numpy()
                
                rand_index = [[index_range[-1]] + choices(index_range[ge_values[i]!=0],k=n_ge) for i in range(len(ge_values))]
                rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                
                x_genes = torch.tensor(rand_genes).to(self.device); 
                x_scales = torch.tensor(rand_scales, dtype=torch.float).to(self.device);
                
                ge_x_genes = x_genes[n_samples:,:]
                ge_x_scales = x_scales[n_samples:,:]
                
                logits, labels, pred_y = self.model(x_genes,x_scales, ge_x_genes)
                
                ge_loss = mse_fun(pred_y, ge_x_scales)
                cl_loss = ce_fun(logits, labels)
                
                loss = ge_loss + cl_loss
                loss.backward()
                optimizer.step()
                    
                sum_cl_loss += cl_loss.item(); 
                sum_ge_loss += ge_loss.item(); 
                
                    
            cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
            print("Train cl_loss: ", cl_loss);  print("Train ge_loss: ", ge_loss);  
            train_cl_loss.append(cl_loss); train_ge_loss.append(ge_loss);
                
            self.model.eval(); step =0; sum_cl_loss = 0.0; sum_ge_loss = 0.0;
            
            for ge_values, in test_dataloader:
                n_samples = len(ge_values); step +=1;
                ge_values = ge_values.repeat(2,1).numpy()
                
                rand_index = [[index_range[-1]] + choices(index_range[ge_values[i]!=0],k=n_ge) for i in range(len(ge_values))]
                rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                
                x_genes = torch.tensor(rand_genes).to(self.device);
                x_scales = torch.tensor(rand_scales, dtype=torch.float).to(self.device);
                
                ge_x_genes = x_genes[n_samples:,:]
                ge_x_scales = x_scales[n_samples:,:]
                
                logits, labels, pred_y = self.model(x_genes, x_scales, ge_x_genes)
                
                ge_loss = mse_fun(pred_y, ge_x_scales)
                cl_loss = ce_fun(logits, labels)
                   
                sum_cl_loss += cl_loss.item(); 
                sum_ge_loss += ge_loss.item(); 
                
                    
            cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
            print("Test cl_loss: ", cl_loss);  print("Test ge_loss: ", ge_loss);  
            test_cl_loss.append(cl_loss); test_ge_loss.append(ge_loss);
                
            if best_loss > test_cl_loss[-1]:
                self.model = self.model.cpu()
                best_loss = test_cl_loss[-1]                 
                
                torch.save(self.model.state_dict(), save_path+'scRobust_model.pt')
                torch.save(self.model.encoder.state_dict(), save_path+'scRobust_model_encoder.pt')
                    
                self.model = self.model.to(self.device)
            
            
            if test_cl_loss[-1] > train_cl_loss[-1]: stop_count += 1
            if stop_count > 10: break;
            
                
        return train_cl_loss, train_ge_loss, test_cl_loss, test_ge_loss
    
    def load_encoder_weight(self, weight_path):
        state_dict = torch.load(weight_path)
        self.encoder.load_state_dict(state_dict)
    
    def load_model_weight(self, weight_path):
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict)
    
    def get_cell_embeddings(self, adata = False, n_ge = 400, batch_size = 64):
        
        if adata: self.adata = adata
            
        self.data_df = self.set_df(self.adata)
        self.set_vocab()
            
        self.data_df = self.sorted_by_highly_unique_genes(self.data_df)
        
        self.transform_data_df()
        self.data_df[4] = 0.
        
        all_genes = self.data_df.columns
        index_range = np.array(range(len(all_genes)))

        ds = TensorDataset(torch.tensor(self.data_df.values))
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        self.encoder.eval();
        
        xs = []
        
        for ge_values, in dataloader:
            ge_values = ge_values.numpy()
                            
            rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                          if sum(ge_values[i]!=0) > n_ge
                          else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                          for i in range(len(ge_values))]
                            
            rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
            rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                                
            x_genes = torch.tensor(rand_genes).to(self.device); x_scales = torch.tensor(rand_scales).to(self.device);
            
            x = self.encoder(x_genes,x_scales)[:,0,:]
            xs.append(x.cpu().detach().numpy())
        
        cell_embeddings = np.concatenate(xs)
        
        return cell_embeddings
    
    def get_cell_adata(self, cell_embeddings, umap = True, tsne = True, leiden = True, 
                       n_comps = 50, n_neighbors=10, n_pcs=50):
        
        cell_df = pd.DataFrame(data = cell_embeddings, index = self.adata.obs.index)
        scRobust_adata = sc.AnnData(cell_df)   
        
        sc.pp.pca(scRobust_adata, n_comps=n_comps)
        sc.pp.neighbors(scRobust_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        if umap: sc.tl.umap(scRobust_adata)
        if tsne: sc.tl.tsne(scRobust_adata)
        if leiden: sc.tl.leiden(scRobust_adata)
        
        return scRobust_adata
    
    

        
        