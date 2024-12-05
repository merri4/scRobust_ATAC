import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
#import copy
from utils import *
from models import *
from random import choices
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,\
                            roc_auc_score, confusion_matrix, precision_recall_curve

class scRobust():
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.device = device
        self.adata = None
        self.data_df = None #copy.deepcopy(data_df)
        self.gene_vocab = None
        self.tokenizer = None
        self.vocab_size = None
        self.embedding = None
        self.encoder = None
        self.model = None
    
    def load_vocab(self,vocab_path):
        self.gene_vocab = pd.read_csv(vocab_path)
        self.tokenizer = Tokenizer2(self.gene_vocab)
        self.vocab_size = self.gene_vocab.shape[0]
        
        return self.gene_vocab, self.tokenizer
        
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
    
    def set_encoder(self, hidden, n_layers, attn_heads):
        self.embedding = Gene_Embedding(vocab_size=self.vocab_size, embed_size=hidden)
        self.encoder = GeneBERT(embedding = self.embedding, hidden=hidden, n_layers= n_layers, attn_heads=attn_heads)
        
        return self.encoder
    
    def set_pretraining_model(self, hidden, att_dropout = 0.3):
        self.model = CL_GE_BERT(y_dim = hidden, dropout_ratio = att_dropout,
                           device = self.device, encoder = self.encoder).to(self.device)
        
        return self.model
    
    def set_downstream_model(self, hidden, n_clssses, att_dropout = 0.3):
        self.model = Downstream_BERT(y_dim = hidden, o_dim = n_clssses, dropout_ratio = att_dropout,
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
        
        if self.gene_vocab is not None:
            common_genes = list(set(self.data_df.columns).intersection(self.gene_vocab['SYMBOL']))
            self.data_df = self.data_df[common_genes]
            
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
    
    def sorted_by_HVGs(self, data_df):
        std = np.std(data_df)
        data_df = data_df[std.sort_values(ascending = False).index]
        
        return data_df
    
    
    def train_SSL(self, epoch = 1000, lr = 0.00005, batch_size = 128, n_ge = 250, save_path = './', pooling = ''):
        sum_pooling = False; max_pooling = False; mean_pooling = False
        if pooling == 'sum': sum_pooling = True
        if pooling == 'max': max_pooling = True
        if pooling == 'mean': mean_pooling = True
        
        
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
                
                
                logits, labels, pred_y = self.model(x_genes,x_scales, ge_x_genes, sum_pooling = sum_pooling,
                                                    max_pooling = max_pooling, mean_pooling = mean_pooling)
                
                ge_loss = mse_fun(pred_y, ge_x_scales)
                cl_loss = ce_fun(logits, labels)
                
                loss = ge_loss + cl_loss
                loss.backward()
                optimizer.step()
                    
                sum_cl_loss += cl_loss.item(); 
                sum_ge_loss += ge_loss.item(); 
                
                    
            cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
            print("Epoch:",ep," Train cl_loss: ", round(cl_loss,4), "Train ge_loss: ", round(ge_loss,4));  
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
                
                logits, labels, pred_y = self.model(x_genes,x_scales, ge_x_genes, sum_pooling = sum_pooling,
                                                    max_pooling = max_pooling, mean_pooling = mean_pooling)
                
                
                ge_loss = mse_fun(pred_y, ge_x_scales)
                cl_loss = ce_fun(logits, labels)
                   
                sum_cl_loss += cl_loss.item(); 
                sum_ge_loss += ge_loss.item(); 
                
                    
            cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
            print("Epoch:",ep," Test cl_loss: ", round(cl_loss,4), "Test ge_loss: ", round(ge_loss,4));  
            test_cl_loss.append(cl_loss); test_ge_loss.append(ge_loss);
                
            if best_loss > test_cl_loss[-1]:
                self.model = self.model.cpu()
                best_loss = test_cl_loss[-1]                 
                
                torch.save(self.model.state_dict(), save_path+'scRobust_model.pt')
                torch.save(self.model.encoder.state_dict(), save_path+'scRobust_model_encoder.pt')
                    
                self.model = self.model.to(self.device)
            
            
            #if test_cl_loss[-1] > train_cl_loss[-1]: stop_count += 1
            #if stop_count > 10: break;
            
                
        return train_cl_loss, train_ge_loss, test_cl_loss, test_ge_loss
    
    
    
    def get_labels(self, ):
        label_codes = list(set(self.adata.obs['label']))
        label_codes.sort()
        labels = np.array([label_codes.index(self.adata.obs['label'][i]) for i in range(len(self.adata.obs['label']))])
        
        return labels
    
    def train_DS(self, epoch = 20, lr = 5e-5, batch_size = 64, n_ge = 800, pooling = ''):
        sum_pooling = False; max_pooling = False; mean_pooling = False
        if pooling == 'sum': sum_pooling = True
        if pooling == 'max': max_pooling = True
        if pooling == 'mean': mean_pooling = True
        
        self.data_df = self.set_df(self.adata)
        #self.set_vocab()
            
        self.data_df = self.sorted_by_highly_unique_genes(self.data_df)
        self.transform_data_df()
        self.data_df[4] = 0.
        
        labels = self.get_labels()
        all_genes = self.data_df.columns
        index_range = np.array(range(len(all_genes)))
        all_cells = self.data_df.index
        
        loss_fun = torch.nn.CrossEntropyLoss()
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        
        total_val_auc = []; total_val_loss = []; total_val_f1 = []; total_val_acc = []
        total_test_auc = []; total_test_loss = []; total_test_f1 = []; total_test_acc = []
        
        fold = 0; 
        for train_index, test_index in kfold.split(all_cells, labels):
            fold += 1;
            
            try:
                train_index, val_index = train_test_split(train_index,test_size = 0.1, stratify = labels.iloc[train_index])
            except:
                train_index, val_index = train_test_split(train_index,test_size = 0.1)
                
            train_ds = TensorDataset(torch.tensor(self.data_df.iloc[train_index].values),
                                     torch.tensor(labels[train_index]))
            val_ds = TensorDataset(torch.tensor(self.data_df.iloc[val_index].values),
                                   torch.tensor(labels[val_index]))
            test_ds = TensorDataset(torch.tensor(self.data_df.iloc[test_index].values),
                                    torch.tensor(labels[test_index]))
            
            train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
           
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) #0.00005
            
            
            train_auc = []; test_auc = []; val_auc = [];              
            train_f1 = []; test_f1 = []; val_f1 = []; train_acc = []; test_acc = [];val_acc = [];                        
            train_acc = []; val_acc = []; test_acc = []; 
            train_loss = []; val_loss = []; test_loss = []; 
        
            best_error = 100;
            
            for ep in range(epoch):
                self.model.train();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0; 
                all_y = []; all_pred_y = []; all_prob_y = [];
                for ge_values, y in train_dataloader:
                    n_samples = len(ge_values)
                    optimizer.zero_grad(); step +=1;
                    
                    y = y.to(self.device, dtype=torch.int64)
                    
                    ge_values = ge_values.numpy()
                    
                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]
                    
                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                    
                    x_genes = torch.tensor(rand_genes).to(self.device); 
                    x_scales = torch.tensor(rand_scales, dtype=torch.float).to(self.device);
                    
                    pred_y = self.model(x_genes,x_scales, sum_pooling = sum_pooling,
                                        max_pooling = max_pooling, mean_pooling = mean_pooling)
                    
                    soft_y = F.softmax(pred_y,dim = 1)
                    prob, predicted = torch.max(soft_y, 1)
                    
                    acc = (predicted == y).sum().item()
                    accuracy += acc
                
                    loss = loss_fun(pred_y,y)
                
                    loss.backward(); optimizer.step();
                    
                    sum_loss += loss.item(); count += len(y)
    
                    all_prob_y.append(soft_y.cpu().detach().numpy())
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(predicted.cpu().detach().numpy())
                    
                    if (step+1) %100 ==0: print('Step ', step, ' Acc: ', acc/len(y))
                    
                    
                loss_train = sum_loss/(step+1)
                ACC = accuracy/count
                weighted_F1 = f1_score(all_y, all_pred_y, average='weighted')
                F1 = f1_score(all_y, all_pred_y, average='macro')
                
                try:
                    AUC = roc_auc_score(all_y, np.concatenate(all_prob_y), average='weighted', multi_class = 'ovo')
                except:
                    AUC = 1
                    
                print("Epoch: ", ep, "Train ACC: ", ACC, "Train F1: ", F1, "Train AUC: ", AUC, "Train weighted F1: ", weighted_F1)
                train_acc.append(ACC); train_f1.append(F1); train_auc.append(AUC); train_loss.append(loss_train)
                
                self.model.eval();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0; 
                all_prob_y = []; all_y = []; all_pred_y = [];
                for ge_values, y in val_dataloader:
                    step +=1; ge_values = ge_values.numpy()
                    y = y.to(self.device, dtype=torch.int64)
                    
                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]
                    
                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                        
                    x_genes = torch.tensor(rand_genes).to(self.device); x_scales = torch.tensor(rand_scales).to(self.device);
                    
                    pred_y = self.model(x_genes,x_scales)
                    soft_y = F.softmax(pred_y,dim = 1)
                    prob, predicted = torch.max(soft_y, 1)
                    accuracy += (predicted == y).sum().item()
                
                    loss = loss_fun(pred_y,y)
                    sum_loss += loss.item(); count += len(y)
                    
                    all_prob_y.append(soft_y.cpu().detach().numpy())
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(predicted.cpu().detach().numpy())
                    
                loss_val = sum_loss/(step+1)
                ACC = accuracy/count
                weighted_F1 = f1_score(all_y, all_pred_y, average='weighted')
                F1 = f1_score(all_y, all_pred_y, average='macro')
                try:
                    AUC = roc_auc_score(all_y, np.concatenate(all_prob_y), average='weighted', 
                                        multi_class = 'ovo')#, labels = np.array(range(33)))
                except:
                    AUC = 1
                    
                print("Val ACC: ", ACC, "Val F1: ", F1, "Val AUC: ", AUC, "Val weighted F1: ", weighted_F1)
                val_acc.append(ACC); val_f1.append(F1); val_auc.append(AUC); val_loss.append(loss_val)
                
                self.model.eval();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0;
                all_prob_y = []; all_y = []; all_pred_y = [];
                for ge_values, y in test_dataloader:
                    step +=1;ge_values = ge_values.numpy()
                    y = y.to(self.device, dtype=torch.int64)
                    
                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]
                    
                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]
                            
                    x_genes = torch.tensor(rand_genes).to(self.device); x_scales = torch.tensor(rand_scales).to(self.device);
                    
                    pred_y = self.model(x_genes,x_scales)
                    
                    soft_y = F.softmax(pred_y,dim = 1)
                    
                    prob, predicted = torch.max(soft_y, 1)
                    
                    accuracy += (predicted == y).sum().item()
                
                    loss = loss_fun(pred_y,y)
                    
                    sum_loss += loss.item(); count += len(y)
    
                    all_prob_y.append(soft_y.cpu().detach().numpy())
                    all_y += list(y.cpu().detach().numpy())
                    all_pred_y += list(predicted.cpu().detach().numpy())
                
                all_prob_y = np.concatenate(all_prob_y)
                target_probs, _ = torch.max(torch.tensor(all_prob_y),dim=1)
                
                loss_test = sum_loss/(step+1)
                ACC = accuracy/count
                weighted_F1 = f1_score(all_y, all_pred_y, average='weighted')
                F1 = f1_score(all_y, all_pred_y, average='macro')
                
                try:
                    AUC = roc_auc_score(all_y, all_prob_y, average='weighted', multi_class = 'ovo')
                except:
                    AUC = 1
                    
                print("Test ACC: ", ACC, "Test F1: ", F1, "Test AUC: ", AUC, "Test weighted F1: ", weighted_F1)
                test_acc.append(ACC); test_f1.append(F1); test_auc.append(AUC); test_loss.append(loss_test)
                
            best_index = np.argsort(val_f1)[-1]
            
            AUC = val_auc[best_index]; loss = val_loss[best_index]; F1 = val_f1[best_index]; ACC = val_acc[best_index]; 
            total_val_auc.append(AUC); total_val_loss.append(loss); total_val_f1.append(F1); total_val_acc.append(ACC)
            
            AUC = test_auc[best_index]; loss = test_loss[best_index]; F1 = test_f1[best_index]; ACC = test_acc[best_index]; 
            total_test_auc.append(AUC); total_test_loss.append(loss); total_test_f1.append(F1);total_test_acc.append(ACC)
            
        return total_val_auc, total_val_loss, total_val_f1, total_val_acc, \
                total_test_auc, total_test_loss, total_test_f1, total_test_acc
        
        
            
    def load_encoder_weight(self, weight_path):
        state_dict = torch.load(weight_path)
        self.encoder.load_state_dict(state_dict, strict = False)
    
    def load_model_weight(self, weight_path):
        state_dict = torch.load(weight_path)
        self.model.load_state_dict(state_dict, strict = False)
    
    def get_cell_embeddings(self, n_ge = 400, batch_size = 64, use_HUGs = True, use_HVGs = False, pooling = ''):
        
        self.model.eval()
        self.data_df = self.set_df(self.adata)
        #self.set_vocab()
        if use_HUGs:
            self.data_df = self.sorted_by_highly_unique_genes(self.data_df)
        elif use_HVGs:
            self.data_df = self.sorted_by_HVGs(self.data_df)
        
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
            
            if pooling == 'sum':
                x = torch.sum(self.encoder(x_genes,x_scales),axis = 1)
            elif pooling == 'max':
                x = torch.max(self.encoder(x_genes,x_scales),axis = 1)[0]
            elif pooling == 'mean':
                x = torch.mean(self.encoder(x_genes,x_scales),axis = 1)
            else:
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
    
    def cell_type_annotation_with_pathway(self,pathway_dict):
                
        n_samples = self.adata.obs.shape[0]
        
        pathway_xs = []
        dot_data = []
        for i in pathway_dict:
            pathway_genes = pathway_dict[i]
            
            pathway_ids = [1]+self.tokenizer.convert_symb_to_id(set(self.adata.var.index).intersection(pathway_genes))
            pathway_ids = np.array(pathway_ids)
            pathway_ids = pathway_ids[pathway_ids!=2]
            
            x_scales = np.ones(len(pathway_ids))*1
            x_genes = torch.tensor(pathway_ids).long().to(self.device)
            x_scales = torch.tensor(x_scales).long().to(self.device)
        
            xs = self.encoder(x_genes.unsqueeze(0),x_scales.unsqueeze(0)).cpu().detach().numpy()[:,0,:][0]
            
            pathway_xs.append(xs)
        
            pathway_cells = []
            batch_size = 64;
            for batch in range(int(n_samples/batch_size)+1):
                
                if (batch+1)*batch_size < n_samples:
                    x_scales = torch.tensor(self.data_df[pathway_ids].iloc[batch*batch_size:(batch+1)*batch_size].values).to(self.device);
                    
                else:
                    x_scales = torch.tensor(self.data_df[pathway_ids].iloc[batch*batch_size:].values).to(self.device);
                    
                rand_genes = torch.tensor(pathway_ids).repeat(x_scales.shape[0])
                
                x_genes = rand_genes.reshape(x_scales.shape[0], len(pathway_ids)).to(self.device)
                x = self.encoder(x_genes,x_scales).cpu().detach().numpy()[:,0,:]
                pathway_cells.append(x)
            
            pathway_cells = np.concatenate(pathway_cells)
            sims = np.dot(pathway_cells, xs)/np.dot(xs,xs)
            
            dot_data.append(sims)
            
        dot_df = pd.DataFrame(data = dot_data,index = list(pathway_dict.keys())).T
        pred_y = np.array([list(pathway_dict.keys())[np.argmax(dot_df.iloc[i])] for i in range(dot_df.shape[0])])
    
        return dot_df, pred_y
