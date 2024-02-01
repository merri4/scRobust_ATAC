import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import torch.nn as nn
import torch
from torch import optim

from utils import *
from models import *
import matplotlib.pyplot as plt
from random import choices
import random

from torch.utils.data.sampler import WeightedRandomSampler

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:3" if cuda_condition else "cpu")


data_names = ['Baron_Pancreas', 'SortedPBMC', '68K_PBMC', 'TM',
              # 0                 1                 2       3
              'Muraro_Pancreas', 'Segerstolpe_Pancreas', 'Xin_Pancreas',
              # 4                           5                   6
              'Baron_MousePancreas', 'MacParland']
              #        7               8

data_name = data_names[8]

zero = False; zero_ratio = 0.; random_genes = False; n_fold = 5;

if zero:
    data_df, labels, gene_vocab, encoder_dict  = get_zero_ratio_dataset(data_name,zero_ratio)
else:
    data_df, labels, gene_vocab, encoder_dict = get_dataset(data_name)


if random_genes:
    data_df = data_df[shuffle(data_df.columns)]
else:
    zero_count = (np.sum(data_df==0)/data_df.shape[0])
    high_zero_genes = zero_count.sort_values(ascending=False).index
    data_df = data_df[high_zero_genes]

n_clssses = len(set(labels['y']))

tokenizer = Tokenizer2(gene_vocab)
vocab_size = gene_vocab.shape[0]; vocab_size = len(gene_vocab)

ex_genes = tokenizer.convert_symb_to_id(data_df.columns)
ex_genes = np.array(ex_genes)

data_df.columns = ex_genes
data_df = data_df.T[data_df.columns!=2].T

data_df.index = list(range(0,len(data_df.index)))

data_df[1] = 1.
data_df[4] = 0.

all_cells = data_df.index
all_genes = data_df.columns
index_range = np.array(range(len(all_genes)))

save_path = './weights/'; img_path = './imgs/'; result_path = './results/'

pre_train = True;

att_dropout = 0.3; epoch = 10; lr = 0.00005; batch_size = 64;  ge_weight = 1
d = 64; attn_heads = 8; hidden = d*attn_heads; n_layers = 2; n_ge = 800;

if data_name == 'Baron_Pancreas' or  data_name ==  'Segerstolpe_Pancreas' or  data_name ==  'Baron_MousePancreas' or  \
   data_name == 'Xin_Pancreas' or data_name == 'Muraro_Pancreas' or data_name == 'MacParland':
       n_layers = 1

if zero:
    name = data_name+'_zero_ratio_'+str(zero_ratio)+'_RG_'+str(random_genes)+'_BERT_PT_'+str(pre_train)+'_hid_'+str(hidden)\
            +'_ly_'+str(n_layers)+'_nG_'+str(n_ge)
else:
    name = data_name+'_BERT_PT_'+str(pre_train)+'_RG_'+str(random_genes)+'_hid_'+str(hidden)\
            +'_ly_'+str(n_layers)+'_nG_'+str(n_ge)

loss_fun = torch.nn.CrossEntropyLoss()


if pre_train:
    hyperparameter_sets = [
    {'att_dropout' : 0.3, 'epoch': 10, 'lr': 5e-5, 'batch_size': 32},
    #{'att_dropout' : 0.3, 'epoch': 50, 'lr': 1e-5, 'batch_size': 64},
    #{'att_dropout' : 0.3, 'epoch': 50, 'lr': 5e-6, 'batch_size': 64},
    ]

else:
    hyperparameter_sets = [
    {'att_dropout' : 0.3, 'epoch': 30, 'lr': 5e-5, 'batch_size': 32},
    #{'att_dropout' : 0.3, 'epoch': 50, 'lr': 1e-5, 'batch_size': 64},
    #{'att_dropout' : 0.3, 'epoch': 50, 'lr': 5e-6, 'batch_size': 32},
    ]


if data_name == 'TM' or data_name == '68K_PBMC' or data_name == 'SortedPBMC':
    n_fold = 1

total_hypersets_val_auc = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_val_loss = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_val_f1 = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_val_acc = [[] for i in range(len(hyperparameter_sets))]

total_hypersets_test_auc = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_test_loss = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_test_f1 = [[] for i in range(len(hyperparameter_sets))]
total_hypersets_test_acc = [[] for i in range(len(hyperparameter_sets))]

for main_fold in range(n_fold):
    fold = 0;

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in kfold.split(all_cells, labels['y']):
        fold += 1;

        try:
            train_index, val_index = train_test_split(train_index,test_size = 0.1, stratify = labels['y'].iloc[train_index])
        except:
            train_index, val_index = train_test_split(train_index,test_size = 0.1)

        train_ds = TensorDataset(torch.tensor(data_df.iloc[train_index].values),
                                 torch.tensor(labels['y'].iloc[train_index].values))

        val_ds = TensorDataset(torch.tensor(data_df.iloc[val_index].values),
                               torch.tensor(labels['y'].iloc[val_index].values))

        test_ds = TensorDataset(torch.tensor(data_df.iloc[test_index].values),
                                torch.tensor(labels['y'].iloc[test_index].values))


        for set_index in range(len(hyperparameter_sets)):

            att_dropout = hyperparameter_sets[set_index]['att_dropout']
            epoch = hyperparameter_sets[set_index]['epoch']
            batch_size = hyperparameter_sets[set_index]['batch_size']
            lr = hyperparameter_sets[set_index]['lr']

            title = name+'_set_'+str(set_index)+'_bt_'+str(batch_size)+'_lr_'+str(lr)

            train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            embedding = Gene_Embedding(vocab_size=vocab_size, embed_size=hidden)
            encoder = GeneBERT(embedding = embedding, hidden=hidden, n_layers= n_layers, attn_heads=attn_heads)

            if pre_train:
                encoder.load_state_dict(torch.load(encoder_dict))

            model = Downstream_BERT(y_dim = hidden, o_dim = n_clssses, dropout_ratio = att_dropout,
                                    device = device, encoder = encoder).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr) #0.00005


            train_auc = []; test_auc = []; val_auc = [];
            train_f1 = []; test_f1 = []; val_f1 = []; train_acc = []; test_acc = [];val_acc = [];
            train_acc = []; val_acc = []; test_acc = [];
            train_loss = []; val_loss = []; test_loss = [];

            best_error = 100;

            for ep in range(epoch):
                model.train();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0;
                all_y = []; all_pred_y = []; all_prob_y = [];
                for ge_values, y in train_dataloader:
                    n_samples = len(ge_values)
                    optimizer.zero_grad(); step +=1;

                    y = y.to(device, dtype=torch.int64)

                    ge_values = ge_values.numpy()

                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]

                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]

                    x_genes = torch.tensor(rand_genes).to(device);
                    x_scales = torch.tensor(rand_scales, dtype=torch.float).to(device);

                    x = model.encoder(x_genes,x_scales)

                    pred_y = model(x_genes,x_scales)

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

                    if (step+1) %100 ==0:
                        print('Step ', step, ' Acc: ', acc/len(y))


                loss_train = sum_loss/(step+1)
                ACC = accuracy/count
                weighted_F1 = f1_score(all_y, all_pred_y, average='weighted')
                F1 = f1_score(all_y, all_pred_y, average='macro')

                try:
                    AUC = roc_auc_score(all_y, np.concatenate(all_prob_y), average='weighted',
                                        multi_class = 'ovo')#, labels = np.array(range(33)))
                except:
                    AUC = 1
                print("Epoch: ", ep, " ", title)
                print("Train ACC: ", ACC); print("Train F1: ", F1); print("Train AUC: ", AUC)
                print("Train weighted F1: ", weighted_F1)

                train_acc.append(ACC); train_f1.append(F1); train_auc.append(AUC);
                train_loss.append(loss_train)

                model.eval();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0;
                all_prob_y = []; all_y = []; all_pred_y = [];
                for ge_values, y in val_dataloader:
                    n_samples = len(ge_values); step +=1;

                    ge_values = ge_values.numpy()

                    y = y.to(device, dtype=torch.int64)

                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]

                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]

                    x_genes = torch.tensor(rand_genes).to(device); x_scales = torch.tensor(rand_scales).to(device);

                    pred_y = model(x_genes,x_scales)

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
                                        multi_class = 'ovo')
                except:
                    AUC = 1

                print("Val ACC: ", ACC); print("Val F1: ", F1); print("Val AUC: ", AUC)
                print("Val weighted F1: ", weighted_F1)

                val_acc.append(ACC); val_f1.append(F1); val_auc.append(AUC);
                val_loss.append(loss_val)

                model.eval();
                step =0; sum_loss = 0.0; count = 0; accuracy = 0;
                all_prob_y = []; all_y = []; all_pred_y = [];
                for ge_values, y in test_dataloader:
                    n_samples = len(ge_values); step +=1;

                    ge_values = ge_values.numpy()
                    y = y.to(device, dtype=torch.int64)

                    rand_index = [[-2]+index_range[ge_values[i]!=0][:n_ge].tolist()
                                  if sum(ge_values[i]!=0) > n_ge
                                  else [-2] + index_range[ge_values[i]!=0].tolist()+[-1]*(n_ge-sum(ge_values[i]!=0))
                                  for i in range(len(ge_values))]

                    rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
                    rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]

                    x_genes = torch.tensor(rand_genes).to(device); x_scales = torch.tensor(rand_scales).to(device);

                    pred_y = model(x_genes,x_scales)

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

                wrong_indices = (torch.tensor(all_y) != torch.tensor(all_pred_y))

                loss_test = sum_loss/(step+1)
                ACC = accuracy/count
                weighted_F1 = f1_score(all_y, all_pred_y, average='weighted')
                F1 = f1_score(all_y, all_pred_y, average='macro')

                try:
                    AUC = roc_auc_score(all_y, all_prob_y, average='weighted',
                                        multi_class = 'ovo')
                except:
                    AUC = 1

                print("Test ACC: ", ACC); print("Test F1: ", F1); print("Test AUC: ", AUC)
                print("Test weighted F1: ", weighted_F1)
                test_acc.append(ACC); test_f1.append(F1); test_auc.append(AUC);
                test_loss.append(loss_test)

                if (ep+1)%10 == 0:
                    plt.plot(train_acc)
                    plt.plot(val_acc)
                    plt.plot(test_acc)

                    plt.legend(['train ACC', 'val ACC','test ACC'])
                    plt.title(str(main_fold)+'-'+str(fold)+' ACC '+title)
                    plt.show()

                    plt.plot(train_f1)
                    plt.plot(val_f1)
                    plt.plot(test_f1)

                    plt.legend(['train F1', 'val F1','test F1'])
                    plt.title(str(main_fold)+'-'+str(fold)+' F1 '+title)
                    plt.show()

                    plt.plot(train_auc)
                    plt.plot(val_auc)
                    plt.plot(test_auc)

                    plt.legend(['train AUC', 'val AUC','test AUC'])
                    plt.title(str(main_fold)+'-'+str(fold)+' AUC '+title)
                    plt.show()

                    plt.plot(train_loss)
                    plt.plot(val_loss)
                    plt.plot(test_loss)

                    plt.legend(['train Loss', 'val Loss','test Loss'])
                    plt.title(str(main_fold)+'-'+str(fold)+' Loss '+title)
                    plt.show()


            best_index = np.argsort(val_f1)[-1]

            AUC = val_auc[best_index]; loss = val_loss[best_index];
            F1 = val_f1[best_index]; ACC = val_acc[best_index];

            total_hypersets_val_auc[set_index].append(AUC)
            total_hypersets_val_loss[set_index].append(loss)
            total_hypersets_val_f1[set_index].append(F1)
            total_hypersets_val_acc[set_index].append(ACC)

            AUC = test_auc[best_index]; loss = test_loss[best_index];
            F1 = test_f1[best_index]; ACC = test_acc[best_index];


            total_hypersets_test_auc[set_index].append(AUC)
            total_hypersets_test_loss[set_index].append(loss)
            total_hypersets_test_f1[set_index].append(F1)
            total_hypersets_test_acc[set_index].append(ACC)

            df_val_auc = pd.DataFrame(data = total_hypersets_val_auc,
                                       columns = [str(i)+' Fold val AUC' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_val_loss = pd.DataFrame(data = total_hypersets_val_loss,
                                       columns = [str(i)+' Fold val loss' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_val_f1 = pd.DataFrame(data = total_hypersets_val_f1,
                                       columns = [str(i)+' Fold val F1' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_val_acc = pd.DataFrame(data = total_hypersets_val_acc,
                                       columns = [str(i)+' Fold val ACC' for i in range(len(total_hypersets_val_auc[set_index]))])

            df_test_auc = pd.DataFrame(data = total_hypersets_test_auc,
                                       columns = [str(i)+' Fold test AUC' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_test_loss = pd.DataFrame(data = total_hypersets_test_loss,
                                       columns = [str(i)+' Fold test loss' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_test_f1 = pd.DataFrame(data = total_hypersets_test_f1,
                                       columns = [str(i)+' Fold test F1' for i in range(len(total_hypersets_val_auc[set_index]))])
            df_test_acc = pd.DataFrame(data = total_hypersets_test_acc,
                                       columns = [str(i)+' Fold test ACC' for i in range(len(total_hypersets_val_auc[set_index]))])

            total_df = pd.concat([df_val_auc, df_val_loss, df_val_f1, df_val_acc,
                                  df_test_auc, df_test_loss, df_test_f1, df_test_acc],axis = 1)

            total_df.to_csv(result_path+name+'_Fold_'+str(fold)+'_summary.csv')

            best_set_indices = [np.argmax(total_df[str(i)+' Fold val F1'])for i in range(len(total_hypersets_val_auc[set_index]))]
            best_test_auc = [total_df[str(i)+' Fold test AUC'].iloc[best_set_indices[i]] for i in range(len(total_hypersets_val_auc[set_index]))]
            best_test_loss = [total_df[str(i)+' Fold test loss'].iloc[best_set_indices[i]] for i in range(len(total_hypersets_val_auc[set_index]))]
            best_test_f1 = [total_df[str(i)+' Fold test F1'].iloc[best_set_indices[i]] for i in range(len(total_hypersets_val_auc[set_index]))]
            best_test_acc = [total_df[str(i)+' Fold test ACC'].iloc[best_set_indices[i]] for i in range(len(total_hypersets_val_auc[set_index]))]

            df_best_test_auc = pd.DataFrame(data=[np.average(best_test_auc)]+best_test_auc).T
            df_best_test_loss = pd.DataFrame(data=[np.average(best_test_loss)]+best_test_loss).T
            df_best_test_f1 = pd.DataFrame(data=[np.average(best_test_f1)]+best_test_f1).T
            df_best_test_acc = pd.DataFrame(data=[np.average(best_test_acc)]+best_test_acc).T

            total_df = pd.concat([df_best_test_f1, df_best_test_auc, df_best_test_loss, df_best_test_acc])
            total_df.index = ['F1','AUC','loss','ACC']

            total_df.to_csv(result_path+name+'_Times_'+str(main_fold)+'_Fold_'+str(fold)+'_final_summary.csv')
