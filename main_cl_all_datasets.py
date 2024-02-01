import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch
from torch import optim

from utils import *
from models import *
import matplotlib.pyplot as plt
from random import choices
import random

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:9" if cuda_condition else "cpu")


data_names = ['Baron_Pancreas', 'SortedPBMC', '68K_PBMC', 'TM',
              # 0                 1                 2       3
              'Muraro_Pancreas', 'Segerstolpe_Pancreas', 'Xin_Pancreas',
              # 4                           5                   6
              'Baron_MousePancreas', 'MacParland']
              #        7               8

data_name = data_names[1]

zero_ratio = 0

if zero_ratio:
    data_df, _, gene_vocab, _ = get_zero_ratio_dataset(data_name,zero_ratio)

else:
    data_df, y_df, gene_vocab, _ = get_dataset(data_name)

tokenizer = Tokenizer2(gene_vocab)

vocab_size = gene_vocab.shape[0]; vocab_size = len(gene_vocab)

ex_genes = tokenizer.convert_symb_to_id(data_df.columns)

ex_genes = np.array(ex_genes)

data_df.index = list(range(0,len(data_df.index)))
data_df.columns = ex_genes

data_df = data_df[ex_genes[ex_genes!=2]]
data_df[1] = 1.0

all_genes = data_df.columns
index_range = np.array(range(len(all_genes)))

all_cells = data_df.index

save_path = './weights/'; img_path = './imgs/'; result_path = './results/'

att_dropout = 0.3; epoch = 5000; lr = 0.00005; batch_size = 256;  ge_weight = 1
d = 64; attn_heads = 8; hidden = d*attn_heads; n_layers = 2; n_ge = 50;

ce_fun = torch.nn.CrossEntropyLoss(); mse_fun = torch.nn.MSELoss();
total_train_losses = [];total_test_losses = [];total_val_losses = [];

from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.05)
split = ss.split(all_cells)

if zero_ratio:
    title = data_name+'_zero_ratio_'+str(zero_ratio)+'_CL_GE_BERT_Hid_'+str(hidden)\
            +'_Att_'+str(attn_heads)+'_nGenes_'+str(n_ge)+'_ly_'+str(n_layers)+'_bt_'+str(batch_size)
else:
    title = data_name+'_CL_GE_BERT_Hid_'+str(hidden)\
        +'_Att_'+str(attn_heads)+'_nGenes_'+str(n_ge)+'_ly_'+str(n_layers)+'_bt_'+str(batch_size)

for k in range(1):

    train_cells, test_cells = train_test_split(all_cells,test_size = 0.1)
    train_df = data_df.loc[train_cells]; test_df = data_df.loc[test_cells];

    train_ds = TensorDataset(torch.tensor(train_df.values))
    test_ds = TensorDataset(torch.tensor(test_df.values))

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    embedding = Gene_Embedding(vocab_size=vocab_size, embed_size=hidden)
    encoder = GeneBERT(embedding = embedding, hidden=hidden, n_layers= n_layers, attn_heads=attn_heads)


    model = CL_GE_BERT(y_dim = hidden, dropout_ratio = att_dropout,
                       device = device, encoder = encoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) #0.00005

    train_cl_loss = []; test_cl_loss = []; train_ge_loss = []; test_ge_loss = [];

    best_loss = 100.

    for ep in range(epoch):

        step =0; sum_cl_loss = 0.0; sum_ge_loss = 0.0;
        model.train();
        for ge_values, in train_dataloader:

            n_samples = len(ge_values)

            optimizer.zero_grad(); step +=1;

            ge_values = ge_values.repeat(2,1).numpy()

            rand_index = [[index_range[-1]] + choices(index_range[ge_values[i]!=0],k=n_ge) for i in range(len(ge_values))]
            rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
            rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]

            x_genes = torch.tensor(rand_genes).to(device);
            x_scales = torch.tensor(rand_scales, dtype=torch.float).to(device);

            ge_x_genes = x_genes[n_samples:,:]
            ge_x_scales = x_scales[n_samples:,:]

            logits, labels, pred_y = model(x_genes,x_scales, ge_x_genes)

            ge_loss = mse_fun(pred_y, ge_x_scales)*ge_weight
            cl_loss = ce_fun(logits, labels)

            loss = ge_loss + cl_loss
            loss.backward()
            optimizer.step()

            sum_cl_loss += cl_loss.item();
            sum_ge_loss += ge_loss.item();

            if (step+1) %50 ==0:
                print("training step: ", step)
                print("cl training loss: ", sum_cl_loss/(step+1))
                print("ge training loss: ", sum_ge_loss/(step+1))

        cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
        print(title)
        print("Train cl_loss: ", cl_loss);  print("Train ge_loss: ", ge_loss);
        train_cl_loss.append(cl_loss); train_ge_loss.append(ge_loss);

        model.eval(); step =0; sum_cl_loss = 0.0; sum_ge_loss = 0.0;

        for ge_values, in test_dataloader:
            n_samples = len(ge_values)
            step +=1;

            ge_values = ge_values.repeat(2,1).numpy()

            rand_index = [[index_range[-1]] + choices(index_range[ge_values[i]!=0],k=n_ge) for i in range(len(ge_values))]
            rand_genes = [all_genes[rand_index[i]].tolist() for i in range(len(ge_values))]
            rand_scales = [ge_values[i][rand_index[i]] for i in range(len(ge_values))]

            x_genes = torch.tensor(rand_genes).to(device);
            x_scales = torch.tensor(rand_scales, dtype=torch.float).to(device);

            ge_x_genes = x_genes[n_samples:,:]
            ge_x_scales = x_scales[n_samples:,:]

            logits, labels, pred_y = model(x_genes, x_scales, ge_x_genes)

            ge_loss = mse_fun(pred_y, ge_x_scales)*ge_weight
            cl_loss = ce_fun(logits, labels)

            sum_cl_loss += cl_loss.item();
            sum_ge_loss += ge_loss.item();


        cl_loss = sum_cl_loss/(step+1); ge_loss = sum_ge_loss/(step+1);
        print("Test cl_loss: ", cl_loss);  print("Test ge_loss: ", ge_loss);
        test_cl_loss.append(cl_loss); test_ge_loss.append(ge_loss);

        if best_loss > test_cl_loss[-1]:
            model = model.cpu()
            best_loss = test_cl_loss[-1]

            torch.save(model.state_dict(), save_path+title+'.pt')
            torch.save(model.encoder.state_dict(), save_path+title+'_encoder.pt')

            model = model.to(device)

        if (ep+1) %50 == 0:
            input_title = 'Epoch'+str(epoch)+'_CL_Loss_'+title
            plt.plot(train_cl_loss)
            plt.plot(test_cl_loss)
            plt.legend(['train','test'])
            plt.title(title)
            plt.show()

            input_title = 'Epoch'+str(epoch)+'_GE_Loss_'+title
            plt.plot(train_ge_loss)
            plt.plot(test_ge_loss)
            plt.legend(['train','test'])
            plt.title(title)

            plt.show()

torch.save(model.state_dict(), save_path+title+'_ep_'+str(len(train_cl_loss))+'.pt')
torch.save(model.encoder.state_dict(), save_path+title+'_encoder_ep_'+str(len(train_cl_loss))+'.pt')

input_title = 'Epoch'+str(len(train_cl_loss))+'_CL_Loss_'+title
plt.plot(train_cl_loss)
plt.plot(test_cl_loss)
plt.legend(['train','test'])
plt.title(title)
plt.savefig('./imgs/'+title+'_'+str(len(train_cl_loss))+'_CL_loss.png')
plt.show()

input_title = 'Epoch'+str(len(train_cl_loss))+'_GE_Loss_'+title
plt.plot(train_ge_loss)
plt.plot(test_ge_loss)
plt.legend(['train','test'])
plt.title(title)
plt.savefig('./imgs/'+title+'_'+str(len(train_cl_loss))+'_GE_loss.png')
plt.show()
