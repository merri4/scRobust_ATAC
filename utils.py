from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,\
                            roc_auc_score, confusion_matrix, precision_recall_curve
import csv
import pandas as pd
import hickle as hkl
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt

import torch

import random, os

import scanpy as sc
from scipy.sparse import csr_matrix

def read_mtx_file(file_dir = 'data/filtered_matrices_mex/hg19/',
                  min_genes=200, min_cells=3):
    data = sc.read_10x_mtx(
        file_dir,  # the directory with the `.mtx` file
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)                              # write a cache file for faster subsequent reading

    sc.pp.filter_cells(data, min_genes=min_genes)
    sc.pp.filter_genes(data, min_cells=min_cells)

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)

    X = csr_matrix(data.X)
    df = pd.DataFrame(data = X.toarray(), columns = data.var.index, index = data.obs.index)

    return df

def get_dataset(data_name):

    data_df, labels, gene_vocab, encoder_dict = None, None, None, None

    if data_name == 'Baron_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_Baron_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Baron_HumanPancreas_gene_vocab.csv',sep=',')
        genes = [i.upper() for i in features.var.index.tolist()]
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Baron_Human_CL_GE_BERT_Hid_512_Att_8_nGenes_100_ly_1_bt_256_encoder.pt'

    elif data_name == 'Muraro_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_Muraro_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Muraro_HumanPancreas_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Muraro_HumanPancreas_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_64_encoder.pt'

    elif data_name == 'Segerstolpe_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Segerstolpe_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Segerstolpe_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_128_encoder.pt'

    elif data_name == 'Xin_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_Xin_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Xin_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Xin_CL_GE_BERT_Hid_512_Att_8_nGenes_500_ly_1_bt_128_encoder.pt'

    elif data_name == 'SortedPBMC':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_DownSampled_SortedPBMC_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/sorted_pmbc_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/SortedPBMC_CL_GE_BERT_Hid_512_Att_8_nGenes_50_ly_2_bt_256_encoder.pt'

    elif data_name == '68K_PBMC':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_68K_PBMC_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/68K_PBMC_gene_vocab.csv',sep=',')
        genes = features.var['gene_ids']
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/68K_PBMC_CL_GE_BERT_Hid_512_Att_8_nGenes_100_ly_2_bt_256_encoder.pt'

    elif data_name == 'TM':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_TM_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/TM_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/TM_CL_GE_BERT_Hid_512_Att_8_nGenes_50_ly_2_bt_256_encoder.pt'

    elif data_name == 'Baron_MousePancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_Baron_MousePancreas.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Baron_MousePancreas_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Baron_HumanMousePancreas_CL_GE_BERT_Hid_512_Att_8_nGenes_100_ly_1_bt_256_nG_100_encoder.pt'

    elif data_name == 'MacParland':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Processed_Filtered_MacParland_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/MacParland_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/MacParland_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'

    X = csr_matrix(features.X)

    data_df = pd.DataFrame(data = X.toarray(), columns = genes, index = features.obs.index)
    labels = pd.DataFrame(data = list(features.obs['label']), columns = ['celltype'])

    if data_name == 'Segerstolpe_Pancreas':
        indices = ~labels['celltype'].isin(['co-expression','unclassified endocrine']).values
        data_df = data_df[indices].reset_index(drop=True)
        labels = labels[indices].reset_index(drop=True)
        clas_names = list(set(labels['celltype']))
        ys = [clas_names.index(i) for i in labels['celltype']]
        labels['y'] = ys

    label_codes = list(set(labels['celltype']))
    label_codes.sort()
    labels['y'] = [label_codes.index(labels['celltype'][i]) for i in range(len(labels))]

    return data_df, labels, gene_vocab, encoder_dict


def get_zero_ratio_dataset(data_name, zero_ratio= 0.5):
    data_df, labels, gene_vocab, encoder_dict = None, None, None, None

    if data_name == 'Baron_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Baron_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Baron_HumanPancreas_gene_vocab.csv',sep=',')
        genes = [i.upper() for i in features.var.index.tolist()]
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Baron_Pancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_100_ly_1_bt_256_encoder.pt'

    elif data_name == 'Muraro_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Muraro_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Muraro_HumanPancreas_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Muraro_Pancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'

    elif data_name == 'Segerstolpe_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Segerstolpe_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Segerstolpe_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Segerstolpe_Pancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'

    elif data_name == 'Xin_Pancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Xin_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Xin_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Xin_Pancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'

    elif data_name == 'SortedPBMC':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_DownSampled_SortedPBMC_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/sorted_pmbc_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/SortedPBMC_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_50_ly_2_bt_256_encoder.pt'


    elif data_name == '68K_PBMC':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_68K_PBMC_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/68K_PBMC_gene_vocab.csv',sep=',')
        genes = features.var['gene_ids']
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/68K_PBMC_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_50_ly_2_bt_256_encoder.pt'

    elif data_name == 'TM':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_TM_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/TM_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/TM_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_50_ly_2_bt_256_encoder.pt'

    elif data_name == 'Baron_MousePancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Baron_HumanPancreas_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Baron_HumanPancreas_gene_vocab.csv',sep=',')
        genes = [i.upper() for i in features.var.index.tolist()]
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Baron_Pancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_100_ly_1_bt_256_encoder.pt'

    elif data_name == 'Baron_MousePancreas':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_Baron_MousePancreas.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/Baron_MousePancreas_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/Baron_MousePancreas_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'

    elif data_name == 'MacParland':
        features = sc.read_h5ad("/NAS_Storage4/leo8544/SingleCell/data/Zero_ratio_"+str(zero_ratio)+"_Processed_Filtered_MacParland_data.h5ad")
        gene_vocab = pd.read_csv('/NAS_Storage4/leo8544/SingleCell/data/MacParland_gene_vocab.csv',sep=',')
        genes = features.var.index
        encoder_dict = '/NAS_Storage4/leo8544/SingleCell/weights/MacParland_zero_ratio_'+str(zero_ratio)+\
                        '_CL_GE_BERT_Hid_512_Att_8_nGenes_200_ly_1_bt_256_encoder.pt'


    X = csr_matrix(features.X)

    data_df = pd.DataFrame(data = X.toarray(), columns = genes, index = features.obs.index)
    labels = pd.DataFrame(data = list(features.obs['label']), columns = ['celltype'])

    if data_name == 'Segerstolpe_Pancreas':
        indices = ~labels['celltype'].isin(['co-expression','unclassified endocrine']).values
        data_df = data_df[indices].reset_index(drop=True)
        labels = labels[indices].reset_index(drop=True)
        clas_names = list(set(labels['celltype']))
        ys = [clas_names.index(i) for i in labels['celltype']]
        labels['y'] = ys

    label_codes = list(set(labels['celltype']))
    label_codes.sort()
    labels['y'] = [label_codes.index(labels['celltype'][i]) for i in range(len(labels))]

    return data_df, labels, gene_vocab, encoder_dict

def show_picture(train,val, test, title, path='' ,save=False):
    plt.plot(train)
    plt.plot(val)
    plt.plot(test)
    plt.legend(['train','val','test'])
    plt.title(title)
    if save:
        plt.savefig(path+title+'.png')
    plt.show()

def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0], recall[0, 0], specificity[0, 0], precision[0, 0]
