import torch
import torch.nn as nn

import torch.nn.functional as F
import math

import collections

from model.transformer import TransformerBlock

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Tokenizer2():
    def __init__(self, Gene_vocab, shuf= True, pad_token=0, sep_token=1, unk_token=2, cls_token=3, mask_token=4, **kwargs) :
        super().__init__()
        self.Gene_vocab = Gene_vocab
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.shuf = shuf

        self.special_tokens = {
            'UNK' : self.unk_token,
            'SEP' : self.sep_token,
            'PAD' : self.pad_token,
            'CLS' : self.cls_token,
            'MASK': self.mask_token,
            }

        self.symb_to_id = collections.OrderedDict([(SYMBOL, ID) for ID, SYMBOL in self.Gene_vocab.values])

    @property
    def vocab_size(self):
        return len(self.Gene_vocab)

    def get_vocab(self):
        return self.Gene_vocab

    def check_unk(self,genes):
        genes = [gene if gene is not None else self.special_tokens['UNK'] for gene in genes]
        return genes

    def check_mis_scale(self,scales):
        scales = [scale if scale > 1e-12 else 1.0 for scale in scales]
        return scales

    def convert_symb_to_id(self, symbs):
        return [self.symb_to_id.get(symb) if self.symb_to_id.get(symb) is not None else self.unk_token for symb in symbs]

    def convert_id_to_symb(self, indices):
        return [list(self.symb_to_id.keys())[list(self.symb_to_id.values()).index(index)] for index in indices]


class Gene_Embedding(nn.Module):
    def __init__(self, vocab_size= None,embed_size=None):
        super(Gene_Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dim = embed_size
        self.eps = 1e-12

    def forward(self, genes=None,scales=None):
        x = self.embedding(genes)
        x = self.unit(x)
        x *= scales.unsqueeze(-1)
        return x

    def unit(self,x):
        return (x+self.eps)/(torch.norm(x,dim=2).unsqueeze(-1)+self.eps)


class GeneBERT(nn.Module):
    def __init__(self, embedding = None, vocab_size = 35420, hidden=512, n_layers=3, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        self.embedding = embedding

        # multi-layers transformer blocks, deep network
        # TODO : transformerBlock으로 되어있음! 이거 분해하기.
        self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, layer_type='PFN') for i in range(n_layers)])

    def forward(self, genes=None,scales=None):#, segment_label = None):
        x = self.get_embedding(genes,scales)
        x = self.get_transformer(x)
        return x

    def get_attention(self, genes=None,scales=None):#, segment_label = None):
        mask = None#(genes > 0).unsqueeze(1).repeat(1, genes.size(1), 1).unsqueeze(1)
        x = self.get_embedding(genes,scales)

        for transformer in self.transformer_blocks[:-1]:
            x = transformer.forward(x, mask)

        att = self.transformer_blocks[-1].get_attention(x,mask)

        return att

    def get_embedding(self, genes=None,scales=None):
        return self.embedding(genes, scales).float()

    def get_transformer(self, x):
        mask = None#(genes > 0).unsqueeze(1).repeat(1, genes.size(1), 1).unsqueeze(1)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def get_transformer_one_layer(self, x):
        mask = None#(genes > 0).unsqueeze(1).repeat(1, genes.size(1), 1).unsqueeze(1)
        #for transformer in self.transformer_blocks:
        x = self.transformer_blocks[0].forward(x, mask)

        return x


class FC(torch.nn.Module):
    def __init__(self, x_dim = 512, y_dim = 512, dropout_ratio = 0.1):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(x_dim, y_dim)
        )

    def forward(self, x):
        return self.fc(x)


class Downstream_BERT(torch.nn.Module):
    def __init__(self, y_dim = 512, o_dim = 11, dropout_ratio = 0.3, temperature = 0.07,
                 device = 'cpu', encoder = None):
        super().__init__()
        self.device = device

        self.fc = FC(y_dim,o_dim)

        self.encoder = encoder

        self.dropout_ratio = dropout_ratio
        self.y_dim = y_dim

        self.do = nn.Dropout(self.dropout_ratio)

    def dropout_forward(self, x_genes,x_expres):
        x = self.encoder(x_genes,x_expres)
        pred_y = self.fc(F.dropout(x[:,0,:],p=0.8))

        return pred_y

    def forward_one_layer(self, x_genes,x_expres):
        x = self.encoder.get_embedding(x_genes,x_expres)
        x = self.encoder.get_transformer_one_layer(x)
        pred_y = self.fc(x[:,0,:])

        return pred_y

    def forward(self, x_genes,x_expres):
        x = self.encoder(x_genes,x_expres)
        pred_y = self.fc(x[:,0,:])

        return pred_y

class CL_BERT(torch.nn.Module):
    def __init__(self, y_dim = 512, dropout_ratio = 0.3, temperature = 0.07,
                 device = 'cpu', encoder = None):
        super().__init__()
        self.device = device
        self.temperature = temperature

        self.cl_fc = FC(y_dim,y_dim)

        self.encoder = encoder

        self.dropout_ratio = dropout_ratio
        self.y_dim = y_dim

        self.do = nn.Dropout(self.dropout_ratio)

    def info_nce_loss(self, y):
        labels = torch.cat([torch.arange(int(y.shape[0]/2)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)

        features = F.normalize(y, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature

        return logits, labels


    def forward(self, x_genes,x_expres):

        x = self.encoder(x_genes,x_expres)
        cl_y = self.cl_fc(x[:,0,:])
        logits, labels = self.info_nce_loss(cl_y)

        return logits, labels

class CL_GE_BERT(torch.nn.Module):
    def __init__(self, y_dim = 512, dropout_ratio = 0.3, temperature = 0.07, device = 'cpu', encoder = None) :
        super().__init__()
        self.device = device
        self.temperature = temperature

        self.cl_fc = FC(y_dim,y_dim)
        self.ge_fc = FC(y_dim,y_dim)

        self.encoder = encoder

        self.dropout_ratio = dropout_ratio
        self.y_dim = y_dim

        self.do = nn.Dropout(self.dropout_ratio)

    def info_nce_loss(self, y):
        labels = torch.cat([torch.arange(int(y.shape[0]/2)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)

        features = F.normalize(y, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature

        return logits, labels

    def predict_gene_expresion(self, ge_y, ge_x_genes):
        embedding_matrix = self.encoder.embedding.embedding.weight.clone()
        pred_y = torch.bmm(embedding_matrix[ge_x_genes],ge_y.unsqueeze(-1)).squeeze(-1)

        return pred_y

    def forward(self, x_genes,x_scales, ge_x_genes):
        x = self.encoder(x_genes,x_scales)
        cl_y = self.cl_fc(x[:,0,:])
        logits, labels = self.info_nce_loss(cl_y)

        ge_y = self.ge_fc(x[:int(len(x)/2),0,:])
        pred_y = self.predict_gene_expresion(ge_y, ge_x_genes)

        return logits, labels, pred_y
