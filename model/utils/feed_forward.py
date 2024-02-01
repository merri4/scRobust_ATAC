import torch.nn as nn
from .gelu import GELU

class TokenwiseFeedForward(nn.Module):
    "Implements TFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(TokenwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Conv1d(d_ff, d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        
        init_eps = 1e-3
        init_eps /= d_model
        nn.init.uniform_(self.w_2.weight, -init_eps, init_eps)
        nn.init.constant_(self.w_2.bias, 1.)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))).permute(0,2,1)).permute(0,2,1)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
   