import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from datetime import datetime
import os
import numpy as np
import time
from tqdm import tqdm
import copy
import pickle


## ==================================== Positional Encoding ======================================
# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    



## ===================================== TimeSpace Embed =========================================

"""

Create a new embedding strategy for time and space embedding

@ yuningw

"""

class TimeSpaceEmbedding(nn.Module):
    
    """"

    A embedding module based on both time and space
    Args:

    d_input : The input size of timedelay embedding

    n_mode : The number of modes/dynamics in the time series 

    d_expand : The projection along the time

    d_model : The projection along the space 

    """

    def __init__(self, time_lag, input_dim,
                d_expand, d_model):

        super(TimeSpaceEmbedding, self).__init__()

        self.spac_proj      = nn.Linear(input_dim,d_model)

        self.time_proj      = nn.Conv1d(time_lag, d_expand,1)

        self.time_avgpool   = nn.AvgPool1d(2)
        self.time_maxpool   = nn.MaxPool1d(2)
        self.time_compress  = nn.Linear(d_model, d_model)
        self.act            = nn.Identity()

        nn.init.xavier_uniform_(self.spac_proj.weight)
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.xavier_uniform_(self.time_compress.weight)
    
    def forward(self, x):
        
        # Along space projection
        x       = self.spac_proj(x)
        
        # Along the time embedding 
        x       = self.time_proj(x)
        timeavg = self.time_avgpool(x)
        timemax = self.time_maxpool(x)
        tau     = torch.cat([timeavg, timemax],-1)
        out     = self.act(self.time_compress(tau))
        return out

######################################
# SwiGLU Activation Function for FFN #
######################################
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu
    
## ====================================== Transformer ============================================
# Define the Transformer Encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(self, time_lag, input_dim, d_model=256, ff_dim=2048, nhead=4, num_layers=4, embed='lin', activation='relu', pre_norm=False):
        super(TransformerEncoderModel, self).__init__()
        if embed == 'TS':
            self.positional_encoding = nn.Identity()
            self.input_projection = TimeSpaceEmbedding(time_lag, input_dim, d_expand=2 * time_lag, d_model=d_model)
        elif embed == 'lin':
            self.positional_encoding = PositionalEncoding(d_model, max_len=time_lag)
            self.input_projection = nn.Linear(input_dim, d_model)

        if isinstance(activation, str):
            activation = activation.lower()
            if activation not in {'relu', 'gelu', 'swiglu'}:
                raise RuntimeError(f"activation should be relu/gelu/swiglu, not {activation}")

        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            base_activation = 'relu' if activation == 'swiglu' else activation
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                dim_feedforward=ff_dim,
                nhead=nhead,
                batch_first=True,
                activation=base_activation,
                norm_first=pre_norm,
            )
            if activation == 'swiglu':
                layer.activation = SwiGLU(ff_dim)
            self.encoder_layers.append(layer)
        self.fc = nn.Linear(d_model, input_dim)

        # Attention outputs storage
        self.encoder_attn_outputs = {}
        self.patch_attention()

    def patch_attention_layer(self, m):
        """Monkey-patch the attention layer to save attention weights."""
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return forward_orig(*args, **kwargs)

        m.forward = wrap

    def patch_attention(self):
        """Patch all attention layers in the encoder."""
        for i, layer in enumerate(self.encoder_layers):
            self.patch_attention_layer(layer.self_attn)
            layer.self_attn.register_forward_hook(partial(self.save_output_encoder, label=f's{i}'))

    def save_output_encoder(self, m, i, o, label='0'):
        """Save the attention weights from the encoder."""
        self.encoder_attn_outputs[label] = o[1].cpu().detach()

    def get_attn(self):
        """Retrieve the saved attention weights."""
        return self.encoder_attn_outputs.copy()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.fc(x[:, -1, :])
        return x
    

## ====================================== LSTM Model ============================================
class LSTMModel(nn.Module):
    def __init__(self, time_lag, input_dim, hidden_dim=256, num_layers=2, batch_size = 256):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=time_lag)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        # Initialize hidden and cell states
        hidden, cell = self.init_hidden(x.shape[0], x.device)
        lstm_out, _ = self.lstm(x, (hidden.detach(), cell.detach()))
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def init_hidden(self,batch_size,device):
        hidden = torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_dim).to(device)
                    
        cell  =  torch.zeros(self.num_layers,
                                batch_size,
                                self.hidden_dim).to(device) 
                    
        return hidden, cell