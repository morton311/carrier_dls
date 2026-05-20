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

##############################
# SwiGLU Activation Function #
##############################
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
    

#####################
# FFN_SwiGLU module #
#####################
class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.W = nn.Linear(d_model, ff_dim, bias=False)
        self.V = nn.Linear(d_model, ff_dim, bias=False)
        self.W2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        x = self.W(x)
        swish = x * torch.sigmoid(x)
        x = swish * self.V(x)
        out = self.W2(x)

        return out
    
##############################################
# Rotary Positional Embeddings for attention #
##############################################
class RotaryPositionalEmbeddings(nn.Module):

  def __init__(self, d: int, base: int = 10_000):

    super().__init__()
    self.base = base
    self.d = d
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):

    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return

    seq_len = x.shape[0]

    theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]

    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]


    self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):

    d_2 = self.d // 2 #

    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


  def forward(self, x: torch.Tensor):

    self._build_cache(x)

    neg_half_x = self._neg_half(x)

    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

    return x_rope

###################################
# RoPE Multihead Attention module #
###################################
class RoPEMultiheadAttention(nn.Module):
    pass
    
    
########################################
# RoPE-based Transformer Encoder Layer #
########################################
class RoPETransformerEncoderLayer(nn.Module):
    pass
    
## ====================================== Transformer ============================================
# Define the Transformer Encoder model
class TransformerEncoderModel(nn.Module):
    def __init__(self, time_lag, input_dim, d_model=256, ff_dim=2048, nhead=4, num_layers=4, embed='lin', activation='relu', pre_norm=False):
        super(TransformerEncoderModel, self).__init__()
        self.use_rope = False
        if embed == 'TS':
            self.positional_encoding = nn.Identity()
            self.input_projection = TimeSpaceEmbedding(time_lag, input_dim, d_expand=2 * time_lag, d_model=d_model)
        elif embed == 'lin':
            self.positional_encoding = PositionalEncoding(d_model, max_len=time_lag)
            self.input_projection = nn.Linear(input_dim, d_model)
        elif embed == 'alibi':
            # ALiBi: no explicit positional embeddings; use identity and mark flag
            self.positional_encoding = nn.Identity()
            self.input_projection = nn.Linear(input_dim, d_model)
            self.use_alibi = True
        elif embed == 'rope':
            # RoPE: use sinusoidal positional encoding as a fallback placeholder
            self.positional_encoding = nn.Identity()
            self.input_projection = nn.Linear(input_dim, d_model)
            self.use_rope = True
        else:
            raise RuntimeError(f"embed should be one of 'TS'/'lin'/'alibi'/'rope', not {embed}")

        if isinstance(activation, str):
            activation = activation.lower()
            if activation not in {'relu', 'gelu', 'swiglu'}:
                raise RuntimeError(f"activation should be relu/gelu/swiglu, not {activation}")

        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            
            if self.use_rope:
                layer = RoPETransformerEncoderLayer(d_model=d_model, nhead=nhead, ff_dim=ff_dim)
            else:
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

            # If ALiBi is enabled, add an attention mask bias
            if getattr(self, 'use_alibi', False):
                # args[0] is query of shape (batch, seq_len, dim) in batch_first=True
                q = args[0]
                tgt_len = q.size(1)
                src_len = tgt_len
                nhead = m.num_heads

                # compute slopes per head (approximation)
                def get_slopes(n):
                    def get_pow(i):
                        return 2 ** (-(2 ** -(math.log2(n) - 3)) * i)
                    return [get_pow(i) for i in range(n)]

                slopes = torch.tensor(get_slopes(nhead), device=q.device).unsqueeze(1).unsqueeze(2)
                pos = torch.arange(src_len, device=q.device).unsqueeze(0) - torch.arange(tgt_len, device=q.device).unsqueeze(1)
                pos = pos.abs().unsqueeze(0)
                alibi = -slopes * pos  # (nhead, tgt_len, src_len)

                kwargs['attn_mask'] = alibi

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