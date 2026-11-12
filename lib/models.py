import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


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
    

def _use_swiglu(layer, ff_dim):
    """Swap a TransformerEncoderLayer's activation for SwiGLU.

    The layer was built with ReLU, and its fused inference kernel (taken in eval
    mode with grad and autocast off) reads `activation_relu_or_gelu` set at
    construction, not `layer.activation`. Left at 1 it silently applies ReLU at
    inference while training used SwiGLU; 0 rules the fused path out.
    """
    layer.activation = SwiGLU(ff_dim)
    layer.activation_relu_or_gelu = 0


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
        w = self.W(x)
        swish = w * torch.sigmoid(w)
        out = self.W2(swish * self.V(x))

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
                raise NotImplementedError("RoPE transformer encoder is not yet implemented")
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
                    _use_swiglu(layer, ff_dim)
            self.encoder_layers.append(layer)
        self.fc = nn.Linear(d_model, input_dim)

        # Attention outputs storage. Capture is off by default: asking for the
        # weights forces need_weights=True, which disables the fused attention
        # kernel, and the hook copies them to host on every forward. Enable it
        # only around the forward whose weights you want (see get_attn).
        self.encoder_attn_outputs = {}
        self.capture_attn = False
        self.patch_attention()

    def patch_attention_layer(self, m):
        """Monkey-patch the attention layer to save attention weights."""
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            if self.capture_attn:
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
        if not self.capture_attn or o[1] is None:
            return
        self.encoder_attn_outputs[label] = o[1].cpu().detach()

    def attn_capture(self):
        """Context manager enabling attention capture for the forwards inside it."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            prev = self.capture_attn
            self.capture_attn = True
            try:
                yield
            finally:
                self.capture_attn = prev
        return _ctx()

    def get_attn(self, x=None):
        """Retrieve the saved attention weights.

        Pass an input tensor to run one capturing forward first; with no
        argument this returns whatever a previous capturing forward stored.
        """
        if x is not None:
            with self.attn_capture(), torch.no_grad():
                self(x)
        return self.encoder_attn_outputs.copy()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.fc(x[:, -1, :])
        return x
    

## ============================ Global-Local Encoder-Decoder =====================================
def _make_encoder_layer(d_model, ff_dim, nhead, activation, pre_norm):
    """Build a TransformerEncoderLayer with the same SwiGLU substitution used by TransformerEncoderModel."""
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
        _use_swiglu(layer, ff_dim)
    return layer


class CoordinateEmbedding(nn.Module):
    """Fourier-feature embedding of continuous element centroid coordinates in [0, 1]."""
    def __init__(self, d_model, spatial_dim=2, num_freqs=6):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs)) * math.pi
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(2 * num_freqs * spatial_dim, d_model)

    def forward(self, coords):
        # coords: (E, spatial_dim) -> (E, d_model)
        ang = coords.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feats.flatten(start_dim=-2))


class SpatialEncoder(nn.Module):
    """Transformer encoder over the set of GFEM element tokens at one global time.

    tokens: (G, E, token_dim), coords: (E, spatial_dim) -> context (G, S, d_model)
    where S = E, or S = num_context_tokens when perceiver-style pooling is enabled.
    """
    def __init__(self, token_dim, d_model, nhead=4, num_layers=2, ff_dim=1024,
                 activation='swiglu', pre_norm=True, spatial_dim=2, num_freqs=6,
                 num_context_tokens=0):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.coord_embed = CoordinateEmbedding(d_model, spatial_dim, num_freqs)
        self.encoder_layers = nn.ModuleList([
            _make_encoder_layer(d_model, ff_dim, nhead, activation, pre_norm)
            for _ in range(num_layers)
        ])
        # prenorm layers leave the residual stream unnormalized; context feeds cross-attention K/V
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.num_context_tokens = num_context_tokens
        if num_context_tokens > 0:
            self.latent_queries = nn.Parameter(torch.randn(num_context_tokens, d_model) * d_model ** -0.5)
            self.pool_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, tokens, coords):
        x = self.token_proj(tokens) + self.coord_embed(coords).unsqueeze(0)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        if self.num_context_tokens > 0:
            q = self.latent_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
            x, _ = self.pool_attn(q, x, x, need_weights=False)
        return x


class GlobalLocalDecoderLayer(nn.Module):
    """Decoder layer: temporal self-attention per element + cross-attention to the shared context.

    x: (G, E, tl, d_model), context: (G, S, d_model). Cross-attention reshapes queries to
    (G, E*tl, d_model) so all E elements of a snapshot attend to one context without K/V duplication.
    """
    def __init__(self, d_model, nhead, ff_dim, activation='swiglu', pre_norm=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if activation == 'swiglu':
            self.ffn = FFN_SwiGLU(d_model, ff_dim)
        else:
            act = nn.ReLU() if activation == 'relu' else nn.GELU()
            self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), act, nn.Linear(ff_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, x, context, attn_mask=None):
        G, E, tl, d = x.shape
        h = x.reshape(G * E, tl, d)
        if self.pre_norm:
            h2 = self.norm1(h)
            a, _ = self.self_attn(h2, h2, h2, attn_mask=attn_mask, need_weights=False)
            h = h + a
            q = h.reshape(G, E * tl, d)
            a, _ = self.cross_attn(self.norm2(q), context, context, need_weights=False)
            q = q + a
            q = q + self.ffn(self.norm3(q))
        else:
            a, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
            h = self.norm1(h + a)
            q = h.reshape(G, E * tl, d)
            a, _ = self.cross_attn(q, context, context, need_weights=False)
            q = self.norm2(q + a)
            q = self.norm3(q + self.ffn(q))
        return q.reshape(G, E, tl, d)


class GlobalLocalTransformer(nn.Module):
    """Globally-aware local dynamics model.

    encode: element tokens (G, E, context_window*input_dim) + coords (E, spatial_dim)
            -> context (G, S, d_model), refreshed on the slow global timescale.
    decode: local windows (G, E, time_lag, input_dim) + context -> next step (G, E, input_dim).

    With `residual` the head predicts the increment on the last frame instead of
    the whole next state. On the LDC cavity the field moves ~2% of its norm per
    step, so a direct head spends its error budget re-synthesizing a frame it
    was handed (one-step floor ~9%); an increment head only has to get the 2%.
    """
    def __init__(self, time_lag, input_dim, d_model=256, ff_dim=1024, nhead=4, num_layers=4,
                 enc_num_layers=2, enc_nhead=4, enc_ff_dim=1024, activation='swiglu',
                 pre_norm=True, spatial_dim=2, num_freqs=6, context_window=1,
                 num_context_tokens=0, causal=True, context_dim=None, residual=False):
        super().__init__()
        # context_dim lets the encoder read something other than the local DOFs
        # -- e.g. global POD coefficients broadcast to every element, which is a
        # genuinely global state at a fraction of the width
        self.context_dim = context_dim if context_dim is not None else input_dim
        self.encoder = SpatialEncoder(
            token_dim=self.context_dim * context_window, d_model=d_model, nhead=enc_nhead,
            num_layers=enc_num_layers, ff_dim=enc_ff_dim, activation=activation,
            pre_norm=pre_norm, spatial_dim=spatial_dim, num_freqs=num_freqs,
            num_context_tokens=num_context_tokens,
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=time_lag)
        self.decoder_layers = nn.ModuleList([
            GlobalLocalDecoderLayer(d_model, nhead, ff_dim, activation, pre_norm)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, input_dim)
        self.residual = residual
        if residual:
            # start as exact persistence and learn corrections from there; the
            # gradient into fc is nonzero, so this only fixes the starting point
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        if causal:
            mask = torch.triu(torch.ones(time_lag, time_lag, dtype=torch.bool), diagonal=1)
            self.register_buffer('causal_mask', mask)
        else:
            self.causal_mask = None

    def encode(self, tokens, coords):
        return self.encoder(tokens, coords)

    def decode(self, x, context):
        G, E, tl, _ = x.shape
        h = self.input_projection(x)
        h = self.positional_encoding(h.reshape(G * E, tl, -1)).reshape(G, E, tl, -1)
        for layer in self.decoder_layers:
            h = layer(h, context, attn_mask=self.causal_mask)
        out = self.fc(h[:, :, -1, :])
        if self.residual:
            out = out + x[:, :, -1, :]
        return out

    def forward(self, x, tokens, coords):
        return self.decode(x, self.encode(tokens, coords))


## ======================================= Spatial Transformer ===================================
class SpatialTransformer(nn.Module):
    """Spatial-attention model over GFEM element tokens.

    One token per element, carrying its `time_lag` most recent frames of concatenated
    u, v(, w) local DOFs. Attention runs over the element (spatial) axis rather than
    the time axis, and the model predicts the next snapshot in one shot.

    x: (G, E, time_lag, input_dim), coords: (E, spatial_dim) -> (G, E, input_dim)
    """
    def __init__(self, time_lag, input_dim, d_model=256, ff_dim=1024, nhead=4, num_layers=4,
                 activation='swiglu', pre_norm=True, spatial_dim=2, num_freqs=6,
                 residual=False):
        super().__init__()
        # num_context_tokens must stay 0: the head needs one output token per element
        self.encoder = SpatialEncoder(
            token_dim=input_dim * time_lag, d_model=d_model, nhead=nhead,
            num_layers=num_layers, ff_dim=ff_dim, activation=activation,
            pre_norm=pre_norm, spatial_dim=spatial_dim, num_freqs=num_freqs,
            num_context_tokens=0,
        )
        self.fc = nn.Linear(d_model, input_dim)
        self.residual = residual

    def forward(self, x, coords):
        G, E, tl, F_dim = x.shape
        h = self.encoder(x.reshape(G, E, tl * F_dim), coords)
        out = self.fc(h)
        if self.residual:
            out = out + x[:, :, -1, :]
        return out


## ================================ Diffusion Spatial Transformer ================================
class DiffusionTimeEmbedding(nn.Module):
    """Sinusoidal features of the diffusion time t in [0, 1], then a 2-layer MLP (as in DiT)."""
    def __init__(self, d_model, freq_dim=256):
        super().__init__()
        half = freq_dim // 2
        self.register_buffer('freqs', torch.exp(-math.log(10000.0) * torch.arange(half) / half))
        self.mlp = nn.Sequential(nn.Linear(2 * half, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t):
        ang = (1000.0 * t.float()).unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1))


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block whose norms are modulated by the time embedding (adaLN-Zero).

    The shift, scale and residual gates come from the conditioning vector c and
    start at zero, so every block starts as the identity (Peebles & Xie 2023).
    x: (G, E, d_model), c: (G, d_model).
    """
    def __init__(self, d_model, nhead, ff_dim, activation='swiglu', dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if activation == 'swiglu':
            self.ffn = FFN_SwiGLU(d_model, ff_dim)
        else:
            act = nn.ReLU() if activation == 'relu' else nn.GELU()
            self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), act, nn.Linear(ff_dim, d_model))
        self.drop = nn.Dropout(dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), sh1, sc1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * self.drop(a)
        x = x + g2.unsqueeze(1) * self.drop(self.ffn(_modulate(self.norm2(x), sh2, sc2)))
        return x


class _DiffusionCore(nn.Module):
    def __init__(self, time_lag, input_dim, d_model, ff_dim, nhead, num_layers, activation,
                 spatial_dim, num_freqs, dropout):
        super().__init__()
        # one token per element: its conditioning frames and its slice of the noisy increment
        self.token_proj = nn.Linear(input_dim * (time_lag + 1), d_model)
        self.coord_embed = CoordinateEmbedding(d_model, spatial_dim, num_freqs)
        self.time_embed = DiffusionTimeEmbedding(d_model)
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, nhead, ff_dim, activation, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.fc = nn.Linear(d_model, input_dim)
        for lin in (self.ada[1], self.fc):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, cond, z, t, coords):
        G, E, tl, F_dim = cond.shape
        x = self.token_proj(torch.cat([cond.reshape(G, E, tl * F_dim), z], dim=-1))
        x = x + self.coord_embed(coords).unsqueeze(0)
        c = self.time_embed(t)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.fc(_modulate(self.norm(x), shift, scale))


class DiffusionSpatialTransformer(nn.Module):
    """Denoiser for the conditional diffusion model (lib/diffusion.py).

    Spatial attention over the GFEM element tokens, as in SpatialTransformer,
    with each token also carrying its slice of the noisy next-frame increment,
    and every block conditioned on the diffusion time through adaLN-Zero.

    cond: (G, E, time_lag, input_dim) normalized frames, z: (G, E, input_dim)
    noisy increment, t: (G,) diffusion time, coords: (E, spatial_dim)
    -> (G, E, input_dim) element-local v prediction.

    With ema, an exponential moving average of the weights is kept as `ema` (no
    gradients; updated by update_ema after each optimizer step) and is what
    eval mode runs. It lives in the state dict, so checkpoints and the saved
    model carry it.
    """
    def __init__(self, time_lag, input_dim, d_model=256, ff_dim=768, nhead=4, num_layers=4,
                 activation='swiglu', spatial_dim=2, num_freqs=6, dropout=0.1, ema=True):
        super().__init__()
        self.net = _DiffusionCore(time_lag, input_dim, d_model, ff_dim, nhead, num_layers,
                                  activation, spatial_dim, num_freqs, dropout)
        self.ema = None
        if ema:
            import copy
            self.ema = copy.deepcopy(self.net)
            for p in self.ema.parameters():
                p.requires_grad_(False)
            self.register_buffer('ema_steps', torch.zeros((), dtype=torch.long))

    def forward(self, cond, z, t, coords):
        net = self.ema if (self.ema is not None and not self.training) else self.net
        return net(cond, z, t, coords)

    @torch.no_grad()
    def update_ema(self, decay):
        if self.ema is None:
            return
        # warm-up: early on the average follows the weights closely. decay 0
        # keeps the copy equal to the weights, so eval mode runs them unaveraged
        n = int(self.ema_steps)
        d = 0.0 if decay <= 0 else min(decay, (1 + n) / (10 + n))
        for pe, p in zip(self.ema.parameters(), self.net.parameters()):
            pe.lerp_(p, 1.0 - d)
        for be, b in zip(self.ema.buffers(), self.net.buffers()):
            be.copy_(b)
        self.ema_steps += 1


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