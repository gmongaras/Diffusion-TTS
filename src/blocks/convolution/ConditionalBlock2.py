import torch
import torch.nn as nn
from .MaskedInstanceNorm1d import MaskedInstanceNorm1d

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d




class ConditionalBlock2(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = 8
        
        # Query and keys come from the conditional dimension
        self.q_proj = WeightStandardizedConv1d(cond_dim, embed_dim, 1)
        self.k_proj = WeightStandardizedConv1d(cond_dim, embed_dim, 1)
        
        # Values come from the input dimension
        self.v_proj = WeightStandardizedConv1d(embed_dim, embed_dim, 1)
        
        # Ouptut is the same size as the input
        self.out_proj = WeightStandardizedConv1d(embed_dim, embed_dim, 1)
        
        # self.layer_norm = nn.GroupNorm(1, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def _split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        return x.reshape(x.shape[0], self.num_heads, x.shape[1]//self.num_heads, -1)
    
    def _combine_heads(self, x):
        # Combine the last two dimensions into (depth, )
        return x.reshape(x.shape[0], self.embed_dim, -1)
        
        
    def forward(self, x, y, mask=None, mask_cond=None):
        # Project the conditional information to queries, keys
        # and the input to values
        q, k, v = self.q_proj(y, mask_cond), self.k_proj(y, mask_cond), self.v_proj(x, mask)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Compute attention along the embedding dimension
        scores = ((q@k.transpose(-1, -2)) / (self.embed_dim//self.num_heads ** 0.5)).softmax(-1)
        
        # Compute the output
        out = scores@v
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        return self.norm((self.out_proj(out, mask) + x).transpose(-1, -2)).transpose(-1, -2) * (mask if type(mask) == torch.Tensor else 1)
