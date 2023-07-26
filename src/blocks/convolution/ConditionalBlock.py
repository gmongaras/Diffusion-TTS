import torch
import torch.nn as nn
from ..transformer.PositionalEncoding import PositionalEncoding
from .MultiHeadAttention import MultiHeadAttention
from .wideResNet import ResnetBlock
from .MaskedInstanceNorm1d import MaskedInstanceNorm1d




class ConditionalBlock(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        
        self.cond_dim = cond_dim
        self.embed_dim = embed_dim
        
        self.attn1 = MultiHeadAttention(embed_dim, 8, query_dim=cond_dim, key_dim=embed_dim, value_dim=embed_dim, output_dim=embed_dim)
        # self.layer_norm1 = nn.GroupNorm(1, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.resBlock = ResnetBlock(embed_dim, embed_dim)
        
        self.attn2 = MultiHeadAttention(embed_dim, 8, query_dim=embed_dim, key_dim=cond_dim, value_dim=embed_dim, output_dim=embed_dim)
        # self.layer_norm2 = nn.GroupNorm(1, embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, x, y=None, mask=None, mask_cond=None):
        # Identity if no conditional information
        if type(y) == type(None):
            return x
        
        # Residual connection
        res = x.clone()
        
        # First MHA, cross where the queries are the input sequence.
        # Let's say our input sequence, X, is of shape (E, T)
        # and our conditional information, y, is of shape (E, T_hat)
        # The first score matrix we construct should be of shape (T_hat, T)
        # This matrix is basically saying we want to transform our input sequence
        # from (E, T) to (E, T_hat). So, we need to have T_hat linear combinations
        # of T values. This can be done by having the input sequnec be the keys
        # and the conditional information be the queries. The values are the input as well
        # (N, E, T) -> (N, E, T_hat)
        # x = self.layer_norm1(self.attn1(y, x, x, mask_cond, mask, mask)) * (mask_cond if type(mask_cond) == torch.Tensor else 1)
        x = self.attn1(y, x, x, mask_cond, mask, mask)
        x = self.layer_norm1(x.transpose(-1, -2)).transpose(-1, -2) * (mask_cond if type(mask_cond) == torch.Tensor else 1)
        
        # Resnet block to extract features from the input
        x = self.resBlock(x, mask=mask_cond)
        
        # Second MHA, cross where the queries are the keys information,
        # the queries are the input sequence and the values are the intermediate output
        # This is the reverse of the MHA where the linear combination is of the
        # conditional information, not the input sequence.
        # (N, E, T_hat) -> (N, E, T)
        # return self.layer_norm2(self.attn2(res, y, x, mask, mask_cond, mask_cond) + res) * (mask if type(mask) == torch.Tensor else 1)
        x = self.attn2(res, y, x, mask, mask_cond, mask_cond)
        x = self.layer_norm2((x+res).transpose(-1, -2)).transpose(-1, -2) * (mask if type(mask) == torch.Tensor else 1)
        return x
        