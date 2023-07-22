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
        
        self.attn1 = MultiHeadAttention(embed_dim, 8, query_dim=embed_dim, key_dim=cond_dim, value_dim=embed_dim, output_dim=embed_dim)
        # self.layer_norm1 = nn.GroupNorm(1, embed_dim)
        self.layer_norm1 = MaskedInstanceNorm1d(embed_dim)
        
        self.resBlock = ResnetBlock(embed_dim, embed_dim)
        
        self.attn2 = MultiHeadAttention(embed_dim, 8, query_dim=cond_dim, key_dim=embed_dim, value_dim=embed_dim, output_dim=embed_dim)
        # self.layer_norm2 = nn.GroupNorm(1, embed_dim)
        self.layer_norm2 = MaskedInstanceNorm1d(embed_dim)
        
        
    def forward(self, x, y=None, mask=None, mask_cond=None):
        # Identity if no conditional information
        if type(y) == type(None):
            return x
        
        # Residual connection
        res = x.clone()
        
        # First MHA, cross where the queries are the input sequence,
        # the keys are the conditional information and the values are the input sequence
        # Basically, the input sequence is conditioned on the conditional information
        x = self.layer_norm1(self.attn1(x, y, x, mask, mask_cond, mask), mask_cond) * (mask_cond if type(mask_cond) == torch.Tensor else 1)
        
        # Resnet block to extract features from the input
        x = self.resBlock(x, mask=mask_cond)
        
        # Second MHA, cross where the queries are the conditional information,
        # the keys are the input sequence and the values are the intermediate output
        # Basically, the conditional information is conditioned on the input sequence
        # and the intermediate output is transformed on this conditional matrix
        return self.layer_norm2(self.attn2(y, res, x, mask_cond, mask, mask_cond) + res, mask) * (mask if type(mask) == torch.Tensor else 1)
        