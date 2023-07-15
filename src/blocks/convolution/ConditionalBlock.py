import torch
import torch.nn as nn
from ..transformer.PositionalEncoding import PositionalEncoding
from .MultiHeadAttention import MultiHeadAttention
from .wideResNet import ResnetBlock




class ConditionalBlock(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        
        self.cond_dim = cond_dim
        self.embed_dim = embed_dim
        
        self.attn1 = MultiHeadAttention(embed_dim, 8, query_dim=cond_dim, key_dim=embed_dim, value_dim=embed_dim, output_dim=embed_dim)
        self.layer_norm1 = nn.GroupNorm(1, embed_dim)
        
        self.resBlock = ResnetBlock(embed_dim, embed_dim)
        
        self.attn2 = MultiHeadAttention(embed_dim, 8, query_dim=cond_dim, key_dim=embed_dim, value_dim=embed_dim, output_dim=embed_dim)
        self.layer_norm2 = nn.GroupNorm(1, embed_dim)
        
        
    def forward(self, x, y):
        # Residual connection
        res = x.clone()
        
        # First MHA, cross where the queries are the conditional information
        # and the keys (and values) are the input
        x = self.layer_norm1(self.attn1(y, x, x))
        
        # Resnet block to extract features from the input
        x = self.resBlock(x)
        
        # Second MHA, cross where the keys are the conditional information,
        # the queries are the input and the values are the output of the resnet block
        return self.layer_norm2(self.attn2(y, res, x) + res)
        