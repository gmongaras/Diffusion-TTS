import torch
import torch.nn as nn
from .MaskedInstanceNorm1d import MaskedInstanceNorm1d

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d




class ContextBlock(nn.Module):
    def __init__(self, embed_dim, context_dim, name=None):
        super().__init__()
        
        # Following Stable Diffusion, the queries are the data embeddings
        # and the keys and values are the context embeddings
        self.embed_dim = embed_dim
        self.num_heads = 8
        self.name = name
        
        # Keys and values come from the conditional dimension
        self.k_proj = WeightStandardizedConv1d(context_dim, embed_dim, 1)
        self.v_proj = WeightStandardizedConv1d(context_dim, embed_dim, 1)
        
        # Queries come from the input dimension
        self.q_proj = WeightStandardizedConv1d(embed_dim, embed_dim, 1)
        
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
        
    def forward(self, x, context, mask=None, mask_ctx=None):
        # Project the conditional information to queries, keys
        # and the input to values
        q, k, v = self.q_proj(x, mask), self.k_proj(context, mask_ctx), self.v_proj(context, mask_ctx)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Compute attention along the embedding dimension
        scores = ((k.transpose(-1, -2)@q) / (self.embed_dim//self.num_heads ** 0.5))
        
        # Mask scores along input (queries) dimension
        if type(mask) == torch.Tensor:
            scores = scores.masked_fill(~mask.unsqueeze(-2), -1e9)
        
        # Apply softmax
        scores = scores.softmax(-1)
        
        # Apply mask along the context (keys) dimension
        if type(mask_ctx) == torch.Tensor:
            scores = scores.masked_fill(~mask_ctx.unsqueeze(-1), 0)
        
        # Compute the output
        # out = (scores@v.transpose(-1, -2)).transpose(-1, -2)
        out = v@scores
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        return self.norm((self.out_proj(out, mask) + x).transpose(-1, -2)).transpose(-1, -2) * (mask if type(mask) == torch.Tensor else 1)
