import torch
from torch import nn
from .MultiHeadAttention import MultiHeadAttention




class FullContextBlock(nn.Module):
    def __init__(self, embed_dim, cond_dim, ctx_dim, num_heads=8, norm_type="middle_norm", proj_scale=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.ctx_dim = ctx_dim
        
        # Self attention layer
        self.atn = MultiHeadAttention(embed_dim, num_heads, norm_type=norm_type)
        
        # Conditioning cross-attention layer
        self.cond_atn = MultiHeadAttention(embed_dim, 8, norm_type=norm_type, query_dim=embed_dim, key_dim=cond_dim, value_dim=cond_dim)
        
        # Context cross-attention layer
        self.ctx_atn = MultiHeadAttention(embed_dim, 8, norm_type=norm_type, query_dim=embed_dim, key_dim=ctx_dim, value_dim=ctx_dim)
        
        # Output linear projection
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * proj_scale),
            nn.SiLU(),
            nn.Linear(embed_dim * proj_scale, embed_dim)
        )
        
        
    def forward(self, X, y, context, mask=None, mask_cond=None, mask_context=None):
        # Transpose to (batch, seq_len, embed_dim)
        transpose_output = False
        if X.shape[1] == self.embed_dim:
            transpose_output = True
            X = X.transpose(1, 2)
            if mask is not None:
                mask = mask.transpose(1, 2)
        if y.shape[1] == self.cond_dim:
            y = y.transpose(1, 2)
            if mask_cond is not None:
                mask_cond = mask_cond.transpose(1, 2)
        if context.shape[1] == self.ctx_dim:
            context = context.transpose(1, 2)
            if mask_context is not None:
                mask_context = mask_context.transpose(1, 2)
        
        # Self attention
        X = self.atn(X, X, X, mask, mask, mask, res=X)
        
        # Conditioning cross-attention
        X = self.cond_atn(X, y, y, mask, mask_cond, mask_cond, res=X)
        
        # Context cross-attention
        X = self.ctx_atn(X, context, context, mask, mask_context, mask_context, res=X)
        
        # Output projection
        X = self.proj(X) + X
        
        # Transpose back to (batch, embed_dim, seq_len)
        if transpose_output:
            X = X.transpose(1, 2)
            
        return X