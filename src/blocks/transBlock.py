import torch
import torch.nn as nn





class transBlock(nn.Module):
    def __init__(self, emb_dim, linear_dim=None, num_heads=8):
        super().__init__()
        
        if not linear_dim:
            linear_dim = emb_dim * 4
            
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads)
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, emb_dim)
        )
        
        
    def forward(self, x, y=None):
        res = x.clone()
        x = self.norm1(x)
        x = self.attn(x, x, x)[0] if y is None else self.attn(y.T, y.T, x)[0]
        x = self.norm2(x + res)
        res = x.clone()
        x = self.linear(x)
        return x + res