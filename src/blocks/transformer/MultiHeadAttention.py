import torch
import torch.nn as nn




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, is_cross=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.is_cross = is_cross
        
        if not is_cross:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        
    def _split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        return x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1]//self.num_heads).permute(0, 2, 1, 3)
    
    def _combine_heads(self, x):
        # Combine the last two dimensions into (depth, )
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-1]*self.num_heads)
        
        
    def forward(self, q, k, v):
        # Project the queries, keys and values
        if not self.is_cross:
            q, k = self.q_proj(q), self.k_proj(k)
        v = self.v_proj(v)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Compute the attention scores
        if self.is_cross:
            # Computes the attention scores along the embedidng dimension unlike normal attention
            scores = (torch.matmul(q.permute(0, 1, 3, 2), k) / (self.embed_dim ** 0.5)).softmax(dim=-1)
            
            # Compute the output
            out = v@scores
        else:
            # Compute attention like normal
            scores = (torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.embed_dim ** 0.5)).softmax(dim=-1)
            
            # Compute the output
            out = torch.matmul(scores, v)
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        return self.out_proj(out)
        