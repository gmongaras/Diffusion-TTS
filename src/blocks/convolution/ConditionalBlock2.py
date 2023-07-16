import torch
import torch.nn as nn




class ConditionalBlock2(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = 8
        
        # Query and keys come from the conditional dimension
        self.q_proj = nn.Conv1d(cond_dim, embed_dim, 1)
        self.k_proj = nn.Conv1d(cond_dim, embed_dim, 1)
        
        # Values come from the input dimension
        self.v_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        
        # Ouptut is the same size as the input
        self.out_proj = nn.Conv1d(embed_dim, embed_dim, 1)
        
        self.layer_norm = nn.GroupNorm(1, embed_dim)
        
    def _split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        return x.reshape(x.shape[0], self.num_heads, x.shape[1]//self.num_heads, -1)
    
    def _combine_heads(self, x):
        # Combine the last two dimensions into (depth, )
        return x.reshape(x.shape[0], self.embed_dim, -1)
        
        
    def forward(self, x, y):
        # Project the conditional infomration to queries, keys
        # and the input to values
        q, k, v = self.q_proj(y), self.k_proj(y), self.v_proj(x)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Compute attention like normal
        scores = ((q@k.transpose(-1, -2)) / (self.embed_dim ** 0.5)).softmax(dim=-1)
        
        # Compute the output
        out = scores@v
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        return self.layer_norm(self.out_proj(out) + x)
