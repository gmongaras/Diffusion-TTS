import torch
import torch.nn as nn




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, query_dim=None, key_dim=None, value_dim=None, output_dim=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_dim = query_dim if query_dim else embed_dim
        self.key_dim = key_dim if key_dim else embed_dim
        self.value_dim = value_dim if value_dim else embed_dim
        self.output_dim = output_dim if output_dim else embed_dim
        
        self.q_proj = nn.Conv1d(self.query_dim, embed_dim, 1)
        self.k_proj = nn.Conv1d(self.key_dim, embed_dim, 1)
        self.v_proj = nn.Conv1d(self.value_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, self.output_dim, 1)
        
        
    def _split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        return x.reshape(x.shape[0], self.num_heads, x.shape[1]//self.num_heads, -1)
    
    def _combine_heads(self, x):
        # Combine the last two dimensions into (depth, )
        return x.reshape(x.shape[0], self.embed_dim, -1)
        
        
    def forward(self, q, k, v, transpose_scores=False):
        # Project the queries, keys and values
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Compute attention like normal
        scores = ((k.permute(0, 1, 3, 2)@q) / (self.embed_dim ** 0.5)).softmax(dim=-1)
        
        # Compute the output
        if transpose_scores:
            out = v@scores.transpose(-1, -2)
        else:
            out = v@scores
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        return self.out_proj(out)
        