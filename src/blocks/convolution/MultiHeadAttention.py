import torch
import torch.nn as nn

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, norm_type="middle_norm", query_dim=None, key_dim=None, value_dim=None, output_dim=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.query_dim = query_dim if query_dim else embed_dim
        self.key_dim = key_dim if key_dim else embed_dim
        self.value_dim = value_dim if value_dim else embed_dim
        self.output_dim = output_dim if output_dim else embed_dim
        
        # self.q_proj = nn.Conv1d(self.query_dim, embed_dim, 1)
        # self.k_proj = nn.Conv1d(self.key_dim, embed_dim, 1)
        # self.v_proj = nn.Conv1d(self.value_dim, embed_dim, 1)
        # self.out_proj = nn.Conv1d(embed_dim, self.output_dim, 1)
        self.q_proj = WeightStandardizedConv1d(self.query_dim, embed_dim, 1)
        self.k_proj = WeightStandardizedConv1d(self.key_dim, embed_dim, 1)
        self.v_proj = WeightStandardizedConv1d(self.value_dim, embed_dim, 1)
        self.out_proj = WeightStandardizedConv1d(embed_dim, self.output_dim, 1)
        
        if norm_type == "pre_norm":
            self.norm_q = nn.LayerNorm(embed_dim)
            self.norm_k = nn.LayerNorm(embed_dim)
            self.norm_v = nn.LayerNorm(embed_dim)
        elif norm_type == "post_norm":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_type == "middle_norm":
            # raise NotImplementedError("GN not implemented. Has issues with masks")
            self.norm_q = nn.LayerNorm(embed_dim//num_heads)
            self.norm_k = nn.LayerNorm(embed_dim//num_heads)
            self.norm_v = nn.LayerNorm(embed_dim//num_heads)
            # self.norm_q = nn.GroupNorm(16, embed_dim//num_heads)
            # self.norm_k = nn.GroupNorm(16, embed_dim//num_heads)
            # self.norm_v = nn.GroupNorm(16, embed_dim//num_heads)
        else:
            raise ValueError(f"norm_type must be 'pre_norm', 'post_norm', or 'middle_norm', not {norm_type}")
        
        
    def _split_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        return x.reshape(x.shape[0], self.num_heads, x.shape[1]//self.num_heads, -1)
    
    def _combine_heads(self, x):
        # Combine the last two dimensions into (depth, )
        return x.reshape(x.shape[0], self.embed_dim, -1)
        
        
    def forward(self, q, k, v, query_mask=None, key_mask=None, value_mask=None, res=None, transpose_scores=False):
        if self.norm_type == "pre_norm":
            q = self.norm_q(q.transpose(-1, -2)).transpose(-1, -2)
            k = self.norm_k(k.transpose(-1, -2)).transpose(-1, -2)
            v = self.norm_v(v.transpose(-1, -2)).transpose(-1, -2)
        
        # Project the queries, keys and values
        # Note that the masked states are retained
        q, k, v = self.q_proj(q, query_mask), self.k_proj(k, key_mask), self.v_proj(v, value_mask)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Normalize each head
        if self.norm_type == "middle_norm":
            q, k, v = self.norm_q(q.transpose(-1, -2)).transpose(-1, -2), self.norm_k(k.transpose(-1, -2)).transpose(-1, -2), self.norm_v(v.transpose(-1, -2)).transpose(-1, -2)
            # q, k, v = self.norm_q(q.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_k(k.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_v(v.flatten(0, 1)).unflatten(0, (-1, self.num_heads))
        
        # Compute attention like normal (along time dimension)
        # Note that the queries are transposed because the input
        # sequence is (E, T) not (T, E) like normal.
        # scores = ((q.permute(0, 1, 3, 2)@k) 
        #             * (query_mask.unsqueeze(-1) if type(query_mask) == torch.Tensor else 1)
        #             * (key_mask.unsqueeze(-2) if type(key_mask) == torch.Tensor else 1)
        #           / (self.embed_dim ** 0.5)).softmax(dim=-1) \
        #               * (key_mask.unsqueeze(-2) if type(key_mask) == torch.Tensor else 1)
        scores = torch.matmul(q.transpose(-1, -2), k) / (self.embed_dim//self.num_heads ** 0.5)
        
        # Key padding mask applied to scores
        if type(key_mask) == torch.Tensor:
            scores = scores.masked_fill(key_mask.unsqueeze(-2) == 0, -1e9)
            
        # Softmax along the keys
        scores = scores.softmax(dim=-1)
        
        # Query padding mask applied to scores after the
        # softmax is applied as the result is always
        # 0 for padded queries
        if type(query_mask) == torch.Tensor:
            scores = scores.masked_fill(query_mask.unsqueeze(-1) == 0, 0)
        
        # Compute the output
        # Again, we need a transpose because the input sequence
        # is (E, T) not (T, E) like normal.
        # Note that the output will be masked along the query mask
        if transpose_scores:
            out = (scores.transpose(-1, -2)@v.transpose(-1, -2)).transpose(-1, -2)
        else:
            out = (scores@v.transpose(-1, -2)).transpose(-1, -2)
        
        # Compute the output and remove the heads
        out = self._combine_heads(out)
        
        # Project output and return
        out = self.out_proj(out, query_mask if not transpose_scores else key_mask)
        if type(res) == torch.Tensor:
            out = out + res
        if self.norm_type == "post_norm":
            out = self.norm(out.transpose(-1, -2)).transpose(-1, -2)
        return out
        