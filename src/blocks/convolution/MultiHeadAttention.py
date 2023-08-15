import torch
import torch.nn as nn

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d




class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, norm_type="middle_norm", query_dim=None, key_dim=None, value_dim=None, output_dim=None, name=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm_type = norm_type
        self.query_dim = query_dim if query_dim else embed_dim
        self.key_dim = key_dim if key_dim else embed_dim
        self.value_dim = value_dim if value_dim else embed_dim
        self.output_dim = output_dim if output_dim else embed_dim
        self.name = name
        
        self.q_proj = nn.Linear(self.query_dim, embed_dim)
        self.k_proj = nn.Linear(self.key_dim, embed_dim)
        self.v_proj = nn.Linear(self.value_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.output_dim)
        
        if norm_type == "pre_norm":
            self.norm_q = nn.LayerNorm(embed_dim)
            self.norm_k = nn.LayerNorm(self.key_dim)
            self.norm_v = nn.LayerNorm(self.value_dim)
        elif norm_type == "post_norm":
            self.norm = nn.LayerNorm(self.output_dim)
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
        # Split into self.heads (N, T, E) -> (N, H, T, E/H)
        return x.reshape(x.shape[0], -1, self.num_heads, x.shape[2]//self.num_heads).transpose(1, 2)
    
    def _combine_heads(self, x):
        # Combine from (N, H, T, E/H) -> (N, T, E)
        return x.transpose(1, 2).reshape(x.shape[0], -1, self.embed_dim)
        
        
    # q - tensor of shape (N, E_q, T) or (N, T, E_q)
    # k - tensor of shape (N, E_k, T2) or (N, T2, E_k)
    # v - tensor of shape (N, E_v, T2) or (N, T2, E_v)
    #    - Where T does not have to equal T2
    # query_mask - tensor of shape (N, 1, T) or (N, T, 1)
    # key_mask - tensor of shape (N, 1, T2) or (N, T2, 1)
    # value_mask - tensor of shape (N, 1, T2) or (N, T2, 1)
    # res - tensor of shape (N, E, T) or (N, T, E)
    def forward(self, q, k, v, query_mask=1, key_mask=1, value_mask=1, res=None):
        if type(query_mask) == type(None):
            query_mask = 1
        if type(key_mask) == type(None):
            key_mask = 1
        if type(value_mask) == type(None):
            value_mask = 1
        
        # Reshape to (N, T, E)
        reshape_output = False
        if q.shape[1] == self.query_dim:
            q = q.transpose(-1, -2)
            if type(query_mask) == torch.Tensor:
                query_mask = query_mask.transpose(-1, -2)
        if k.shape[1] == self.key_dim:
            k = k.transpose(-1, -2)
            if type(key_mask) == torch.Tensor:
                key_mask = key_mask.transpose(-1, -2)
        if v.shape[1] == self.value_dim:
            reshape_output = True
            v = v.transpose(-1, -2)
            if type(value_mask) == torch.Tensor:
                value_mask = value_mask.transpose(-1, -2)
        
        # Pre normalization
        if self.norm_type == "pre_norm":
            q = self.norm_q(q)
            k = self.norm_k(k)
            v = self.norm_v(v)
        
        # Project the queries, keys and values
        # Note that the masked states are retained
        q, k, v = self.q_proj(q)*query_mask, self.k_proj(k)*key_mask, self.v_proj(v)*value_mask
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Normalize each head
        if self.norm_type == "middle_norm":
            q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
            # q, k, v = self.norm_q(q.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_k(k.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_v(v.flatten(0, 1)).unflatten(0, (-1, self.num_heads))
        
        # Compute attention like normal (along time dimension)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.embed_dim//self.num_heads ** 0.5)
        
        # Key padding mask applied to scores
        if type(key_mask) == torch.Tensor:
            scores = scores.masked_fill(~key_mask.unsqueeze(1).transpose(-1, -2), -1e9)
            
        # Softmax along the keys
        scores = scores.softmax(dim=-1)
        
        # Query padding mask applied to scores after the
        # softmax is applied as the result is always
        # 0 for padded queries
        if type(query_mask) == torch.Tensor:
            scores = scores.masked_fill(~query_mask.unsqueeze(1), 0)
        
        # Compute the output attention scores
        out = scores@v
        
        # Remove the heads
        out = self._combine_heads(out)
        
        # Project output. Mask by queries
        out = self.out_proj(out) * query_mask
        
        # Residual
        if type(res) == torch.Tensor:
            out = out + (res if not reshape_output else res.transpose(-1, -2))
            
        # Post norm
        if self.norm_type == "post_norm":
            out = self.norm(out)
            
        # Reshape to (N, E, T)
        if reshape_output:
            out = out.transpose(-1, -2)
            
        return out
        