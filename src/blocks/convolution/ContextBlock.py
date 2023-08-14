import torch
import torch.nn as nn
from .MaskedInstanceNorm1d import MaskedInstanceNorm1d

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d




class ContextBlock(nn.Module):
    def __init__(self, embed_dim, context_dim, name=None, num_heads=8, norm_type="middle_norm"):
        super().__init__()
        
        # Following Stable Diffusion, the queries are the data embeddings
        # and the keys and values are the context embeddings
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.name = name
        self.norm_type = norm_type
        
        # Keys and values come from the conditional dimension
        self.k_proj = WeightStandardizedConv1d(context_dim, embed_dim, 1)
        self.v_proj = WeightStandardizedConv1d(context_dim, embed_dim, 1)
        
        # Queries come from the input dimension
        self.q_proj = WeightStandardizedConv1d(embed_dim, embed_dim, 1)
        
        # Ouptut is the same size as the input
        self.out_proj = WeightStandardizedConv1d(embed_dim, embed_dim, 1)
        
        if norm_type == "pre_norm":
            self.norm_x = nn.LayerNorm(embed_dim)
            self.norm_ctx = nn.LayerNorm(context_dim)
        elif norm_type == "post_norm":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_type == "middle_norm":
            # raise NotImplementedError("middle_norm not implemented. Has issues with masks")
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
        
    def forward(self, x, context, mask=None, mask_ctx=None):
        # Normalize input
        if self.norm_type == "pre_norm":
            x = self.norm_x(x.transpose(-1, -2)).transpose(-1, -2) * (mask if type(mask) == torch.Tensor else 1)
            context = self.norm_ctx(context.transpose(-1, -2)).transpose(-1, -2) * (mask_ctx if type(mask_ctx) == torch.Tensor else 1)
        
        # Project the conditional information to queries, keys
        # and the input to values
        q, k, v = self.q_proj(x, mask), self.k_proj(context, mask_ctx), self.v_proj(context, mask_ctx)
        
        # Split each embedding into self.num_heads pieces
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        
        # Normalize each head
        if self.norm_type == "middle_norm":
            q, k, v = self.norm_q(q.transpose(-1, -2)).transpose(-1, -2), self.norm_k(k.transpose(-1, -2)).transpose(-1, -2), self.norm_v(v.transpose(-1, -2)).transpose(-1, -2)
            # q = self.norm_q(q.transpose(-1, -2)).transpose(-1, -2) * (mask.unsqueeze(-2) if type(mask) == torch.Tensor else 1)
            # k = self.norm_k(k.transpose(-1, -2)).transpose(-1, -2) * (mask_ctx.unsqueeze(-2) if type(mask_ctx) == torch.Tensor else 1)
            # v = self.norm_v(v.transpose(-1, -2)).transpose(-1, -2) * (mask_ctx.unsqueeze(-2) if type(mask_ctx) == torch.Tensor else 1)
            
            # q, k, v = self.norm_q(q.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_k(k.flatten(0, 1)).unflatten(0, (-1, self.num_heads)), self.norm_v(v.flatten(0, 1)).unflatten(0, (-1, self.num_heads))
            # q = self.norm_q(q.flatten(0, 1)).unflatten(0, (-1, self.num_heads)) * (mask.unsqueeze(-2) if type(mask) == torch.Tensor else 1)
            # k = self.norm_k(k.flatten(0, 1)).unflatten(0, (-1, self.num_heads)) * (mask_ctx.unsqueeze(-2) if type(mask_ctx) == torch.Tensor else 1)
            # v = self.norm_v(v.flatten(0, 1)).unflatten(0, (-1, self.num_heads)) * (mask_ctx.unsqueeze(-2) if type(mask_ctx) == torch.Tensor else 1)
        
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
        out = self.out_proj(out, mask) + x
        if self.norm_type == "post_norm":
            out = self.norm(out.transpose(-1, -2)).transpose(-1, -2)
        return out
