import torch
from torch import nn
from einops import rearrange
from src.blocks.convolution.MaskedInstanceNorm1d import MaskedInstanceNorm1d

try:
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
except ModuleNotFoundError:
    from .WeightStandardizedConv1d import WeightStandardizedConv1d



# Note, these blocks are from https://huggingface.co/blog/annotated-diffusion




# Check if something exists. Return None if it doesn't
def exists(x):
    return x is not None



class Block(nn.Module):
    """
    Each block consists of:
        A weight standardized convolution (3x3)
        A group norm block
        A Silu block
    
    The original convolution was conv 3x3 -> ReLU,
    but it was found that group norm + weight standardization
    improves the performance of the model.
    """
    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        # self.norm = nn.GroupNorm(groups, dim_out)
        self.norm = MaskedInstanceNorm1d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, t_mul=None, t_add=None, mask=None):
        # Project and normalize the embeddings
        x = self.proj(x, mask=mask)
        if type(mask) != torch.Tensor:
            x = self.norm(x)
        else:
            x = self.norm(x, mask)

        # To add the class and time information, the
        # embedding is scaled by the time embeddings
        # and shifted by the class embeddings. In the
        # diffusion models beat gans on image synthesis paper
        # and is called Adaptive Group Normalization
        # https://arxiv.org/abs/2105.05233
        if exists(t_mul):
            x = x * t_mul
        if exists(t_add):
            x = x + t_add

        # Apply the SiLU layer to the embeddings
        x = self.act(x)
        return x * mask if type(mask) == torch.Tensor else x



class ResnetBlock(nn.Module):
    """
    https://arxiv.org/abs/1512.03385
    This resnet block consits of:
        1 residual block with 8 groups (or 1 group) using cls and time info
        1 residual block with 8 groups (or 1 group) not using cls and time info
        one output convolution to project the embeddings from 
            the input channels to the output channels
    
    For the time and class embeddings, the embeddings
    is projected using a linear layer and SiLU layer
    before entering the residual block
    """
    def __init__(self, inCh, outCh, t_dim=None, dropoutRate=0.0):
        # Projections with time and class info
        super().__init__()
        self.t_mlp_mul = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, outCh))
            if exists(t_dim)
            else None
        )
        self.t_mlp_add = (
            nn.Sequential(nn.SiLU(), nn.Linear(t_dim, outCh))
            if exists(t_dim)
            else None
        )

        # Convolutional blocks for residual and projections
        self.block1 = Block(inCh, outCh, groups=8 if inCh > 4 and inCh%8==0 and outCh%8==0 else 1)
        self.block2 = Block(outCh, outCh, groups=8 if outCh > 4 and outCh%8==0 else 1)
        self.res_conv = WeightStandardizedConv1d(inCh, outCh, 1) if inCh != outCh else nn.Identity()

    def forward(self, x, t=None, mask=None):
        # Apply the class and time projections
        t_mul, t_add = None, None
        if exists(self.t_mlp_mul) and exists(t):
            t_mul = self.t_mlp_mul(t)
            t_mul = rearrange(t_mul, "b c -> b c 1")
            
            t_add = self.t_mlp_add(t)
            t_add = rearrange(t_add, "b c -> b c 1")

        # Apply the convolutional blocks and
        # output projection with a residual connection
        h = self.block1(x, t_mul, t_add, mask)
        h = self.block2(h, mask=mask)
        return h + self.res_conv(x) if x.shape == h.shape else self.res_conv(x, mask=mask)