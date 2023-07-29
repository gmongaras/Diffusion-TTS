import torch
from torch import nn
import math




# https://huggingface.co/blog/annotated-diffusion#position-embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        self.embeddings = math.log(10000) / (self.half_dim - 1)
        self.embeddings = torch.exp(torch.arange(self.half_dim) * -self.embeddings)

    def forward(self, time):
        embeddings = time[:, None] * self.embeddings[None, :].to(time.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings