import torch
import torch.nn as nn
from src.blocks.transBlock import transBlock
from src.blocks.PositionalEncoding import PositionalEncoding






class Transformer(nn.Module):
    def __init__(self, input_dim, emb_dim, num_layers):
        super().__init__()
        
        self.proj_input = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.encoder = nn.Sequential(
            *[
                transBlock(emb_dim) for _ in range(num_layers)
            ]
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, input_dim)
        )
        
        self.pos_enc = PositionalEncoding(emb_dim, 0.0)
        
        
    def forward(self, x, y=None):
        x = self.proj_input(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.out_proj(x)