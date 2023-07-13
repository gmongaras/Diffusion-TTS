import torch
import torch.nn as nn
from src.blocks.transformer.transBlock import transBlock
from src.blocks.transformer.PositionalEncoding import PositionalEncoding






class Transformer(nn.Module):
    def __init__(self, input_dim, input_dim_cond, emb_dim, num_layers):
        super().__init__()
        
        self.proj_input = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.encoder = []
        for _ in range(num_layers):
            self.encoder.append(transBlock(emb_dim))    # Self attention
            self.encoder.append(nn.Sequential(          # Conditiong embedding
                    nn.Linear(input_dim_cond, emb_dim),
                    nn.ReLU(),
            ))
            self.encoder.append(transBlock(emb_dim, is_cross=True))    # Cross attention
        self.encoder = nn.Sequential(*self.encoder)
        
        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, input_dim)
        )
        
        self.pos_enc = PositionalEncoding(emb_dim, 0.0)
        
        
        
    # x: (B, T, E)
    # y: (B, T2, E)
    def forward(self, x, y=None):
        # Embed the inputs
        x = self.proj_input(x)
        x = self.pos_enc(x)
        
        # Iterate over the encoder layers
        i = 0
        while i < len(self.encoder):
            x = self.encoder[i](x) # Self attention
            x = self.encoder[i+2](x, self.encoder[i+1](y)) # Cross attention
            i += 3
            
        return self.out_proj(x)