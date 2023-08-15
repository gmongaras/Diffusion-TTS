
# # Realtive import
# import sys
# sys.path.append('../blocks')

import torch
from torch import nn
try:
    from blocks.convolution.unetBlock import unetBlock
    from blocks.convolution.wideResNet import WeightStandardizedConv1d
    from blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
    from blocks.convolution.WeightStandardizedConvTranspose1d import WeightStandardizedConvTranspose1d
except ModuleNotFoundError:
    from src.blocks.convolution.unetBlock import unetBlock
    from src.blocks.convolution.wideResNet import WeightStandardizedConv1d
    from src.blocks.convolution.WeightStandardizedConv1d import WeightStandardizedConv1d
    from src.blocks.convolution.WeightStandardizedConvTranspose1d import WeightStandardizedConvTranspose1d
    








class U_Net(nn.Module):
    # inCh - Number of input channels in the input batch
    # outCh - Number of output channels in the output batch
    # embCh - Number of channels to embed the batch to
    # chMult - Multiplier to scale the number of channels by
    #          for each up/down sampling block
    # num_blocks - Number of blocks on the up/down path
    # blk_types - How should the residual block be structured 
    #             (list of "res", "conv", "clsAtn", and/or "chnAtn". 
    #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    # cond_dim - (optional) Vector size for the supplied cond vector
    # t_dim - (optional) Vector size for the supplied t vector
    # c_dim - (optional) Vector size for the supplied context vector
    # dropoutRate - Rate to apply dropout in the model
    # atn_resolution - Resolution of the attention blocks
    def __init__(self, inCh, outCh, embCh, chMult, num_blocks, blk_types, cond_dim=None, t_dim=None, c_dim=None, dropoutRate=0.0, atn_resolution=16):
        super(U_Net, self).__init__()

        self.cond_dim = cond_dim

        # Input convolution
        self.inConv = WeightStandardizedConv1d(inCh, embCh, 7, padding=3)
        
        # Downsampling
        # (N, inCh, L, W) -> (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        blocks = []
        curCh = embCh
        for i in range(1, num_blocks+1):
            blocks.append(unetBlock(curCh, embCh*(2**(chMult*i)), blk_types[i-1], cond_dim, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
            if i != num_blocks+1:
                blocks.append(WeightStandardizedConv1d(embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), kernel_size=3, stride=2, padding=1))
            curCh = embCh*(2**(chMult*i))
        self.downBlocks = nn.Sequential(
            *blocks
        )
        
        
        # Intermediate blocks
        # (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        # -> (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks))
        intermediateCh = curCh
        self.intermediate = nn.Sequential(
            # convNext(intermediateCh, intermediateCh, t_dim, dropoutRate=dropoutRate),
            unetBlock(intermediateCh, intermediateCh, ["res", "atn", "res"], cond_dim, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution),
            # Efficient_Channel_Attention(intermediateCh),
            # convNext(intermediateCh, intermediateCh, t_dim, dropoutRate=dropoutRate)
            # unetBlock(intermediateCh, intermediateCh, blk_types, cond_dim, t_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution),
        )
        
        
        # Upsample
        # (N, embCh^(chMult*num_blocks), L/(2^num_blocks), W/(2^num_blocks)) -> (N, inCh, L, W)
        blocks = []
        for i in range(num_blocks, -1, -1):
            if i == 0:
                # blocks.append(unetBlock(embCh*(2**(chMult*i)), embCh*(2**(chMult*i)), blk_types, cond_dim, t_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
                blocks.append(unetBlock(embCh*(2**(chMult*i)), embCh, ["res"], cond_dim, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
            else:
                blocks.append(WeightStandardizedConvTranspose1d(embCh*(2**(chMult*(i))), embCh*(2**(chMult*(i))), kernel_size=4, stride=2, padding=1))
                blocks.append(unetBlock(2*embCh*(2**(chMult*i)), embCh*(2**(chMult*(i-1))), blk_types[-min(num_blocks-i+2, num_blocks)], cond_dim, t_dim, c_dim, dropoutRate=dropoutRate, atn_resolution=atn_resolution))
        self.upBlocks = nn.Sequential(
            *blocks
        )
        
        # Final output block
        self.out = nn.Sequential(
            nn.SiLU(),
            WeightStandardizedConv1d(embCh, outCh, 3, padding=1)
        )

        # Down/up sample blocks
        self.downSamp = nn.AvgPool1d(2) 
        self.upSamp = nn.Upsample(scale_factor=2)

        # Time embeddings
        if t_dim is not None:
            self.t_emb = nn.Sequential(
                    nn.Linear(t_dim, t_dim),
                    nn.SiLU(),
                    nn.Linear(t_dim, t_dim),
                )
            
        # Context embeddings
        if c_dim is not None:
            self.c_emb = nn.Sequential(
                    nn.Linear(c_dim, c_dim),
                    nn.SiLU(),
                    nn.Linear(c_dim, c_dim),
                )
    
    
    # Input:
    #   X - Tensor of shape (N, E, T)
    #   y - (optional) Batch of encoded conditonal information
    #       of shape (N, E, T2)
    #   t - (optional) Batch of encoded t values for each 
    #       X value of shape (N, t_dim)
    #   context - (optional) Batch of encoded context information. This
    #             is the text conditioning information. Shape (N, T3, c_dim)
    #   masks - (optional) Batch of masks for each X value
    #       of shape (N, 1, T)
    #   masks_cond - (optional) Batch of masks for each c value
    #       of shape (N, 1, T2)
    #   masks_context - (optional) Batch of masks for each context value
    #       of shape (N, 1, T3)
    def forward(self, X, y=None, t=None, context=None, masks=None, masks_cond=None, masks_context=None):
        # conditional information assertion
        if type(y) != type(None):
            assert type(self.cond_dim) != type(None), "cond_dim must be specified when using condtional information."

        # Encode the time embeddings
        if t is not None:
            t = self.t_emb(t)
            
        # Encode the contextual information
        if context is not None:
            context = self.c_emb(context).transpose(-1, -2)

        # Saved residuals to add to the upsampling
        residuals = []
        residual_masks = []
        
        # Pre masking
        if type(masks) == torch.Tensor:
            X = X * masks
        if type(masks_cond) == torch.Tensor:
            y = y * masks_cond

        # Send the input through the input convolution
        X = self.inConv(X, masks)
        
        # Send the input through the downsampling blocks
        # while saving the output of each one
        # for residual connections
        b = 0
        while b < len(self.downBlocks):
            # Convoltuion blocks
            X = self.downBlocks[b](X, y, t, context=context, mask=masks, mask_cond=masks_cond, mask_context=masks_context)
            
            # Save residual from convolutions
            residuals.append(X.clone())
            
            # Final block is a downsampling block
            b += 1
            if b < len(self.downBlocks) and type(self.downBlocks[b]) == WeightStandardizedConv1d:
                # Reduce mask size and save old masks for later
                residual_masks.append(masks.clone() if type(masks) == torch.Tensor else None)
                masks = masks[:, :, ::2] if type(masks) == torch.Tensor else None
                
                # Downsample the input
                X = self.downBlocks[b](X, masks)
                b += 1
            
        # Reverse the residuals
        residuals = residuals[::-1] 
        
        # Send the output of the downsampling block
        # through the intermediate blocks
        # return X
        for b in self.intermediate:
            X = b(X, y=y, t=t, context=context, mask=masks, mask_cond=masks_cond, mask_context=masks_context)
        
        # Send the intermediate batch through the upsampling
        # block to get the original shape
        b = 0
        while b < len(self.upBlocks):
            # Upsample the input and get he masks
            if b < len(self.upBlocks) and type(self.upBlocks[b]) == WeightStandardizedConvTranspose1d:
                # Get masks for this layer
                masks = residual_masks.pop()
                
                # Upsample
                X = self.upBlocks[b](X, masks)
                b += 1
                
            # Other residual blocks
            if len(residuals) > 0:
                X = self.upBlocks[b](torch.cat((X[:, :, :residuals[0].shape[-1]], residuals[0]), dim=1), y, t, context, masks, masks_cond, masks_context)
                
            # Final residual block
            else:
                X = self.upBlocks[b](X, y, t, context, masks, masks_cond, masks_context)
            b += 1
            residuals = residuals[1:]
        
        # Send the output through the final block
        # and return the output
        return self.out[1](self.out[0](X) * masks, masks) if type(masks) == torch.Tensor else self.out(X) 