from torch import nn
from .convNext import convNext
from .Efficient_Channel_Attention import Efficient_Channel_Attention
from .clsAttn import clsAttn, clsAttn_Linear, Efficient_Cls_Attention
from .wideResNet import ResnetBlock
from .MultiHeadAttention import MultiHeadAttention
from .ConditionalBlock import ConditionalBlock
from .ConditionalBlock2 import ConditionalBlock2






# # Map from string form of a block to object form
# str_to_blk = dict(
#     res=ResnetBlock,
#     cond=ConditionalBlock,
#     conv=convNext,
#     clsAtn=clsAttn,
#     chnAtn=Efficient_Channel_Attention,
#     atn=Multihead_Attn,
# )



class unetBlock(nn.Module):
    # Inputs:
    #   inCh - Number of channels the input batch has
    #   outCh - Number of chanels the ouput batch should have
    #   blk_types - How should the residual block be structured 
    #             (list of "res", "conv", "atn", "clsAtn", and/or "chnAtn". 
    #              Ex: ["res", "res", "conv", "clsAtn", "chnAtn"] 
    #   cond_dim - (optional) Vector size for the supplied cond vector
    #   t_dim - (optional) Number of dimensions in the time input embedding
    #   atn_resolution - (optional) Resolution for the attention ("atn") blocks if used
    #   dropoutRate - (optional) Rate to apply dropout in the convnext blocks
    def __init__(self, inCh, outCh, blk_types, cond_dim=None, t_dim=None, atn_resolution=None, dropoutRate=0.0):
        super(unetBlock, self).__init__()

        self.useCls = False if cond_dim == None else True

        # Generate the blocks. THe first blocks goes from inCh->outCh.
        # The rest goes from outCh->outCh
        blocks = []
        curCh = inCh
        curCh1 = outCh
        for blk in blk_types:
            if blk == "res":
                blocks.append(ResnetBlock(curCh, curCh1, t_dim, dropoutRate))
            if blk == "cond":
                blocks.append(ConditionalBlock(cond_dim, curCh))
            if blk == "cond2":
                blocks.append(ConditionalBlock2(cond_dim, curCh))
            if blk == "atn":
                blocks.append(MultiHeadAttention(curCh, 8))

            curCh = curCh1

        self.block = nn.Sequential(*blocks)


    # Input:
    #   X - Tensor of shape (N, inCh, T)
    #   y - (optional) Tensor of shape (N, cond_dim, T)
    #   t - (optional) Tensor of shape (N, t_dim)
    #   mask - (optional) Tensor of shape (N, 1, T)
    #   mask_cond - (optional) Tensor of shape (N, 1, T)
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, y=None, t=None, mask=None, mask_cond=None):
        # Class assertion
        if y != None:
            assert self.useCls == True, \
                "cond_dim cannot be None if using conditional information"

        for b in self.block:
            if type(b) == ResnetBlock:
                X = b(X, t, mask)
            elif type(b) == ConditionalBlock or type(b) == ConditionalBlock2:
                X = b(X, y, mask, mask_cond)
            elif type(b) == MultiHeadAttention:
                X = b(X, X, X, mask, mask, mask)
                # X = b(X.transpose(-1, -2), X.transpose(-1, -2), X.transpose(-1, -2))[0].transpose(-1, -2) 
            else:
                X = b(X)
        return X