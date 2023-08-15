from torch import nn
from .wideResNet import ResnetBlock
from .MultiHeadAttention import MultiHeadAttention
from .ConditionalBlock import ConditionalBlock
from .ConditionalBlock2 import ConditionalBlock2
from .ContextBlock import ContextBlock
from .FullContextBlock import FullContextBlock






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
    #   c_dim - (optional) Number of dimensions in the context (text) input embedding
    #   atn_resolution - (optional) Resolution for the attention ("atn") blocks if used
    #   dropoutRate - (optional) Rate to apply dropout in the convnext blocks
    def __init__(self, inCh, outCh, blk_types, cond_dim=None, t_dim=None, c_dim=None, atn_resolution=None, dropoutRate=0.0):
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
            elif blk == "cond":
                blocks.append(ConditionalBlock(cond_dim, curCh))
            elif blk == "cond2":
                blocks.append(ConditionalBlock2(cond_dim, curCh))
            elif blk == "cond3":
                # blocks.append(ContextBlock(curCh, cond_dim, name="cond", num_heads=8, norm_type="middle_norm"))
                blocks.append(MultiHeadAttention(curCh, 8, norm_type="middle_norm", query_dim=curCh, key_dim=cond_dim, value_dim=cond_dim, name="cond"))
            elif blk == "atn":
                blocks.append(MultiHeadAttention(curCh, 8, norm_type="middle_norm", name="atn"))
            elif blk == "ctx":
                # blocks.append(ContextBlock(curCh, c_dim, name="text_context", num_heads=8, norm_type="post_norm"))
                blocks.append(MultiHeadAttention(curCh, 8, norm_type="middle_norm", query_dim=curCh, key_dim=c_dim, value_dim=c_dim, name="ctx"))
            elif blk == "fullctx":
                blocks.append(FullContextBlock(curCh, cond_dim, c_dim, 8, norm_type="pre_norm"))

            curCh = curCh1

        self.block = nn.Sequential(*blocks)


    # Input:
    #   X - Tensor of shape (N, inCh, T)
    #   y - (optional) Tensor of shape (N, cond_dim, T2)
    #   t - (optional) Tensor of shape (N, t_dim)
    #   context - (optional) Tensor of shape (N, c_dim, T3)
    #   mask - (optional) Tensor of shape (N, 1, T)
    #   mask_cond - (optional) Tensor of shape (N, 1, T2)
    #   masks_context - (optional) Tensor of shape (N, 1, T3)
    # Output:
    #   Tensor of shape (N, outCh, L, W)
    def forward(self, X, y=None, t=None, context=None, mask=None, mask_cond=None, mask_context=None):
        # Class assertion
        if y != None:
            assert self.useCls == True, \
                "cond_dim cannot be None if using conditional information"

        for b in self.block:
            if type(b) == ResnetBlock:
                X = b(X, t, mask)
            elif type(b) == ConditionalBlock or type(b) == ConditionalBlock2 or (type(b) == ContextBlock and b.name=="cond"):
                X = b(X, y, mask, mask_cond)
            elif type(b) == ContextBlock:
                X = b(X, context, mask, mask_context)
                
            elif (type(b) == MultiHeadAttention and b.name == "atn"):
                X = b(X, X, X, mask, mask, mask, res=X)
            elif (type(b) == MultiHeadAttention and b.name == "cond"):
                X = b(X, y, y, mask, mask_cond, mask_cond, res=X)
            elif (type(b) == MultiHeadAttention and b.name == "ctx"):
                X = b(X, context, context, mask, mask_context, mask_context, res=X)
                
            elif type(b) == FullContextBlock:
                X = b(X, y, context, mask, mask_cond, mask_context)

            else:
                X = b(X)
        return X