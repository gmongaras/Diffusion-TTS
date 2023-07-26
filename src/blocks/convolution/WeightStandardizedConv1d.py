import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce
from functools import partial





# Weight standardization is shown to improve convolutions
# when using groupNorm.
class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def __init__(self, *args, norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, x, mask=None):
        if self.norm:
            eps = 1e-5 if x.dtype == torch.float32 else 1e-3
            weight = self.weight
            mean = reduce(weight, "o ... -> o 1 1", "mean")
            var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
            normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight if self.norm else self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        ) * (mask if type(mask) == torch.Tensor else 1)