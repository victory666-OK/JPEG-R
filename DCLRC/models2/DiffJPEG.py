# Pytorch
import torch
import torch.nn as nn
# Local
from models.modules import decompress_jpeg
from models.utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = 1
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, y, cb, cr,quantization):
        recovered = self.decompress(y, cb, cr,quantization)
        return recovered
        
class CoefficientsRearrange(nn.Module):
    def __init__(self):
        super(CoefficientsRearrange, self).__init__()

    def forward(self, coefficient):
        k = 8
        # 假设coefficient的形状是 [N, C, D, H, W, S], 其中S是额外的维度
        coefficient_1 = coefficient.view(
            coefficient.shape[0], 
            coefficient.shape[1], 
            coefficient.shape[2], 
            coefficient.shape[3], 
            coefficient.shape[4]*coefficient.shape[5]
        )
        coefficient_2 = coefficient_1.permute(0, 1, 4, 2, 3)
        coefficient_3 = coefficient_2.contiguous().view(
            coefficient_2.shape[0], 
            coefficient_2.shape[1]*coefficient_2.shape[2], 
            coefficient_2.shape[3],
            coefficient_2.shape[4]
        )
        return coefficient_3

class CoefficientsRearrangeInverse(nn.Module):
    def __init__(self):
        super(CoefficientsRearrangeInverse, self).__init__()

    def forward(self, coefficient):
        N,C,H,W = coefficient.shape
        k = 8
        
        coefficient_11 = coefficient.permute(0,2,3,1)
        coefficient_22 = coefficient_11.contiguous().view(coefficient_11.shape[0],coefficient_11.shape[1],coefficient_11.shape[2],k,k)
        coefficient_33 = coefficient_22.contiguous().view(coefficient_22.shape[0],coefficient_22.shape[1]*coefficient_22.shape[2],k,k)
        
        return coefficient_33

