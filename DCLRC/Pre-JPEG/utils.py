# Standard libraries
import numpy as np
# PyTorch
import torch
import torch.nn as nn

y_table = np.array(
        [[13,  9,  8, 13, 19, 32, 41, 49],
         [10, 10, 11, 15, 21, 46, 48, 44],
         [11, 10, 13, 19, 32, 46, 55, 45],
         [11, 14, 18, 23, 41, 70, 64, 50],
         [14, 18, 30, 45, 54, 87, 82, 62],
         [19, 28, 44, 51, 65, 83, 90, 74],
         [39, 51, 62, 70, 82, 97, 96, 81],
         [58, 74, 76, 78, 90, 80, 82, 79]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.array(
        [[14, 14, 19, 38, 79, 79, 79, 79],
         [14, 17, 21, 53, 79, 79, 79, 79],
         [19, 21, 45, 79, 79, 79, 79, 79],
         [38, 53, 79, 79, 79, 79, 79, 79],
         [79, 79, 79, 79, 79, 79, 79, 79],
         [79, 79, 79, 79, 79, 79, 79, 79],
         [79, 79, 79, 79, 79, 79, 79, 79],
         [79, 79, 79, 79, 79, 79, 79, 79]],
    dtype=np.float32).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
