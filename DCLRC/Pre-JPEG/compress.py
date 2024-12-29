# -*- coding: utf-8 -*-
import argparse

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import torchvision

import torchjpeg.codec
import torch
import os

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def crop_tensor_to_divisible(tensor, divisor):
    _, height, width = tensor.size()
    new_height = height - (height % divisor)
    new_width = width - (width % divisor)
    start_h = (height - new_height) // 2
    start_w = (width - new_width) // 2
    end_h = start_h + new_height
    end_w = start_w + new_width
    cropped_tensor = tensor[:, start_h:end_h, start_w:end_w]
    return cropped_tensor

mse = torch.nn.MSELoss(reduction='mean')
    
parser = argparse.ArgumentParser("Tests the pytorch DCT loader by reading and image, quantizing its pixels, and writing the DCT coefficients to a JPEG")
parser.add_argument("input_file", help="Input lossless image file")
#parser.add_argument("output_file", help="output lossless image file")
parser.add_argument("quality", type=int, help="Output quality on the 0-100 scale")
#parser.add_argument("color_samp_factor_vertical", type=int, nargs="?", default=2, help="Vertical chroma subsampling factor. Defaults to 2.")
#parser.add_argument("color_samp_factor_horizontal", type=int, nargs="?", default=2, help="Horizontal chroma subsampling factor. Defaults to 2.")
args = parser.parse_args()

for fileName in os.listdir(args.input_file):
    path = args.input_file + "//" + fileName
    #im = to_tensor(Image.open(path))
    
    
    im = Image.open(path)
    im = im.convert('RGB')
    im = np.array(im)
    im = to_tensor(im)
    
    im = crop_tensor_to_divisible(im, 16)
    #im = TF.center_crop(im, (370, 740))
    #im = TF.resize(im, (128,256))

    if im.shape[0] > 3:
        im = im[:3]
            
    dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im, args.quality)
    original_shape = Y_coefficients.shape
    original_shape_CbCr = CbCr_coefficients.shape
    Cb_coefficients = CbCr_coefficients[0:1]
    Cr_coefficients = CbCr_coefficients[1:2]
    original_shape_Cb = Cb_coefficients.shape
    original_shape_Cr = Cr_coefficients.shape
            
    tensor = np.transpose(Y_coefficients, (0, 1, 3, 2, 4))
    tensor_Cb = np.transpose(Cb_coefficients, (0, 1, 3, 2, 4))
    tensor_Cr = np.transpose(Cr_coefficients, (0, 1, 3, 2, 4))
    new_tensor = np.reshape(tensor, (original_shape[0], original_shape[1] * original_shape[3], original_shape[2] * original_shape[4]))
    new_tensor_Cb = np.reshape(tensor_Cb, (original_shape_Cb[0], original_shape_Cb[1] * original_shape_Cb[3], original_shape_Cb[2] * original_shape_Cb[4]))
    new_tensor_Cr = np.reshape(tensor_Cr, (original_shape_Cr[0], original_shape_Cr[1] * original_shape_Cr[3], original_shape_Cr[2] * original_shape_Cr[4]))
    torch.set_printoptions(threshold=10_00000000000)
    tensor = torch.abs(new_tensor)
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_tensor = normalized_tensor.float()

    scale_factor = 20  
    scaled_tensor = torch.where(normalized_tensor > 0, normalized_tensor * scale_factor, normalized_tensor)
    scaled_tensor = torch.clamp(scaled_tensor, 0, 1)
    torchvision.utils.save_image(scaled_tensor, 'scaled_image.png')
    



