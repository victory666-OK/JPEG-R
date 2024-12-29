import argparse

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

import torchjpeg.codec
import torch
import os

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    
parser = argparse.ArgumentParser("Tests the pytorch DCT loader by reading and image, quantizing its pixels, and writing the DCT coefficients to a JPEG")
#parser.add_argument("input", help="Input image, should be lossless")
parser.add_argument("input_file", help="Input lossless image file")
#parser.add_argument("output", help="Output image, must be a JPEG")
parser.add_argument("output_file", help="output lossless image file")
#parser.add_argument("quality", type=int, help="Output quality on the 0-100 scale")
parser.add_argument("color_samp_factor_vertical", type=int, nargs="?", default=2, help="Vertical chroma subsampling factor. Defaults to 2.")
parser.add_argument("color_samp_factor_horizontal", type=int, nargs="?", default=2, help="Horizontal chroma subsampling factor. Defaults to 2.")
args = parser.parse_args()

for fileName in os.listdir(args.input_file):
    img1 = Image.open(args.input_file + "/" + fileName)
    img2 = Image.open(args.output_file + "/" + fileName.replace("png","jpg"))
    
    im1 = to_tensor(img1)
    im2 = to_tensor(img2)
    
    print(im1)
    print(im2)
    




