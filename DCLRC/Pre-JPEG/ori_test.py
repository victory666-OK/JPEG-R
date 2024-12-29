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
    img = Image.open(args.input_file + "//" + fileName)
    img = TF.center_crop(img, (370, 740))
    img = TF.resize(img, (128,256))
    
    im = to_tensor(img)
    
    _,_,width = im.size()
    
    if im.shape[0] > 3:
        im = im[:3]

    #compress ori_image
    
    toPIL = transforms.ToPILImage()
    img = im.cpu().clone()
    img = img.squeeze(0)
    pic = toPIL(img)
    pic.save(args.output_file + "//" + fileName)
    




