import argparse

from torchvision.transforms.functional import to_pil_image
import os
import torchjpeg.codec
from DiffJPEG import DiffJPEG
from PIL import Image
from torchvision.transforms import transforms

path = "/home/whut1/wsp/torchjpeg-master/examples/sample.jpg"
dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(path)
spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

img_ori = Image.open(path)
img_ori = transforms.ToTensor()(img_ori)
img_ori = img_ori.unsqueeze(0)

Y_coefficients = Y_coefficients.view(Y_coefficients.shape[0],Y_coefficients.shape[1]*Y_coefficients.shape[2],Y_coefficients.shape[3],Y_coefficients.shape[4])

Cb_coefficients = CbCr_coefficients[0:1]
Cr_coefficients = CbCr_coefficients[1:2]

Cb_coefficients = Cb_coefficients.view(Cb_coefficients.shape[0],Cb_coefficients.shape[1]*Cb_coefficients.shape[2],Cb_coefficients.shape[3],Cb_coefficients.shape[4])
Cr_coefficients = Cr_coefficients.view(Cr_coefficients.shape[0],Cr_coefficients.shape[1]*Cr_coefficients.shape[2],Cr_coefficients.shape[3],Cr_coefficients.shape[4])

jpeg = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
out = jpeg(Y_coefficients,Cb_coefficients,Cr_coefficients)
print(out)
print(spatial)