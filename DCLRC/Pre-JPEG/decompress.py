import argparse

from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
import numpy as np
import torchjpeg.codec
import torch.nn.functional as F
import torch
import os
from PIL import Image

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by using it along with a custom DCT routine to decompress a JPEG")
#parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("input_file", help="Input lossless image file")
parser.add_argument("dct_file", help="dct_file")
#parser.add_argument("output", help="Output image, should be lossless for best results")
args = parser.parse_args()

def numerical_sort(file_name):
    file_name = file_name.replace("_","")
    try:
        return int(file_name.split('.')[0])
    except ValueError:
        return float('inf')

input_folder = args.input_file
dct_folder = args.dct_file

input_files = os.listdir(input_folder)
input_files = sorted(input_files, key=numerical_sort)
#input_files = input_files.sort(reverse=True)
dct_files = os.listdir(dct_folder)
dct_files = sorted(dct_files, key=numerical_sort)

for fileName,fileName2 in zip(input_files,dct_files):
  dimensions, quantization, Y, CbCr = torchjpeg.codec.read_coefficients(args.input_file+"/"+fileName)
  coefficients = torch.load(args.dct_file+"/"+fileName2,map_location=torch.device('cpu'))
  print(dimensions.shape)
  print(quantization.shape)
  #print(coefficients.type())
  #print(coefficients)
  ioj = to_tensor(Image.open(args.input_file+"/"+fileName))
  #print(fileName)
  #print(fileName2)
  
  original_shape = Y.shape
  coefficients = coefficients.squeeze(0)
  Y_coefficients = coefficients[0:1]
  Y_coefficients = np.reshape(Y_coefficients, (original_shape[0], original_shape[1], original_shape[3], original_shape[2], original_shape[4]))
  Y_coefficients = np.transpose(Y_coefficients,(0,1,3,2,4))
  #print(Y_coefficients)
  
  Cb_shape = CbCr.shape
  #print(Cb_shape)
  
  Cb_coefficients = coefficients[1:2]
  #print(Cb_coefficients.shape)
  Cb_coefficients = Cb_coefficients.float()
  Cb_coefficients = Cb_coefficients.unsqueeze(0)
  Cb_coefficients = F.interpolate(Cb_coefficients, size=(int(Cb_coefficients.shape[2]/2),int(Cb_coefficients.shape[3]/2)), mode='nearest')
  Cb_coefficients = Cb_coefficients.squeeze(0).squeeze(0)
  Cb_coefficients = Cb_coefficients.to(torch.int16)
  #print(Cb_coefficients.shape)
  Cb_coefficients = np.reshape(Cb_coefficients, (original_shape[0], int(original_shape[1]/2), original_shape[3] , int(original_shape[2]/2), original_shape[4]))
  Cb_coefficients = np.transpose(Cb_coefficients,(0,1,3,2,4))
  #print(Cb_coefficients)
  #print(Cb_coefficients.shape)
  
  Cr_coefficients = coefficients[2:3]
  #print(Cr_coefficients.shape)
  Cr_coefficients = Cr_coefficients.float()
  Cr_coefficients = Cr_coefficients.unsqueeze(0)
  Cr_coefficients = F.interpolate(Cr_coefficients, size=(int(Cr_coefficients.shape[2]/2),int(Cr_coefficients.shape[3]/2)), mode='nearest')
  Cr_coefficients = Cr_coefficients.squeeze(0).squeeze(0)
  Cr_coefficients = Cr_coefficients.to(torch.int16)
  #print(Cr_coefficients.shape)
  Cr_coefficients = np.reshape(Cr_coefficients, (original_shape[0], int(original_shape[1]/2), original_shape[3], int(original_shape[2]/2), original_shape[4]))
  Cr_coefficients = np.transpose(Cr_coefficients,(0,1,3,2,4))
  #print(Cr_coefficients)
  #print(Cr_coefficients.shape)
  
  CbCr_coefficients = torch.cat((Cb_coefficients,Cr_coefficients),dim=0)
  #print(CbCr_coefficients)
  #print(CbCr)
  print(Y_coefficients.shape)
  print(CbCr_coefficients.shape)
  #spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

  spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)
  #print(spatial)
  to_pil_image(ioj).save("pt_out/" + fileName.replace(".jpg",".png"))
