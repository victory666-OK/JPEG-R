import argparse

from torchvision.transforms.functional import to_pil_image
import numpy as np
import torchjpeg.codec
import torch.nn.functional as F
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from DiffJPEG import DiffJPEG
from torch import nn
from pytorch_msssim import ms_ssim

def chroma_subsampling(image):
        """ Chroma subsampling on CbCr channels
        Input:
            image(tensor): batch x channels x height x width
        Output:
            y(tensor): batch x channels x height x width
            cb(tensor): batch x channels x height/2 x width/2
            cr(tensor): batch x channels x height/2 x width/2
        """
        # Assuming image is already in the shape of (batch, channels, height, width)
        
        # Luma (Y) channel doesn't change in size
        #y = image[:, 0:1, :, :]  # Take the first channel for Y
        # Subsample the Cb and Cr channels
        #cb = self.pool(image[:, 1:2, :, :])  # Take the second channel for Cb
        #cr = self.pool(image[:, 2:3, :, :])  # Take the third channel for Cr
        
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        y_ori, cb_ori, cr_ori = torch.chunk(image, 3, dim=1)
        y = y_ori
        cb = pool(cb_ori.float())
        cr = pool(cr_ori.float())
        return y, cb, cr
        
def block_splitting(image):
        """ Splitting image into 8x8 patches
        Input:
            image(tensor): batch x channels x height x width
        Output: 
            patches(tensor): batch x channels x num_patches x patch_height x patch_width
        """
        k = 8  # The size of the block (patch)
        batch_size, channels, height, width = image.shape
        # Calculate the number of patches along height and width
        h_patches = height // k
        w_patches = width // k
        num_patches = h_patches * w_patches
    
        # Ensure the image dimensions are divisible by the block size
        if height % k != 0 or width % k != 0:
            raise ValueError("Image dimensions must be divisible by the block size")
    
        # Reshape and permute to get the patches in the desired output shape
        indices_list = [0,1,5,6,14,15,27,28,2,4,7,13,16,26,29,42,3,8,12,17,25,30,41,43,9,11,18,24,31,40,44,53,10,19,23,32,39,45,52,54,20,22,33,38,46,51,55,60,21,34,37,47,50,56,59,61,35,36,48,49,57,58,62,63]
        inverse_indices_list = [0] * 64
        for new_index, original_index in enumerate(indices_list):
             inverse_indices_list[original_index] = new_index
        indices = torch.tensor(indices_list)
        image = torch.index_select(image, 1, indices)
        image_transposed = image.permute(0, 2, 3, 1)
        patches = image_transposed.contiguous().view(batch_size, height*width, k, k)
        #patches = patches.squeeze(0)
    
        return patches
        
indices_list = [0,1,5,6,14,15,27,28,
                2,4,7,13,16,26,29,42,
                3,8,12,17,25,30,41,43,
                9,11,18,24,31,40,44,53,
                10,19,23,32,39,45,52,54,
                20,22,33,38,46,51,55,60,
                21,34,37,47,50,56,59,61,
                35,36,48,49,57,58,62,63]
inverse_indices_list = [0] * 64
for new_index, original_index in enumerate(indices_list):
    inverse_indices_list[original_index] = new_index

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by using it along with a custom DCT routine to decompress a JPEG")
#parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("input_file", help="Input lossless image file")
#parser.add_argument("output", help="Output image, should be lossless for best results")
args = parser.parse_args()

for fileName in os.listdir(args.input_file):
  torch.set_printoptions(threshold=100_000)
  dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input_file+"/"+fileName)
  original_shape = Y_coefficients.shape
  original_shape_CbCr = CbCr_coefficients.shape
  Cb_coefficients = CbCr_coefficients[0:1]
  Cr_coefficients = CbCr_coefficients[1:2]
  original_shape_Cb = Cb_coefficients.shape
  original_shape_Cr = Cr_coefficients.shape
            
  tensor = np.reshape(Y_coefficients, (original_shape[0], original_shape[1], original_shape[2],original_shape[3]*original_shape[4]))
  tensor_Cb = np.reshape(Cb_coefficients, (original_shape_Cb[0], original_shape_Cb[1], original_shape_Cb[2],original_shape_Cb[3]*original_shape_Cb[4]))
  tensor_Cr = np.reshape(Cr_coefficients, (original_shape_Cr[0], original_shape_Cr[1], original_shape_Cr[2],original_shape_Cr[3]*original_shape_Cr[4]))
  tensor = np.transpose(tensor, (0, 3, 1, 2))
  tensor_Cb = np.transpose(tensor_Cb, (0, 3, 1, 2))
  tensor_Cr = np.transpose(tensor_Cr, (0, 3, 1, 2))
  
  indices = torch.tensor(indices_list)
  inverse_indices_tensor = torch.tensor(inverse_indices_list)
  tensor = torch.index_select(tensor, 1, inverse_indices_tensor)
  tensor_Cb = torch.index_select(tensor_Cb, 1, inverse_indices_tensor)
  tensor_Cr = torch.index_select(tensor_Cr, 1, inverse_indices_tensor)
  
  tensor,_ = torch.chunk(tensor, 2, dim=1)
  tensor_Cb, _ = torch.chunk(tensor_Cb, 2, dim=1)
  tensor_Cr, _ = torch.chunk(tensor_Cr, 2, dim=1)
  
  new_tensor = np.reshape(tensor, (tensor.shape[1] , tensor.shape[2],tensor.shape[3]))
  new_tensor_Cb = np.reshape(tensor_Cb, (tensor_Cb.shape[1] , tensor_Cb.shape[2],tensor_Cb.shape[3]))
  new_tensor_Cr = np.reshape(tensor_Cr, (tensor_Cr.shape[1] , tensor_Cr.shape[2],tensor_Cr.shape[3]))
            
  img = new_tensor.cpu().clone()
  img = img.squeeze(0)
            
  img_Cb = new_tensor_Cb.cpu().clone()
  img_Cb = img_Cb.squeeze(0)
  img_Cb = img_Cb.float()
  target_size = (img_Cb.shape[0],img_Cb.shape[1]*2,img_Cb.shape[2]*2)
  img_Cb = img_Cb.unsqueeze(0).unsqueeze(0)
  img_Cb = F.interpolate(img_Cb, size=target_size, mode='nearest')
  img_Cb = img_Cb.squeeze().squeeze()
  img_Cb = img_Cb.to(torch.int16)
            
  img_Cr = new_tensor_Cr.cpu().clone()
  img_Cr = img_Cr.squeeze(0)
  img_Cr = img_Cr.float()
  target_size = (img_Cr.shape[0],img_Cr.shape[1]*2,img_Cr.shape[2]*2)
  img_Cr = img_Cr.unsqueeze(0).unsqueeze(0)
  img_Cr = F.interpolate(img_Cr, size=target_size, mode='nearest')
  img_Cr = img_Cr.squeeze().squeeze()
  img_Cr = img_Cr.to(torch.int16)
            
  out_img = torch.cat((img, img_Cb, img_Cr), dim=0)
  #print(img)
  '''
  toPIL = transforms.ToPILImage()
  img_PIL = toPIL(out_img.float())
  img_PIL.save('random.jpg')
  '''
  out_img = out_img.unsqueeze(0)
  
  y,cb,cr = chroma_subsampling(out_img)
  
  y_comp = block_splitting(y)
  cb_comp = block_splitting(cb)
  cr_comp = block_splitting(cr)
  
  jpeg = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
  cor_img_ori = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)
  x_out = jpeg(y_comp,cb_comp,cr_comp,quantization)
  cor_img_ori = cor_img_ori.unsqueeze(0)
  msssim = ms_ssim(x_out.clone().cpu(), cor_img_ori.clone().cpu(), data_range=1.0, size_average=True,win_size=7)
  print(msssim)

  ''' 
  original_shape = Y_coefficients.shape
  original_shape_CbCr = CbCr_coefficients.shape
  Cb_coefficients = CbCr_coefficients[0:1]
  Cr_coefficients = CbCr_coefficients[1:2]
  original_shape_Cb = Cb_coefficients.shape
  original_shape_Cr = Cr_coefficients.shape
  
  tensor = np.transpose(Y_coefficients, (0, 3, 1, 4, 2))
  tensor_Cb = np.transpose(Cb_coefficients, (0, 3, 1, 4, 2))
  tensor_Cr = np.transpose(Cr_coefficients, (0, 3, 1, 4, 2))
  new_tensor = np.reshape(tensor, (original_shape[0], original_shape[1] * original_shape[3], original_shape[2] * original_shape[4]))
  new_tensor_Cb = np.reshape(tensor_Cb, (original_shape_Cb[0], original_shape_Cb[1] * original_shape_Cb[3], original_shape_Cb[2] * original_shape_Cb[4]))
  new_tensor_Cr = np.reshape(tensor_Cr, (original_shape_Cr[0], original_shape_Cr[1] * original_shape_Cr[3], original_shape_Cr[2] * original_shape_Cr[4]))
  
  img = new_tensor.cpu().clone()
  img = img.squeeze(0)
  
  img_Cb = new_tensor_Cb.cpu().clone()
  img_Cb = img_Cb.squeeze(0)
  img_Cb = img_Cb.float()
  target_size = (img_Cb.shape[0]*2,img_Cb.shape[1]*2)
  img_Cb = img_Cb.unsqueeze(0).unsqueeze(0)
  img_Cb = F.interpolate(img_Cb, size=target_size, mode='nearest')
  img_Cb = img_Cb.squeeze().squeeze()
  img_Cb = img_Cb.to(torch.int16)
  
  img_Cr = new_tensor_Cr.cpu().clone()
  img_Cr = img_Cr.squeeze(0)
  img_Cr = img_Cr.float()
  target_size = (img_Cr.shape[0]*2,img_Cr.shape[1]*2)
  img_Cr = img_Cr.unsqueeze(0).unsqueeze(0)
  img_Cr = F.interpolate(img_Cr, size=target_size, mode='nearest')
  img_Cr = img_Cr.squeeze().squeeze()
  img_Cr = img_Cr.to(torch.int16)
  
  img = img.unsqueeze(0)
  img_Cb = img_Cb.unsqueeze(0)
  img_Cr = img_Cr.unsqueeze(0)
  out_img = torch.cat((img, img_Cb, img_Cr), dim=0)
  torch.set_printoptions(threshold=50_000)
  
  #print(Y_coefficients)
  

  #torch.save(image, "stereoDCT2/" + fileName.replace(".jpg","") + ".png")
  
#print(out_img)

#spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

#to_pil_image(spatial).save(args.output)
'''