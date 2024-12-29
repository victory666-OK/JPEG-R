import torch
import math
import numpy as np
import torch.nn.functional as F
import subprocess
import torchjpeg
from pytorch_msssim import ms_ssim
from models.balle2017 import entropy_model, gdn
from torch import nn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck
from torch import Tensor
from typing import Tuple
from typing import Optional
from models.DiffJPEG import DiffJPEG

lower_bound = entropy_model.lower_bound_fn.apply

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out

class JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super(JointContextTransfer, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//8, 
            head_count=2, value_channels=channels//4)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x_left, x_right):
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))
        A_right_to_left, A_left_to_right = self.attn(x_left, x_right), self.attn(x_right, x_left)
        compact_left = identity_left + self.refine(torch.cat((A_right_to_left, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((A_left_to_right, x_right), dim=1))
        return compact_left, compact_right

class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32, head_count=8, value_channels=64):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ input_

        return attention

class HyperPriorDistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(HyperPriorDistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.ha_primary_image = HyperAnalysisTransform(num_filters)
        self.hs_primary_image = HyperSynthesisTransform(num_filters)

        self.ha_cor_image = HyperAnalysisTransform(num_filters)
        self.hs_cor_image = HyperSynthesisTransform(num_filters)

        self.entropy_bottleneck_sigma_x = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_sigma_y = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_common_info = entropy_model.EntropyBottleneck(num_filters, quantize=False)

        self.conditional_entropy_bottleneck_hx = ConditionalEntropyBottleneck()
        self.conditional_entropy_bottleneck_hy = ConditionalEntropyBottleneck(cor_input=True)

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.bound = bound
        
        self.change = torch.tensor
        self.jpeg = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
        self.jpeg_cor = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_cor = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_y = nn.Conv2d(1, num_filters//3, 5, stride=2, padding=2)
        self.conv1_cbcr = nn.Conv2d(2, num_filters*2//3, 3, stride=1, padding=1)
        self.gdn1_ycbcr = gdn.GDN(num_filters)
        self.igdn1_ycbcr = gdn.GDN(num_filters,inverse=True)
        self.deconv1_y = nn.ConvTranspose2d(num_filters//3, 1, 5, stride=2, padding=2, output_padding=1)
        self.deconv1_cbcr = nn.ConvTranspose2d(num_filters*2//3, 2, 3, stride=1, padding=1)
        
        self.conv1_y_cor = nn.Conv2d(1, num_filters//3, 5, stride=2, padding=2)
        self.conv1_cbcr_cor = nn.Conv2d(2, num_filters*2//3, 3, stride=1, padding=1)
        self.gdn1_ycbcr_cor = gdn.GDN(num_filters)
        self.igdn1_ycbcr_cor = gdn.GDN(num_filters,inverse=True)
        self.deconv1_y_cor = nn.ConvTranspose2d(num_filters//3, 1, 5, stride=2, padding=2, output_padding=1)
        self.deconv1_cbcr_cor = nn.ConvTranspose2d(num_filters*2//3, 2, 3, stride=1, padding=1)

    def encode(self, x):
        #x = self.conv1(x)
        #x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        #x = self.conv1_cor(x)
        #x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        #x = self.conv1_w(x)
        #x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        #x = self.igdn3(x)
        #x = self.deconv4(x)

        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        #x = self.igdn3_cor(x)
        #x = self.deconv4_cor(x)

        return x
        
    def chroma_subsampling(self,image):
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
        y = image[:, 0:1, :, :]  # Take the first channel for Y
        
        # Subsample the Cb and Cr channels
        cb = self.pool(image[:, 1:2, :, :])  # Take the second channel for Cb
        cr = self.pool(image[:, 2:3, :, :])  # Take the third channel for Cr
        
        return y, cb, cr
    
    def chroma_subsampling_cor(self,image):
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
        y = image[:, 0:1, :, :]  # Take the first channel for Y
        
        # Subsample the Cb and Cr channels
        cb = self.pool_cor(image[:, 1:2, :, :])  # Take the second channel for Cb
        cr = self.pool_cor(image[:, 2:3, :, :])  # Take the third channel for Cr
        
        return y, cb, cr
        
    def block_splitting(self,image):
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
        image_reshaped = image.view(batch_size, channels, h_patches, k, -1, k)
        image_transposed = image_reshaped.permute(0, 1, 2, 4, 3, 5)
        patches = image_transposed.contiguous().view(batch_size, channels, num_patches, k, k)
        patches = patches.squeeze(0)
    
        return patches
    
    def cfm(self,y,cb,cr):
        y = self.conv1_y(y)
        cbcr = torch.concat((cb,cr),1)
        cbcr = self.conv1_cbcr(cbcr)
        x = torch.concat((y,cbcr),1)
        x = self.gdn1_ycbcr(x)
        return x

    def cfm_cor(self,y,cb,cr):
        y = self.conv1_y_cor(y)
        cbcr = torch.concat((cb,cr),1)
        cbcr = self.conv1_cbcr_cor(cbcr)
        x = torch.concat((y,cbcr),1)
        x = self.gdn1_ycbcr_cor(x)
        return x

    def cpsm(self,x):
        x = self.igdn1_ycbcr(x)
        y,cb,cr = torch.chunk(x, 3, dim=1)
        cbcr = torch.concat((cb,cr),1)
        y = self.deconv1_y(y)
        cbcr = self.deconv1_cbcr(cbcr)
        cb,cr = torch.chunk(cbcr, 2, dim=1)
        return y,cb,cr

    def cpsm_cor(self,x):
        x = self.igdn1_ycbcr_cor(x)
        y,cb,cr = torch.chunk(x, 3, dim=1)
        cbcr = torch.concat((cb,cr),1)
        y = self.deconv1_y_cor(y)
        cbcr = self.deconv1_cbcr_cor(cbcr)
        cb,cr = torch.chunk(cbcr, 2, dim=1)
        return y,cb,cr

    def forward(self,Y,Cb,Cr,Y_cor,Cb_cor,Cr_cor,img_ori,img_side_ori,quantization,dimensions):
        quantization = quantization.cuda()
        x = self.cfm(Y,Cb,Cr)
        y = self.cfm_cor(Y_cor,Cb_cor,Cr_cor)
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        z = self.ha_primary_image(abs(hx))
        z_tilde, z_likelihoods = self.entropy_bottleneck_sigma_x(z)
        sigma = self.hs_primary_image(z_tilde)
        sigma_lower_bounded = lower_bound(sigma, self.bound)
        
        z_cor = self.ha_cor_image(abs(hy))
        z_tilde_cor, z_likelihoods_cor = self.entropy_bottleneck_sigma_y(
            z_cor)
        sigma_cor = self.hs_cor_image(z_tilde_cor)
        sigma_cor_lower_bounded = lower_bound(sigma_cor, self.bound)

        hx_tilde, x_likelihoods = self.conditional_entropy_bottleneck_hx(hx, sigma_lower_bounded)
        hy_tilde, y_likelihoods = self.conditional_entropy_bottleneck_hy(hy, sigma_cor_lower_bounded)

        w = self.encode_w(y)  # p(w|y), i.e. the "common variable"
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding a small Gaussian noise improves stability of the training
        w_tilde, w_likelihoods = self.entropy_bottleneck_common_info(w)

        x_tilde = self.decode(hx_tilde, w_tilde)
        y_tilde = self.decode_cor(hy_tilde, w_tilde)
        
        y, cb, cr = self.cpsm(x_tilde)
        y_cor,cb_cor,cr_cor = self.cpsm_cor(y_tilde)
        
        y_comp = self.block_splitting(y)
        cb_comp = self.block_splitting(cb)
        cr_comp = self.block_splitting(cr)
        y_cor_comp = self.block_splitting(y_cor)
        cb_cor_comp = self.block_splitting(cb_cor)
        cr_cor_comp = self.block_splitting(cr_cor)
        
        x_out = self.jpeg(y_comp,cb_comp,cr_comp,quantization)
        y_out = self.jpeg_cor(y_cor_comp,cb_cor_comp,cr_cor_comp,quantization)
        
        x_out = x_out.cuda()
        y_out = y_out.cuda()

        return x_out, y_out, x_likelihoods, y_likelihoods, z_likelihoods, \
               z_likelihoods_cor, w_likelihoods

class DistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11,decode_atten=JointContextTransfer):
        super(DistributedAutoEncoder, self).__init__()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters)

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)

        self.bound = bound
        
        self.change = torch.tensor
        self.jpeg = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
        self.jpeg_cor = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_cor = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_y = nn.Conv2d(1, num_filters//3, 5, stride=2, padding=2)
        self.conv1_cbcr = nn.Conv2d(2, num_filters*2//3, 3, stride=1, padding=1)
        self.gdn1_ycbcr = gdn.GDN(num_filters)
        self.igdn1_ycbcr = gdn.GDN(num_filters,inverse=True)
        self.deconv1_y = nn.ConvTranspose2d(num_filters//3, 1, 5, stride=2, padding=2, output_padding=1)
        self.deconv1_cbcr = nn.ConvTranspose2d(num_filters*2//3, 2, 3, stride=1, padding=1)
        
        self.conv1_y_cor = nn.Conv2d(1, num_filters//3, 5, stride=2, padding=2)
        self.conv1_cbcr_cor = nn.Conv2d(2, num_filters*2//3, 3, stride=1, padding=1)
        self.gdn1_ycbcr_cor = gdn.GDN(num_filters)
        self.igdn1_ycbcr_cor = gdn.GDN(num_filters,inverse=True)
        self.deconv1_y_cor = nn.ConvTranspose2d(num_filters//3, 1, 5, stride=2, padding=2, output_padding=1)
        self.deconv1_cbcr_cor = nn.ConvTranspose2d(num_filters*2//3, 2, 3, stride=1, padding=1)

        self.atten_1 = decode_atten(num_filters)
        self.atten_2 = decode_atten(num_filters)
        
    def encode(self, x):
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decoder_1(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        return x
        
    def decoder_2(self, x):
        x = self.deconv3(x)
        x = self.igdn1_ycbcr(x)
        return x

    def decoder_cor_1(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        return x
        
    def decoder_cor_2(self, x):
        x = self.deconv3_cor(x)
        return x
        
    def chroma_subsampling(self,image):
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
        y = image[:, 0:1, :, :]  # Take the first channel for Y
        
        # Subsample the Cb and Cr channels
        cb = self.pool(image[:, 1:2, :, :])  # Take the second channel for Cb
        cr = self.pool(image[:, 2:3, :, :])  # Take the third channel for Cr
        
        return y, cb, cr
    
    def chroma_subsampling_cor(self,image):
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
        y = image[:, 0:1, :, :]  # Take the first channel for Y
        
        # Subsample the Cb and Cr channels
        cb = self.pool_cor(image[:, 1:2, :, :])  # Take the second channel for Cb
        cr = self.pool_cor(image[:, 2:3, :, :])  # Take the third channel for Cr
        
        return y, cb, cr
        
    def block_splitting(self,image):
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
        image_reshaped = image.view(batch_size, channels, h_patches, k, -1, k)
        image_transposed = image_reshaped.permute(0, 1, 2, 4, 3, 5)
        patches = image_transposed.contiguous().view(batch_size, channels, num_patches, k, k)
        patches = patches.squeeze(0)
    
        return patches
    
    def cfm(self,y,cb,cr):
        y = self.conv1_y(y)
        cbcr = torch.concat((cb,cr),1)
        cbcr = self.conv1_cbcr(cbcr)
        x = torch.concat((y,cbcr),1)
        x = self.gdn1_ycbcr(x)
        return x

    def cfm_cor(self,y,cb,cr):
        y = self.conv1_y_cor(y)
        cbcr = torch.concat((cb,cr),1)
        cbcr = self.conv1_cbcr_cor(cbcr)
        x = torch.concat((y,cbcr),1)
        x = self.gdn1_ycbcr_cor(x)
        return x

    def cpsm(self,x):
        y,cb,cr = torch.chunk(x, 3, dim=1)
        cbcr = torch.concat((cb,cr),1)
        y = self.deconv1_y(y)
        cbcr = self.deconv1_cbcr(cbcr)
        cb,cr = torch.chunk(cbcr, 2, dim=1)
        return y,cb,cr

    def cpsm_cor(self,x):
        x = self.igdn1_ycbcr_cor(x)
        y,cb,cr = torch.chunk(x, 3, dim=1)
        cbcr = torch.concat((cb,cr),1)
        y = self.deconv1_y_cor(y)
        cbcr = self.deconv1_cbcr_cor(cbcr)
        cb,cr = torch.chunk(cbcr, 2, dim=1)
        return y,cb,cr

    def forward(self,Y,Cb,Cr,Y_cor,Cb_cor,Cr_cor,img_ori,img_side_ori,quantization,dimensions):
        quantization = quantization.cuda()
        x = self.cfm(Y,Cb,Cr)
        y = self.cfm(Y_cor,Cb_cor,Cr_cor)

        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        hy_tilde, y_likelihoods = self.entropy_bottleneck_hx(hy)
        
        hx_tilde,hy_tilde= self.atten_1(hx_tilde,hy_tilde)
        y_left,y_right = self.atten_2(self.decoder_1(hx_tilde), self.decoder_1(hy_tilde))
        x_tilde, y_tilde = self.decoder_2(y_left), self.decoder_2(y_right)

        y, cb, cr = self.cpsm(x_tilde)
        y_cor,cb_cor,cr_cor = self.cpsm(y_tilde)
        
        y_comp = self.block_splitting(y)
        cb_comp = self.block_splitting(cb)
        cr_comp = self.block_splitting(cr)
        y_cor_comp = self.block_splitting(y_cor)
        cb_cor_comp = self.block_splitting(cb_cor)
        cr_cor_comp = self.block_splitting(cr_cor)
        
        x_out = self.jpeg(y_comp,cb_comp,cr_comp,quantization)
        y_out = self.jpeg(y_cor_comp,cb_cor_comp,cr_cor_comp,quantization)
        
        x_out = x_out.cuda()
        y_out = y_out.cuda()
        
        return x_out, y_out, x_likelihoods, y_likelihoods,x_tilde,y_tilde 

if __name__ == '__main__':
    net = DistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)