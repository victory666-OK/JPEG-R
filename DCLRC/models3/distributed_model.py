import torch
import math
import numpy as np #多的
import torch.nn.functional as F # 多的
import subprocess # 多的
import torchjpeg # 多的
from pytorch_msssim import ms_ssim # 多的
from models.balle2017 import entropy_model, gdn
from torch import nn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck

lower_bound = entropy_model.lower_bound_fn.apply


'''
The following model is based on the balle2018 model, which uses scale hyperpriors (z). 
'''


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

        self.entropy_bottleneck_sigma_x = entropy_model.EntropyBottleneck(num_filters, quantize=False)
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

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
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
        x = self.igdn3(x)
        x = self.deconv4(x)

        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)

        return x
        
    #后加开始
    def idct_2d(self,block, norm='ortho'):
        def idct_1d(tensor, norm='ortho'):
            x = torch.fft.irfft(torch.fft.rfft(tensor, n=8, dim=-1, norm=norm), n=8, dim=-1, norm=norm)
            return x
        return idct_1d(idct_1d(block, norm).transpose(-2, -1), norm).transpose(-2, -1)
        
    def idct(self,tensor):
        blocks = tensor.view(tensor.shape[0], tensor.shape[1], tensor.shape[2] // 8, 8, tensor.shape[3] // 8, 8)
        # 应用 2D IDCT 到每个块
        idct_blocks = torch.zeros_like(blocks)
        for i in range(blocks.shape[2]):
            for j in range(blocks.shape[4]):
                block = blocks[:, :, i, :, j, :]
                idct_blocks[:, :, i, :, j, :] = self.idct_2d(block)
        result_tensor = idct_blocks.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
        return result_tensor
        
    def ycbcr_to_rgb_jpeg(self,image):
        """ 
        Converts YCbCr image to RGB JPEG
        Input:
            image(tensor): batch x 3 x height x width
        Output:
            result(tensor): batch x 3 x height x width
        """
        image = image.permute(0, 2, 3, 1).cuda()  # Change from Bx3xHxW to BxHxWx3 #加了.cuda()
        matrix = torch.tensor(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=torch.float32).T  # Convert numpy array to tensor
        matrix = matrix.cuda()
        shift = torch.tensor([0, -128, -128], dtype=torch.float32).cuda()  # Convert list to tensor
    
        result = torch.tensordot(image + shift, matrix, dims=1)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)  # Change back to Bx3xHxW
    #后加结束

    def forward(self, x, y):
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        z = self.ha_primary_image(abs(hx))
        z_tilde, z_likelihoods = self.entropy_bottleneck_sigma_x(z) #z_tilde是目标图像量化后的存在， z_likelihoods是目标图片量化中的概率分布
        sigma = self.hs_primary_image(z_tilde) #主图像熵模型
        sigma_lower_bounded = lower_bound(sigma, self.bound)
        
        z_cor = self.ha_cor_image(abs(hy))
        z_tilde_cor, z_likelihoods_cor = self.entropy_bottleneck_sigma_y(z_cor)
        sigma_cor = self.hs_cor_image(z_tilde_cor)
        sigma_cor_lower_bounded = lower_bound(sigma_cor, self.bound)

        hx_tilde, x_likelihoods = self.conditional_entropy_bottleneck_hx(hx, sigma_lower_bounded) #hx_tilde是主图熵编码
        hy_tilde, y_likelihoods = self.conditional_entropy_bottleneck_hy(hy, sigma_cor_lower_bounded)

        w = self.encode_w(y)  # p(w|y), i.e. the "common variable" #对关联图像 y 进一步编码，提取出关联图像和主图像共享的“公共变量” w，它代表与 y 相关的全局或共享信息
        if self.training: #如果模型处于训练状态，则对相关信息添加噪声
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding a small Gaussian noise improves stability of the training
        _, w_likelihoods = self.entropy_bottleneck_common_info(w)

        x_tilde = self.decode(hx_tilde, w) 
        y_tilde = self.decode_cor(hy_tilde, w)
        
        #通道划分
        Y_x_tilde  = x_tilde[:, 0:1, :, :]
        Cb_x_tilde = x_tilde[:, 1:2, :, :]
        Cr_x_tilde = x_tilde[:, 2:3, :, :]
        Y_y_tilde  = y_tilde[:, 0:1, :, :]
        Cb_y_tilde = y_tilde[:, 1:2, :, :]
        Cr_y_tilde = y_tilde[:, 2:3, :, :]
        
        
        #下采样
        target_size = (int(Cb_x_tilde.shape[2]/2),int(Cb_x_tilde.shape[3]/2))
        Cb_x_tilde = F.interpolate(Cb_x_tilde, size=target_size, mode='nearest')
        Cr_x_tilde = F.interpolate(Cr_x_tilde, size=target_size, mode='nearest')
        Cb_y_tilde = F.interpolate(Cb_y_tilde, size=target_size, mode='nearest')
        Cr_y_tilde = F.interpolate(Cr_y_tilde, size=target_size, mode='nearest')
        
        #CbCr_x_coefficients = torch.cat((Cb_x_tilde, Cr_x_tilde), dim=0) #后加
        #CbCr_y_coefficients = torch.cat((Cb_y_tilde, Cr_y_tilde), dim=0) #后加
        
        #x_out = torchjpeg.codec.reconstruct_full_image(Y_x_tilde, quantization, CbCr_x_coefficients, dimensions) #后加
        #y_out = torchjpeg.codec.reconstruct_full_image(Y_y_tilde, quantization, CbCr_y_coefficients, dimensions) #后加
        #逆离散余弦变换
        Y_x_idct  = self.idct(Y_x_tilde)
        Cb_x_idct = self.idct(Cb_x_tilde)
        Cr_x_idct = self.idct(Cr_x_tilde)
        Y_y_idct  = self.idct(Y_y_tilde)
        Cb_y_idct = self.idct(Cb_y_tilde)
        Cr_y_idct = self.idct(Cr_y_tilde)
        
        #色度上采样
        target_size_2 = (Cb_x_idct.shape[2]*2,Cb_x_idct.shape[3]*2)
        Cb_x_idct = F.interpolate(Cb_x_idct, size=target_size_2, mode='nearest')
        Cr_x_idct = F.interpolate(Cr_x_idct, size=target_size_2, mode='nearest')
        Cb_y_idct = F.interpolate(Cb_y_idct, size=target_size_2, mode='nearest')
        Cr_y_idct = F.interpolate(Cr_y_idct, size=target_size_2, mode='nearest')
        
        #CbCr_x_coefficients = torch.cat((Cb_x_idct, Cr_x_idct), dim=0) #后加
        #CbCr_y_coefficients = torch.cat((Cb_y_idct, Cr_y_idct), dim=0) #后加
        
        
        
        #通道拼接
        x_YCbCr = torch.cat((Y_x_idct, Cb_x_idct, Cr_x_idct), dim=1)
        y_YCbCr = torch.cat((Y_y_idct, Cb_y_idct, Cr_y_idct), dim=1)
        
        #YCbCrZ转rgb
        x_out = self.ycbcr_to_rgb_jpeg(x_YCbCr)
        y_out = self.ycbcr_to_rgb_jpeg(y_YCbCr)
        
        x_out = torch.min(255 * torch.ones_like(x_out), torch.max(torch.zeros_like(x_out), x_out)) / 255
        y_out = torch.min(255 * torch.ones_like(y_out), torch.max(torch.zeros_like(y_out), y_out)) / 255
        
        
        #x_left_hat = self.dct_to_img(x_left_hat,quantization,dimensions)
        #x_right_hat = self.dct_to_img(x_right_hat,quantization,dimensions)
        
        
        x_out = x_out.cuda()
        y_out = y_out.cuda()

        '''return x_tilde, y_tilde, x_likelihoods, y_likelihoods, z_likelihoods, \
               z_likelihoods_cor, w_likelihoods'''
        return {
            "x_hat": [x_out, y_out],
            "likelihoods": [{"y": x_likelihoods, "z": z_likelihoods}, {"y":y_likelihoods, "z":z_likelihoods_cor}],
            "likelihoods_w": [{"w": w_likelihoods}], 
            #"feature": [y_left_ste, y_right_ste, z_left_hat, z_right_hat, left_means_hat, right_means_hat],
        }


'''
This model is based on balle2017 model.
'''


class DistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(DistributedAutoEncoder, self).__init__()
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

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)

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

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
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
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x

    def idct_2d(self,block, norm='ortho'):
        def idct_1d(tensor, norm='ortho'):
            x = torch.fft.irfft(torch.fft.rfft(tensor, n=8, dim=-1, norm=norm), n=8, dim=-1, norm=norm)
            return x
        return idct_1d(idct_1d(block, norm).transpose(-2, -1), norm).transpose(-2, -1)
        
    def idct(self,tensor):
        blocks = tensor.view(tensor.shape[0], tensor.shape[1], tensor.shape[2] // 8, 8, tensor.shape[3] // 8, 8)
        # 应用 2D IDCT 到每个块
        idct_blocks = torch.zeros_like(blocks)
        for i in range(blocks.shape[2]):
            for j in range(blocks.shape[4]):
                block = blocks[:, :, i, :, j, :]
                idct_blocks[:, :, i, :, j, :] = self.idct_2d(block)
        result_tensor = idct_blocks.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
        return result_tensor
  
    def ycbcr_to_rgb_jpeg(self,image):
        """ 
        Converts YCbCr image to RGB JPEG
        Input:
            image(tensor): batch x 3 x height x width
        Output:
            result(tensor): batch x 3 x height x width
        """
        image = image.permute(0, 2, 3, 1)  # Change from Bx3xHxW to BxHxWx3
        matrix = torch.tensor(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=torch.float32).T  # Convert numpy array to tensor
        matrix = matrix.cuda()
        shift = torch.tensor([0, -128, -128], dtype=torch.float32).cuda()  # Convert list to tensor
    
        result = torch.tensordot(image + shift, matrix, dims=1)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)  # Change back to Bx3xHxW
    
    def forward(self, x, y,img_ori,img_side_ori):
        w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding small Gaussian noise improves the stability of training
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        hy_tilde, y_likelihoods = self.entropy_bottleneck_hy(hy)
        _, w_likelihoods = self.entropy_bottleneck(w)

        x_tilde = self.decode(hx_tilde, w)
        y_tilde = self.decode_cor(hy_tilde, w)
        
        #通道划分
        Y_x_tilde  = x_tilde[:, 0:1, :, :]
        Cb_x_tilde = x_tilde[:, 1:2, :, :]
        Cr_x_tilde = x_tilde[:, 2:3, :, :]
        Y_y_tilde  = y_tilde[:, 0:1, :, :]
        Cb_y_tilde = y_tilde[:, 1:2, :, :]
        Cr_y_tilde = y_tilde[:, 2:3, :, :]
        
        #下采样
        target_size = (int(Cb_x_tilde.shape[2]/2),int(Cb_x_tilde.shape[3]/2))
        Cb_x_tilde = F.interpolate(Cb_x_tilde, size=target_size, mode='nearest')
        Cr_x_tilde = F.interpolate(Cr_x_tilde, size=target_size, mode='nearest')
        Cb_y_tilde = F.interpolate(Cb_y_tilde, size=target_size, mode='nearest')
        Cr_y_tilde = F.interpolate(Cr_y_tilde, size=target_size, mode='nearest')
        
        #逆离散余弦变换
        Y_x_idct  = self.idct(Y_x_tilde)
        Cb_x_idct = self.idct(Cb_x_tilde)
        Cr_x_idct = self.idct(Cr_x_tilde)
        Y_y_idct  = self.idct(Y_y_tilde)
        Cb_y_idct = self.idct(Cb_y_tilde)
        Cr_y_idct = self.idct(Cr_y_tilde)
        
        #色度上采样
        target_size_2 = (Cb_x_idct.shape[2]*2,Cb_x_idct.shape[3]*2)
        Cb_x_idct = F.interpolate(Cb_x_idct, size=target_size_2, mode='nearest')
        Cr_x_idct = F.interpolate(Cr_x_idct, size=target_size_2, mode='nearest')
        Cb_y_idct = F.interpolate(Cb_y_idct, size=target_size_2, mode='nearest')
        Cr_y_idct = F.interpolate(Cr_y_idct, size=target_size_2, mode='nearest')
        
        #通道拼接
        x_YCbCr = torch.cat((Y_x_idct, Cb_x_idct, Cr_x_idct), dim=1)
        y_YCbCr = torch.cat((Y_y_idct, Cb_y_idct, Cr_y_idct), dim=1)
        
        #YCbCrZ转rgb
        x_out = self.ycbcr_to_rgb_jpeg(x_YCbCr)
        y_out = self.ycbcr_to_rgb_jpeg(y_YCbCr)
        
        x = x.cuda()
        y = y.cuda()
        
        return x_out, y_out, x_likelihoods, y_likelihoods, w_likelihoods
        
        
class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        #self.lmbda = lmbda #原本注释

    def forward(self, output, target, lmbda): #将原本lmbda改为canshu用以同时传送lmbda和alpha
        target1, target2 = target[0], target[1]
        #lmbda, alpha = canshu[0], canshu[1]
        #target1, target2 = self.target1, self.target2
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W
        
        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())  
        out['bpp_w'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods_w'][1].values()) 
                  
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2 +  out['bpp_w'] #原本
        
        out["transmited_bpp"] = out['bpp0'] #目标图片的压缩效果
        #out["bpp_loss"] = out["transmited_bpp"]
        out["mse0"] = self.mse(output['x_hat'][0], target1)
        out["mse1"] = self.mse(output['x_hat'][1], target2)
        out["transmited_mse"] = self.mse(output['x_hat'][0], target1) #目标图片的还原效果
        
        if isinstance(lmbda, list):
            out['mse_loss'] = (lmbda[0] * out["mse0"] + lmbda[1] * out["mse1"])/2 
        else:
            out['mse_loss'] = (out["mse0"] + out["mse1"])/2        #end to end   #原本
            #out['mse_loss'] = out["transmited_mse"]
        out['loss'] = out['mse_loss'] + lmbda * out['bpp_loss']  

        return out

class MS_SSIM_Loss(nn.Module):
    def __init__(self, device, size_average=True, max_val=1):
        super().__init__()
        self.ms_ssim = MS_SSIM(size_average, max_val).to(device)
        #self.lmbda = lmbda

    def forward(self, output, target, lmbda):
        target1, target2 = target[0], target[1]
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W

        # 计算误差
        out['bpp0'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][0].values())
        out['bpp1'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'][1].values())        
        out["bpp_loss"] = (out['bpp0'] + out['bpp1'])/2  #end to end  这是原本利用相关图片的
        #out["bpp_loss"] = out['bpp0']

        out["ms_ssim0"] = 1 - self.ms_ssim(output['x_hat'][0], target1)
        out["ms_ssim1"] = 1 - self.ms_ssim(output['x_hat'][1], target2)
 
        out['ms_ssim_loss'] = (out["ms_ssim0"] + out["ms_ssim1"])/2        #end to end  这是原本利用相关图片的
        #out['ms_ssim_loss'] = out["ms_ssim0"]
        out['loss'] = out['ms_ssim_loss'] + lmbda * out['bpp_loss']
        return out


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, max_val=255, device_id=0):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device_id = device_id

    def _ssim(self, img1, img2):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = create_window(window_size, sigma, self.channel)
        if self.device_id != None:
            window = window.cuda(self.device_id)

        mu1 = F.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        msssim=Variable(torch.Tensor(levels,))
        mcs=Variable(torch.Tensor(levels,))
        # if self.device_id != None:
        #     weight = weight.cuda(self.device_id)
        #     weight = msssim.cuda(self.device_id)
        #     weight = mcs.cuda(self.device_id)
        #     print(weight.device)

        for i in range(levels):
            ssim_map, mcs_map=self._ssim(img1, img2)
            msssim[i]=ssim_map
            mcs[i]=mcs_map
            filtered_im1=F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2=F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1=filtered_im1
            img2=filtered_im2

        value=(torch.prod(mcs[0:levels-1]**weight[0:levels-1]) *
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2, levels=5):
        return self.ms_ssim(img1, img2, levels)


if __name__ == '__main__':
    net = HyperPriorDistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
    net = DistributedAutoEncoder().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
