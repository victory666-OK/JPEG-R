import argparse
import math
import random
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchjpeg

from torch.utils.data import DataLoader
from torchvision import transforms

from models.ldmic import *
#from models.dclrc import * #后面加的！！！
from lib.dclrc_utils import get_output_folder, AverageMeter, save_checkpoint, StereoImageDataset
import numpy as np

from PIL import Image
import yaml
import wandb
import os
from tqdm import tqdm
from pytorch_msssim import ms_ssim
os.environ["WANDB_API_KEY"] = "70fb0a09c73d46772536e8e0de96abdc16229c96" # write your own wandb id

def compute_aux_loss(aux_list: List, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
        if backward is True:
            aux_loss.backward()

    return aux_loss_sum

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(p for p in net.named_parameters() if p[1].requires_grad)
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=args.learning_rate,
    )
    return optimizer, aux_optimizer

'''def dct_to_img(recon, quantization, dimensions):
    
    coefficients_s = recon.to(device='cpu')
    
    #coefficients = coefficients.detach().cpu().numpy() #后加
    
    original_shape = (1, int(coefficients_s.shape[2] / 8), int(coefficients_s.shape[3] / 8), 8, 8)
    
    a = torch.empty((16, 3, 128, 256)) #作为返回的张量 后加

    for index in range(coefficients_s.size(0)): #后加
    
        #coefficients = coefficients.squeeze(0) #去掉第一个维度 ，这里注释掉，因为下一句已经将第一位维度消掉
        coefficients =  coefficients_s[index] #后加
        print(f"coefficients.shape：{coefficients.shape}")
    
        Y_coefficients = coefficients[0:1]
        
        Y_coefficients = np.reshape(Y_coefficients.detach().numpy(), (
        original_shape[0], original_shape[1], original_shape[3], original_shape[2], original_shape[4]))
       
        Y_coefficients = np.transpose(Y_coefficients, (0, 1, 3, 2, 4))
        print(f"Y_coefficients.shape：{Y_coefficients.shape}")
    
        Cb_coefficients = coefficients[1:2]
        Cb_coefficients = Cb_coefficients.float()
        Cb_coefficients = Cb_coefficients.unsqueeze(0)
        Cb_coefficients = F.interpolate(Cb_coefficients,
                                        size=(int(Cb_coefficients.shape[2] / 2), int(Cb_coefficients.shape[3] / 2)),
                                        mode='nearest')
        Cb_coefficients = Cb_coefficients.squeeze(0).squeeze(0)
        Cb_coefficients = Cb_coefficients.to(torch.int16)
        Cb_coefficients = np.reshape(Cb_coefficients, (
        original_shape[0], int(original_shape[1] / 2), original_shape[3], int(original_shape[2] / 2), original_shape[4]))
        Cb_coefficients = np.transpose(Cb_coefficients, (0, 1, 3, 2, 4))
    
        Cr_coefficients = coefficients[2:3]
        Cr_coefficients = Cr_coefficients.float()
        Cr_coefficients = Cr_coefficients.unsqueeze(0)
        Cr_coefficients = F.interpolate(Cr_coefficients,
                                        size=(int(Cr_coefficients.shape[2] / 2), int(Cr_coefficients.shape[3] / 2)),
                                        mode='nearest')
        Cr_coefficients = Cr_coefficients.squeeze(0).squeeze(0)
        Cr_coefficients = Cr_coefficients.to(torch.int16) 
        Cr_coefficients = np.reshape(Cr_coefficients, (
        original_shape[0], int(original_shape[1] / 2), original_shape[3], int(original_shape[2] / 2), original_shape[4]))
        Cr_coefficients = np.transpose(Cr_coefficients, (0, 1, 3, 2, 4))
        
        #Cb_coefficients, Cr_coefficients = torch.from_numpy(Cb_coefficients), torch.from_numpy(Cr_coefficients) #后加
        #Y_coefficients = torch.from_numpy(Y_coefficients) #后加
    
        print(f"Cb_coefficients.shape：{Cb_coefficients.shape}")
        print(f"Cr_coefficients.shape：{Cr_coefficients.shape}")
        CbCr_coefficients = torch.cat((Cb_coefficients, Cr_coefficients), dim=0)
        print(f"CbCr_coefficients.shape：{CbCr_coefficients.shape}")
    
        spatial = torchjpeg.codec.reconstruct_full_image(torch.from_numpy(Y_coefficients), quantization, CbCr_coefficients, dimensions)
        print(f"spatial.shape：{spatial.shape}")
        
        a[index] = spatial

    return a'''
    
    
def dct_to_img(recon, quantization, dimensions):
    
    coefficients = recon.to(device='cpu')
    
    #a = torch.empty((1, 3, 128, 256)) #作为返回的张量 后加

    original_shape = (1, int(coefficients.shape[2] / 8), int(coefficients.shape[3] / 8), 8, 8)
 
    
    coefficients = coefficients.squeeze(0) 

    Y_coefficients = coefficients[0:1]
    
    Y_coefficients = np.reshape(Y_coefficients.detach().numpy(), (
    original_shape[0], original_shape[1], original_shape[3], original_shape[2], original_shape[4]))
   
    Y_coefficients = np.transpose(Y_coefficients, (0, 1, 3, 2, 4))

    Cb_coefficients = coefficients[1:2]
    Cb_coefficients = Cb_coefficients.float()
    Cb_coefficients = Cb_coefficients.unsqueeze(0)
    Cb_coefficients = F.interpolate(Cb_coefficients,
                                    size=(int(Cb_coefficients.shape[2] / 2), int(Cb_coefficients.shape[3] / 2)),
                                    mode='nearest')
    Cb_coefficients = Cb_coefficients.squeeze(0).squeeze(0)
    Cb_coefficients = Cb_coefficients.to(torch.int16)
    Cb_coefficients = np.reshape(Cb_coefficients, (
    original_shape[0], int(original_shape[1] / 2), original_shape[3], int(original_shape[2] / 2), original_shape[4]))
    Cb_coefficients = np.transpose(Cb_coefficients, (0, 1, 3, 2, 4))

    Cr_coefficients = coefficients[2:3]
    Cr_coefficients = Cr_coefficients.float()
    Cr_coefficients = Cr_coefficients.unsqueeze(0)
    Cr_coefficients = F.interpolate(Cr_coefficients,
                                    size=(int(Cr_coefficients.shape[2] / 2), int(Cr_coefficients.shape[3] / 2)),
                                    mode='nearest')
    Cr_coefficients = Cr_coefficients.squeeze(0).squeeze(0)
    Cr_coefficients = Cr_coefficients.to(torch.int16) 
    Cr_coefficients = np.reshape(Cr_coefficients, (
    original_shape[0], int(original_shape[1] / 2), original_shape[3], int(original_shape[2] / 2), original_shape[4]))
    Cr_coefficients = np.transpose(Cr_coefficients, (0, 1, 3, 2, 4))
    

    CbCr_coefficients = torch.cat((Cb_coefficients, Cr_coefficients), dim=0)
    spatial = torchjpeg.codec.reconstruct_full_image(torch.from_numpy(Y_coefficients), quantization, CbCr_coefficients, dimensions)
    
    #a[0] = spatial
    
    return spatial
    

def get_distortion(args, out, img, cor_img, mse): #从ndlrc模拟过来
    distortion = None
    alpha = 1 #这里本来需要config超参数定义，我先设置为1
    x_recon, y_recon = out[0], out[1]
    if args.metric == "mse":
        distortion = mse(img, x_recon)
        distortion += alpha * mse(cor_img, y_recon)
    else:
        distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
        distortion += alpha * (1 - ms_ssim(cor_img.cpu(), y_recon.cpu(), data_range=1.0, size_average=True,
                                               win_size=7))

    return distortion
    
#后加开始
def idct_2d(block, norm='ortho'):
    def idct_1d(tensor, norm='ortho'):
        x = torch.fft.irfft(torch.fft.rfft(tensor, n=8, dim=-1, norm=norm), n=8, dim=-1, norm=norm)
        return x
    return idct_1d(idct_1d(block, norm).transpose(-2, -1), norm).transpose(-2, -1)
    
def idct(tensor):
    blocks = tensor.view(tensor.shape[0], tensor.shape[1], tensor.shape[2] // 8, 8, tensor.shape[3] // 8, 8)
    # 应用 2D IDCT 到每个块
    idct_blocks = torch.zeros_like(blocks)
    for i in range(blocks.shape[2]):
        for j in range(blocks.shape[4]):
            block = blocks[:, :, i, :, j, :]
            idct_blocks[:, :, i, :, j, :] = idct_2d(block)
    result_tensor = idct_blocks.view(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])
    return result_tensor
    
def ycbcr_to_rgb_jpeg(image):
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
    
def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().detach().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().detach().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
    if not os.path.exists(path):
        os.makedirs(path)
    img_final.save(os.path.join(path, name + '.png'))

def train_one_epoch(model, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"
    #transmited_mse_loss = AverageMeter(transmited_mse, ':.4e')#记录例如mse或是ms_ssim
    #transmited_mse_loss_h = AverageMeter(transmited_mse, ':.4e')

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    #transmited_bpp_loss = AverageMeter('Transmited_BppLoss', ':.4e')

    train_dataloader = tqdm(train_dataloader)
    print('Train epoch:', epoch)
    #接下来大量引入ndlrc
    for i, batch in enumerate(train_dataloader):
    
        #image_list, img, cor_img, img_ori,img_side_ori,dimensions,quantization = batch 原版！！！
        
        img, cor_img, _, _,img_ori,img_side_ori,dimensions,quantization = batch
        #print(f"dim {dimensions}")
        #print(f"qua {quantization}")
        #print(f"img_ori {img_ori.mode}")
        dimensions = dimensions.squeeze(0) #原本无
        quantization = quantization.squeeze(0) #原本无
        
        
        #d = [frame.to(device) for frame in batch]
        #dimensions = dimensions.squeeze(0)
        #quantization = quantization.squeeze(0)

        #将数据转移到GPU，顺便转为float型
        img = img.cuda().float() if args.cuda else img.float() # 后加！！！
        cor_img = cor_img.cuda().float() if args.cuda else cor_img.float() # 后加！！！
        img_ori = img_ori.cuda().float() if args.cuda else img.float() # 后加！！！
        img_side_ori = img_side_ori.cuda().float() if args.cuda else img.float() # 后加！！！
        
        
        d = torch.stack([img, cor_img], dim=0).to(torch.float32)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()
        #aux_optimizer.zero_grad()

        d.float()
        
        out_net = model(d)
        
        '''out_net['x_hat'][0] = dct_to_img(out_net['x_hat'][0], quantization, dimensions)
        out_net['x_hat'][1] = dct_to_img(out_net['x_hat'][1], quantization, dimensions)
        out_net['x_hat'][0] = out_net['x_hat'][0].unsqueeze(0).cuda()
        out_net['x_hat'][1] = out_net['x_hat'][1].unsqueeze(0).cuda()
        img_ori = dct_to_img(img_ori, quantization, dimensions)
        img_side_ori = dct_to_img(img_side_ori, quantization, dimensions)
        img_ori = img_ori.unsqueeze(0).cuda()
        img_side_ori = img_side_ori.unsqueeze(0).cuda()'''
        
        '''#将参照图片转为RGB
        #img_ori = torch.from_numpy(img_ori)
        #img_side_ori = torch.from_numpy(img_side_ori)
        #通道划分
        Y_x_tilde  = img_ori[:, 0:1, :, :]
        Cb_x_tilde = img_ori[:, 1:2, :, :]
        Cr_x_tilde = img_ori[:, 2:3, :, :]
        Y_y_tilde  = img_side_ori[:, 0:1, :, :]
        Cb_y_tilde = img_side_ori[:, 1:2, :, :]
        Cr_y_tilde = img_side_ori[:, 2:3, :, :]
        
        
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
        Y_x_idct  = idct(Y_x_tilde)
        Cb_x_idct = idct(Cb_x_tilde)
        Cr_x_idct = idct(Cr_x_tilde)
        Y_y_idct  = idct(Y_y_tilde)
        Cb_y_idct = idct(Cb_y_tilde)
        Cr_y_idct = idct(Cr_y_tilde)
        
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
        x_out = ycbcr_to_rgb_jpeg(x_YCbCr)
        y_out = ycbcr_to_rgb_jpeg(y_YCbCr)
        
        #x_out = torch.min(255 * torch.ones_like(x_out), torch.max(torch.zeros_like(x_out), x_out)) / 255
        #y_out = torch.min(255 * torch.ones_like(y_out), torch.max(torch.zeros_like(y_out), y_out)) / 255
        img_ori, img_side_ori = x_out, y_out
        #转换结束'''
        
        #确定损失函数***    原本在main中
        if args.metric == "mse":
            criterion = MSE_Loss() #原本MSE_Loss()
        else:
            criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lambda)
        
        out_criterion = criterion(out_net, (img_ori, img_side_ori), args.lmbda) #设置类似损失函数的对象，  注意这里的d

        out_criterion["loss"].backward()#反向传播计算模型个参数梯度，用于后续优化器利用该梯度优化神经网络模型
        
        if clip_max_norm > 0:#防止梯度爆炸，对超过设定的梯度做减少
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()#优化器优化神经网络参数

        if aux_optimizer is not None:#如果存在辅助优化器，则计算辅助损失，并执行反向传播       参数更新
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
            aux_optimizer.step()
        else:#如果没有辅助优化器，则计算辅助损失但不执行反向传播   不更新参数
            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
        

        print(out_criterion["loss"])
        loss.update(out_criterion["loss"].item())#更新总损失：将当前损失值添加到总损失的平均计算中
        bpp_loss.update((out_criterion["bpp_loss"]).item())
        aux_loss.update(out_aux_loss.item())
        metric_loss.update(out_criterion[metric_name].item())
        left_bpp.update(out_criterion["bpp0"].item())
        right_bpp.update(out_criterion["bpp1"].item())
        #transmited_bpp_loss.update(out_criterion["transmited_bpp"].item())
        #transmited_mse_loss.update(out_criterion["transmited_mse"].item())

        if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
            left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
            right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
            left_db.update(left_metric)
            right_db.update(right_metric)
            metric_dB.update((left_metric+right_metric)/2) #这个是两张图片的psnr
            #transmited_mse_loss = 10 * (torch.log10(1 / out_criterion["transmited_mse"])).mean().item()
            

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        '''train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
            metric_dB_name:metric_dB.avg, "transmited_bpp":left_bpp.avg, "transmited_psnr":left_db.avg})'''
        train_dataloader.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
            metric_dB_name:metric_dB.avg})
            
        #print(out_net["x_hat"][0].shape)
        #print(img_ori.shape)
        
        '''for index in range(out_net["x_hat"][0].size(0)):
            x_recon = out_net["x_hat"][0][index]
            x_ori = img_ori[index]
            save_image(x_recon, x_ori, os.path.join("/home/whut4/Zhangbenyi/DCLRC-main/output/images_output", str(i)),
                               str(index))'''

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    return out

def test_epoch(epoch, val_dataloader, model, args):
    model.eval()
    device = next(model.parameters()).device

    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    left_db, right_db = AverageMeter(left_db_name, ':.4e'), AverageMeter(right_db_name, ':.4e')
    metric0, metric1 = args.metric+"0", args.metric+"1"

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')
    left_bpp, right_bpp = AverageMeter('LBpp', ':.4e'), AverageMeter('RBpp', ':.4e')
    loop = tqdm(val_dataloader)

    with torch.no_grad():
        for i, batch in enumerate(loop):
            #d = [frame.to(device) for frame in batch] 原本***
            
            '''image_list, img, cor_img, img_ori,img_side_ori,dimensions,quantization = batch
            d = image_list'''
            
            #image_list, img, cor_img, img_ori,img_side_ori,dimensions,quantization = batch 原版！！！
            
            img, cor_img, _, _,img_ori,img_side_ori,dimensions,quantization = batch
            dimensions = dimensions.squeeze(0) #原本无
            quantization = quantization.squeeze(0) #原本无
        
        
            #d = [frame.to(device) for frame in batch]
            #dimensions = dimensions.squeeze(0)
            #quantization = quantization.squeeze(0)

            img = img.cuda().float() if args.cuda else img.float() # 后加！！！
            cor_img = cor_img.cuda().float() if args.cuda else cor_img.float() # 后加！！！
        
            d = torch.stack([img, cor_img], dim=0).to(torch.float32)
            d.float()

            img_ori = img_ori.cuda().float() if args.cuda else img.float() # 后加！！！
            img_side_ori = img_side_ori.cuda().float() if args.cuda else img.float() # 后加！！！
            
            out_net = model(d)
        
        
        
            #将参照图片转为RGB
            #img_ori = torch.from_numpy(img_ori)
            #img_side_ori = torch.from_numpy(img_side_ori)
            
            #首先将img_ori和img_side_ori从JPE图片转换为DCT系数
            
            
            #接下来将DCT系数转为RGB图片
            #通道划分
            Y_x_tilde  = img_ori[:, 0:1, :, :]
            Cb_x_tilde = img_ori[:, 1:2, :, :]
            Cr_x_tilde = img_ori[:, 2:3, :, :]
            Y_y_tilde  = img_side_ori[:, 0:1, :, :]
            Cb_y_tilde = img_side_ori[:, 1:2, :, :]
            Cr_y_tilde = img_side_ori[:, 2:3, :, :]
            
            
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
            Y_x_idct  = idct(Y_x_tilde)
            Cb_x_idct = idct(Cb_x_tilde)
            Cr_x_idct = idct(Cr_x_tilde)
            Y_y_idct  = idct(Y_y_tilde)
            Cb_y_idct = idct(Cb_y_tilde)
            Cr_y_idct = idct(Cr_y_tilde)
            
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
            x_out = ycbcr_to_rgb_jpeg(x_YCbCr)
            y_out = ycbcr_to_rgb_jpeg(y_YCbCr)
            
            #x_out = torch.min(255 * torch.ones_like(x_out), torch.max(torch.zeros_like(x_out), x_out)) / 255
            #y_out = torch.min(255 * torch.ones_like(y_out), torch.max(torch.zeros_like(y_out), y_out)) / 255
            img_ori, img_side_ori = x_out, y_out
            #转换结束
              
            #确定损失函数***    原本在main中
            if args.metric == "mse":
                criterion = MSE_Loss() #原本MSE_Loss()
            else:
                criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lambda)
                
            #out_criterion = criterion(out_net, (img_ori, img_side_ori), args.lmbda) #设置类似损失函数的对象，  注意这里的d
            
            out_criterion = criterion(out_net, (img_ori, img_side_ori), args.lmbda)

            out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
        
            left_bpp.update(out_criterion["bpp0"].item())
            right_bpp.update(out_criterion["bpp1"].item())

            if out_criterion[metric0] > 0 and out_criterion[metric1] > 0:
                left_metric = 10 * (torch.log10(1 / out_criterion[metric0])).mean().item()
                right_metric = 10 * (torch.log10(1 / out_criterion[metric1])).mean().item()
                left_db.update(left_metric)
                right_db.update(right_metric)
                metric_dB.update((left_metric+right_metric)/2)

            loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
            '''loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg, "transmited_bpp":left_bpp.avg, "transmited_psnr":left_db.avg})'''
            loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg,
                metric_dB_name:metric_dB.avg})
            
            
            for index in range(out_net["x_hat"][0].size(0)):
                x_recon = out_net["x_hat"][0][index]
                x_ori = img_ori[index]
                save_image(x_recon, x_ori, os.path.join("/home/whut4/Zhangbenyi/DCLRC-main/output/images_output", str(i)),
                                   str(index))

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "bpp_loss": bpp_loss.avg, 
            "aux_loss":aux_loss.avg, metric_dB_name: metric_dB.avg, "left_bpp": left_bpp.avg, "right_bpp": right_bpp.avg,
            left_db_name:left_db.avg, right_db_name: right_db.avg,}

    return out

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/whut4/Zhangbenyi/DCLRC-main/dataset/', help="Training dataset"
    )
    parser.add_argument(
        "--data-name", type=str, default='kitti', help="Training dataset"
    )
    parser.add_argument(
        "--model-name", type=str, default='LDMIC', help="Training dataset"
    )

    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=40e-4, #2024
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)" #原本为16
    )
    parser.add_argument(
        "--epochs", type=int, default=61, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64, #原本为64
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", help="Save model to disk"
    )
    parser.add_argument(
        "--resize", default = (128, 256), action="store_true", help="training use resize or randomcrop"
    ) #在命令后指定--save则会保存权重
    parser.add_argument(
        "--seed", type=float, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--i_model_path", type=str, default = "/home/whut4/Zhangbenyi/DCLRC-main/output/weight/DistributedAutoEncoder_KITTI_stereo_MSE_11.pt", help="Path to a checkpoint")
    #parser.add_argument("--i_model_path", type=str, help="Path to a checkpoint")
    parser.add_argument("--metric", type=str, default="mse", help="metric: mse, ms_ssim")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        #default=7.5e-5,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    #获取指令
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    #保证研究结果公平性、可重复性
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    #获取数据集***   这里先只考虑kitti
    resize = tuple(args.resize)
    path = "/home/whut4/Zhangbenyi/DCLRC-main/dataset"
    # Warning, the order of the transform composition should be kept.
    train_dataset = StereoImageDataset(ds_type='train', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)
    test_dataset = StereoImageDataset(ds_type='test', ds_name=args.data_name, root=args.dataset, crop_size=args.patch_size, resize=args.resize)
    
    

    #数据集迭代***
    #train_dataset和test_dataset都只是StereoImageDataset类型对象，其成员变量所包含的是左右图片的路径，同时他们有成员函数，接下来DataLoader一定会调用train_dataset和test_dataset中的成员函数
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    #确定网络模型***
    #一般来说下面会比上面效果好
    if args.model_name == "LDMIC":
        net = LDMIC(N=192, M=320, decode_atten=JointContextTransfer) #M原本也为192
    elif args.model_name == "LDMIC_checkboard":
        net = LDMIC_checkboard(N=192, M=192, decode_atten=JointContextTransfer)
    net = net.to(device) 

    #确定优化器***
    optimizer, aux_optimizer = configure_optimizers(net, args)
    #学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 30, 45, 60], 0.5) #原本为0.5 #optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    '''#确定损失函数***
    if args.metric == "mse":
        criterion = MSE_Loss() #原本MSE_Loss()
    else:
        criterion = MS_SSIM_Loss(device) #(device, lmbda=args.lambda)''' #先将这部分移动到训练函数中

    #保存之前训练的状态
    last_epoch = 0
    best_loss = float("inf")
    '''if args.i_model_path:  #load from previous checkpoint
        print("Loading model: ", args.i_model_path)
        checkpoint = torch.load(args.i_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])   
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        #optimizer.param_groups[0]['lr'] = 0.01 #后加的 ！！！
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        #best_b_model_path = os.path.join(os.path.split(args.i_model_path)[0], 'ckpt.best.pth.tar')
        #best_loss = torch.load(best_b_model_path)["loss"]'''
        

    #为训练过程设置日志目录、实验ID以及显示名称和标签，以便组织和标识实验结果
    log_dir, experiment_id = get_output_folder('./checkpoints/{}/{}/{}/lamda{}/'.format(args.data_name, args.metric, args.model_name, int(args.lmbda)), 'train')
    display_name = "{}_{}_lmbda{}".format(args.model_name, args.metric, int(args.lmbda))
    tags = "lmbda{}".format(args.lmbda)

    #将命令行参数写入yaml文件
    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    #实验中进行可视化监控、追踪和记录
    project_name = "DSIC_" + args.data_name
    wandb.init(project=project_name, name=display_name, tags=[tags],) #notes="lmbda{}".format(args.lmbda))
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    wandb.config.update(args) # config is a variable that holds and saves hyper parameters and inputs

    #根据损失类型确定使用哪一组性能度量指标名称
    if args.metric == "mse":
        metric_dB_name, left_db_name, right_db_name = 'psnr', "left_PSNR", "right_PSNR"
        metric_name = "mse_loss" 
    else:
        metric_dB_name, left_db_name, right_db_name = "ms_db", "left_ms_db", "right_ms_db"
        metric_name = "ms_ssim_loss"

    #开始训练
    #val_loss = test_epoch(0, test_dataloader, net, criterion, args)
    for epoch in range(last_epoch, args.epochs):
        #adjust_learning_rate(optimizer, aux_optimizer, epoch, args)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss = train_one_epoch(net, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args)
        lr_scheduler.step()#根据预先制定策略调整学习率

        wandb.log({"train": {"loss": train_loss["loss"], metric_name: train_loss[metric_name], "bpp_loss": train_loss["bpp_loss"],
            "aux_loss": train_loss["aux_loss"], metric_dB_name: train_loss[metric_dB_name], "left_bpp": train_loss["left_bpp"], "right_bpp": train_loss["right_bpp"],
            left_db_name:train_loss[left_db_name], right_db_name: train_loss[right_db_name]}, }
        )
        if epoch%10==0 and epoch!=0:
            val_loss = test_epoch(epoch, test_dataloader, net, args)
            wandb.log({ 
                "test": {"loss": val_loss["loss"], metric_name: val_loss[metric_name], "bpp_loss": val_loss["bpp_loss"],
                "aux_loss": val_loss["aux_loss"], metric_dB_name: val_loss[metric_dB_name], "left_bpp": val_loss["left_bpp"], "right_bpp": val_loss["right_bpp"],
                left_db_name:val_loss[left_db_name], right_db_name: val_loss[right_db_name],}
                })
        
            loss = val_loss["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False
        '''if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                is_best, log_dir
            )'''
        if args.save:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                args.i_model_path
            )

if __name__ == "__main__":
    main(sys.argv[1:])