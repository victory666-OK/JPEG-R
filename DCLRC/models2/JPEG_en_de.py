import torch
from torch import nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import decompression
import compression

# 假设前面提到的各个函数已经定义，特别是：
# rgb_to_ycbcr_jpeg, compress_jpeg, decompress_jpeg, ycbcr_to_rgb_jpeg 等。

# PNG to RGB Conversion
def png_to_rgb(png_image_path):
    # 读取PNG图像并转换为RGB
    img = Image.open(png_image_path).convert('RGB')
    transform = transforms.ToTensor()  # 将图片转换为Tensor格式 (C, H, W)
    img_tensor = transform(img).unsqueeze(0)  # 增加批量维度 (1, C, H, W)
    return img_tensor

# Convert RGB to YCbCr and compress it to JPEG
def rgb_to_jpeg_compressed(img_tensor, factor=1):
    # 将RGB图像转换为YCbCr并进行JPEG压缩
    # img_tensor is a tensor of shape (batch, C, H, W)
    y, cb, cr = compress_jpeg(img_tensor, rounding=torch.round, factor=factor)
    return y, cb, cr

# Decompress JPEG and get RGB image back
def jpeg_to_rgb(y, cb, cr, height, width, factor=1):
    # 使用压缩的Y, Cb, Cr通道还原为RGB图像
    img_reconstructed = decompress_jpeg(y, cb, cr, height, width, rounding=torch.round, factor=factor)
    return img_reconstructed

# Save the image as PNG
def save_image_as_png(tensor, output_path):
    # 将Tensor转换为PIL图像并保存为PNG
    tensor = tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp values between [0, 1]
    transform = transforms.ToPILImage()  # To convert Tensor back to PIL Image
    img = transform(tensor)
    img.save(output_path)

# Main Function: PNG -> JPEG -> PNG (lossless roundtrip for demonstration)
def convert_png_to_jpeg_and_back(png_input_path, png_output_path, quality_factor=1):
    # Step 1: Load PNG image and convert to RGB Tensor
    rgb_tensor = png_to_rgb(png_input_path)
    height, width = rgb_tensor.shape[2], rgb_tensor.shape[3]  # Get the height and width of the image

    # Step 2: Compress the image to JPEG (Convert RGB -> YCbCr, Quantize, DCT)
    y, cb, cr = rgb_to_jpeg_compressed(rgb_tensor, factor=quality_factor)

    # Step 3: Decompress the JPEG and get the image back
    rgb_reconstructed = jpeg_to_rgb(y, cb, cr, height, width, factor=quality_factor)

    # Step 4: Save the decompressed image as PNG
    save_image_as_png(rgb_reconstructed, png_output_path)

    print(f"Image successfully converted from PNG -> JPEG -> PNG. Saved at {png_output_path}")

