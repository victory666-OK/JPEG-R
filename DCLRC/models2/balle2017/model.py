import torch
from torch import nn
from models.balle2017 import entropy_model, gdn
from models.DiffJPEG import DiffJPEG
from compressai.entropy_models import EntropyBottleneck

class BLS2017Model(nn.Module):
    def __init__(self, num_filters=192):
        super(BLS2017Model, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 9, stride=4, padding=4)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters)

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, 3, 9, stride=4, padding=4, output_padding=3)
        
        self.jpeg = DiffJPEG(height=128, width=256, differentiable=True, quality=60)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv1_y = nn.Conv2d(1, num_filters//3, 5, stride=2, padding=2)
        self.conv1_cbcr = nn.Conv2d(2, num_filters*2//3, 3, stride=1, padding=1)
        self.gdn1_ycbcr = gdn.GDN(num_filters)
        self.igdn1_ycbcr = gdn.GDN(num_filters,inverse=True)
        self.deconv1_y = nn.ConvTranspose2d(num_filters//3, 1, 5, stride=2, padding=2, output_padding=1)
        self.deconv1_cbcr = nn.ConvTranspose2d(num_filters*2//3, 2, 3, stride=1, padding=1)
        
        self.test_entropy = EntropyBottleneck.compress

    def encode(self, x):
        #x = self.conv1(x)
        #x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn2(x)
        x = self.deconv2(x)
        #x = self.igdn3(x)
        #x = self.deconv3(x)
        return x

    def cfm(self,y,cb,cr):
        y = self.conv1_y(y)
        cbcr = torch.concat((cb,cr),1)
        cbcr = self.conv1_cbcr(cbcr)
        x = torch.concat((y,cbcr),1)
        x = self.gdn1_ycbcr(x)
        return x
    
    def cpsm(self,x):
        x = self.igdn1_ycbcr(x)
        y,cb,cr = torch.chunk(x, 3, dim=1)
        cbcr = torch.concat((cb,cr),1)
        y = self.deconv1_y(y)
        cbcr = self.deconv1_cbcr(cbcr)
        cb,cr = torch.chunk(cbcr, 2, dim=1)
        return y,cb,cr
    
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
    
    def forward(self,Y,Cb,Cr,img_ori,quantization,dimensions):
        x = self.cfm(Y,Cb,Cr)
        quantization = quantization.cuda()
        
        y = self.encode(x)
        out = EntropyBottleneck(192).compress(y.cpu())
        print(out)
        y_tilde, likelihoods = self.entropy_bottleneck(y)
        x_tilde = self.decode(y_tilde)
        
        y, cb, cr = self.cpsm(x_tilde)
        y_comp = self.block_splitting(y)
        cb_comp = self.block_splitting(cb)
        cr_comp = self.block_splitting(cr)
        x_out = self.jpeg(y_comp,cb_comp,cr_comp,quantization)
        
        return x_out, likelihoods


if __name__ == '__main__':
    net = BLS2017Model().cuda()
    print(net(torch.randn(1, 3, 256, 256).cuda())[0].shape)
