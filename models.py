import torch
import torch.nn as nn
import numpy as np


class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class single_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.conv(x)
        return x


class Vox_Att(nn.Module):
    
    def __init__(self, in_ch):
        super(Vox_Att, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, int(in_ch/2), kernel_size=1, padding=0),
            nn.GroupNorm(8, int(in_ch/2)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(in_ch/2), int(in_ch/2), kernel_size=3, padding=5, dilation=5),
            nn.GroupNorm(8, int(in_ch/2)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(in_ch/2), in_ch, kernel_size=1, padding=0),
            nn.GroupNorm(8, in_ch),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sigmoid()
        
    def forward(self, in_x):
        
        x = self.conv1(in_x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.output(x)
        
        return in_x * x + in_x

class UNet0(nn.Module):

    def __init__(self):
        super(UNet0, self).__init__()

        self.Dconv1 = double_conv(1, 16)
        self.maxpool1 = nn.MaxPool2d(2)

        self.Dconv2 = double_conv(16, 32)
        self.maxpool2 = nn.MaxPool2d(2)

        self.Dconv3 = double_conv(32, 64)
        self.maxpool3 = nn.MaxPool2d(2)

        self.Dconv4 = double_conv(64, 128)
        self.maxpool4 = nn.MaxPool2d(2)

        self.Dconv5 = double_conv(128, 256)

        self.Tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.Dconv6 = double_conv(256, 128)

        self.Tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.Dconv7 = double_conv(128, 64)

        self.Tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.Dconv8 = double_conv(64, 32)

        self.Tconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.Dconv9 = double_conv(32, 16)

        self.Fconv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x):

        c1 = self.Dconv1(x)  # 16
        p1 = self.maxpool1(c1)

        c2 = self.Dconv2(p1)  # 32
        p2 = self.maxpool2(c2)

        c3 = self.Dconv3(p2)  # 64
        p3 = self.maxpool3(c3)

        c4 = self.Dconv4(p3)  # 128
        p4 = self.maxpool4(c4)

        c5 = self.Dconv5(p4)  # 256

        t1 = self.Tconv1(c5)  # 128
        x1 = torch.cat([c4, t1], dim=1)  # 128 + 128
        c6 = self.Dconv6(x1)  # 128

        t2 = self.Tconv2(c6)    # 64
        x2 = torch.cat([c3, t2], dim=1)  # 64 + 64
        c7 = self.Dconv7(x2)  # 64

        t3 = self.Tconv3(c7)    # 32
        x3 = torch.cat([c2, t3], dim=1)  # 32 + 32
        c8 = self.Dconv8(x3)  # 32

        t4 = self.Tconv4(c8)   # 16
        x4 = torch.cat([c1, t4], dim=1)  # 16 + 16
        c9 = self.Dconv9(x4)  # 16

        x = self.Fconv(c9)

        output = self.output(x)

        return output


class UNet1(nn.Module):

    def __init__(self):
        super(UNet1, self).__init__()
        self.Dconv1 = double_conv(1, 16)
        self.maxpool1 = nn.MaxPool2d(2)

        self.Dconv2 = double_conv(16, 32)
        self.maxpool2 = nn.MaxPool2d(2)

        self.Dconv3 = double_conv(32, 64)
        self.maxpool3 = nn.MaxPool2d(2)

        self.Dconv4 = double_conv(64, 128)
        self.maxpool4 = nn.MaxPool2d(2)

        self.Dconv5 = double_conv(128, 256)

        self.Tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.Vatt1 = Vox_Att(256)
        self.Dconv6 = double_conv(256, 128)

        self.Tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.Vatt2 = Vox_Att(128)
        self.Dconv7 = double_conv(128, 64)

        self.Tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.Vatt3 = Vox_Att(64)
        self.Dconv8 = double_conv(64, 32)

        self.Tconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.Vatt4 = Vox_Att(32)
        self.Dconv9 = double_conv(32, 16)

        self.Fconv = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()


    def forward(self, x):

        c1 = self.Dconv1(x)  # 16
        p1 = self.maxpool1(c1)

        c2 = self.Dconv2(p1)  # 32
        p2 = self.maxpool2(c2)

        c3 = self.Dconv3(p2)  # 64
        p3 = self.maxpool3(c3)

        c4 = self.Dconv4(p3)  # 128
        p4 = self.maxpool4(c4)

        c5 = self.Dconv5(p4)  # 256

        t1 = self.Tconv1(c5)  # 128
        x1 = torch.cat([c4, t1], dim=1)  # 128 + 128
        x1 = self.Vatt1(x1)
        c6 = self.Dconv6(x1)  # 128

        t2 = self.Tconv2(c6)    # 64
        x2 = torch.cat([c3, t2], dim=1)  # 64 + 64
        x2 = self.Vatt2(x2)
        c7 = self.Dconv7(x2)  # 64

        t3 = self.Tconv3(c7)    # 32
        x3 = torch.cat([c2, t3], dim=1)  # 32 + 32
        x3 = self.Vatt3(x3)
        c8 = self.Dconv8(x3)  # 32

        t4 = self.Tconv4(c8)   # 16
        x4 = torch.cat([c1, t4], dim=1)  # 16 + 16
        x4 = self.Vatt4(x4)
        c9 = self.Dconv9(x4)  # 16

        x = self.Fconv(c9)

        output = self.output(x)

        return output




class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, params):
        super(UNet2D, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p = 0.0) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p = 0.0) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p = 0.0) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p = 0.0) 
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size = 3, padding = 1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.out_conv(x)
        output = nn.Sigmoid()(x)

        return output









    

    