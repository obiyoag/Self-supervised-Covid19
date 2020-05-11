import torch
import torch.nn as nn


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
