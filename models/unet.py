import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    # in each stage of downsampling, two convolution layers are applied in a layer
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x
    
class DownSample(nn.Module):
    """
        Downsampling after double_conv, then approach next stage of double_conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_doubleconv(x)
        return x

class UpSample(nn.Module):
    """
        upsample and double_conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # forward
        x1 = self.up(x1)
        
        # Skip Connection
        # input (c, H, W)
        # calculate the dimension difference between downsample feature map
        # and up sample feature map (we want to concat to for skip connection)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        
        # padding to make the feature map have same size
        # [left, right, top, bottom] side of the tensor
        paddings = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        x1 = F.pad(x1, paddings)    
        
        # Concatenate along the channel dimension (dimension 1)
        x = torch.cat((x1, x2), dim=1)
        
        # forward
        x = self.double_conv(x)
        
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.in_conv = DoubleConv(n_channels, 64)
        self.dowsample1 = DownSample(64, 128)
        self.dowsample2 = DownSample(128, 256)
        self.dowsample3 = DownSample(256, 512)
        self.dowsample4 = DownSample(512, 1024)
        
        self.upsample1 = UpSample(1024, 512)
        self.upsample2 = UpSample(512, 256)
        self.upsample3 = UpSample(256, 128)
        self.upsample4 = UpSample(128, 64)
        self.out_conv = OutConv(64, n_classes)
        
    def forward(self, x):
        # Encoder
        x1 = self.in_conv(x)
        x2 = self.dowsample1(x1)
        x3 = self.dowsample2(x2)
        x4 = self.dowsample3(x3)
        x5 = self.dowsample4(x4)
        # Decoder
        x = self.upsample1(x5, x4)
        x = self.upsample2(x, x3)
        x = self.upsample3(x, x2)
        x = self.upsample4(x, x1)
        logits = self.out_conv(x)
        
        return logits