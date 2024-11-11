import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution Blocks found in U-Net architecture
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # same convolution (padding=1)
            nn.BatchNorm2d(out_channels), # requires bias=False
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    """
    out_channels=1: if binary segmentation
    features=[64, 128, 256, 512]: intermediate channels
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down half of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottom level of UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up half of UNET
        for feature in reversed(features):
            # Up + skip connections (hence feature * 2), 2 convolutions
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
            in_channels = feature

        # Compress into result using conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        conns = [] # high to low resolution

        # Downwards 
        for down in self.downs:
            x = down(x)
            conns.append(x)
            self.pool(x)
        
        # Bottom
        x = self.bottleneck(x)
        conns = conns[::-1] # read it from low to high res

        # Upwards
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)

            # Get skip connection
            conn = conns[idx//2]

            # If input dims not divisible by 16:
            if x.shape != conn.shape:
                # resize (x is always equal to or smaller than conn)
                x = F.resize(x, size=conn.shape[2:]) # HW only

            # Concatenate
            concat = torch.cat((conn, x), dim=1) # BCHW, add along batch dim

            # Go right (double conv)
            x = self.ups[idx+1](concat)
        
        return self.final(x)
    

