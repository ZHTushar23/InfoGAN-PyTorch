""" Full assembly of the parts to form the complete network """
import torch.nn as nn
from models.unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512,1024)

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear,dp=True)
        self.up2 = Up(512, 256 // factor, bilinear,dp=True)
        self.up3 = Up(256, 128 // factor, bilinear,dp=True)
        self.up4 = Up(128, 64, bilinear,dp=True)
        self.outc = OutConv(64, n_classes)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__=="__main__":
    model = UNet(n_channels=2,n_classes=2)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))
    x = torch.rand(5, 2, 64, 64)
    print(model(x).shape)
    print("Done!")