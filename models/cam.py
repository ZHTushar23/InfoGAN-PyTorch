import torch
import torch.nn as nn
from models.cbam import * 
from models.unet_parts import *
# from cbam import * 
# from unet_parts import *

class CAM(nn.Module):
    def __init__(self,in_channels,gate_channels,n_classes=1):
        super().__init__()
        bilinear=False
        self.inc = DoubleConv(in_channels, gate_channels//2)
        self.down1 = Down(gate_channels//2, gate_channels)
        self.down2 = Down(gate_channels, gate_channels*2)
        self.attn1 = CBAM(gate_channels=gate_channels//2)
        self.attn2 = CBAM(gate_channels=gate_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.up2 = Up(gate_channels*2, gate_channels, bilinear,dp=True)
        self.up1 = Up(gate_channels, gate_channels//2, bilinear,dp=True)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1a = self.attn1(x1)
        # x1a = self.dropout(x1a)
        x2 = self.down1(x1)
        x2a = self.attn2(x2)
        # x2a = self.dropout(x2a)
        x3 = self.down2(x2)
        # x3 = self.dropout(x3)
        x  = self.up2(x3,x2a)
        x = self.up1(x,x1a)
        x = self.outc(x)
        return x
    

    
def check_parameters(model):
    for param in model.parameters():
        if param.requires_grad:
            mean = torch.mean(param.data)
            std = torch.std(param.data)
            if mean != 0 or std != 1:
                return False

    return True

if __name__=="__main__":
    x = torch.rand(1, 1, 72, 72)
    model = CAM(in_channels=1,gate_channels=64)
    # print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))   
    print(model(x).shape)

    model = initialize_weights(model)

    if check_parameters(model):
        print("All parameters are initialized with zero mean and unit std.")
    else:
        print("Some parameters are not initialized with zero mean and unit std.")
    