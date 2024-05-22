import torch
import torch.nn as nn
from diffusion_unet_parts import *

def initialize_weights(model):
    # Iterate over the model's parameters and initialize them
    for param in model.parameters():
        nn.init.normal_(param, mean=0, std=1)
    return model

def initialize_weights_xavier(model):
    # Iterate over the model's parameters and initialize them
    for param in model.parameters():
        nn.init.xavier_normal_(param)
    return model
    
def check_parameters(model):
    for param in model.parameters():
        if param.requires_grad:
            mean = torch.mean(param.data)
            std = torch.std(param.data)
            if mean != 0 or std != 1:
                return False

    return True

class UNET(nn.Module):
    def __init__(self,in_channels,interim_channels=32):
        super().__init__()
        self.encoders = nn.ModuleList([
            # 5(Batch_Size, 1, height, width) -> (Batch_Size, interim_channels, height, width)
             SwitchSequentialI2(nn.Conv2d(in_channels, interim_channels, kernel_size=3, padding=1)),
            
            # 4(Batch_Size, interim_channels, height, width) -> # (Batch_Size, interim_channels, height, width) -> (Batch_Size, interim_channels, height, width)
             SwitchSequentialI2(UNET_AttentionBlock(4, 8) ),
            

            
            # 3(Batch_Size, interim_channels, height, width) -> (Batch_Size, interim_channels, Height / 2, Width / 2)
             SwitchSequentialI2(nn.Conv2d(interim_channels, interim_channels*2, kernel_size=3, stride=2, padding=1)),
            
            # 2(Batch_Size, interim_channels, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*2, Height / 2, Width / 2)
             SwitchSequentialI2( UNET_AttentionBlock(4, 16)),
            

            
            # 1(Batch_Size, interim_channels*2, Height / 2, Width / 2) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
             SwitchSequentialI2(nn.Conv2d(interim_channels*2, interim_channels*4, kernel_size=3, stride=2, padding=1)),
            
            # 0(Batch_Size, interim_channels*2, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
             SwitchSequentialI2(  UNET_AttentionBlock(4, 32)),
            
            # # (Batch_Size, interim_channels*8 Height / 4, Width / 4) -> (Batch_Size, interim_channels*8, Height / 8, Width / 8)
            #  SwitchSequentialI2(nn.Conv2d(interim_channels*4, interim_channels*8, kernel_size=3, stride=2, padding=1)),
            
        ])

        # (Batch_Size, interim_channels*4, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
        self.bottleneck =  SwitchSequentialI2(UNET_AttentionBlock(4, 32))


        self.decoders = nn.ModuleList([
            
            # 0(Batch_Size, interim_channels*8, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 4, Width / 4)
             SwitchSequentialI2(nn.Conv2d(interim_channels*8, interim_channels*4, kernel_size=3, padding=1)),

            # 1(Batch_Size, interim_channels*8, Height / 4, Width / 4) -> (Batch_Size, interim_channels*4, Height / 2, Width / 2)
             SwitchSequentialI2(nn.Conv2d(interim_channels*8, interim_channels*4, kernel_size=3, padding=1), Upsample(interim_channels*4)),


            # 2(Batch_Size, interim_channels*8, Height / 2, Width / 2) -> (Batch_Size, interim_channels*4, Height / 2, Width / 2)
             SwitchSequentialI2(nn.Conv2d(interim_channels*6, interim_channels*2, kernel_size=3, padding=1),UNET_AttentionBlock(4, 16)),

            # 3(Batch_Size, interim_channels*4, Height / 2, Width / 2) -> (Batch_Size, interim_channels*4, Height, Width )
             SwitchSequentialI2(nn.Conv2d(interim_channels*4, interim_channels*2, kernel_size=3, padding=1), Upsample(interim_channels*2)),

            # 4(Batch_Size, interim_channels*3, Height, Width) -> (Batch_Size, interim_channels, Height, Width)
             SwitchSequentialI2(nn.Conv2d(interim_channels*3, interim_channels, kernel_size=3, padding=1),UNET_AttentionBlock(4, 8) ),

            # 5(Batch_Size, interim_channels*2, Height, Width) -> (Batch_Size, interim_channels, Height, Width)
             SwitchSequentialI2(nn.Conv2d(interim_channels*2, interim_channels, kernel_size=3, padding=1),UNET_AttentionBlock(4, 8) ),
        ])

    def forward(self, x,  context):
        # x: (Batch_Size, 4, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, interim_channels*4)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x,  context)
            skip_connections.append(x)

        x = self.bottleneck(x, context)

        # k=1

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            # print("Decoder Layer: ",k, x.shape)
            # k=k+1

            x = layers(x,  context)  
                
        return x


class AttentionUNet(nn.Module):
    def __init__(self,in_channels,interim_channels,out_channels):
        super().__init__()
        self.unet = UNET(in_channels,interim_channels)
        self.final = UNET_OutputLayer(interim_channels, out_channels)
    
    def forward(self, latent, context):
        # latent: (Batch_Size, in_channels, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        
        # (Batch, in_channels, Height , Width) -> (Batch, interim_channels, Height, Width)
        output = self.unet(latent, context)
        
        # (Batch, interim_channels, Height, Width) -> (Batch,out_channels, Height, Width)
        output = self.final(output)
        
        # (Batch, out_channels, Height, Width)
        return output




if __name__=="__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")  

    context = torch.rand((1,1, 2, 100),device=device)
    x = torch.rand((1, 1, 72, 72),device=device)


    model = AttentionUNet(in_channels=1,interim_channels=32,out_channels=1).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))   

    print(model(x,context).shape)

    # model = initialize_weights(model)

    # if check_parameters(model):
    #     print("All parameters are initialized with zero mean and unit std.")
    # else:
    #     print("Some parameters are not initialized with zero mean and unit std.")
    